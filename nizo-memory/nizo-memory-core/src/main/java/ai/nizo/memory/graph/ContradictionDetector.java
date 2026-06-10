package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Detects and resolves contradictions in the knowledge graph.
 *
 * <p>Contradictions arise when a new edge or property conflicts with an
 * existing one -- for example, "works_at Company-A" vs "works_at Company-B".
 * Resolution follows a priority hierarchy: user-stated facts trump
 * conversation-inferred facts, which trump extracted facts. Equal priority
 * is resolved by recency (newer wins).
 *
 * <p>Also provides a health scan that identifies duplicate nodes, conflicting
 * edges, and stale data.
 *
 * <p>De-Springified port from Kimaya's {@code ContradictionDetector}. Plain
 * Java 21, no framework dependencies.
 */
public final class ContradictionDetector {

    private static final Logger LOG = Logger.getLogger(ContradictionDetector.class.getName());
    private static final Duration STALE_THRESHOLD = Duration.ofDays(90);

    private final SqliteGraphStore store;

    public ContradictionDetector(SqliteGraphStore store) {
        this.store = Objects.requireNonNull(store, "store");
    }

    // ===== Result records ===================================================

    /**
     * Outcome of a contradiction check between a proposed edge and existing edges.
     *
     * @param hasContradiction whether a contradiction was found
     * @param type             contradiction category (e.g. "relationship_conflict")
     * @param description      human-readable explanation
     * @param resolution       how it was resolved ("kept_existing", "kept_new", or null)
     * @param invalidatedEdgeId id of the edge that was invalidated, or null
     */
    public record ContradictionResult(
            boolean hasContradiction,
            String type,
            String description,
            String resolution,
            String invalidatedEdgeId
    ) {
        /** Factory for the no-contradiction case. */
        public static ContradictionResult none() {
            return new ContradictionResult(false, null, null, null, null);
        }
    }

    /**
     * A single property-level conflict between old and new values.
     *
     * @param key        property name
     * @param oldValue   existing value
     * @param newValue   proposed value
     * @param resolution how it was resolved ("kept_existing" or "kept_new")
     */
    public record PropertyConflict(String key, Object oldValue, Object newValue, String resolution) { }

    /**
     * Summary of graph health issues found during a scan.
     *
     * @param totalNodes       total accessible nodes
     * @param totalEdges       total current edges
     * @param duplicateNodes   nodes sharing a label across different categories
     * @param conflictingEdges edges with same source+relationship but different targets
     * @param staleNodes       nodes not confirmed in over 90 days
     */
    public record GraphHealthReport(
            int totalNodes,
            int totalEdges,
            int duplicateNodes,
            int conflictingEdges,
            int staleNodes
    ) {
        /** Returns {@code true} if any issue was found. */
        public boolean hasIssues() {
            return duplicateNodes > 0 || conflictingEdges > 0 || staleNodes > 0;
        }
    }

    // ===== Core contradiction detection =====================================

    /**
     * Checks whether creating an edge from {@code sourceNodeId} to
     * {@code targetNodeId} with the given {@code relationship} contradicts any
     * existing edge, and resolves the conflict if so.
     *
     * <p>A contradiction exists when the source node already has a current edge
     * with the same relationship pointing to a <em>different</em> target.
     *
     * @param userId       owner
     * @param sourceNodeId origin of the proposed edge
     * @param targetNodeId destination of the proposed edge
     * @param relationship edge type (e.g. "works_at")
     * @param newSource    provenance of the proposed edge
     * @return contradiction result with resolution details
     */
    public ContradictionResult checkAndResolve(String userId, String sourceNodeId,
                                               String targetNodeId, String relationship,
                                               String newSource) {
        List<Edge> currentEdges = store.findCurrentEdgesFromNode(sourceNodeId);

        // F10: When concurrent sessions raced to insert conflicting edges,
        // multiple current rows with the same (source, rel) but different
        // targets can exist. The old implementation returned after the FIRST
        // conflict and left the rest of the divergent edges alive. We now
        // walk the FULL list, decide the resolution on the highest-priority
        // existing edge, and invalidate ALL losing ones in a single pass.
        List<Edge> conflicts = new ArrayList<>();
        for (Edge existing : currentEdges) {
            if (!existing.relationship().equals(relationship)) continue;
            if (existing.targetNodeId().equals(targetNodeId)) continue;
            conflicts.add(existing);
        }
        if (conflicts.isEmpty()) return ContradictionResult.none();

        int newPriority = sourcePriority(newSource);
        // Determine the highest priority among current conflicting edges.
        int maxExistingPriority = conflicts.stream()
                .mapToInt(e -> sourcePriority(e.source()))
                .max().orElse(0);

        String description = "Conflicting '" + relationship + "': "
                + conflicts.size() + " existing target(s) vs new target [" + targetNodeId + "]";

        if (newPriority < maxExistingPriority) {
            // New loses to at least one higher-priority existing edge.
            LOG.fine(() -> "Contradiction resolved: kept existing (higher priority).");
            return new ContradictionResult(true, "relationship_conflict", description,
                    "kept_existing", null);
        }

        // New wins (strictly greater, or ties with newer semantic).
        // Invalidate ALL existing conflicts — not just the first one. This
        // is what the concurrent-session stress test exposed.
        String firstInvalidatedId = null;
        for (Edge existing : conflicts) {
            store.updateEdge(existing.invalidated());
            if (firstInvalidatedId == null) firstInvalidatedId = existing.id();
        }
        LOG.fine(() -> "Contradiction resolved: kept new (priority/ties). Invalidated "
                + conflicts.size() + " edge(s).");
        return new ContradictionResult(true, "relationship_conflict", description,
                "kept_new", firstInvalidatedId);
    }

    /**
     * Compares a node's existing properties against proposed new properties,
     * returning any conflicts found.
     *
     * <p>Keys starting with {@code "_"} are internal metadata and are skipped.
     *
     * @param node          the existing node
     * @param newProperties proposed property updates
     * @param newSource     provenance of the proposed update
     * @return list of property conflicts (empty if none)
     */
    public List<PropertyConflict> checkNodeProperties(Node node, Map<String, Object> newProperties,
                                                      String newSource) {
        if (node.properties() == null || newProperties == null) return List.of();

        List<PropertyConflict> conflicts = new ArrayList<>();
        Map<String, Object> oldProps = node.properties();

        for (Map.Entry<String, Object> entry : newProperties.entrySet()) {
            String key = entry.getKey();
            if (key.startsWith("_")) continue;

            Object oldVal = oldProps.get(key);
            Object newVal = entry.getValue();

            if (oldVal != null && !oldVal.equals(newVal)) {
                int existingPriority = sourcePriority(node.source());
                int incomingPriority = sourcePriority(newSource);
                String resolution = incomingPriority >= existingPriority ? "kept_new" : "kept_existing";
                conflicts.add(new PropertyConflict(key, oldVal, newVal, resolution));
            }
        }

        return Collections.unmodifiableList(conflicts);
    }

    // ===== Health scan ======================================================

    /**
     * Scans the user's graph for structural issues: duplicate labels, edge
     * conflicts, and stale nodes.
     *
     * @param userId owner
     * @return a report summarizing any issues found
     */
    public GraphHealthReport scanForIssues(String userId) {
        List<Node> allNodes = store.findAllAccessible(userId);
        List<Edge> allEdges = store.findAllCurrentEdges(userId);

        int duplicates = countDuplicateLabels(allNodes);
        int conflicts = countEdgeConflicts(allEdges);
        int stale = countStaleNodes(allNodes);

        GraphHealthReport report = new GraphHealthReport(
                allNodes.size(), allEdges.size(), duplicates, conflicts, stale);

        if (report.hasIssues()) {
            LOG.info(() -> "Graph health scan for user [" + userId + "]: "
                    + duplicates + " duplicate labels, "
                    + conflicts + " edge conflicts, "
                    + stale + " stale nodes");
        }

        return report;
    }

    // ===== Internals ========================================================

    /**
     * Returns a numeric priority for source provenance. Higher is more
     * authoritative.
     */
    static int sourcePriority(String source) {
        if (source == null) return 0;
        return switch (source) {
            case "user_stated" -> 3;
            case "conversation" -> 2;
            case "extracted" -> 1;
            default -> 0;
        };
    }

    /**
     * Counts nodes that share the same label but belong to different categories
     * (potential duplicates that should be merged or disambiguated).
     */
    private static int countDuplicateLabels(List<Node> nodes) {
        Map<String, Set<String>> labelToCategories = new HashMap<>();
        for (Node node : nodes) {
            labelToCategories
                    .computeIfAbsent(node.label().toLowerCase(), k -> new HashSet<>())
                    .add(node.category());
        }
        return (int) labelToCategories.values().stream()
                .filter(cats -> cats.size() > 1)
                .count();
    }

    /**
     * Counts edge pairs where the same source+relationship points to different targets.
     */
    private static int countEdgeConflicts(List<Edge> edges) {
        // Group by (sourceNodeId, relationship) and check for multiple distinct targets
        Map<String, Set<String>> grouped = new HashMap<>();
        for (Edge edge : edges) {
            String key = edge.sourceNodeId() + "|" + edge.relationship();
            grouped.computeIfAbsent(key, k -> new HashSet<>()).add(edge.targetNodeId());
        }
        return (int) grouped.values().stream()
                .filter(targets -> targets.size() > 1)
                .count();
    }

    /**
     * Counts nodes whose {@link Node#lastConfirmedAt()} is more than 90 days ago.
     */
    private static int countStaleNodes(List<Node> nodes) {
        Instant cutoff = Instant.now().minus(STALE_THRESHOLD);
        return (int) nodes.stream()
                .filter(n -> n.lastConfirmedAt().isBefore(cutoff))
                .count();
    }
}
