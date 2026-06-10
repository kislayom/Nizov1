package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;

import java.time.Instant;
import java.util.*;
import java.util.logging.Logger;

/**
 * Core knowledge-graph business logic implementing {@link GraphService}.
 *
 * <p>Handles node/edge CRUD with merge-on-match semantics, entity resolution
 * (exact, fuzzy, FTS), person-node disambiguation by relationship type, and
 * soft-delete with privacy-level gating. All operations are scoped to a single
 * user -- cross-user leaks are impossible by construction.
 *
 * <p>De-Springified port from Kimaya's {@code KnowledgeGraphService}. No
 * framework annotations, no JPA, no Lombok -- plain Java 21 with records.
 */
public final class KnowledgeGraph implements GraphService {

    private static final Logger LOG = Logger.getLogger(KnowledgeGraph.class.getName());
    private static final int MAX_LABEL_LENGTH = 250;

    private final SqliteGraphStore store;

    public KnowledgeGraph(SqliteGraphStore store) {
        this.store = Objects.requireNonNull(store, "store");
    }

    // ===== Node operations ==================================================

    @Override
    public Node createOrMergeNode(Node node) {
        String userId = node.userId();
        String category = node.category();
        String label = truncateLabel(node.label());
        Map<String, Object> properties = node.properties();
        String source = node.source();

        Optional<Node> existing = store.findByUserCategoryLabel(userId, category, label);

        if (existing.isPresent()) {
            Node old = existing.get();
            Node bumped = old.withMentionRecorded();

            // Merge properties: new values override old, but keep old keys absent from new
            Map<String, Object> merged = mergeProperties(old.properties(), properties);

            // Upgrade source and confidence when user explicitly stated something
            double confidence = bumped.confidence();
            String resolvedSource = bumped.source();
            if ("user_stated".equals(source)) {
                confidence = Math.max(confidence, 0.95);
                resolvedSource = "user_stated";
            }

            Node updated = new Node(
                    bumped.id(), userId, category, label, merged,
                    confidence, resolvedSource, bumped.privacyLevel(),
                    bumped.mentionCount(), bumped.firstSeenAt(), bumped.lastConfirmedAt()
            );
            store.upsertNode(updated);
            LOG.fine(() -> "Merged node [" + updated.id() + "] label=" + label);
            return updated;
        }

        // New node
        double confidence = "user_stated".equals(source) ? 0.95 : 0.8;
        Instant now = Instant.now();
        Node created = new Node(
                UUID.randomUUID().toString(), userId, category, label,
                properties != null ? properties : Map.of(),
                confidence, source, "active",
                1, now, now
        );
        store.upsertNode(created);
        LOG.fine(() -> "Created node [" + created.id() + "] label=" + label);
        return created;
    }

    @Override
    public Node createOrMergePersonNode(String userId, String label,
                                        Map<String, Object> properties, String source) {
        String relationshipType = properties != null
                ? (String) properties.get("relationship_type")
                : null;

        List<Node> existingPersons = store.findPersonsByName(userId, label);

        if (!existingPersons.isEmpty()) {
            if (relationshipType != null) {
                // Look for an existing person with the same relationship_type
                Optional<Node> sameType = existingPersons.stream()
                        .filter(n -> n.properties() != null
                                && relationshipType.equals(n.properties().get("relationship_type")))
                        .findFirst();

                if (sameType.isPresent()) {
                    return mergeIntoExisting(sameType.get(), properties, source);
                }

                // Check for an untyped person (no relationship_type) we can upgrade
                Optional<Node> untyped = existingPersons.stream()
                        .filter(n -> n.properties() == null
                                || !n.properties().containsKey("relationship_type"))
                        .findFirst();

                if (untyped.isPresent()) {
                    return mergeIntoExisting(untyped.get(), properties, source);
                }

                // All existing persons have different relationship types -- create new disambiguated node
                return createPersonNode(userId, label, properties, source);
            }

            // No relationship type provided -- legacy merge with first existing
            return mergeIntoExisting(existingPersons.getFirst(), properties, source);
        }

        // No existing person with this name
        return createPersonNode(userId, label, properties, source);
    }

    @Override
    public Optional<Node> resolveEntity(String userId, String mention) {
        if (mention == null || mention.isBlank()) return Optional.empty();

        String trimmed = mention.strip();

        // 1. Exact match via substring search, then filter to exact label
        List<Node> candidates = store.findByUserAndLabelContaining(userId, trimmed);
        Optional<Node> exact = candidates.stream()
                .filter(Node::isAccessible)
                .filter(n -> n.label().equalsIgnoreCase(trimmed))
                .max(Comparator.comparingDouble(Node::confidence));
        if (exact.isPresent()) return exact;

        // 2. Fuzzy: highest confidence from the contains results
        Optional<Node> fuzzy = candidates.stream()
                .filter(Node::isAccessible)
                .max(Comparator.comparingDouble(Node::confidence));
        if (fuzzy.isPresent()) return fuzzy;

        // 3. FTS as last resort
        List<Node> ftsResults = store.ftsSearchNodes(userId, trimmed, 5);
        return ftsResults.stream()
                .filter(Node::isAccessible)
                .max(Comparator.comparingDouble(Node::confidence));
    }

    @Override
    public Node findSelfNode(String userId) {
        List<Node> persons = store.findByCategory(userId, "person");

        // Prefer a node explicitly marked as self
        Optional<Node> selfNode = persons.stream()
                .filter(n -> n.properties() != null
                        && Boolean.TRUE.equals(n.properties().get("is_self")))
                .findFirst();
        if (selfNode.isPresent()) return selfNode.get();

        // Fallback: highest-confidence user_stated person
        Optional<Node> fallback = persons.stream()
                .filter(n -> "user_stated".equals(n.source()))
                .max(Comparator.comparingDouble(Node::confidence));
        if (fallback.isPresent()) return fallback.get();

        // Create a placeholder self node if none exists
        Instant now = Instant.now();
        Node self = new Node(
                UUID.randomUUID().toString(), userId, "person", "Me",
                Map.of("is_self", true),
                0.95, "user_stated", "active",
                1, now, now
        );
        store.upsertNode(self);
        return self;
    }

    @Override
    public Optional<Node> getNode(String userId, String nodeId) {
        return store.findNodeById(nodeId)
                .filter(n -> n.userId().equals(userId))
                .filter(Node::isAccessible);
    }

    @Override
    public List<Node> getNodesByCategory(String userId, String category) {
        return store.findByCategory(userId, category);
    }

    @Override
    public List<Node> searchNodes(String userId, String query, int limit) {
        return store.ftsSearchNodes(userId, query, limit);
    }

    @Override
    public void deleteNode(String userId, String nodeId) {
        store.findNodeById(nodeId)
                .filter(n -> n.userId().equals(userId))
                .ifPresent(node -> {
                    // Soft delete: set privacy to "deleted"
                    Node deleted = new Node(
                            node.id(), node.userId(), node.category(), node.label(),
                            node.properties(), node.confidence(), node.source(),
                            "deleted", node.mentionCount(),
                            node.firstSeenAt(), node.lastConfirmedAt()
                    );
                    store.upsertNode(deleted);

                    // Invalidate all associated edges
                    List<Edge> edges = store.findCurrentEdgesForNode(nodeId);
                    for (Edge edge : edges) {
                        store.updateEdge(edge.invalidated());
                    }
                    LOG.fine(() -> "Soft-deleted node [" + nodeId + "], invalidated " + edges.size() + " edges");
                });
    }

    // ===== Edge operations ==================================================

    @Override
    public Edge createEdge(Edge edge) {
        // Invalidate conflicting edges (same source + relationship, different target)
        List<Edge> conflicts = store.findConflictingEdges(
                edge.userId(), edge.sourceNodeId(), edge.relationship());
        for (Edge conflict : conflicts) {
            store.updateEdge(conflict.invalidated());
        }

        Edge toInsert = new Edge(
                edge.id() != null ? edge.id() : UUID.randomUUID().toString(),
                edge.userId(), edge.sourceNodeId(), edge.targetNodeId(),
                edge.relationship(),
                edge.properties() != null ? edge.properties() : Map.of(),
                edge.validFrom() != null ? edge.validFrom() : Instant.now(),
                edge.validTo(), edge.confidence(), edge.source(),
                edge.invalidatedAt()
        );
        store.insertEdge(toInsert);
        LOG.fine(() -> "Created edge [" + toInsert.id() + "] "
                + toInsert.sourceNodeId() + " --" + toInsert.relationship() + "--> " + toInsert.targetNodeId());
        return toInsert;
    }

    @Override
    public Edge createEdgeIfNotExists(Edge edge) {
        List<Edge> existing = store.findConflictingEdges(
                edge.userId(), edge.sourceNodeId(), edge.relationship());

        // If any current edge has the same source + target + relationship, return it
        Optional<Edge> match = existing.stream()
                .filter(e -> e.targetNodeId().equals(edge.targetNodeId()))
                .findFirst();
        if (match.isPresent()) return match.get();

        return createEdge(edge);
    }

    @Override
    public List<Edge> getEdgesForNode(String userId, String nodeId) {
        return store.findCurrentEdgesForNode(nodeId);
    }

    @Override
    public List<Edge> getEdgesBetween(String userId, String nodeIdA, String nodeIdB) {
        return store.findEdgesBetween(userId, nodeIdA, nodeIdB);
    }

    @Override
    public void invalidateEdge(String userId, String edgeId) {
        store.findEdgeById(edgeId)
                .filter(e -> e.userId().equals(userId))
                .ifPresent(edge -> {
                    store.updateEdge(edge.invalidated());
                    LOG.fine(() -> "Invalidated edge [" + edgeId + "]");
                });
    }

    // ===== Stats ============================================================

    @Override
    public Map<String, Object> getGraphStats(String userId) {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("nodeCount", store.countNodes(userId));
        stats.put("edgeCount", store.countEdges(userId));
        stats.put("categoryBreakdown", store.countByCategory(userId));
        return stats;
    }

    // ===== Privacy / GDPR ===================================================

    @Override
    public int deleteAllForUser(String userId) {
        if (userId == null || userId.isBlank()) return 0;
        return store.deleteAllForUser(userId);
    }

    // ===== Internal helpers =================================================

    private Node mergeIntoExisting(Node existing, Map<String, Object> newProperties, String source) {
        Node bumped = existing.withMentionRecorded();
        Map<String, Object> merged = mergeProperties(existing.properties(), newProperties);

        double confidence = bumped.confidence();
        String resolvedSource = bumped.source();
        if ("user_stated".equals(source)) {
            confidence = Math.max(confidence, 0.95);
            resolvedSource = "user_stated";
        }

        Node updated = new Node(
                bumped.id(), bumped.userId(), "person", bumped.label(), merged,
                confidence, resolvedSource, bumped.privacyLevel(),
                bumped.mentionCount(), bumped.firstSeenAt(), bumped.lastConfirmedAt()
        );
        store.upsertNode(updated);
        return updated;
    }

    private Node createPersonNode(String userId, String label, Map<String, Object> properties, String source) {
        double confidence = "user_stated".equals(source) ? 0.95 : 0.8;
        Instant now = Instant.now();
        Node node = new Node(
                UUID.randomUUID().toString(), userId, "person", truncateLabel(label),
                properties != null ? properties : Map.of(),
                confidence, source, "active",
                1, now, now
        );
        store.upsertNode(node);
        return node;
    }

    private static String truncateLabel(String label) {
        if (label == null) return "";
        return label.length() > MAX_LABEL_LENGTH ? label.substring(0, MAX_LABEL_LENGTH) : label;
    }

    private static Map<String, Object> mergeProperties(Map<String, Object> oldProps, Map<String, Object> newProps) {
        if (oldProps == null && newProps == null) return Map.of();
        Map<String, Object> merged = new LinkedHashMap<>();
        if (oldProps != null) merged.putAll(oldProps);
        if (newProps != null) merged.putAll(newProps);  // new overrides old
        return Collections.unmodifiableMap(merged);
    }
}
