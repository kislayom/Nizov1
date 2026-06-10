package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.GraphTraversal;
import ai.nizo.memory.api.graph.Node;

import java.util.*;
import java.util.logging.Logger;

/**
 * BFS-based graph traversal engine implementing {@link GraphTraversal}.
 *
 * <p>Given a set of seed nodes (or raw text mentions that are first resolved
 * to nodes), walks outward through edges up to {@code maxHops}, collecting
 * neighbours scored by a composite of node confidence, edge confidence, and
 * distance decay. Results feed the recall ranker so the agent sees the local
 * neighbourhood of mentioned entities, not just the entities themselves.
 *
 * <p>De-Springified port from Kimaya's {@code GraphTraversalService}. Plain
 * Java 21, no framework dependencies.
 */
public final class GraphTraversalEngine implements GraphTraversal {

    private static final Logger LOG = Logger.getLogger(GraphTraversalEngine.class.getName());

    /** Confidence decay at 1 hop from a seed node. */
    private static final double HOP_1_DECAY = 0.8;
    /** Confidence decay at 2 hops from a seed node. */
    private static final double HOP_2_DECAY = 0.5;

    private final SqliteGraphStore store;
    private final GraphService graphService;

    public GraphTraversalEngine(SqliteGraphStore store, GraphService graphService) {
        this.store = Objects.requireNonNull(store, "store");
        this.graphService = Objects.requireNonNull(graphService, "graphService");
    }

    @Override
    public List<GraphNeighbor> expandFromNodes(String userId, Set<Node> seedNodes, int maxHops) {
        if (seedNodes == null || seedNodes.isEmpty() || maxHops <= 0) return List.of();

        List<GraphNeighbor> results = new ArrayList<>();
        Set<String> visited = new HashSet<>();

        // Mark all seeds as visited so we don't return them as neighbours
        for (Node seed : seedNodes) {
            visited.add(seed.id());
        }

        // BFS frontier: (nodeId, current hop distance)
        Deque<HopEntry> frontier = new ArrayDeque<>();
        for (Node seed : seedNodes) {
            frontier.add(new HopEntry(seed.id(), 0));
        }

        while (!frontier.isEmpty()) {
            HopEntry current = frontier.poll();
            if (current.hop >= maxHops) continue;

            int nextHop = current.hop + 1;
            double hopDecay = hopDecayFactor(nextHop);

            List<Edge> edges = store.findCurrentEdgesForNode(current.nodeId);

            for (Edge edge : edges) {
                // Determine the neighbour on the other side of the edge
                String neighbourId = edge.sourceNodeId().equals(current.nodeId)
                        ? edge.targetNodeId()
                        : edge.sourceNodeId();

                if (visited.contains(neighbourId)) continue;
                visited.add(neighbourId);

                Optional<Node> neighbourOpt = store.findNodeById(neighbourId);
                if (neighbourOpt.isEmpty()) continue;

                Node neighbour = neighbourOpt.get();
                if (!neighbour.isAccessible()) continue;
                if (!neighbour.userId().equals(userId)) continue;

                double score = neighbour.confidence() * hopDecay * edge.confidence();
                results.add(new GraphNeighbor(neighbour, edge, nextHop, score));

                // Enqueue for further expansion if we haven't hit max hops
                if (nextHop < maxHops) {
                    frontier.add(new HopEntry(neighbourId, nextHop));
                }
            }
        }

        // Sort by score descending
        results.sort(Comparator.comparingDouble(GraphNeighbor::score).reversed());

        LOG.fine(() -> "Expanded " + seedNodes.size() + " seed(s), maxHops=" + maxHops
                + " -> " + results.size() + " neighbour(s)");

        return Collections.unmodifiableList(results);
    }

    @Override
    public List<GraphNeighbor> expandFromMentions(String userId, Set<String> mentions, int maxHops) {
        if (mentions == null || mentions.isEmpty()) return List.of();

        Set<Node> seedNodes = new LinkedHashSet<>();
        for (String mention : mentions) {
            graphService.resolveEntity(userId, mention).ifPresent(seedNodes::add);
        }

        if (seedNodes.isEmpty()) {
            LOG.fine(() -> "No mentions resolved to nodes for user [" + userId + "]");
            return List.of();
        }

        return expandFromNodes(userId, seedNodes, maxHops);
    }

    // ===== Internals ========================================================

    private static double hopDecayFactor(int hop) {
        return switch (hop) {
            case 1 -> HOP_1_DECAY;
            case 2 -> HOP_2_DECAY;
            default -> HOP_2_DECAY * Math.pow(0.5, hop - 2); // further decay for 3+
        };
    }

    /**
     * Internal BFS queue entry: a node id and its current distance from the
     * nearest seed.
     */
    private record HopEntry(String nodeId, int hop) { }
}
