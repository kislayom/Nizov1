package ai.nizo.memory.api.graph;

import java.util.List;
import java.util.Set;

/**
 * Graph traversal operations for context expansion.
 *
 * <p>Given a set of seed nodes (or raw text mentions), the traversal walks
 * outward through edges up to {@code maxHops}, collecting neighbours scored
 * by distance and edge confidence. The result feeds the recall ranker so
 * the agent sees not just directly mentioned entities but their local
 * neighbourhood in the knowledge graph.
 */
public interface GraphTraversal {

    /**
     * A node discovered during graph expansion, together with the edge that
     * led to it and a composite relevance score.
     *
     * @param node        the discovered neighbour
     * @param edge        the edge connecting it to the previous hop
     * @param hopDistance  number of hops from the nearest seed node (1-based)
     * @param score       composite relevance score (higher = more relevant)
     */
    record GraphNeighbor(Node node, Edge edge, int hopDistance, double score) { }

    /**
     * Expands outward from a set of seed nodes, collecting accessible
     * neighbours up to {@code maxHops} edges away.
     *
     * @param userId    owner; traversal never crosses user boundaries
     * @param seedNodes starting points for expansion
     * @param maxHops   maximum edge distance from any seed node
     * @return neighbours ordered by descending {@link GraphNeighbor#score}
     */
    List<GraphNeighbor> expandFromNodes(String userId, Set<Node> seedNodes, int maxHops);

    /**
     * Resolves raw text mentions to nodes via entity resolution, then
     * expands outward as in {@link #expandFromNodes}.
     *
     * @param userId   owner
     * @param mentions raw text mentions (e.g. "Ravi", "my Zerodha account")
     * @param maxHops  maximum edge distance
     * @return neighbours ordered by descending {@link GraphNeighbor#score}
     */
    List<GraphNeighbor> expandFromMentions(String userId, Set<String> mentions, int maxHops);
}
