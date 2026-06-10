package ai.nizo.memory.api.graph;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * CRUD and query facade for the personal knowledge graph.
 *
 * <p>Every operation is scoped to a {@code userId}. Two users' graphs are
 * completely isolated: they share the same backing store but never leak
 * nodes or edges across user boundaries.
 *
 * <p>Implementations are expected to handle deduplication (merge-on-match)
 * for both nodes and edges, so callers can fire-and-forget extractions
 * without worrying about duplicates.
 */
public interface GraphService {

    // ── Node operations ─────────────────────────────────────────────────

    /**
     * Creates a node if no matching entity exists, or merges properties and
     * records a mention on the existing one.
     *
     * @return the created or updated node
     */
    Node createOrMergeNode(Node node);

    /**
     * Convenience method for person entities. Delegates to
     * {@link #createOrMergeNode(Node)} with {@code category = "person"}.
     *
     * @param userId     owner
     * @param label      display name of the person
     * @param properties additional attributes (e.g. role, company)
     * @param source     provenance tag
     * @return the created or updated person node
     */
    Node createOrMergePersonNode(String userId, String label,
                                 Map<String, Object> properties, String source);

    /** Retrieves a single node by id, scoped to the given user. */
    Optional<Node> getNode(String userId, String nodeId);

    /** Returns all accessible nodes in a category for the given user. */
    List<Node> getNodesByCategory(String userId, String category);

    /**
     * Free-text search across node labels and properties.
     *
     * @param userId owner
     * @param query  search term
     * @param limit  maximum results to return
     * @return matching nodes, ranked by relevance
     */
    List<Node> searchNodes(String userId, String query, int limit);

    /**
     * Attempts to resolve an ambiguous mention to an existing node. Returns
     * the best-matching node if confidence is above the implementation's
     * threshold, or empty if no match is found.
     *
     * @param userId  owner
     * @param mention raw text mention (e.g. "my wife", "AAPL")
     * @return resolved node, or empty
     */
    Optional<Node> resolveEntity(String userId, String mention);

    /**
     * Returns the "self" node for a user -- the node representing the user
     * themselves. Created lazily on first access.
     */
    Node findSelfNode(String userId);

    /**
     * Soft-deletes a node by setting its privacy level to {@code "deleted"}.
     * Associated edges are invalidated.
     */
    void deleteNode(String userId, String nodeId);

    // ── Edge operations ─────────────────────────────────────────────────

    /** Creates a new edge between two nodes. */
    Edge createEdge(Edge edge);

    /**
     * Creates an edge only if no current edge with the same source, target,
     * and relationship already exists for the user.
     *
     * @return the existing or newly created edge
     */
    Edge createEdgeIfNotExists(Edge edge);

    /** Returns all current edges originating from or targeting the given node. */
    List<Edge> getEdgesForNode(String userId, String nodeId);

    /** Returns all current edges between two specific nodes (in either direction). */
    List<Edge> getEdgesBetween(String userId, String nodeIdA, String nodeIdB);

    /** Invalidates (soft-deletes) an edge by id. */
    void invalidateEdge(String userId, String edgeId);

    // ── Stats ───────────────────────────────────────────────────────────

    /**
     * Returns high-level statistics about the user's graph.
     *
     * <p>Expected keys include {@code "nodeCount"}, {@code "edgeCount"},
     * {@code "categoryBreakdown"}, etc. The exact shape is implementation-defined.
     */
    Map<String, Object> getGraphStats(String userId);

    // ── Privacy / GDPR ──────────────────────────────────────────────────

    /**
     * Hard-deletes every node and every edge belonging to {@code userId}.
     * Used by the cascade path of {@code MemoryService.forgetUser()}.
     *
     * <p>Unlike {@link #deleteNode} which soft-deletes by flipping
     * {@code privacy_level = "deleted"}, this operation is irreversible and
     * removes the rows entirely so a user's data is truly gone.
     *
     * @return total rows deleted (nodes + edges)
     */
    default int deleteAllForUser(String userId) { return 0; }
}
