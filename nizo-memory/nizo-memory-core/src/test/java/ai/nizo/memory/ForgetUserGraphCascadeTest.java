package ai.nizo.memory;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.graph.GraphTraversalEngine;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * F9: {@code forgetUser} must cascade into the knowledge graph, purging the
 * user's nodes + edges. Prior to this fix, the graph survived a GDPR-style
 * forget request and could leak via graph-channel recall.
 */
class ForgetUserGraphCascadeTest {

    @Test
    void forgetUser_removesGraphNodesAndEdges(@TempDir Path tmp) {
        Path dbPath = tmp.resolve("mem.db");
        SqliteMemoryStore memStore = new SqliteMemoryStore(dbPath);
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(List.of("alice", "acme", "bob"));
        SqliteGraphStore graphStore = new SqliteGraphStore(dbPath);
        GraphService graph = new KnowledgeGraph(graphStore);
        GraphTraversalEngine traversal = new GraphTraversalEngine(graphStore, graph);

        LayeredMemoryService svc = new LayeredMemoryService(
                memStore, index, embedder, null, graph, traversal,
                100, 0.0, 0.01, 0.0);

        // Seed user-A with memory items + graph entities
        String uidA = "userA";
        svc.learnFact(uidA, "Alice works at Acme", "user_stated", 0.9);
        svc.remember(uidA, "I had lunch with Bob", Map.of(), "user");

        Node alice = graph.createOrMergePersonNode(uidA, "Alice", Map.of(), "user_stated");
        Node bob   = graph.createOrMergePersonNode(uidA, "Bob",   Map.of(), "user_stated");
        Edge e = graph.createEdgeIfNotExists(new Edge(
                UUID.randomUUID().toString(), uidA,
                alice.id(), bob.id(), "knows",
                Map.of(), Instant.now(), null, 0.9, "user_stated", null));
        assertNotNull(e);

        // Seed user-B with its own graph so we can verify isolation
        String uidB = "userB";
        Node other = graph.createOrMergePersonNode(uidB, "Charlie", Map.of(), "user_stated");
        assertNotNull(other);

        // Precondition: A has 2 nodes + 1 edge
        Map<String, Object> statsABefore = graph.getGraphStats(uidA);
        long nodesABefore = ((Number) statsABefore.get("nodeCount")).longValue();
        long edgesABefore = ((Number) statsABefore.get("edgeCount")).longValue();
        assertTrue(nodesABefore >= 2, "A should have at least 2 nodes before forget");
        assertTrue(edgesABefore >= 1, "A should have at least 1 edge before forget");

        // Call forgetUser on A
        int removed = svc.forgetUser(uidA);
        assertTrue(removed > 0, "forgetUser must report a non-zero purge count");

        // A's graph must be empty
        Map<String, Object> statsAAfter = graph.getGraphStats(uidA);
        assertEquals(0L, ((Number) statsAAfter.get("nodeCount")).longValue(),
                "all of A's nodes must be deleted");
        assertEquals(0L, ((Number) statsAAfter.get("edgeCount")).longValue(),
                "all of A's edges must be deleted");

        // B's graph must be untouched
        Map<String, Object> statsBAfter = graph.getGraphStats(uidB);
        assertTrue(((Number) statsBAfter.get("nodeCount")).longValue() >= 1,
                "forgetUser(A) must not touch B's graph");

        // And A's memory items are gone too
        assertTrue(svc.inspect(uidA, 100).isEmpty(),
                "A's memory items must be purged");
    }

    @Test
    void forgetUser_withoutGraph_isStillSafe(@TempDir Path tmp) {
        // Regression: forgetUser must not throw when graphService is null.
        SqliteMemoryStore memStore = new SqliteMemoryStore(tmp.resolve("mem.db"));
        InMemoryVectorIndex index = new InMemoryVectorIndex();

        LayeredMemoryService svc = new LayeredMemoryService(
                memStore, index, null, null, 100, 0.0);

        svc.learnFact("u1", "some fact", "user_stated", 0.9);
        int removed = svc.forgetUser("u1");
        assertTrue(removed >= 1);
    }
}
