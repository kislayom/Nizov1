package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphTraversal.GraphNeighbor;
import ai.nizo.memory.api.graph.Node;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class GraphTraversalTest {

    private static final String USER = "user-1";
    private static final String USER_B = "user-2";

    private SqliteGraphStore store;
    private KnowledgeGraph kg;
    private GraphTraversalEngine engine;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        store = new SqliteGraphStore(tmp.resolve("gt.db"));
        kg = new KnowledgeGraph(store);
        engine = new GraphTraversalEngine(store, kg);
    }

    @AfterEach
    void tearDown() {
        if (store != null) store.close();
    }

    // ---- helpers ----

    private Node insertNode(String userId, String category, String label) {
        Instant now = Instant.now();
        Node n = new Node(UUID.randomUUID().toString(), userId, category, label,
                Map.of(), 0.8, "extracted", "active", 1, now, now);
        store.upsertNode(n);
        return n;
    }

    private void insertEdge(String userId, String sourceId, String targetId, String rel) {
        Edge e = new Edge(UUID.randomUUID().toString(), userId, sourceId, targetId,
                rel, Map.of(), Instant.now(), null, 0.9, "extracted", null);
        store.insertEdge(e);
    }

    // ===== 1-hop expansion =====

    @Test
    void oneHopExpansionFindsDirectNeighbors() {
        Node self = insertNode(USER, "person", "Me");
        Node alice = insertNode(USER, "person", "Alice");
        Node bob = insertNode(USER, "person", "Bob");

        insertEdge(USER, self.id(), alice.id(), "knows");
        insertEdge(USER, self.id(), bob.id(), "works_with");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(self), 1);

        assertEquals(2, results.size());
        Set<String> labels = new HashSet<>();
        for (GraphNeighbor n : results) labels.add(n.node().label());
        assertTrue(labels.contains("Alice"));
        assertTrue(labels.contains("Bob"));

        // All neighbors should be at hop distance 1
        for (GraphNeighbor n : results) {
            assertEquals(1, n.hopDistance());
        }
    }

    // ===== 2-hop expansion =====

    @Test
    void twoHopExpansionFindsSecondDegreeNeighbors() {
        Node self = insertNode(USER, "person", "Me");
        Node alice = insertNode(USER, "person", "Alice");
        Node charlie = insertNode(USER, "person", "Charlie");

        insertEdge(USER, self.id(), alice.id(), "knows");
        insertEdge(USER, alice.id(), charlie.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(self), 2);

        assertEquals(2, results.size());
        // Alice at hop 1, Charlie at hop 2
        Optional<GraphNeighbor> aliceN = results.stream()
                .filter(n -> n.node().label().equals("Alice")).findFirst();
        Optional<GraphNeighbor> charlieN = results.stream()
                .filter(n -> n.node().label().equals("Charlie")).findFirst();

        assertTrue(aliceN.isPresent());
        assertTrue(charlieN.isPresent());
        assertEquals(1, aliceN.get().hopDistance());
        assertEquals(2, charlieN.get().hopDistance());
    }

    // ===== Confidence decay =====

    @Test
    void confidenceDecay_hop1and2() {
        Node self = insertNode(USER, "person", "Me");
        Node alice = insertNode(USER, "person", "Alice");
        Node charlie = insertNode(USER, "person", "Charlie");

        insertEdge(USER, self.id(), alice.id(), "knows");
        insertEdge(USER, alice.id(), charlie.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(self), 2);

        GraphNeighbor hop1 = results.stream()
                .filter(n -> n.hopDistance() == 1).findFirst().orElseThrow();
        GraphNeighbor hop2 = results.stream()
                .filter(n -> n.hopDistance() == 2).findFirst().orElseThrow();

        // hop1 score = node.confidence * 0.8 * edge.confidence = 0.8 * 0.8 * 0.9 = 0.576
        // hop2 score = node.confidence * 0.5 * edge.confidence = 0.8 * 0.5 * 0.9 = 0.36
        assertTrue(hop1.score() > hop2.score(),
                "Hop 1 score (%f) should be greater than hop 2 score (%f)"
                        .formatted(hop1.score(), hop2.score()));

        // Verify decay factors are applied: hop1 uses 0.8, hop2 uses 0.5
        double expectedHop1 = 0.8 * 0.8 * 0.9; // nodeConf * decay * edgeConf
        double expectedHop2 = 0.8 * 0.5 * 0.9;
        assertEquals(expectedHop1, hop1.score(), 0.01);
        assertEquals(expectedHop2, hop2.score(), 0.01);
    }

    // ===== Results sorted by score descending =====

    @Test
    void resultsSortedByScoreDescending() {
        Node self = insertNode(USER, "person", "Me");
        Node alice = insertNode(USER, "person", "Alice");
        Node charlie = insertNode(USER, "person", "Charlie");

        insertEdge(USER, self.id(), alice.id(), "knows");
        insertEdge(USER, alice.id(), charlie.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(self), 2);

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).score() >= results.get(i).score(),
                    "Results should be sorted by score descending");
        }
    }

    // ===== Visited set prevents cycles =====

    @Test
    void visitedSetPreventsCycles() {
        Node a = insertNode(USER, "person", "A");
        Node b = insertNode(USER, "person", "B");
        Node c = insertNode(USER, "person", "C");

        // A -> B -> C -> A (cycle)
        insertEdge(USER, a.id(), b.id(), "knows");
        insertEdge(USER, b.id(), c.id(), "knows");
        insertEdge(USER, c.id(), a.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(a), 3);

        // Should find B and C, but never revisit A
        assertEquals(2, results.size());
        Set<String> foundLabels = new HashSet<>();
        for (GraphNeighbor n : results) foundLabels.add(n.node().label());
        assertTrue(foundLabels.contains("B"));
        assertTrue(foundLabels.contains("C"));
        assertFalse(foundLabels.contains("A"));
    }

    // ===== Empty graph =====

    @Test
    void emptyGraphReturnsEmpty() {
        Node self = insertNode(USER, "person", "Me");
        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(self), 2);
        assertTrue(results.isEmpty());
    }

    @Test
    void nullOrEmptySeedsReturnsEmpty() {
        assertTrue(engine.expandFromNodes(USER, Set.of(), 1).isEmpty());
        assertTrue(engine.expandFromNodes(USER, null, 1).isEmpty());
    }

    @Test
    void zeroMaxHopsReturnsEmpty() {
        Node self = insertNode(USER, "person", "Me");
        assertTrue(engine.expandFromNodes(USER, Set.of(self), 0).isEmpty());
    }

    // ===== expandFromMentions =====

    @Test
    void expandFromMentions_resolvesThenExpands() {
        Node alice = insertNode(USER, "person", "Alice");
        Node bob = insertNode(USER, "person", "Bob");
        insertEdge(USER, alice.id(), bob.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromMentions(USER, Set.of("Alice"), 1);
        assertEquals(1, results.size());
        assertEquals("Bob", results.getFirst().node().label());
    }

    @Test
    void expandFromMentions_unresolvedMentionsReturnEmpty() {
        List<GraphNeighbor> results = engine.expandFromMentions(
                USER, Set.of("CompletelyUnknownPerson"), 1);
        assertTrue(results.isEmpty());
    }

    // ===== User isolation =====

    @Test
    void userIsolation_doesNotCrossUserBoundaries() {
        Node selfA = insertNode(USER, "person", "SelfA");
        Node secretB = insertNode(USER_B, "person", "SecretB");

        // Edge exists but crosses user boundary (in terms of node ownership)
        insertEdge(USER, selfA.id(), secretB.id(), "knows");

        List<GraphNeighbor> results = engine.expandFromNodes(USER, Set.of(selfA), 1);
        // SecretB belongs to USER_B, so it should be excluded
        assertTrue(results.isEmpty());
    }
}
