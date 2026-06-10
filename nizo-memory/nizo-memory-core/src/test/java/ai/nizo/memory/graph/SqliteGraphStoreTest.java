package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class SqliteGraphStoreTest {

    private static final String USER = "user-1";
    private static final String USER_B = "user-2";

    // ---- helpers ----

    private static Node node(String userId, String category, String label) {
        return node(userId, category, label, Map.of(), 0.8, "extracted", "active");
    }

    private static Node node(String userId, String category, String label,
                             Map<String, Object> props, double confidence,
                             String source, String privacy) {
        Instant now = Instant.now();
        return new Node(UUID.randomUUID().toString(), userId, category, label,
                props, confidence, source, privacy, 1, now, now);
    }

    private static Edge edge(String userId, String sourceId, String targetId,
                             String relationship) {
        return new Edge(UUID.randomUUID().toString(), userId, sourceId, targetId,
                relationship, Map.of(), Instant.now(), null, 0.9, "extracted", null);
    }

    // ---- Node CRUD ----

    @Test
    void insertAndFindById(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node n = node(USER, "person", "Alice");
            store.upsertNode(n);

            Optional<Node> found = store.findNodeById(n.id());
            assertTrue(found.isPresent());
            assertEquals("Alice", found.get().label());
            assertEquals("person", found.get().category());
            assertEquals(USER, found.get().userId());
        }
    }

    @Test
    void findByUserCategoryLabelIsCaseInsensitive(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node n = node(USER, "person", "Alice Smith");
            store.upsertNode(n);

            Optional<Node> found = store.findByUserCategoryLabel(USER, "person", "alice smith");
            assertTrue(found.isPresent());
            assertEquals(n.id(), found.get().id());

            Optional<Node> upper = store.findByUserCategoryLabel(USER, "person", "ALICE SMITH");
            assertTrue(upper.isPresent());
        }
    }

    @Test
    void ftsSearchOnNodeLabels(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            store.upsertNode(node(USER, "person", "Sarah Johnson"));
            store.upsertNode(node(USER, "person", "Bob Williams"));
            store.upsertNode(node(USER, "organization", "Acme Corp"));

            List<Node> results = store.ftsSearchNodes(USER, "Sarah", 10);
            assertFalse(results.isEmpty());
            assertTrue(results.stream().anyMatch(n -> n.label().equals("Sarah Johnson")));
        }
    }

    @Test
    void insertEdgeAndFindCurrentEdgesBidirectional(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node a = node(USER, "person", "Alice");
            Node b = node(USER, "person", "Bob");
            store.upsertNode(a);
            store.upsertNode(b);

            Edge e = edge(USER, a.id(), b.id(), "knows");
            store.insertEdge(e);

            // Bidirectional: find from source side
            List<Edge> fromA = store.findCurrentEdgesForNode(a.id());
            assertEquals(1, fromA.size());
            assertEquals("knows", fromA.getFirst().relationship());

            // Bidirectional: find from target side
            List<Edge> fromB = store.findCurrentEdgesForNode(b.id());
            assertEquals(1, fromB.size());
        }
    }

    @Test
    void findConflictingEdges(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node self = node(USER, "person", "Me");
            Node compA = node(USER, "organization", "Google");
            Node compB = node(USER, "organization", "Meta");
            store.upsertNode(self);
            store.upsertNode(compA);
            store.upsertNode(compB);

            store.insertEdge(edge(USER, self.id(), compA.id(), "works_at"));

            List<Edge> conflicts = store.findConflictingEdges(USER, self.id(), "works_at");
            assertEquals(1, conflicts.size());
            assertEquals(compA.id(), conflicts.getFirst().targetNodeId());
        }
    }

    @Test
    void userIsolation(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node nA = node(USER, "person", "Secret Person");
            Node nB = node(USER_B, "person", "Other Person");
            store.upsertNode(nA);
            store.upsertNode(nB);

            // User A cannot see User B's nodes
            Optional<Node> aSearch = store.findByUserCategoryLabel(USER, "person", "Other Person");
            assertTrue(aSearch.isEmpty());

            // User B cannot see User A's nodes
            Optional<Node> bSearch = store.findByUserCategoryLabel(USER_B, "person", "Secret Person");
            assertTrue(bSearch.isEmpty());

            // FTS also isolated
            List<Node> ftsA = store.ftsSearchNodes(USER, "Other", 10);
            assertTrue(ftsA.isEmpty());
        }
    }

    @Test
    void privacyFilteringExcludesDeletedAndRedacted(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            store.upsertNode(node(USER, "person", "Active Person", Map.of(), 0.8, "extracted", "active"));
            store.upsertNode(node(USER, "person", "Deleted Person", Map.of(), 0.8, "extracted", "deleted"));
            store.upsertNode(node(USER, "person", "Redacted Person", Map.of(), 0.8, "extracted", "redacted"));

            List<Node> accessible = store.findAllAccessible(USER);
            assertEquals(1, accessible.size());
            assertEquals("Active Person", accessible.getFirst().label());

            List<Node> byCategory = store.findByCategory(USER, "person");
            assertEquals(1, byCategory.size());
        }
    }

    @Test
    void edgeInvalidation(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node a = node(USER, "person", "Alice");
            Node b = node(USER, "organization", "Google");
            store.upsertNode(a);
            store.upsertNode(b);

            Edge e = edge(USER, a.id(), b.id(), "works_at");
            store.insertEdge(e);

            // Invalidate the edge
            Edge invalidated = e.invalidated();
            store.updateEdge(invalidated);

            // Should no longer appear in current edges
            List<Edge> current = store.findCurrentEdgesForNode(a.id());
            assertTrue(current.isEmpty());

            // But should still exist in the database
            Optional<Edge> raw = store.findEdgeById(e.id());
            assertTrue(raw.isPresent());
            assertNotNull(raw.get().invalidatedAt());
        }
    }

    @Test
    void countByCategoryAndCounts(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            store.upsertNode(node(USER, "person", "Alice"));
            store.upsertNode(node(USER, "person", "Bob"));
            store.upsertNode(node(USER, "organization", "Google"));
            store.upsertNode(node(USER, "preference", "Dark mode"));

            Map<String, Long> counts = store.countByCategory(USER);
            assertEquals(2L, counts.get("person"));
            assertEquals(1L, counts.get("organization"));
            assertEquals(1L, counts.get("preference"));

            assertEquals(4L, store.countNodes(USER));
            assertEquals(0L, store.countEdges(USER));

            // Add an edge and verify edge count
            Node a = store.findByUserCategoryLabel(USER, "person", "Alice").orElseThrow();
            Node b = store.findByUserCategoryLabel(USER, "organization", "Google").orElseThrow();
            store.insertEdge(edge(USER, a.id(), b.id(), "works_at"));
            assertEquals(1L, store.countEdges(USER));
        }
    }

    @Test
    void deleteNodeCascadesEdges(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node a = node(USER, "person", "Alice");
            Node b = node(USER, "person", "Bob");
            store.upsertNode(a);
            store.upsertNode(b);

            Edge e = edge(USER, a.id(), b.id(), "knows");
            store.insertEdge(e);

            // Cascade delete edges for node a
            store.deleteEdgesForNode(a.id());
            store.deleteNode(a.id());

            assertTrue(store.findNodeById(a.id()).isEmpty());
            assertTrue(store.findCurrentEdgesForNode(b.id()).isEmpty());
        }
    }

    @Test
    void upsertNodeUpdatesExisting(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node original = node(USER, "person", "Alice");
            store.upsertNode(original);

            // Update with same id but new properties
            Node updated = new Node(
                    original.id(), USER, "person", "Alice",
                    Map.of("company", "Google"), 0.95, "user_stated", "active",
                    3, original.firstSeenAt(), Instant.now());
            store.upsertNode(updated);

            Node loaded = store.findNodeById(original.id()).orElseThrow();
            assertEquals(0.95, loaded.confidence(), 0.01);
            assertEquals("user_stated", loaded.source());
            assertEquals(3, loaded.mentionCount());
            assertEquals("Google", loaded.properties().get("company"));
        }
    }

    @Test
    void findByUserAndLabelContaining(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            store.upsertNode(node(USER, "person", "Sarah Johnson"));
            store.upsertNode(node(USER, "person", "Sarah Williams"));
            store.upsertNode(node(USER, "person", "Bob Smith"));

            List<Node> results = store.findByUserAndLabelContaining(USER, "Sarah");
            assertEquals(2, results.size());
        }
    }

    @Test
    void findEdgesBetweenBidirectional(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            Node a = node(USER, "person", "Alice");
            Node b = node(USER, "person", "Bob");
            store.upsertNode(a);
            store.upsertNode(b);

            store.insertEdge(edge(USER, a.id(), b.id(), "knows"));
            store.insertEdge(edge(USER, b.id(), a.id(), "friend_of"));

            List<Edge> between = store.findEdgesBetween(USER, a.id(), b.id());
            assertEquals(2, between.size());
        }
    }

    @Test
    void findPersonsByName(@TempDir Path tmp) {
        try (SqliteGraphStore store = new SqliteGraphStore(tmp.resolve("g.db"))) {
            store.upsertNode(node(USER, "person", "Sarah"));
            store.upsertNode(node(USER, "person", "sarah")); // different case, same name
            store.upsertNode(node(USER, "person", "Bob"));

            List<Node> sarahs = store.findPersonsByName(USER, "Sarah");
            assertEquals(2, sarahs.size());
        }
    }
}
