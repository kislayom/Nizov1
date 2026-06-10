package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class KnowledgeGraphTest {

    private static final String USER = "user-1";
    private static final String USER_B = "user-2";

    private SqliteGraphStore store;
    private KnowledgeGraph kg;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        store = new SqliteGraphStore(tmp.resolve("kg.db"));
        kg = new KnowledgeGraph(store);
    }

    @AfterEach
    void tearDown() {
        if (store != null) store.close();
    }

    // ---- helpers ----

    private Node newNode(String category, String label, Map<String, Object> props, String source) {
        Instant now = Instant.now();
        return new Node(UUID.randomUUID().toString(), USER, category, label,
                props, 0.8, source, "active", 1, now, now);
    }

    private Edge newEdge(String sourceId, String targetId, String relationship, String source) {
        return new Edge(UUID.randomUUID().toString(), USER, sourceId, targetId,
                relationship, Map.of(), Instant.now(), null, 0.9, source, null);
    }

    // ===== createOrMergeNode =====

    @Test
    void createOrMergeNode_createsNewWhenNoneExisting() {
        Node input = newNode("person", "Alice", Map.of("age", 30), "extracted");
        Node result = kg.createOrMergeNode(input);

        assertNotNull(result.id());
        assertEquals("Alice", result.label());
        assertEquals("person", result.category());
        assertEquals(1, result.mentionCount());
        assertEquals(0.8, result.confidence(), 0.01);
    }

    @Test
    void createOrMergeNode_mergesPropertiesWhenExisting() {
        Node first = newNode("person", "Alice", Map.of("age", 30, "city", "Sydney"), "extracted");
        Node created = kg.createOrMergeNode(first);

        // Second merge with overlapping + new properties
        Node second = newNode("person", "Alice", Map.of("age", 31, "company", "Google"), "extracted");
        Node merged = kg.createOrMergeNode(second);

        assertEquals(created.id(), merged.id());
        assertEquals(31, merged.properties().get("age"));          // new overrides old
        assertEquals("Sydney", merged.properties().get("city"));   // old preserved
        assertEquals("Google", merged.properties().get("company")); // new added
    }

    @Test
    void createOrMergeNode_upgradesSourceWhenUserStated() {
        Node extracted = newNode("person", "Alice", Map.of(), "extracted");
        Node created = kg.createOrMergeNode(extracted);
        assertEquals("extracted", created.source());
        assertEquals(0.8, created.confidence(), 0.01);

        // Now user states the same fact
        Node userStated = newNode("person", "Alice", Map.of(), "user_stated");
        Node merged = kg.createOrMergeNode(userStated);

        assertEquals("user_stated", merged.source());
        assertTrue(merged.confidence() >= 0.95);
    }

    @Test
    void createOrMergeNode_doesNotDowngradeFromUserStated() {
        // First: user_stated
        Node userStated = newNode("person", "Alice", Map.of(), "user_stated");
        Node created = kg.createOrMergeNode(userStated);
        assertEquals("user_stated", created.source());

        // Attempt to "downgrade" via extracted
        Node extracted = newNode("person", "Alice", Map.of(), "extracted");
        Node merged = kg.createOrMergeNode(extracted);

        // Source should remain user_stated, not be downgraded
        assertEquals("user_stated", merged.source());
    }

    @Test
    void createOrMergeNode_incrementsMentionCountOnMerge() {
        Node first = newNode("person", "Alice", Map.of(), "extracted");
        Node created = kg.createOrMergeNode(first);
        assertEquals(1, created.mentionCount());

        Node second = newNode("person", "Alice", Map.of(), "extracted");
        Node merged = kg.createOrMergeNode(second);
        assertEquals(2, merged.mentionCount());

        Node third = newNode("person", "Alice", Map.of(), "extracted");
        Node merged2 = kg.createOrMergeNode(third);
        assertEquals(3, merged2.mentionCount());
    }

    @Test
    void createOrMergeNode_truncatesLabelAt250Chars() {
        String longLabel = "A".repeat(300);
        Node input = newNode("person", longLabel, Map.of(), "extracted");
        Node result = kg.createOrMergeNode(input);

        assertEquals(250, result.label().length());
    }

    // ===== createOrMergePersonNode =====

    @Test
    void createOrMergePersonNode_disambiguationByRelationshipType() {
        // Create "Sarah" as spouse
        Map<String, Object> spouseProps = Map.of("relationship_type", "spouse");
        Node spouse = kg.createOrMergePersonNode(USER, "Sarah", spouseProps, "extracted");

        // Create "Sarah" as colleague -- different relationship_type
        Map<String, Object> colleagueProps = Map.of("relationship_type", "colleague");
        Node colleague = kg.createOrMergePersonNode(USER, "Sarah", colleagueProps, "extracted");

        // Should be different nodes (disambiguated)
        assertNotEquals(spouse.id(), colleague.id());
    }

    @Test
    void createOrMergePersonNode_assignsRelationshipTypeToUntyped() {
        // Create untyped "Sarah"
        Node untyped = kg.createOrMergePersonNode(USER, "Sarah", Map.of(), "extracted");

        // Now add relationship_type -- should merge into untyped
        Map<String, Object> typedProps = Map.of("relationship_type", "spouse");
        Node typed = kg.createOrMergePersonNode(USER, "Sarah", typedProps, "extracted");

        assertEquals(untyped.id(), typed.id());
        assertEquals("spouse", typed.properties().get("relationship_type"));
    }

    @Test
    void createOrMergePersonNode_createsNewWhenAllTypesDiffer() {
        // Create "Sarah" as spouse
        kg.createOrMergePersonNode(USER, "Sarah",
                Map.of("relationship_type", "spouse"), "extracted");

        // Create "Sarah" as colleague
        kg.createOrMergePersonNode(USER, "Sarah",
                Map.of("relationship_type", "colleague"), "extracted");

        // Create "Sarah" as friend -- all existing have types, none match
        Node friend = kg.createOrMergePersonNode(USER, "Sarah",
                Map.of("relationship_type", "friend"), "extracted");

        // Should be a third distinct node
        List<Node> sarahs = store.findPersonsByName(USER, "Sarah");
        assertEquals(3, sarahs.size());
    }

    // ===== resolveEntity =====

    @Test
    void resolveEntity_exactMatchReturnsHighestConfidence() {
        Node n = newNode("person", "Alice Smith", Map.of(), "user_stated");
        kg.createOrMergeNode(n);

        Optional<Node> resolved = kg.resolveEntity(USER, "Alice Smith");
        assertTrue(resolved.isPresent());
        assertEquals("Alice Smith", resolved.get().label());
    }

    @Test
    void resolveEntity_fuzzyMatchWhenExactFails() {
        Node n = newNode("person", "Alice Smith-Jones", Map.of(), "extracted");
        kg.createOrMergeNode(n);

        // "Alice" is a substring, not an exact match
        Optional<Node> resolved = kg.resolveEntity(USER, "Alice");
        assertTrue(resolved.isPresent());
        assertTrue(resolved.get().label().contains("Alice"));
    }

    @Test
    void resolveEntity_ftsFallback() {
        Node n = newNode("event", "Team offsite in Goa", Map.of(), "extracted");
        kg.createOrMergeNode(n);

        Optional<Node> resolved = kg.resolveEntity(USER, "offsite Goa");
        assertTrue(resolved.isPresent());
        assertTrue(resolved.get().label().contains("Goa"));
    }

    @Test
    void resolveEntity_returnsEmptyForUnknownMention() {
        Optional<Node> resolved = kg.resolveEntity(USER, "CompletelyUnknownEntity12345");
        assertTrue(resolved.isEmpty());
    }

    // ===== findSelfNode =====

    @Test
    void findSelfNode_returnsSelfMarkedNode() {
        Map<String, Object> selfProps = Map.of("is_self", true, "name", "Kislay");
        kg.createOrMergePersonNode(USER, "Kislay", selfProps, "user_stated");

        Node self = kg.findSelfNode(USER);
        assertNotNull(self);
        assertEquals(true, self.properties().get("is_self"));
    }

    @Test
    void findSelfNode_fallsBackToHighestConfidenceUserStatedPerson() {
        // Create persons without is_self
        Node n1 = newNode("person", "Kislay", Map.of(), "user_stated");
        kg.createOrMergeNode(n1);

        Node n2 = newNode("person", "Other", Map.of(), "extracted");
        kg.createOrMergeNode(n2);

        Node self = kg.findSelfNode(USER);
        assertNotNull(self);
        // Should be the user_stated person
        assertEquals("user_stated", self.source());
    }

    @Test
    void findSelfNode_createsPlaceholderWhenNoneExist() {
        Node self = kg.findSelfNode(USER);
        assertNotNull(self);
        assertEquals("Me", self.label());
        assertEquals(true, self.properties().get("is_self"));
    }

    // ===== Edge operations =====

    @Test
    void createEdge_invalidatesConflictingEdge() {
        Node self = newNode("person", "Me", Map.of("is_self", true), "user_stated");
        Node compA = newNode("organization", "Google", Map.of(), "extracted");
        Node compB = newNode("organization", "Meta", Map.of(), "extracted");
        kg.createOrMergeNode(self);
        kg.createOrMergeNode(compA);
        kg.createOrMergeNode(compB);

        // Create first edge: Me -> works_at -> Google
        Edge e1 = newEdge(self.id(), compA.id(), "works_at", "extracted");
        kg.createEdge(e1);

        // Create second edge: Me -> works_at -> Meta (conflicts!)
        Edge e2 = newEdge(self.id(), compB.id(), "works_at", "extracted");
        Edge created = kg.createEdge(e2);

        // The old edge should be invalidated
        List<Edge> current = store.findCurrentEdgesForNode(self.id());
        assertEquals(1, current.size());
        assertEquals(compB.id(), current.getFirst().targetNodeId());
    }

    @Test
    void createEdgeIfNotExists_returnsExistingEdgeWhenDuplicate() {
        Node a = newNode("person", "Me", Map.of(), "extracted");
        Node b = newNode("organization", "Google", Map.of(), "extracted");
        kg.createOrMergeNode(a);
        kg.createOrMergeNode(b);

        Edge first = newEdge(a.id(), b.id(), "works_at", "extracted");
        Edge created = kg.createEdgeIfNotExists(first);

        // Same edge again
        Edge second = newEdge(a.id(), b.id(), "works_at", "extracted");
        Edge returned = kg.createEdgeIfNotExists(second);

        assertEquals(created.id(), returned.id());
    }

    @Test
    void createEdgeIfNotExists_createsNewWhenNoneExist() {
        Node a = newNode("person", "Me", Map.of(), "extracted");
        Node b = newNode("organization", "Google", Map.of(), "extracted");
        kg.createOrMergeNode(a);
        kg.createOrMergeNode(b);

        Edge edge = newEdge(a.id(), b.id(), "works_at", "extracted");
        Edge created = kg.createEdgeIfNotExists(edge);

        assertNotNull(created.id());
        assertEquals("works_at", created.relationship());
    }

    // ===== deleteNode (soft delete) =====

    @Test
    void deleteNode_softDeleteViaPrivacy() {
        Node n = newNode("person", "Alice", Map.of(), "extracted");
        Node created = kg.createOrMergeNode(n);

        // Add an edge
        Node b = newNode("person", "Bob", Map.of(), "extracted");
        kg.createOrMergeNode(b);
        kg.createEdge(newEdge(created.id(), b.id(), "knows", "extracted"));

        // Soft delete
        kg.deleteNode(USER, created.id());

        // Node still exists but is not accessible
        Optional<Node> raw = store.findNodeById(created.id());
        assertTrue(raw.isPresent());
        assertEquals("deleted", raw.get().privacyLevel());
        assertFalse(raw.get().isAccessible());

        // Edge should be invalidated
        List<Edge> edges = store.findCurrentEdgesForNode(created.id());
        assertTrue(edges.isEmpty());
    }

    // ===== User isolation =====

    @Test
    void userIsolation_userACantSeeUserBNodes() {
        Node nodeA = newNode("person", "Shared Name", Map.of(), "extracted");
        kg.createOrMergeNode(nodeA);

        // User B creates their own node
        Instant now = Instant.now();
        Node nodeB = new Node(UUID.randomUUID().toString(), USER_B, "person", "Shared Name",
                Map.of(), 0.8, "extracted", "active", 1, now, now);
        kg.createOrMergeNode(nodeB);

        // Resolve for user A should only see user A's node
        Optional<Node> resolvedA = kg.resolveEntity(USER, "Shared Name");
        assertTrue(resolvedA.isPresent());
        assertEquals(USER, resolvedA.get().userId());

        // getNode scoped to user
        Optional<Node> crossUser = kg.getNode(USER, nodeB.id());
        assertTrue(crossUser.isEmpty());
    }

    // ===== Graph stats =====

    @Test
    void graphStatsReturnsCorrectCounts() {
        Node a = newNode("person", "Alice", Map.of(), "extracted");
        Node b = newNode("organization", "Google", Map.of(), "extracted");
        kg.createOrMergeNode(a);
        kg.createOrMergeNode(b);
        kg.createEdge(newEdge(a.id(), b.id(), "works_at", "extracted"));

        Map<String, Object> stats = kg.getGraphStats(USER);
        assertEquals(2L, stats.get("nodeCount"));
        assertEquals(1L, stats.get("edgeCount"));
        assertNotNull(stats.get("categoryBreakdown"));
    }

    // ===== New node gets 0.95 confidence when user_stated =====

    @Test
    void newUserStatedNodeGetsHighConfidence() {
        Node input = newNode("person", "Kislay", Map.of(), "user_stated");
        Node result = kg.createOrMergeNode(input);
        assertEquals(0.95, result.confidence(), 0.01);
    }
}
