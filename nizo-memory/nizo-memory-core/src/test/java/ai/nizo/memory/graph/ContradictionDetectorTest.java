package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class ContradictionDetectorTest {

    private static final String USER = "user-1";

    private SqliteGraphStore store;
    private ContradictionDetector detector;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        store = new SqliteGraphStore(tmp.resolve("cd.db"));
        detector = new ContradictionDetector(store);
    }

    @AfterEach
    void tearDown() {
        if (store != null) store.close();
    }

    // ---- helpers ----

    private Node insertNode(String category, String label, String source) {
        Instant now = Instant.now();
        Node n = new Node(UUID.randomUUID().toString(), USER, category, label,
                Map.of(), 0.8, source, "active", 1, now, now);
        store.upsertNode(n);
        return n;
    }

    private Edge insertEdge(String sourceNodeId, String targetNodeId,
                            String relationship, String source) {
        Edge e = new Edge(UUID.randomUUID().toString(), USER, sourceNodeId, targetNodeId,
                relationship, Map.of(), Instant.now(), null, 0.9, source, null);
        store.insertEdge(e);
        return e;
    }

    // ===== Source priority =====

    @Test
    void sourcePriority_userStatedIsHighest() {
        assertEquals(3, ContradictionDetector.sourcePriority("user_stated"));
        assertEquals(2, ContradictionDetector.sourcePriority("conversation"));
        assertEquals(1, ContradictionDetector.sourcePriority("extracted"));
        assertEquals(0, ContradictionDetector.sourcePriority("unknown"));
        assertEquals(0, ContradictionDetector.sourcePriority(null));
    }

    // ===== DIRECT contradiction: same relationship, different target =====

    @Test
    void directContradiction_newWinsIfHigherPriority() {
        Node self = insertNode("person", "Me", "user_stated");
        Node compA = insertNode("organization", "Google", "extracted");
        Node compB = insertNode("organization", "Meta", "user_stated");

        // Existing edge: extracted source
        insertEdge(self.id(), compA.id(), "works_at", "extracted");

        // New fact: user_stated (higher priority)
        var result = detector.checkAndResolve(USER, self.id(), compB.id(), "works_at", "user_stated");

        assertTrue(result.hasContradiction());
        assertEquals("relationship_conflict", result.type());
        assertEquals("kept_new", result.resolution());
        assertNotNull(result.invalidatedEdgeId());
    }

    @Test
    void samePriorityContradiction_newerWins() {
        Node self = insertNode("person", "Me", "extracted");
        Node compA = insertNode("organization", "Google", "extracted");
        Node compB = insertNode("organization", "Meta", "extracted");

        // Existing edge: extracted source
        insertEdge(self.id(), compA.id(), "works_at", "extracted");

        // New fact: also extracted (same priority) -- newer wins
        var result = detector.checkAndResolve(USER, self.id(), compB.id(), "works_at", "extracted");

        assertTrue(result.hasContradiction());
        assertEquals("kept_new", result.resolution());
        assertNotNull(result.invalidatedEdgeId());
    }

    @Test
    void lowerPriorityNewFact_keepsOld() {
        Node self = insertNode("person", "Me", "user_stated");
        Node compA = insertNode("organization", "Google", "user_stated");
        Node compB = insertNode("organization", "Meta", "extracted");

        // Existing edge: user_stated source (high priority)
        insertEdge(self.id(), compA.id(), "works_at", "user_stated");

        // New fact: extracted (lower priority)
        var result = detector.checkAndResolve(USER, self.id(), compB.id(), "works_at", "extracted");

        assertTrue(result.hasContradiction());
        assertEquals("kept_existing", result.resolution());
        assertNull(result.invalidatedEdgeId());
    }

    @Test
    void sameTargetIsNotContradiction() {
        Node self = insertNode("person", "Me", "extracted");
        Node comp = insertNode("organization", "Google", "extracted");

        // Existing edge
        insertEdge(self.id(), comp.id(), "works_at", "extracted");

        // New fact with same target -- not a contradiction
        var result = detector.checkAndResolve(USER, self.id(), comp.id(), "works_at", "user_stated");

        assertFalse(result.hasContradiction());
        assertNull(result.type());
    }

    // ===== Property conflict detection =====

    @Test
    void propertyConflictDetection() {
        Instant now = Instant.now();
        Node existing = new Node("n1", USER, "person", "Me",
                Map.of("company", "Google", "city", "Sydney"),
                0.8, "extracted", "active", 1, now, now);

        Map<String, Object> newProps = Map.of("company", "Meta", "role", "SWE");

        List<ContradictionDetector.PropertyConflict> conflicts =
                detector.checkNodeProperties(existing, newProps, "extracted");

        assertEquals(1, conflicts.size());
        assertEquals("company", conflicts.getFirst().key());
        assertEquals("Google", conflicts.getFirst().oldValue());
        assertEquals("Meta", conflicts.getFirst().newValue());
    }

    @Test
    void propertyConflictResolutionBySourcePriority() {
        Instant now = Instant.now();
        // Existing node with extracted source
        Node existing = new Node("n1", USER, "person", "Me",
                Map.of("company", "Google"),
                0.8, "extracted", "active", 1, now, now);

        // New props with user_stated source (higher)
        List<ContradictionDetector.PropertyConflict> conflicts =
                detector.checkNodeProperties(existing, Map.of("company", "Meta"), "user_stated");

        assertEquals(1, conflicts.size());
        assertEquals("kept_new", conflicts.getFirst().resolution());

        // Reverse: existing is user_stated, new is extracted
        Node userStated = new Node("n2", USER, "person", "Me",
                Map.of("company", "Google"),
                0.95, "user_stated", "active", 1, now, now);

        List<ContradictionDetector.PropertyConflict> conflicts2 =
                detector.checkNodeProperties(userStated, Map.of("company", "Meta"), "extracted");

        assertEquals(1, conflicts2.size());
        assertEquals("kept_existing", conflicts2.getFirst().resolution());
    }

    @Test
    void propertyConflict_internalKeysSkipped() {
        Instant now = Instant.now();
        Node existing = new Node("n1", USER, "person", "Me",
                Map.of("_internal", "old"),
                0.8, "extracted", "active", 1, now, now);

        List<ContradictionDetector.PropertyConflict> conflicts =
                detector.checkNodeProperties(existing, Map.of("_internal", "new"), "extracted");

        assertTrue(conflicts.isEmpty());
    }

    // ===== Graph health scan =====

    @Test
    void graphHealthScan_detectsDuplicateLabels() {
        // Same label "Sarah" in different categories -> duplicate
        insertNode("person", "Sarah", "extracted");
        insertNode("topic", "Sarah", "extracted");

        var report = detector.scanForIssues(USER);
        assertTrue(report.duplicateNodes() > 0);
        assertTrue(report.hasIssues());
    }

    @Test
    void graphHealthScan_detectsConflictingEdges() {
        Node self = insertNode("person", "Me", "extracted");
        Node compA = insertNode("organization", "Google", "extracted");
        Node compB = insertNode("organization", "Meta", "extracted");

        // Two current edges with same source+relationship but different targets
        insertEdge(self.id(), compA.id(), "works_at", "extracted");
        insertEdge(self.id(), compB.id(), "works_at", "extracted");

        var report = detector.scanForIssues(USER);
        assertTrue(report.conflictingEdges() > 0);
        assertTrue(report.hasIssues());
    }

    @Test
    void graphHealthScan_detectsStaleNodes() {
        Instant staleTime = Instant.now().minus(100, ChronoUnit.DAYS);
        Node stale = new Node(UUID.randomUUID().toString(), USER, "person", "Old Contact",
                Map.of(), 0.5, "extracted", "active", 1, staleTime, staleTime);
        store.upsertNode(stale);

        var report = detector.scanForIssues(USER);
        assertTrue(report.staleNodes() > 0);
    }

    @Test
    void noContradictionWhenNoConflictingEdges() {
        Node self = insertNode("person", "Me", "extracted");
        Node comp = insertNode("organization", "Google", "extracted");

        // No existing edges at all
        var result = detector.checkAndResolve(USER, self.id(), comp.id(), "works_at", "extracted");

        assertFalse(result.hasContradiction());
    }

    @Test
    void graphHealthReport_noIssuesOnCleanGraph() {
        insertNode("person", "Alice", "extracted");
        insertNode("organization", "Google", "extracted");

        var report = detector.scanForIssues(USER);
        assertFalse(report.hasIssues());
        assertEquals(2, report.totalNodes());
    }
}
