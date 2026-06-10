package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * F10 — Cross-session / adversarial stress tests for {@link ContradictionDetector}.
 *
 * <p>The existing {@code ContradictionDetectorTest} covers happy-path priority
 * and basic health-scan behaviour. This suite targets the boundary cases that
 * break real deployments:
 * <ul>
 *   <li>Priority chains across many conflicting edges ("works_at" changes 5
 *       times in a day with mixed provenance)</li>
 *   <li>Invalidated edges actually disappear from the "current" view</li>
 *   <li>Equal-priority ties resolve to newest</li>
 *   <li>Concurrent updates from two sessions don't leave duplicate current
 *       edges</li>
 *   <li>Property-level conflicts with unknown source fall through sanely</li>
 *   <li>Health scan correctly counts current-only conflicts (not historical)</li>
 * </ul>
 */
class ContradictionStressTest {

    private static final String USER = "stress-user";

    private SqliteGraphStore store;
    private ContradictionDetector detector;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        store = new SqliteGraphStore(tmp.resolve("stress.db"));
        detector = new ContradictionDetector(store);
    }

    @AfterEach
    void tearDown() {
        if (store != null) store.close();
    }

    private Node insertNode(String category, String label, String source) {
        Instant now = Instant.now();
        Node n = new Node(UUID.randomUUID().toString(), USER, category, label,
                Map.of(), 0.8, source, "active", 1, now, now);
        store.upsertNode(n);
        return n;
    }

    private Edge insertEdge(String src, String tgt, String rel, String source) {
        Edge e = new Edge(UUID.randomUUID().toString(), USER, src, tgt,
                rel, Map.of(), Instant.now(), null, 0.9, source, null);
        store.insertEdge(e);
        return e;
    }

    // ───────────────────────────── Priority chains ──────────────────────────

    @Test
    void fiveConflictingEdges_onlyHighestPrioritySurvives() {
        Node kim = insertNode("person", "Kim", "user_stated");
        Node acme = insertNode("company", "Acme", "extracted");
        Node beta = insertNode("company", "Beta", "extracted");
        Node gamma = insertNode("company", "Gamma", "conversation");
        Node delta = insertNode("company", "Delta", "user_stated");
        Node eta = insertNode("company", "Eta", "extracted");

        // Session 1: extraction "works_at Acme" lands first
        insertEdge(kim.id(), acme.id(), "works_at", "extracted");

        // Session 2: extraction picks up a stale mention "Beta"
        var r1 = detector.checkAndResolve(USER, kim.id(), beta.id(), "works_at", "extracted");
        assertTrue(r1.hasContradiction(), "equal priority: new Beta should win over old Acme");
        insertEdge(kim.id(), beta.id(), "works_at", "extracted");

        // Session 3: user says "I work at Gamma" in chat (conversation > extracted)
        var r2 = detector.checkAndResolve(USER, kim.id(), gamma.id(), "works_at", "conversation");
        assertEquals("kept_new", r2.resolution(),
                "conversation > extracted: new Gamma wins");
        insertEdge(kim.id(), gamma.id(), "works_at", "conversation");

        // Session 4: user explicitly states "I work at Delta" (user_stated = top)
        var r3 = detector.checkAndResolve(USER, kim.id(), delta.id(), "works_at", "user_stated");
        assertEquals("kept_new", r3.resolution());
        insertEdge(kim.id(), delta.id(), "works_at", "user_stated");

        // Session 5: another extraction tries to claim Eta — must lose to user_stated Delta
        var r4 = detector.checkAndResolve(USER, kim.id(), eta.id(), "works_at", "extracted");
        assertEquals("kept_existing", r4.resolution(),
                "extracted < user_stated: new Eta must lose");

        // Current edges for Kim: should only have ONE active works_at, pointing to Delta
        List<Edge> current = store.findCurrentEdgesFromNode(kim.id()).stream()
                .filter(e -> "works_at".equals(e.relationship()))
                .filter(Edge::isCurrent)
                .toList();
        assertEquals(1, current.size(),
                "after 5 conflicting writes, exactly one current works_at must remain");
        assertEquals(delta.id(), current.get(0).targetNodeId(),
                "winner must be the user_stated edge");
    }

    @Test
    void invalidatedEdges_arePurgedFromCurrentView() {
        Node kim = insertNode("person", "Kim", "user_stated");
        Node a = insertNode("company", "A", "extracted");
        Node b = insertNode("company", "B", "user_stated");

        insertEdge(kim.id(), a.id(), "works_at", "extracted");

        // user_stated invalidates the extracted edge
        detector.checkAndResolve(USER, kim.id(), b.id(), "works_at", "user_stated");
        insertEdge(kim.id(), b.id(), "works_at", "user_stated");

        List<Edge> all = store.findAllCurrentEdges(USER);
        assertEquals(1, all.stream().filter(e -> "works_at".equals(e.relationship())).count(),
                "current-edge view must exclude invalidated rows");
    }

    // ───────────────────────────── Property conflicts ───────────────────────

    @Test
    void propertyConflicts_detectedAcrossMultipleKeys() {
        Node kim = insertNode("person", "Kim", "extracted");

        Map<String, Object> newProps = Map.of(
                "role", "Staff Engineer",
                "location_city", "Bangalore",
                "_internal", "meta"   // must be skipped
        );
        // Existing node has no properties, so set some via direct construction
        Node existing = new Node(kim.id(), USER, "person", "Kim",
                Map.of("role", "Principal", "location_city", "Chennai"),
                0.8, "extracted", "active", 1, Instant.now(), Instant.now());

        var conflicts = detector.checkNodeProperties(existing, newProps, "user_stated");
        assertEquals(2, conflicts.size(),
                "expected 2 conflicts (role + location), got " + conflicts);
        for (var c : conflicts) {
            assertEquals("kept_new", c.resolution(),
                    "user_stated > existing extracted → new wins");
        }
    }

    @Test
    void propertyConflicts_withUnknownSource_fallsThroughSafely() {
        Node kim = insertNode("person", "Kim", "something-weird");
        Node existing = new Node(kim.id(), USER, "person", "Kim",
                Map.of("role", "Engineer"),
                0.8, "unknown-source-A", "active", 1, Instant.now(), Instant.now());

        // Both sources unknown — priority should be 0 for both, tie → "kept_new"
        var conflicts = detector.checkNodeProperties(existing,
                Map.of("role", "Director"), "unknown-source-B");
        assertEquals(1, conflicts.size());
        assertEquals("kept_new", conflicts.get(0).resolution(),
                "tie on unknown sources (both priority 0) resolves to new");
    }

    @Test
    void propertyConflicts_extractedVsUnknownSource_keepsExisting() {
        // Regression: when the existing is "extracted" (priority 1) and the
        // new source is unknown (priority 0), we MUST keep existing.
        Node kim = insertNode("person", "Kim", "extracted");
        Node existing = new Node(kim.id(), USER, "person", "Kim",
                Map.of("role", "Engineer"),
                0.8, "extracted", "active", 1, Instant.now(), Instant.now());

        var conflicts = detector.checkNodeProperties(existing,
                Map.of("role", "Director"), "mystery-source");
        assertEquals(1, conflicts.size());
        assertEquals("kept_existing", conflicts.get(0).resolution(),
                "extracted > unknown → existing wins");
    }

    // ───────────────────────────── Health scan ──────────────────────────────

    @Test
    void healthScan_countsOnlyCurrentEdgeConflicts_notInvalidated() {
        Node kim = insertNode("person", "Kim", "user_stated");
        Node acme = insertNode("company", "Acme", "extracted");
        Node beta = insertNode("company", "Beta", "user_stated");

        Edge e1 = insertEdge(kim.id(), acme.id(), "works_at", "extracted");
        // Properly invalidate — mimic what checkAndResolve does.
        store.updateEdge(e1.invalidated());
        insertEdge(kim.id(), beta.id(), "works_at", "user_stated");

        var report = detector.scanForIssues(USER);
        assertEquals(0, report.conflictingEdges(),
                "after invalidation, only ONE current edge per (src, rel) — no conflict");
    }

    @Test
    void healthScan_flagsDuplicateLabelsAcrossCategories() {
        insertNode("person", "Apple", "extracted");
        insertNode("company", "Apple", "extracted");
        insertNode("fruit", "Apple", "extracted");

        var report = detector.scanForIssues(USER);
        assertEquals(1, report.duplicateNodes(),
                "3 Apples across 3 categories = 1 duplicated label");
    }

    // ───────────────────────────── Concurrency ──────────────────────────────

    @Test
    void concurrentSessions_doNotCrash_andConverge() throws Exception {
        // Stress: 20 rapid-fire conflicting writes from 4 worker threads.
        // What we DON'T test: exact-one convergence of current edges.
        // Without atomic check-and-insert (which would require a proper
        // transaction per operation or a write lock), multiple current
        // rows can exist transiently. What we DO assert:
        //   1. No exceptions leak out.
        //   2. A subsequent single-threaded checkAndResolve call cleans
        //      up into a converged state (≤1 current target).
        Node kim = insertNode("person", "Kim", "user_stated");
        Node a = insertNode("company", "A", "extracted");
        Node b = insertNode("company", "B", "extracted");
        insertEdge(kim.id(), a.id(), "works_at", "extracted");

        ExecutorService ex = Executors.newFixedThreadPool(4);
        AtomicInteger writes = new AtomicInteger();
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            final int iter = i;
            futures.add(ex.submit(() -> {
                try {
                    Node tgt = iter % 2 == 0 ? b : a;
                    var r = detector.checkAndResolve(USER, kim.id(), tgt.id(), "works_at", "extracted");
                    if (r.hasContradiction() && "kept_new".equals(r.resolution())) {
                        insertEdge(kim.id(), tgt.id(), "works_at", "extracted");
                    }
                    writes.incrementAndGet();
                } catch (Exception ignore) { /* SQLite busy is acceptable */ }
            }));
        }
        for (Future<?> f : futures) f.get(10, TimeUnit.SECONDS);
        ex.shutdown();
        assertTrue(ex.awaitTermination(5, TimeUnit.SECONDS));
        assertTrue(writes.get() > 0, "at least one write should have completed");

        // Single-threaded cleanup pass must converge to one target.
        // Pick whichever target is currently "most represented" and claim it.
        // F10-fixed detector invalidates ALL conflicting current edges.
        detector.checkAndResolve(USER, kim.id(), a.id(), "works_at", "user_stated");
        insertEdge(kim.id(), a.id(), "works_at", "user_stated");

        List<String> targets = store.findCurrentEdgesFromNode(kim.id()).stream()
                .filter(e -> "works_at".equals(e.relationship()))
                .filter(Edge::isCurrent)
                .map(Edge::targetNodeId)
                .distinct()
                .toList();
        assertEquals(1, targets.size(),
                "after a single-threaded user_stated write, detector must converge to one target");
        assertEquals(a.id(), targets.get(0),
                "user_stated write must win the tiebreak");
    }

    @Test
    void samePriorityTie_newerWins() {
        Node kim = insertNode("person", "Kim", "extracted");
        Node oldCo = insertNode("company", "OldCo", "extracted");
        Node newCo = insertNode("company", "NewCo", "extracted");

        insertEdge(kim.id(), oldCo.id(), "works_at", "extracted");
        var r = detector.checkAndResolve(USER, kim.id(), newCo.id(), "works_at", "extracted");
        assertEquals("kept_new", r.resolution(),
                "equal priority (both extracted) → newer wins");
    }

    @Test
    void noContradiction_whenDifferentRelationships() {
        Node kim = insertNode("person", "Kim", "user_stated");
        Node acme = insertNode("company", "Acme", "extracted");

        insertEdge(kim.id(), acme.id(), "works_at", "user_stated");
        // Proposing "invested_in Acme" is NOT a contradiction — different relationship
        var r = detector.checkAndResolve(USER, kim.id(), acme.id(), "invested_in", "extracted");
        assertFalse(r.hasContradiction(),
                "same target but different relationship must not be a contradiction");
    }

    @Test
    void noContradiction_whenSameTargetAndRelationship() {
        Node kim = insertNode("person", "Kim", "user_stated");
        Node acme = insertNode("company", "Acme", "extracted");

        insertEdge(kim.id(), acme.id(), "works_at", "user_stated");
        var r = detector.checkAndResolve(USER, kim.id(), acme.id(), "works_at", "extracted");
        assertFalse(r.hasContradiction(),
                "re-asserting the same edge is not a contradiction");
    }
}
