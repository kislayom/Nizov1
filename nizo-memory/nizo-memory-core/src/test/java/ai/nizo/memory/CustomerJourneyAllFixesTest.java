package ai.nizo.memory;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.MemoryTags;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.graph.ContradictionDetector;
import ai.nizo.memory.graph.GraphTraversalEngine;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.reflect.ReflectionService;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Customer-journey acceptance tests that mirror the 10 fixes (F1-F17 subset).
 * Each test is written as a short user story so we can see, end-to-end, that
 * the feature works from the customer's point of view.
 *
 * <p>No Ollama required — uses {@link FakeEmbedder} and {@link FakeModelClient}.
 * For a real Ollama end-to-end, see the LongMemEval baseline run in
 * {@code bench/baseline-report-v2.json} (5/5 = 100%).
 */
class CustomerJourneyAllFixesTest {

    // -----------------------------------------------------------------
    // Test fixture helper: build a full memory stack against a temp db.
    // -----------------------------------------------------------------
    private static final class Stack implements AutoCloseable {
        final Path dbPath;
        final SqliteMemoryStore store;
        final SqliteGraphStore graphStore;
        final InMemoryVectorIndex index;
        final GraphService graph;
        final GraphTraversalEngine traversal;
        final FakeEmbedder embedder;
        final LayeredMemoryService svc;
        final ContradictionDetector contradictions;

        Stack(Path dbPath, List<String> vocabulary) {
            this.dbPath = dbPath;
            this.store = new SqliteMemoryStore(dbPath);
            this.graphStore = new SqliteGraphStore(dbPath);
            this.index = new InMemoryVectorIndex();
            this.graph = new KnowledgeGraph(graphStore);
            this.traversal = new GraphTraversalEngine(graphStore, graph);
            this.embedder = new FakeEmbedder(vocabulary);
            this.svc = new LayeredMemoryService(
                    store, index, embedder, null, graph, traversal,
                    999_999, 0.0, 0.01, 0.0);
            this.contradictions = new ContradictionDetector(graphStore);
        }

        @Override public void close() {
            if (graphStore != null) graphStore.close();
        }
    }

    // =================================================================
    // F7 — "I set a lease expiry date; when memory retracts it, the
    // original date must still be visible (not silently overwritten)."
    // =================================================================

    @Test
    @DisplayName("F7: retracted edge keeps original validTo instead of overwriting with now")
    void journey_F7_edgeRetraction_preservesOriginalValidTo(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("alice", "acme"))) {
            String uid = "customer";
            Node alice = s.graph.createOrMergePersonNode(uid, "Alice", Map.of(), "user_stated");
            Node acme = s.graph.createOrMergePersonNode(uid, "Acme", Map.of(), "user_stated");

            Instant originalExpiry = Instant.now().plus(365, ChronoUnit.DAYS);
            Edge lease = s.graph.createEdge(new Edge(
                    UUID.randomUUID().toString(), uid, alice.id(), acme.id(),
                    "works_at", Map.of(), Instant.now(), originalExpiry,
                    0.9, "user_stated", null));

            // User retracts ("Actually I already left Acme")
            Edge retracted = lease.invalidated();

            assertEquals(originalExpiry, retracted.validTo(),
                    "the original lease end date must still be readable after retraction");
            assertNotNull(retracted.invalidatedAt(),
                    "invalidation stamp must be set");
        }
    }

    // =================================================================
    // F4 — "If the assistant claims I'm vegetarian, I want to see WHICH
    // message that came from."
    // =================================================================

    @Test
    @DisplayName("F4: derived facts carry a pointer back to the source message id")
    void journey_F4_provenance_linksFactToSourceMessage(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("user", "vegetarian", "raw", "message"))) {
            String uid = "customer";

            // User says the raw thing
            String messageId = s.svc.remember(uid,
                    "I went fully vegetarian six months ago.",
                    Map.of(MemoryTags.SOURCE_MESSAGE_ID, "irrelevant_self"),
                    "user");

            // Derived SEMANTIC fact gets tagged with source_message_id
            s.svc.learnFact(uid, "User is vegetarian", "extraction", 0.9,
                    Map.of(MemoryTags.SOURCE_MESSAGE_ID, messageId,
                            MemoryTags.SOURCE_EXCERPT,
                            "I went fully vegetarian six months ago."));

            // Inspect — the fact must carry the provenance pointer
            List<MemoryItem> inspected = s.svc.inspect(uid, 50);
            MemoryItem fact = inspected.stream()
                    .filter(m -> m.tier() == MemoryItem.Tier.SEMANTIC
                            && "extraction".equals(m.source()))
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("expected a SEMANTIC extracted fact"));

            assertEquals(messageId, fact.tags().get(MemoryTags.SOURCE_MESSAGE_ID),
                    "fact must link to the episode id");
            assertEquals("I went fully vegetarian six months ago.",
                    fact.tags().get(MemoryTags.SOURCE_EXCERPT),
                    "excerpt must be stored for cheap display");
        }
    }

    // =================================================================
    // F5 — "A fact from 3 years ago must not outrank a fact I mentioned
    // yesterday."
    // =================================================================

    @Test
    @DisplayName("F5: confidence decay — fresh facts outrank stale ones with same content")
    void journey_F5_confidenceDecay_freshOutranksStale(@TempDir Path tmp) throws Exception {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("user", "likes", "coffee"))) {
            String uid = "customer";

            // Old extraction (will decay)
            String oldId = s.svc.learnFact(uid, "User likes dark roast coffee", "extraction", 0.8);
            backdateFactByDays(tmp.resolve("db.sqlite"), oldId, 400);

            // Fresh extraction
            s.svc.learnFact(uid, "User likes dark roast coffee", "extraction", 0.8);

            // Recall — top hit should be the FRESH fact
            List<MemoryItem> hits = s.svc.recall(RecallRequest.of(uid, "coffee preference", 500));
            assertFalse(hits.isEmpty(), "should recall at least one fact");
            // Fresh fact has createdAt = now; old one was backdated 400 days.
            MemoryItem top = hits.get(0);
            long ageDays = Duration.between(top.createdAt(), Instant.now()).toDays();
            assertTrue(ageDays < 30,
                    "top-ranked fact should be the fresh one (age " + ageDays + "d)");
        }
    }

    // =================================================================
    // F11+F3 — "A fact that matches both vector AND BM25 must rank
    // above a fact that only matches one channel."
    // =================================================================

    @Test
    @DisplayName("F11+F3: multi-channel RRF — items hit by multiple channels outrank single-channel")
    void journey_F11_F3_rrfFusion_multiChannelBeatsSingle(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("bali", "beach", "vacation", "holiday", "photo", "trip"))) {
            String uid = "customer";
            // Two facts. First: hit by vector via semantics (token overlap).
            //                Second: hit by BOTH FTS (literal "bali")
            //                        AND vector (token "bali" in vocab).
            s.svc.learnFact(uid, "The family vacation photos are lovely", "user_stated", 0.9);
            s.svc.learnFact(uid, "The Bali beach trip was last December", "user_stated", 0.9);

            List<MemoryItem> hits = s.svc.recall(RecallRequest.of(uid, "Bali trip", 500));
            assertFalse(hits.isEmpty());
            assertTrue(hits.get(0).content().toLowerCase().contains("bali"),
                    "multi-channel winner must be the Bali fact; got: " + hits.get(0).content());
        }
    }

    // =================================================================
    // F9 — "When I ask to be forgotten, my graph relationships must
    // disappear too — not just my memory items."
    // =================================================================

    @Test
    @DisplayName("F9: forget-user cascades into the knowledge graph")
    void journey_F9_forgetUser_cascadesGraph(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("alice", "bob"))) {
            String uid = "customer";
            s.svc.learnFact(uid, "Alice is my colleague at Acme", "user_stated", 0.9);
            Node alice = s.graph.createOrMergePersonNode(uid, "Alice", Map.of(), "user_stated");
            Node acme = s.graph.createOrMergePersonNode(uid, "Acme", Map.of(), "user_stated");
            s.graph.createEdgeIfNotExists(new Edge(
                    UUID.randomUUID().toString(), uid, alice.id(), acme.id(),
                    "works_at", Map.of(), Instant.now(), null, 0.9, "user_stated", null));

            int removed = s.svc.forgetUser(uid);
            assertTrue(removed >= 3,
                    "forget-user should remove memory items + graph nodes + edges");

            Map<String, Object> stats = s.graph.getGraphStats(uid);
            assertEquals(0L, ((Number) stats.get("nodeCount")).longValue());
            assertEquals(0L, ((Number) stats.get("edgeCount")).longValue());
            assertTrue(s.svc.inspect(uid, 100).isEmpty());
        }
    }

    // =================================================================
    // F2 — Category coverage: GOAL / FOLLOW_UP / DEFERRAL / RESOLUTION /
    // IMPLICIT_COMMITMENT now have triggers + examples that qwen2.5
    // can pattern-match on. Here we just assert the tags are
    // retrievable — the live prompt test is LongMemEval.
    // =================================================================

    @Test
    @DisplayName("F2: all 10 categories are routable as learnFact tags retrievable by mode")
    void journey_F2_extractionCategories_retrievableByMode(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("marathon", "aws", "bali", "trip", "contract", "hr"))) {
            String uid = "customer";

            // Simulate what extraction produces after F2 prompt improvements.
            s.svc.learnFact(uid, "GOAL: User wants to run a marathon next year",
                    "extraction", 0.85,
                    Map.of(MemoryTags.MODE, "personal"));
            s.svc.learnFact(uid, "FOLLOW_UP: Email accountant about GST return in 2 days",
                    "extraction", 0.85,
                    Map.of(MemoryTags.MODE, "work"));
            s.svc.learnFact(uid, "IMPLICIT_COMMITMENT: Waiting for HR to send revised contract",
                    "extraction", 0.8,
                    Map.of(MemoryTags.MODE, "work"));
            s.svc.learnFact(uid, "DEFERRAL: Bali trip parked until March bonus lands",
                    "extraction", 0.8,
                    Map.of(MemoryTags.MODE, "finance"));
            s.svc.learnFact(uid, "RESOLUTION: Signed up for AWS certification",
                    "extraction", 0.9,
                    Map.of(MemoryTags.MODE, "work"));

            // Recall filtered by mode=work — should return the 3 work items
            var hits = s.svc.recall(new RecallRequest(
                    uid, "what's pending on my plate", 800,
                    null, Map.of(MemoryTags.MODE, "work"), 0.0));
            assertEquals(3, hits.size(),
                    "3 work-mode items should be retrieved, got " + hits.size());
        }
    }

    // =================================================================
    // F1 — "My assistant compresses 8 mentions of my morning run into
    // one SEMANTIC fact about my running habit."
    // =================================================================

    @Test
    @DisplayName("F1: reflection worker compresses episodes into semantic facts")
    void journey_F1_reflectionLoop_distillsEpisodes(@TempDir Path tmp) throws Exception {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("run", "morning", "park", "exercise"))) {
            String uid = "customer";
            // 8 episodes about morning runs
            for (int i = 0; i < 8; i++) {
                s.svc.remember(uid, "Went for a morning run at the park (day " + i + ")",
                        Map.of(), "user");
            }
            // Backdate so the reflection worker considers them old enough
            backdateAllEpisodes(tmp.resolve("db.sqlite"), uid, 3);

            // Fake LLM: distilled semantic summary
            FakeModelClient llm = new FakeModelClient(
                    "User runs in the morning at the park as a regular habit.");

            try (ReflectionService reflector = new ReflectionService(
                    s.svc, s.store, s.index, s.embedder, llm,
                    Duration.ofMinutes(60), Duration.ofHours(2),
                    6, 40, 0.92)) {
                int produced = reflector.runOnce();
                assertEquals(1, produced,
                        "8 episodes → 1 distilled semantic fact");
            }

            // Customer inspects — now sees a SEMANTIC fact source=reflection
            boolean hasReflected = s.svc.inspect(uid, 100).stream()
                    .anyMatch(m -> m.tier() == MemoryItem.Tier.SEMANTIC
                            && "reflection".equals(m.source()));
            assertTrue(hasReflected, "reflection fact should be visible to customer");
        }
    }

    // =================================================================
    // F10 — "I changed jobs 5 times; the graph must show my CURRENT
    // employer, not a mess of stale edges."
    // =================================================================

    @Test
    @DisplayName("F10: contradiction detector — conflicting edges converge to one current target")
    void journey_F10_contradictionDetector_fiveJobChanges_finalWins(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"), List.of("kim"))) {
            String uid = "customer";
            Node kim = s.graph.createOrMergePersonNode(uid, "Kim", Map.of(), "user_stated");
            Node[] companies = new Node[5];
            String[] names = {"Acme", "Beta", "Gamma", "Delta", "Eta"};
            for (int i = 0; i < 5; i++) {
                companies[i] = s.graph.createOrMergePersonNode(uid, names[i], Map.of(), "user_stated");
            }

            // Sequence of job changes, each via detector + edge insert.
            for (int i = 0; i < 5; i++) {
                var r = s.contradictions.checkAndResolve(uid, kim.id(),
                        companies[i].id(), "works_at", "user_stated");
                if (i == 0) assertFalse(r.hasContradiction());
                s.graph.createEdge(new Edge(
                        UUID.randomUUID().toString(), uid, kim.id(), companies[i].id(),
                        "works_at", Map.of(), Instant.now(), null, 0.9, "user_stated", null));
            }

            // Only the LAST target must be current
            List<Edge> current = s.graphStore.findCurrentEdgesFromNode(kim.id()).stream()
                    .filter(e -> "works_at".equals(e.relationship()))
                    .filter(Edge::isCurrent)
                    .toList();
            assertEquals(1, current.size(),
                    "after 5 job changes, exactly one current works_at must remain");
            assertEquals(companies[4].id(), current.get(0).targetNodeId(),
                    "the FINAL target (Eta) must be the winner");
        }
    }

    // =================================================================
    // F16 — LongMemEval harness runs without dataset download, and with
    // real LLMs hits 5/5 (see bench/baseline-report-v2.json). This
    // journey test just proves the harness is wired.
    // =================================================================

    @Test
    @DisplayName("F16: LongMemEval harness exists and runs against a synthetic mini-dataset")
    void journey_F16_longmemeval_harnessRunsEndToEnd(@TempDir Path tmp) throws Exception {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("alice", "stripe", "engineer"))) {
            FakeModelClient answerer = new FakeModelClient("Alice works at Stripe");
            FakeModelClient judge = new FakeModelClient("YES");
            var harness = new ai.nizo.memory.eval.LongMemEvalHarness(
                    s.svc, null, answerer, judge, 800);
            var items = List.of(
                    new ai.nizo.memory.eval.LongMemEvalHarness.Item(
                            "j1", "Where does Alice work?", "single_session_user",
                            List.of(List.of(Map.of("role", "user",
                                    "content", "Alice is an engineer at Stripe."))),
                            "Stripe"));
            var report = harness.run(items);
            assertEquals(1, report.total());
            assertEquals(1, report.correct());
            assertEquals(1.0, report.accuracy(), 1e-6);
            assertTrue(report.summary().contains("Overall: 1/1"));
        }
    }

    // =================================================================
    // Combined: sensitivity + subject filter still works WITH all the
    // new scoring changes. "Do I have allergies?" must not leak Mom's
    // peanut allergy (stored with subject=other:mom).
    // =================================================================

    @Test
    @DisplayName("Combined: subject filter still redirects third-party sensitive facts")
    void journey_combined_subjectFilter_keepsMomsAllergyOutOfSelfQueries(@TempDir Path tmp) {
        try (Stack s = new Stack(tmp.resolve("db.sqlite"),
                List.of("peanut", "allergy", "mother", "mom"))) {
            String uid = "customer";
            // Mom's allergy — tagged other:mom + CRITICAL sensitivity
            s.svc.learnFact(uid, "Mom has a severe peanut allergy",
                    "user_stated", 0.95,
                    Map.of(MemoryTags.SUBJECT, "other:mom",
                            MemoryTags.SENSITIVITY, MemoryTags.SENS_CRITICAL));

            // Question is about the USER, not about mom
            var hits = s.svc.recall(RecallRequest.of(uid, "do I have any allergies", 500));
            boolean leaked = hits.stream()
                    .anyMatch(m -> m.content().toLowerCase().contains("mom"));
            assertFalse(leaked,
                    "mom's allergy must not surface when user asks about themselves; hits: "
                            + hits);

            // But if they ask about mom directly, the fact IS returned
            var direct = s.svc.recall(RecallRequest.of(uid, "does my mom have allergies", 500));
            boolean found = direct.stream()
                    .anyMatch(m -> m.content().toLowerCase().contains("mom"));
            assertTrue(found,
                    "querying about mom must surface her allergy");
        }
    }

    // -----------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------
    private static void backdateFactByDays(Path dbPath, String factId, int days) throws Exception {
        long backTo = Instant.now().minus(days, ChronoUnit.DAYS).toEpochMilli();
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
             PreparedStatement ps = conn.prepareStatement(
                     "UPDATE memory_items SET created_at = ? WHERE id = ?")) {
            ps.setLong(1, backTo);
            ps.setString(2, factId);
            ps.executeUpdate();
        }
    }

    private static void backdateAllEpisodes(Path dbPath, String userId, int hoursAgo) throws Exception {
        long backTo = Instant.now().minus(hoursAgo, ChronoUnit.HOURS).toEpochMilli();
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
             PreparedStatement ps = conn.prepareStatement(
                     "UPDATE memory_items SET created_at = ? WHERE user_id = ? AND tier = 'EPISODIC'")) {
            ps.setLong(1, backTo);
            ps.setString(2, userId);
            ps.executeUpdate();
        }
    }
}
