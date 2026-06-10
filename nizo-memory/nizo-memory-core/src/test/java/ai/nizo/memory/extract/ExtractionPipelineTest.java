package ai.nizo.memory.extract;

import ai.nizo.memory.api.extract.ExtractionCategory;
import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.testsupport.FakeModelClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.*;

class ExtractionPipelineTest {

    private static final String USER = "user-1";

    private SqliteGraphStore graphStore;
    private KnowledgeGraph kg;
    private StubMemoryService memory;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        graphStore = new SqliteGraphStore(tmp.resolve("ep.db"));
        kg = new KnowledgeGraph(graphStore);
        memory = new StubMemoryService();
    }

    @AfterEach
    void tearDown() {
        if (graphStore != null) graphStore.close();
    }

    // ---- helpers ----

    private ExtractionPipeline pipeline(String cannedJson) {
        FakeModelClient model = new FakeModelClient(cannedJson);
        GraphFactRouter router = new GraphFactRouter(kg);
        return new ExtractionPipeline(model, router, memory);
    }

    private ExtractionPipeline pipeline(FakeModelClient model) {
        GraphFactRouter router = new GraphFactRouter(kg);
        return new ExtractionPipeline(model, router, memory);
    }

    // ===== isCommandOnly =====

    @Test
    void isCommandOnly_filtersAnalyzeCommand() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly("analyze AAPL"));
    }

    @Test
    void isCommandOnly_filtersShortMessages() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly("hi"));
        assertTrue(ep.isCommandOnly("hello there"));
    }

    @Test
    void isCommandOnly_filtersRemindMe() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly("remind me to buy groceries tomorrow"));
    }

    @Test
    void isCommandOnly_filtersNullAndBlank() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly(null));
        assertTrue(ep.isCommandOnly(""));
        assertTrue(ep.isCommandOnly("   "));
    }

    @Test
    void isCommandOnly_allowsNormalConversation() {
        ExtractionPipeline ep = pipeline("{}");
        assertFalse(ep.isCommandOnly("My wife Sarah and I went to dinner last night at that Italian place"));
    }

    @Test
    void isCommandOnly_filtersSearchCommand() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly("search for restaurants nearby"));
    }

    @Test
    void isCommandOnly_filtersFindCommand() {
        ExtractionPipeline ep = pipeline("{}");
        assertTrue(ep.isCommandOnly("find the best coffee shop"));
    }

    // ===== Extract PROFILE =====

    @Test
    void extractProfile_createsPersonNodeAndStoresInMemory() {
        String json = """
                {"PROFILE": {"name": "Kislay", "company": "Google", "location_city": "Sydney"}}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I'm Kislay, I work at Google in Sydney");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.PROFILE));
        assertEquals(1, result.count());

        // Memory should have profile facts stored
        assertFalse(memory.learnedFacts.isEmpty());
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("Kislay")));
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("Google")));
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("Sydney")));

        // Graph should have a person node with is_self
        Node self = kg.findSelfNode(USER);
        assertNotNull(self);
        assertEquals("Kislay", self.label());
    }

    // ===== Extract RELATIONSHIP =====

    @Test
    void extractRelationship_createsPersonNodeWithRelationshipType() {
        String json = """
                {"RELATIONSHIP": [{"person_name": "Sarah", "relationship_type": "spouse", "context": "lives together"}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "My wife Sarah and I have been living together for 5 years");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.RELATIONSHIP));

        // Memory should have a relationship fact
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("Sarah")));
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("spouse")));

        // Graph should have a person node for Sarah
        Optional<Node> sarah = kg.resolveEntity(USER, "Sarah");
        assertTrue(sarah.isPresent());
        assertEquals("spouse", sarah.get().properties().get("relationship_type"));
    }

    // ===== Extract PREFERENCE =====

    @Test
    void extractPreference_createsPreferenceNode() {
        String json = """
                {"PREFERENCE": [{"subject": "Coffee", "assertion": "Prefers flat white", "domain": "lifestyle"}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I always order a flat white at coffee shops, can't stand drip coffee");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.PREFERENCE));

        // Memory fact
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("Coffee")));
    }

    // ===== Extract EVENT =====

    @Test
    void extractEvent_createsEventNodeAndStoresEpisodicMemory() {
        String json = """
                {"EVENT": [{"summary": "Team dinner at Italian restaurant", "event_type": "experience", "date": "2024-12-15", "participants": ["Sarah", "Bob"], "emotional_valence": "positive"}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "Last night we had a great team dinner at the Italian restaurant with Sarah and Bob");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.EVENT));

        // Episodic memory
        assertFalse(memory.rememberedItems.isEmpty());
        assertTrue(memory.rememberedItems.stream().anyMatch(r -> r.contains("Italian")));
    }

    // ===== Extract GOAL =====

    @Test
    void extractGoal_createsGoalNodeAndStoresSemanticMemory() {
        String json = """
                {"GOAL": [{"title": "Run a marathon", "description": "Complete a full marathon by end of 2025", "category": "health", "priority": "high"}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I really want to run a marathon before the end of next year");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.GOAL));

        // Semantic memory fact
        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("marathon")));
    }

    // ===== Extract DEFERRAL =====

    @Test
    void extractDeferral_storesDeferralInMemory() {
        String json = """
                {"DEFERRAL": [{"decision": "Whether to upgrade to M4 MacBook", "context": "Waiting for reviews", "days_until_followup": 7}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I'll decide on the M4 MacBook later once reviews come out, let me think about it for a week");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.DEFERRAL));

        assertTrue(memory.learnedFacts.stream().anyMatch(f -> f.contains("MacBook")));
    }

    // ===== Extract IMPLICIT_COMMITMENT =====

    @Test
    void extractImplicitCommitment_storesCommitmentInMemory() {
        String json = """
                {"IMPLICIT_COMMITMENT": [{"description": "Review Sarah's proposal", "commitment_type": "will_do", "related_person": "Sarah", "estimated_timeframe": 3}]}
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I'll take a look at Sarah's proposal sometime this week");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.IMPLICIT_COMMITMENT));

        // Episodic memory
        assertTrue(memory.rememberedItems.stream().anyMatch(r -> r.contains("Sarah")));
    }

    // ===== Multiple categories in single extraction =====

    @Test
    void multipleCategoriesInSingleExtraction() {
        String json = """
                {
                    "PROFILE": {"name": "Kislay", "company": "Google"},
                    "RELATIONSHIP": [{"person_name": "Sarah", "relationship_type": "spouse", "context": "wife"}],
                    "PREFERENCE": [{"subject": "Work style", "assertion": "Prefers remote work", "domain": "work"}]
                }
                """;
        ExtractionPipeline ep = pipeline(json);
        ExtractionResult result = ep.extract(USER, "I'm Kislay from Google, my wife Sarah and I both prefer working from home");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.PROFILE));
        assertTrue(result.types().contains(ExtractionCategory.RELATIONSHIP));
        assertTrue(result.types().contains(ExtractionCategory.PREFERENCE));
        assertEquals(3, result.count()); // 1 profile + 1 relationship + 1 preference
    }

    // ===== Malformed JSON =====

    @Test
    void malformedJsonResponse_gracefulDegradation() {
        String malformedJson = "This is not valid JSON at all {broken";
        ExtractionPipeline ep = pipeline(malformedJson);
        ExtractionResult result = ep.extract(USER, "My wife Sarah and I went to dinner");

        // Should return empty rather than throw
        assertFalse(result.hasExtractions());
        assertEquals(0, result.count());
    }

    // ===== Empty extraction response =====

    @Test
    void emptyExtractionResponse_returnsEmpty() {
        ExtractionPipeline ep = pipeline("{}");
        ExtractionResult result = ep.extract(USER, "Just a normal day at the office, nothing special");

        assertFalse(result.hasExtractions());
        assertEquals(0, result.count());
    }

    // ===== Null/blank message =====

    @Test
    void nullMessage_skipped() {
        ExtractionPipeline ep = pipeline("{}");
        ExtractionResult result = ep.extract(USER, null);
        assertFalse(result.hasExtractions());
    }

    @Test
    void blankMessage_skipped() {
        ExtractionPipeline ep = pipeline("{}");
        ExtractionResult result = ep.extract(USER, "   ");
        assertFalse(result.hasExtractions());
    }

    @Test
    void shortMessage_skipped() {
        ExtractionPipeline ep = pipeline("{}");
        ExtractionResult result = ep.extract(USER, "ok thanks");
        assertFalse(result.hasExtractions());
    }

    // ===== JSON with markdown fencing =====

    @Test
    void jsonWithMarkdownFencing_parsedCorrectly() {
        String fencedJson = """
                Here is the extraction:
                ```json
                {"PROFILE": {"name": "Kislay"}}
                ```
                """;
        ExtractionPipeline ep = pipeline(fencedJson);
        ExtractionResult result = ep.extract(USER, "My name is Kislay and I am a software engineer at a major tech company");

        assertTrue(result.hasExtractions());
        assertTrue(result.types().contains(ExtractionCategory.PROFILE));
    }

    // ===== Model invocation tracking =====

    @Test
    void extractionCallsModelOnce() {
        FakeModelClient model = new FakeModelClient("""
                {"PROFILE": {"name": "Alice"}}
                """);
        ExtractionPipeline ep = pipeline(model);
        ep.extract(USER, "I'm Alice and I work at a startup in San Francisco");

        assertEquals(1, model.invocations.get());
    }

    @Test
    void commandOnlyMessage_doesNotCallModel() {
        FakeModelClient model = new FakeModelClient("{}");
        ExtractionPipeline ep = pipeline(model);
        ep.extract(USER, "analyze AAPL");

        assertEquals(0, model.invocations.get());
    }

    // ===== Stub MemoryService =====

    /**
     * A minimal stub that records calls to learnFact and remember for assertion.
     */
    private static final class StubMemoryService implements MemoryService {
        final CopyOnWriteArrayList<String> learnedFacts = new CopyOnWriteArrayList<>();
        final CopyOnWriteArrayList<String> rememberedItems = new CopyOnWriteArrayList<>();

        @Override
        public String remember(String userId, String content, Map<String, String> tags, String source) {
            rememberedItems.add(content);
            return UUID.randomUUID().toString();
        }

        @Override
        public String learnFact(String userId, String fact, String source, double confidence) {
            learnedFacts.add(fact);
            return UUID.randomUUID().toString();
        }

        @Override
        public List<MemoryItem> recall(RecallRequest request) {
            return List.of();
        }

        @Override
        public void consolidate(String userId) { }

        @Override
        public Map<MemoryItem.Tier, Long> stats(String userId) {
            return Map.of();
        }
    }
}
