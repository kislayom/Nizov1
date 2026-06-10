package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.extract.ExtractionPipeline;
import ai.nizo.memory.extract.GraphFactRouter;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Manual scenario test — simulates a real user's life over several weeks.
 *
 * This is NOT a unit test. This exercises the FULL STACK:
 *   User message → Extraction (LLM) → Graph (nodes + edges) → Memory (facts)
 *   Then: Various recall queries → inspect what comes back
 *
 * The FakeModelClient returns realistic extraction JSON for each message,
 * simulating what a real LLM would produce.
 */
class ManualScenarioTest {

    private static final List<String> VOCAB = List.of(
            // people & relationships
            "kislay", "sarah", "wife", "spouse", "rahul", "friend", "boss", "mom", "mother",
            "parent", "family", "colleague", "surgeon",
            // work
            "work", "google", "apple", "engineer", "manager", "project", "promotion",
            "principal", "senior", "left", "joined",
            // finance
            "stock", "hdfc", "reliance", "invest", "portfolio", "bank", "mutual",
            // dates & places
            "birthday", "trip", "japan", "india", "mumbai", "sydney", "doctor", "dentist",
            "december", "january", "friday", "monday", "next", "week", "month", "upcoming",
            // preferences & habits
            "prefer", "dark", "light", "mode", "coffee", "tea", "morning", "evening", "vegetarian",
            "switched", "headache",
            // actions & commitments
            "call", "email", "book", "schedule", "remind", "reminder", "action", "task", "pending",
            "follow", "commitment", "promised", "send", "document", "review",
            // products
            "bluetooth", "speaker", "jbl", "price", "buy", "discount", "expensive",
            // deferred / postponed
            "defer", "deferred", "postpone", "postponed", "pass", "decide", "decided",
            // work events
            "meeting", "standup", "deadline", "report", "quarterly",
            // learning
            "rust", "python", "learn", "course", "programming",
            // briefing
            "briefing", "today", "important", "urgent"
    );

    private SqliteMemoryStore memStore;
    private SqliteGraphStore graphStore;
    private InMemoryVectorIndex index;
    private FakeEmbedder embedder;
    private GraphService graph;
    private MemoryService memory;
    private ExtractionService extraction;

    @TempDir Path tmpDir;

    @BeforeEach
    void setup() {
        Path dbPath = tmpDir.resolve("scenario.db");
        memStore = new SqliteMemoryStore(dbPath);
        graphStore = new SqliteGraphStore(dbPath);
        index = new InMemoryVectorIndex();
        embedder = new FakeEmbedder(VOCAB);
        graph = new KnowledgeGraph(graphStore);

        // The extraction LLM — concatenates ALL messages so our responder
        // can match on user content (FakeModelClient only reads messages[0]).
        ai.nizo.memory.api.model.ModelClient extractionModel = new ai.nizo.memory.api.model.ModelClient() {
            @Override
            public ai.nizo.memory.api.model.ModelCapability capability() {
                return new ai.nizo.memory.api.model.ModelCapability("fake", "fake",
                        Set.of(ai.nizo.memory.api.Modality.TEXT), Set.of(ai.nizo.memory.api.Modality.TEXT),
                        8192, false, true, 0, 0, 10);
            }
            @Override
            public ai.nizo.memory.api.model.ModelResponse complete(ai.nizo.memory.api.model.ModelRequest request) {
                String userMsg = request.messages().isEmpty() ? "" :
                        request.messages().get(request.messages().size() - 1).text();
                return ai.nizo.memory.api.model.ModelResponse.text(
                        extractionResponse(userMsg),
                        ai.nizo.memory.api.model.ModelResponse.Usage.zero());
            }
        };

        // Consolidation summariser
        FakeModelClient summariser = new FakeModelClient(prompt -> "No stable facts extracted");

        memory = new LayeredMemoryService(memStore, index, embedder, summariser,
                graph, null, 100, 0.1);
        extraction = new ExtractionPipeline(extractionModel, new GraphFactRouter(graph), memory);
    }

    @AfterEach
    void teardown() {
        memStore.close();
        graphStore.close();
    }

    @Test
    void simulateRealUserLifeOverWeeks() {
        String userId = "kislay";

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  MANUAL SCENARIO TEST — Simulating real user life");
        System.out.println("=".repeat(80));

        // ===== WEEK 1: User introduces himself =====
        section("WEEK 1 — Day 1: User introduces himself");

        simulate(userId, "I'm Kislay, I work at Google as a senior engineer. I live in Sydney.");
        simulate(userId, "My wife Sarah is a surgeon at Royal Prince Alfred Hospital.");
        simulate(userId, "My friend Rahul from college works at Amazon in Seattle.");

        // Check: who does the system think I am?
        query(userId, "who am I");
        query(userId, "who is Sarah");
        query(userId, "who is Rahul");
        showGraph(userId);

        // ===== WEEK 1: Day 3 — Work stuff =====
        section("WEEK 1 — Day 3: Work context");

        simulate(userId, "I have a meeting with my boss about the promotion on Friday.");
        simulate(userId, "Need to review the quarterly report before the meeting.");

        query(userId, "meeting boss promotion");
        query(userId, "what do I need to do");

        // ===== WEEK 1: Day 5 — Personal stuff =====
        section("WEEK 1 — Day 5: Personal life");

        simulate(userId, "My mom's birthday is December 15th. She lives in Mumbai.");
        simulate(userId, "I should call the dentist this week, been putting it off.");
        simulate(userId, "Sarah and I are thinking about a trip to Japan in December.");

        query(userId, "mom birthday");
        query(userId, "dentist");
        query(userId, "Japan trip");

        // ===== WEEK 2: Preferences and habits =====
        section("WEEK 2 — Preferences and daily life");

        simulate(userId, "I prefer dark mode in all my apps. Can't stand light themes.");
        simulate(userId, "I'm vegetarian and drink coffee every morning, never tea.");
        simulate(userId, "I check my stock portfolio every morning before standup at 10am.");

        query(userId, "user preferences");
        query(userId, "morning routine");
        query(userId, "what does user eat drink");

        // ===== WEEK 2: Investment interests =====
        section("WEEK 2 — Investment and finance");

        simulate(userId, "I've been watching HDFC Bank closely, thinking about adding to my position.");
        simulate(userId, "I follow value investing principles, big fan of Buffett's approach.");
        simulate(userId, "Should I look at Reliance too? But the price seems high right now.");

        query(userId, "stock portfolio investment");
        query(userId, "HDFC Bank");

        // ===== WEEK 3: Deferred interest — the bluetooth speaker =====
        section("WEEK 3 — The bluetooth speaker scenario");

        simulate(userId, "I was looking at the JBL Flip 6 bluetooth speaker. Great reviews.");
        simulate(userId, "But ₹12,999 is too expensive for a bluetooth speaker. I'll pass for now.");

        query(userId, "bluetooth speaker");
        query(userId, "JBL Flip 6");
        query(userId, "things I wanted to buy but didn't");

        // ===== WEEK 3: Commitments and deferrals =====
        section("WEEK 3 — Commitments and deferrals");

        simulate(userId, "I promised Rahul I'd send him the project design doc by next week.");
        simulate(userId, "I'll think about whether to learn Rust or stick with Python. Not sure yet.");

        query(userId, "Rahul promise document");
        query(userId, "learn programming language");

        // ===== WEEK 4: Life update — job change =====
        section("WEEK 4 — Major life change: job switch");

        simulate(userId, "Big news — I left Google and joined Apple last month as a principal engineer!");

        query(userId, "where do I work");
        query(userId, "Google");
        query(userId, "Apple");

        // ===== WEEK 4: Preference change =====
        section("WEEK 4 — Preference change");

        simulate(userId, "Actually I switched to light mode recently. Dark mode was giving me headaches.");

        query(userId, "dark mode light mode preference");

        // ===== CROSS-CUTTING QUERIES =====
        section("CROSS-CUTTING: Complex recall queries");

        query(userId, "what do you know about me");
        query(userId, "my family");
        query(userId, "pending actions and reminders");
        query(userId, "upcoming dates birthdays");
        query(userId, "things I deferred or postponed");
        query(userId, "morning briefing today");

        // ===== FINAL STATE =====
        section("FINAL STATE");
        showGraph(userId);
        showStats(userId);

        System.out.println("\n" + "=".repeat(80));
        System.out.println("  END OF SCENARIO");
        System.out.println("=".repeat(80) + "\n");
    }

    // ===== Helpers =====

    private void simulate(String userId, String message) {
        System.out.println("\n  📝 USER: \"" + message + "\"");
        ExtractionResult result = extraction.extract(userId, message);
        if (result.hasExtractions()) {
            System.out.println("     → Extracted " + result.count() + " items: " + result.types());
        } else {
            System.out.println("     → No extractions");
        }
    }

    private void query(String userId, String queryText) {
        System.out.println("\n  🔍 QUERY: \"" + queryText + "\"");
        List<MemoryItem> results = memory.recall(RecallRequest.of(userId, queryText, 1000));
        if (results.isEmpty()) {
            System.out.println("     → (empty — nothing found)");
        } else {
            for (int i = 0; i < Math.min(results.size(), 5); i++) {
                MemoryItem m = results.get(i);
                String content = m.content().length() > 100
                        ? m.content().substring(0, 100) + "..."
                        : m.content();
                System.out.printf("     %d. [%s|%.2f] %s%n",
                        i + 1, m.tier().name().substring(0, 3), m.confidence(), content);
            }
            if (results.size() > 5) {
                System.out.println("     ... and " + (results.size() - 5) + " more");
            }
        }
    }

    private void showGraph(String userId) {
        System.out.println("\n  📊 KNOWLEDGE GRAPH:");
        List<Node> nodes = graph.getNodesByCategory(userId, "person");
        nodes.addAll(graph.getNodesByCategory(userId, "organization"));
        nodes.addAll(graph.getNodesByCategory(userId, "location"));
        nodes.addAll(graph.getNodesByCategory(userId, "preference"));
        nodes.addAll(graph.getNodesByCategory(userId, "goal"));
        nodes.addAll(graph.getNodesByCategory(userId, "topic"));
        nodes.addAll(graph.getNodesByCategory(userId, "event"));

        if (nodes.isEmpty()) {
            System.out.println("     (empty graph)");
            return;
        }
        System.out.println("     Nodes (" + nodes.size() + "):");
        for (Node n : nodes) {
            String props = n.properties().isEmpty() ? ""
                    : " " + n.properties().entrySet().stream()
                    .filter(e -> e.getValue() != null && !e.getValue().toString().isBlank())
                    .map(e -> e.getKey() + "=" + e.getValue())
                    .collect(Collectors.joining(", ", "{", "}"));
            System.out.printf("       [%s] %s (conf=%.2f, mentions=%d)%s%n",
                    n.category(), n.label(), n.confidence(), n.mentionCount(), props);
        }

        // Collect edges for all nodes
        Set<String> seenEdges = new HashSet<>();
        List<Edge> allEdges = new ArrayList<>();
        for (Node n : nodes) {
            for (Edge e : graph.getEdgesForNode(userId, n.id())) {
                if (e.isCurrent() && seenEdges.add(e.id())) {
                    allEdges.add(e);
                }
            }
        }
        if (!allEdges.isEmpty()) {
            System.out.println("     Edges (" + allEdges.size() + "):");
            for (Edge e : allEdges) {
                String srcLabel = nodes.stream().filter(n -> n.id().equals(e.sourceNodeId()))
                        .map(Node::label).findFirst().orElse(e.sourceNodeId().substring(0, 8));
                String tgtLabel = nodes.stream().filter(n -> n.id().equals(e.targetNodeId()))
                        .map(Node::label).findFirst().orElse(e.targetNodeId().substring(0, 8));
                System.out.printf("       %s -[%s]-> %s%n", srcLabel, e.relationship(), tgtLabel);
            }
        }
    }

    private void showStats(String userId) {
        System.out.println("\n  📈 MEMORY STATS:");
        Map<MemoryItem.Tier, Long> stats = memory.stats(userId);
        for (var e : stats.entrySet()) {
            System.out.printf("     %s: %d%n", e.getKey(), e.getValue());
        }
    }

    private void section(String title) {
        System.out.println("\n" + "-".repeat(70));
        System.out.println("  " + title);
        System.out.println("-".repeat(70));
    }

    /**
     * Simulates what a real LLM would return for each user message.
     * This is the most critical part — the extraction quality determines everything.
     */
    private String extractionResponse(String prompt) {
        // The prompt contains the user message at the end. Extract it.
        String msg = prompt.toLowerCase();

        if (msg.contains("i'm kislay") || msg.contains("i work at google")) {
            return """
                {"PROFILE": {"name": "Kislay", "occupation": "senior engineer", "company": "Google", "location_city": "Sydney", "location_country": "Australia"}}
                """;
        }
        if (msg.contains("wife sarah") && msg.contains("surgeon")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Sarah", "relationship_type": "spouse", "context": "surgeon at Royal Prince Alfred Hospital"}]}
                """;
        }
        if (msg.contains("friend rahul") && msg.contains("amazon")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Rahul", "relationship_type": "friend", "context": "college friend, works at Amazon in Seattle"}]}
                """;
        }
        if (msg.contains("meeting") && msg.contains("boss") && msg.contains("promotion")) {
            return """
                {"EVENT": [{"summary": "Meeting with boss about promotion on Friday", "event_type": "meeting", "participants": ["boss"], "emotional_valence": "mixed"}],
                 "FOLLOW_UP": [{"description": "Prepare for promotion discussion meeting on Friday", "follow_up_days": 2}]}
                """;
        }
        if (msg.contains("quarterly report") && msg.contains("review")) {
            return """
                {"IMPLICIT_COMMITMENT": [{"description": "Review quarterly report before Friday meeting", "commitment_type": "need_to", "estimated_timeframe": 2}]}
                """;
        }
        if (msg.contains("mom") && msg.contains("birthday") && msg.contains("december")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Mom", "relationship_type": "parent", "context": "lives in Mumbai, birthday December 15th"}],
                 "EVENT": [{"summary": "Mom's birthday on December 15th", "event_type": "milestone", "date": "2026-12-15"}]}
                """;
        }
        if (msg.contains("dentist") && msg.contains("call")) {
            return """
                {"FOLLOW_UP": [{"description": "Call the dentist — been putting it off", "follow_up_days": 3}],
                 "IMPLICIT_COMMITMENT": [{"description": "Call dentist this week", "commitment_type": "need_to", "estimated_timeframe": 5}]}
                """;
        }
        if (msg.contains("japan") && msg.contains("trip") && msg.contains("december")) {
            return """
                {"GOAL": [{"title": "Trip to Japan with Sarah in December", "category": "personal", "priority": "high"}],
                 "IMPLICIT_COMMITMENT": [{"description": "Plan Japan trip with Sarah for December", "commitment_type": "planning_to", "related_person": "Sarah", "estimated_timeframe": 30}]}
                """;
        }
        if (msg.contains("light mode") && msg.contains("switched")) {
            return """
                {"PREFERENCE": [{"subject": "UI theme", "assertion": "switched to light mode, dark mode was giving headaches", "domain": "technical"}],
                 "RESOLUTION": [{"decision": "UI theme preference", "choice": "Switched from dark mode to light mode due to headaches"}]}
                """;
        }
        if (msg.contains("dark mode") && msg.contains("prefer")) {
            return """
                {"PREFERENCE": [{"subject": "UI theme", "assertion": "prefers dark mode in all applications", "domain": "technical"}]}
                """;
        }
        if (msg.contains("vegetarian") && msg.contains("coffee")) {
            return """
                {"PREFERENCE": [{"subject": "diet", "assertion": "vegetarian", "domain": "lifestyle"},
                                {"subject": "morning drink", "assertion": "coffee every morning, never tea", "domain": "lifestyle"}]}
                """;
        }
        if (msg.contains("stock portfolio") && msg.contains("standup")) {
            return """
                {"PREFERENCE": [{"subject": "morning routine", "assertion": "checks stock portfolio every morning before 10am standup", "domain": "work"}]}
                """;
        }
        if (msg.contains("hdfc") && msg.contains("watching")) {
            return """
                {"INVESTMENT_INTEREST": {"tickers": ["HDFC"], "style": "value", "risk_appetite": "moderate"}}
                """;
        }
        if (msg.contains("value investing") && msg.contains("buffett")) {
            return """
                {"PREFERENCE": [{"subject": "investing style", "assertion": "follows value investing principles, fan of Buffett", "domain": "finance"}]}
                """;
        }
        if (msg.contains("reliance") && msg.contains("price") && msg.contains("high")) {
            return """
                {"DEFERRAL": [{"decision": "Whether to invest in Reliance stock", "context": "Price seems too high right now", "days_until_followup": 14}]}
                """;
        }
        if (msg.contains("jbl flip 6") && msg.contains("great reviews")) {
            return """
                {"PREFERENCE": [{"subject": "bluetooth speaker", "assertion": "interested in JBL Flip 6, great reviews", "domain": "personal"}]}
                """;
        }
        if (msg.contains("12,999") && msg.contains("expensive") && msg.contains("pass")) {
            return """
                {"DEFERRAL": [{"decision": "Buying JBL Flip 6 bluetooth speaker", "context": "Price ₹12,999 is too expensive, will pass for now", "days_until_followup": 30}]}
                """;
        }
        if (msg.contains("promised rahul") && msg.contains("design doc")) {
            return """
                {"IMPLICIT_COMMITMENT": [{"description": "Send Rahul the project design doc", "commitment_type": "will_do", "related_person": "Rahul", "estimated_timeframe": 7}]}
                """;
        }
        if (msg.contains("rust") && msg.contains("python") && msg.contains("not sure")) {
            return """
                {"DEFERRAL": [{"decision": "Whether to learn Rust or stick with Python", "context": "Undecided, needs more thought", "days_until_followup": 14}]}
                """;
        }
        if (msg.contains("left google") && msg.contains("joined apple")) {
            return """
                {"PROFILE": {"name": "Kislay", "occupation": "principal engineer", "company": "Apple"},
                 "EVENT": [{"summary": "Left Google and joined Apple as principal engineer", "event_type": "milestone", "emotional_valence": "positive"}],
                 "RESOLUTION": [{"decision": "Career change", "choice": "Joined Apple as principal engineer, leaving Google"}]}
                """;
        }
        // (handled above — "light mode" + "switched" check is before "dark mode" + "prefer")

        return "{}";
    }
}
