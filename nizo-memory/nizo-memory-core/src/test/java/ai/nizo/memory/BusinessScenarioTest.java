package ai.nizo.memory;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Business-level tests. Each test simulates a real user scenario and verifies
 * the memory service delivers what a human assistant would:
 *
 * - Remembers people, relationships, preferences from conversation
 * - Surfaces pending actions when asked "what do I need to do?"
 * - Recalls the right facts when context changes
 * - Consolidates scattered mentions into durable knowledge
 * - Doesn't lose important things over time
 * - Handles the "morning briefing" use case: what should I tell the user today?
 */
class BusinessScenarioTest {

    /** Vocabulary that lets FakeEmbedder create meaningful similarity between related topics */
    private static final List<String> VOCAB = List.of(
            // people
            "sarah", "wife", "spouse", "mom", "mother", "dad", "father", "rahul", "friend", "boss",
            // work
            "work", "office", "meeting", "project", "deadline", "promotion", "google", "apple",
            // actions
            "call", "email", "send", "book", "schedule", "remind", "todo", "action", "pending", "task",
            // finance
            "stock", "invest", "portfolio", "hdfc", "reliance", "bank", "mutual fund",
            // personal
            "birthday", "anniversary", "trip", "japan", "vacation", "doctor", "dentist",
            // preferences
            "prefer", "dark mode", "morning", "coffee", "tea", "vegetarian",
            // time
            "today", "tomorrow", "friday", "december", "upcoming", "next week",
            // behavior
            "always", "usually", "every", "daily", "routine", "habit",
            // emotions
            "happy", "worried", "excited", "stressed"
    );

    private SqliteMemoryStore store;
    private InMemoryVectorIndex index;
    private FakeEmbedder embedder;
    private MemoryService memory;

    @TempDir Path tmpDir;

    @BeforeEach
    void setup() {
        store = new SqliteMemoryStore(tmpDir.resolve("test.db"));
        index = new InMemoryVectorIndex();
        embedder = new FakeEmbedder(VOCAB);
        // Summariser for consolidation: extracts facts from episodic memories
        FakeModelClient summariser = new FakeModelClient(prompt -> {
            // Simulate what an LLM would extract as stable facts
            if (prompt.contains("sarah") || prompt.contains("Sarah")) {
                return "Sarah is the user's wife\nUser is planning a trip to Japan with Sarah";
            }
            if (prompt.contains("dentist") || prompt.contains("doctor")) {
                return "User needs to call the dentist\nUser has a doctor appointment pending";
            }
            if (prompt.contains("stock") || prompt.contains("invest")) {
                return "User is interested in value investing\nUser tracks HDFC Bank stock";
            }
            if (prompt.contains("prefer") || prompt.contains("dark mode") || prompt.contains("coffee")) {
                return "User prefers dark mode\nUser drinks coffee in the morning";
            }
            return "No stable facts extracted";
        });
        memory = new LayeredMemoryService(store, index, embedder, summariser, 100, 0.1);
    }

    @AfterEach
    void teardown() { store.close(); }

    // ===== SCENARIO 1: The assistant should remember people =====

    @Test
    @DisplayName("User mentions wife Sarah → recall 'family' surfaces Sarah")
    void remembersPeopleFromConversation() {
        memory.remember("default", "My wife Sarah and I went to dinner last night",
                Map.of("kind", "message"), "conversation");
        memory.learnFact("default", "Sarah is the user's wife", "conversation", 0.9);

        List<MemoryItem> results = memory.recall(RecallRequest.of("who is Sarah", 500));
        assertFalse(results.isEmpty(), "Should recall something about Sarah");
        String combined = results.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(combined.toLowerCase().contains("sarah"), "Results should mention Sarah");
        assertTrue(combined.toLowerCase().contains("wife"), "Results should mention the relationship");
    }

    @Test
    @DisplayName("User mentions multiple people → recall distinguishes them")
    void distinguishesDifferentPeople() {
        memory.learnFact("default", "Sarah is the user's wife", "conversation", 0.9);
        memory.learnFact("default", "Rahul is the user's friend from college", "conversation", 0.85);
        memory.learnFact("default", "Mom's birthday is December 15", "user_stated", 0.95);

        // Query about friend should surface Rahul, not Sarah
        List<MemoryItem> friendResults = memory.recall(RecallRequest.of("my friend", 500));
        String friendText = friendResults.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(friendText.contains("Rahul"), "Friend query should surface Rahul");

        // Query about family should surface wife and mom
        List<MemoryItem> familyResults = memory.recall(RecallRequest.of("my wife", 500));
        String familyText = familyResults.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(familyText.contains("Sarah"), "Wife query should surface Sarah");
    }

    // ===== SCENARIO 2: Pending actions and tasks =====

    @Test
    @DisplayName("User says 'call dentist' → retrieving actions surfaces all of them")
    void surfacesPendingActions() {
        // User mentions actions across different conversations — tagged as actions
        memory.remember("default", "I need to call the dentist this week",
                Map.of("kind", "action"), "conversation");
        memory.remember("default", "Don't forget to email Rahul about the project",
                Map.of("kind", "action"), "conversation");
        memory.remember("default", "Should book flight tickets for the Japan trip",
                Map.of("kind", "action"), "conversation");

        // Backend job retrieves all actions using tag-based query
        List<MemoryItem> actions = memory.recall(new RecallRequest(
                "default", "pending actions", 1000, null, Map.of("kind", "action"), 0.0));
        assertEquals(3, actions.size(), "Should find all 3 tagged actions");

        String allActions = actions.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        assertTrue(allActions.contains("dentist"), "Should include dentist task");
        assertTrue(allActions.contains("email") || allActions.contains("Rahul"),
                "Should include email task");
        assertTrue(allActions.contains("book") || allActions.contains("Japan"),
                "Should include booking task");
    }

    @Test
    @DisplayName("Actions tagged as 'action' can be filtered with requiredTags")
    void filterActionsByTag() {
        memory.remember("default", "Had a great dinner with Sarah", Map.of("kind", "message"), "conversation");
        memory.remember("default", "Need to call the dentist", Map.of("kind", "action"), "conversation");
        memory.remember("default", "User prefers morning meetings", Map.of("kind", "preference"), "extracted");

        // Filter to only actions
        List<MemoryItem> actionsOnly = memory.recall(new RecallRequest(
                "default", "everything", 1000, null, Map.of("kind", "action"), 0.0));
        assertEquals(1, actionsOnly.size(), "Should return exactly the 1 action");
        assertTrue(actionsOnly.get(0).content().contains("dentist"));
    }

    // ===== SCENARIO 3: Reminders and important dates =====

    @Test
    @DisplayName("User mentions mom's birthday → 'upcoming dates' surfaces it")
    void surfacesRemindersAndDates() {
        memory.learnFact("default", "Mom's birthday is December 15", "user_stated", 0.95);
        memory.learnFact("default", "Wedding anniversary is June 3", "user_stated", 0.95);
        memory.learnFact("default", "User has a doctor appointment next Friday", "conversation", 0.8);

        List<MemoryItem> upcoming = memory.recall(
                RecallRequest.of("upcoming dates birthdays anniversaries reminders", 1000));
        assertFalse(upcoming.isEmpty(), "Should surface date-related memories");

        String combined = upcoming.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(combined.contains("birthday") || combined.contains("December"),
                "Should surface birthday");
    }

    // ===== SCENARIO 4: Preferences persist and are recallable =====

    @Test
    @DisplayName("User states preferences → recallable weeks later")
    void preferencesAreRecallable() {
        memory.learnFact("default", "User prefers dark mode in all applications", "user_stated", 0.9);
        memory.learnFact("default", "User is vegetarian", "user_stated", 0.95);
        memory.learnFact("default", "User drinks coffee every morning, never tea", "conversation", 0.8);

        // Simulate "weeks later" — preferences should still be in semantic tier
        List<MemoryItem> prefs = memory.recall(
                RecallRequest.of("what are my preferences", 500));
        assertFalse(prefs.isEmpty(), "Preferences should be recallable");

        String combined = prefs.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        // At least one preference should surface
        boolean hasAnyPref = combined.contains("dark mode")
                || combined.contains("vegetarian")
                || combined.contains("coffee");
        assertTrue(hasAnyPref, "Should recall at least one user preference");
    }

    // ===== SCENARIO 5: Investment interests =====

    @Test
    @DisplayName("User discusses stocks → financial context surfaces on market queries")
    void investmentInterestsRecalled() {
        memory.remember("default", "I've been watching HDFC Bank closely, might add to my position",
                Map.of("kind", "message"), "conversation");
        memory.learnFact("default", "User holds HDFC Bank stock and monitors it actively", "conversation", 0.85);
        memory.learnFact("default", "User follows value investing principles", "extracted", 0.7);

        List<MemoryItem> results = memory.recall(
                RecallRequest.of("stock portfolio investment", 500));
        String combined = results.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(combined.toLowerCase().contains("hdfc") || combined.toLowerCase().contains("invest"),
                "Should recall investment interests when asked about portfolio");
    }

    // ===== SCENARIO 6: Consolidation turns raw conversations into durable knowledge =====

    @Test
    @DisplayName("Multiple mentions of Sarah across conversations → consolidation creates stable fact")
    void consolidationDistillsKnowledge() {
        // Simulate multiple conversations mentioning Sarah
        memory.remember("default", "Sarah called me today about the trip", Map.of(), "conversation");
        memory.remember("default", "Discussed Japan itinerary with Sarah over dinner", Map.of(), "conversation");
        memory.remember("default", "Sarah prefers Kyoto over Tokyo", Map.of(), "conversation");
        memory.remember("default", "Need to check Sarah's passport renewal", Map.of(), "conversation");
        memory.remember("default", "Sarah suggested booking through her travel agent", Map.of(), "conversation");
        memory.remember("default", "We agreed on December for the Japan trip with Sarah", Map.of(), "conversation");
        memory.remember("default", "Sarah is excited about visiting temples", Map.of(), "conversation");
        memory.remember("default", "Should ask Sarah about hotel preferences", Map.of(), "conversation");

        // Before consolidation: only episodic tier has items
        Map<MemoryItem.Tier, Long> before = memory.stats("default");
        assertEquals(8, before.get(MemoryItem.Tier.EPISODIC), "Should have 8 episodic items");

        // Consolidation runs (like a background job would)
        memory.consolidate("default");

        // After: semantic facts should exist
        Map<MemoryItem.Tier, Long> after = memory.stats("default");
        assertTrue(after.get(MemoryItem.Tier.SEMANTIC) > 0,
                "Consolidation should create semantic facts from episodic memories");

        // The consolidated knowledge should be recallable
        List<MemoryItem> results = memory.recall(RecallRequest.of("Sarah trip Japan", 500));
        assertFalse(results.isEmpty(), "Consolidated knowledge should be recallable");
    }

    // ===== SCENARIO 7: Higher-confidence user statements outrank extracted facts =====

    @Test
    @DisplayName("User-stated fact ranks higher than system-extracted fact")
    void userStatedFactsRankHigher() {
        // System extracted a fact with lower confidence
        memory.learnFact("default", "User might work at Google", "extracted", 0.5);
        // User explicitly stated something with high confidence
        memory.learnFact("default", "I work at Apple as a senior engineer", "user_stated", 0.95);

        List<MemoryItem> results = memory.recall(RecallRequest.of("where does user work", 500));
        assertFalse(results.isEmpty());

        // The user-stated fact (Apple) should rank higher than the extracted one (Google)
        MemoryItem top = results.get(0);
        assertTrue(top.content().contains("Apple"),
                "User-stated fact should rank first: got '" + top.content() + "'");
    }

    // ===== SCENARIO 8: Morning briefing — what should the assistant tell the user? =====

    @Test
    @DisplayName("Morning briefing surfaces actions, reminders, and relevant context")
    void morningBriefingScenario() {
        // Build up memory over "several days"
        memory.learnFact("default", "User needs to call dentist by Friday", "conversation", 0.85);
        memory.learnFact("default", "Mom's birthday is coming up on December 15", "user_stated", 0.95);
        memory.learnFact("default", "User has a meeting with boss tomorrow about promotion", "conversation", 0.8);
        memory.learnFact("default", "HDFC Bank stock dropped 3% yesterday, user holds this stock", "system", 0.7);
        memory.remember("default", "User was stressed about the project deadline", Map.of(), "conversation");

        // Morning briefing query — use words that bridge to stored facts
        // (dentist/meeting/boss/promotion/birthday/stock) since FakeEmbedder
        // requires lexical overlap. In production a real embedder would
        // semantically match "morning briefing" to these topics.
        List<MemoryItem> briefing = memory.recall(
                RecallRequest.of("dentist meeting boss promotion birthday stock morning", 2000));

        assertFalse(briefing.isEmpty(), "Morning briefing should have content");
        assertTrue(briefing.size() >= 3,
                "Should surface multiple relevant items for briefing, got " + briefing.size());

        String all = briefing.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        // The briefing should contain actionable/important information
        boolean hasActionable = all.contains("dentist") || all.contains("meeting")
                || all.contains("birthday") || all.contains("HDFC");
        assertTrue(hasActionable, "Briefing should surface actionable items, got:\n" + all);
    }

    // ===== SCENARIO 9: Cross-conversation memory =====

    @Test
    @DisplayName("Fact from conversation 1 is recallable in conversation 2")
    void crossConversationRecall() {
        // Conversation 1: user mentions family
        memory.remember("default", "My wife Sarah works at a hospital as a surgeon",
                Map.of("session", "conv-1"), "conversation");
        memory.learnFact("default", "Sarah is user's wife and works as a surgeon at a hospital",
                "conversation", 0.9);

        // Conversation 2 (completely separate): user asks about Sarah
        List<MemoryItem> results = memory.recall(RecallRequest.of("tell me about Sarah", 500));
        assertFalse(results.isEmpty(), "Should recall facts from a different conversation");

        String combined = results.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(combined.contains("Sarah"), "Should recall Sarah from previous conversation");
        assertTrue(combined.contains("wife") || combined.contains("surgeon"),
                "Should recall relationship or occupation details");
    }

    // ===== SCENARIO 10: Behavioral patterns =====

    @Test
    @DisplayName("Repeated behaviors are captured and recallable")
    void behavioralPatternsRecallable() {
        memory.learnFact("default", "User checks stock portfolio every morning before 9am",
                "extracted", 0.75);
        memory.learnFact("default", "User always has a standup meeting at 10am on weekdays",
                "extracted", 0.8);
        memory.learnFact("default", "User usually goes for a walk after lunch",
                "conversation", 0.7);

        // Query with words that bridge to stored facts (stock/standup/walk/morning).
        // Real embedder would match "daily routine" semantically; FakeEmbedder needs overlap.
        List<MemoryItem> behaviors = memory.recall(
                RecallRequest.of("stock standup walk morning daily", 500));
        assertFalse(behaviors.isEmpty(), "Should recall behavioral patterns");

        String combined = behaviors.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        boolean hasBehavior = combined.contains("stock") || combined.contains("standup")
                || combined.contains("walk");
        assertTrue(hasBehavior, "Should surface at least one behavioral pattern");
    }

    // ===== SCENARIO 11: Token budget prevents context overflow =====

    @Test
    @DisplayName("Large memory doesn't overflow the agent's context window")
    void tokenBudgetPreventsOverflow() {
        // Store 50 facts (simulating months of use)
        for (int i = 0; i < 50; i++) {
            memory.learnFact("default", "Fact number " + i + " about the user's life: " +
                    "this is a moderately long piece of information that takes some tokens " +
                    "and represents real knowledge the system has accumulated over time.",
                    "conversation", 0.7 + (i % 3) * 0.1);
        }

        // Recall with tight budget (like what an agent would use)
        List<MemoryItem> results = memory.recall(RecallRequest.of("user facts", 200));
        int totalTokens = results.stream().mapToInt(MemoryItem::tokens).sum();

        assertTrue(totalTokens <= 200,
                "Total tokens (" + totalTokens + ") should not exceed budget of 200");
        assertFalse(results.isEmpty(), "Should still return some results within budget");
    }

    // ===== SCENARIO 12: Semantic tier outranks episodic for same topic =====

    @Test
    @DisplayName("Consolidated fact outranks raw conversation for same topic")
    void semanticOutranksEpisodicForSameTopic() {
        // Raw conversation (episodic)
        memory.remember("default", "Yeah Sarah and I talked about it, she thinks Japan would be fun in December",
                Map.of(), "conversation");

        // Consolidated fact (semantic) — what a summariser would produce
        memory.learnFact("default", "User is planning a trip to Japan in December with wife Sarah",
                "consolidation", 0.8);

        List<MemoryItem> results = memory.recall(RecallRequest.of("Japan trip Sarah", 500));
        assertFalse(results.isEmpty());

        // The clean semantic fact should rank higher than the raw conversation
        MemoryItem top = results.get(0);
        assertEquals(MemoryItem.Tier.SEMANTIC, top.tier(),
                "Semantic (consolidated) fact should rank above episodic conversation");
    }

    // ===== SCENARIO 13: The "what do you know about me" query =====

    @Test
    @DisplayName("'What do you know about me?' returns a useful profile summary")
    void whatDoYouKnowAboutMe() {
        memory.learnFact("default", "User's name is Kislay", "user_stated", 0.95);
        memory.learnFact("default", "User works at Apple as a senior engineer", "user_stated", 0.95);
        memory.learnFact("default", "User is married to Sarah", "conversation", 0.9);
        memory.learnFact("default", "User is interested in value investing and tracks HDFC Bank",
                "conversation", 0.8);
        memory.learnFact("default", "User prefers dark mode and drinks coffee every morning",
                "extracted", 0.75);

        List<MemoryItem> profile = memory.recall(
                RecallRequest.of("what do you know about me user profile", 1000));

        assertTrue(profile.size() >= 3,
                "Profile query should return multiple facts, got " + profile.size());

        String all = profile.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        // Should have a mix of personal, professional, and preference info
        boolean hasPersonal = all.contains("Kislay") || all.contains("Sarah");
        boolean hasProfessional = all.contains("Apple") || all.contains("engineer");
        boolean hasPreference = all.contains("dark mode") || all.contains("coffee")
                || all.contains("invest");
        int categories = (hasPersonal ? 1 : 0) + (hasProfessional ? 1 : 0) + (hasPreference ? 1 : 0);
        assertTrue(categories >= 2,
                "Profile should cover at least 2 life categories. Got:\n" + all);
    }

    // ===== SCENARIO 14: Emotional context is captured =====

    @Test
    @DisplayName("User's emotional state is captured and recallable")
    void emotionalContextCaptured() {
        memory.remember("default", "I'm really stressed about this deadline, boss is pushing hard",
                Map.of("kind", "message"), "conversation");
        memory.learnFact("default", "User is stressed about a work deadline and under pressure from boss",
                "extracted", 0.75);

        List<MemoryItem> results = memory.recall(
                RecallRequest.of("how is the user feeling stressed worried", 500));
        assertFalse(results.isEmpty(), "Should recall emotional context");

        String combined = results.stream().map(MemoryItem::content).collect(Collectors.joining(" "));
        assertTrue(combined.contains("stressed") || combined.contains("deadline") || combined.contains("pressure"),
                "Should surface stress-related context");
    }

    // ===== SCENARIO 15: Memory doesn't hallucinate — only returns what was stored =====

    @Test
    @DisplayName("Query about unknown topic returns empty, not fabricated results")
    void noHallucinationOnUnknownTopic() {
        memory.learnFact("default", "User works at Apple", "user_stated", 0.9);

        // Query about something never mentioned
        List<MemoryItem> results = memory.recall(
                RecallRequest.of("user's children school grades", 500));

        // Should either return empty or return unrelated items (not fabricated "user has 2 children")
        for (MemoryItem item : results) {
            assertFalse(item.content().contains("children"),
                    "Should not fabricate facts about children");
            assertFalse(item.content().contains("school"),
                    "Should not fabricate facts about school");
        }
    }
}
