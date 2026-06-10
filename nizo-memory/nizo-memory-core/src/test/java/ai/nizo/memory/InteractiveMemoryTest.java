package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelCapability;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.api.model.ModelResponse;
import ai.nizo.memory.api.Modality;
import ai.nizo.memory.extract.ExtractionPipeline;
import ai.nizo.memory.extract.GraphFactRouter;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Interactive memory test — builds a rich, complex memory from scratch,
 * then runs queries. User-driven: the printed output is the test result.
 */
class InteractiveMemoryTest {

    private static final List<String> VOCAB = List.of(
        // people
        "kislay", "priya", "wife", "spouse", "arjun", "son", "asha", "mother", "mom",
        "vikram", "boss", "manager", "neha", "colleague", "amit", "friend", "college",
        "dr", "sharma", "doctor", "dentist", "mehta", "accountant",
        // work
        "work", "stripe", "engineer", "principal", "fintech", "payments", "api",
        "startup", "founded", "quit", "resigned", "promotion", "team", "lead",
        "project", "atlas", "deadline", "sprint", "standup", "review",
        // finance
        "stock", "tata", "motors", "infosys", "reliance", "invest", "portfolio",
        "sip", "mutual", "fund", "zerodha", "demat", "nifty", "sensex",
        "bitcoin", "crypto", "etf", "dividend",
        // personal
        "birthday", "anniversary", "wedding", "trip", "goa", "japan", "bali",
        "bangalore", "mumbai", "singapore", "home", "apartment", "rent", "emi",
        // preferences
        "prefer", "dark", "light", "mode", "coffee", "chai", "tea", "morning",
        "vegetarian", "vegan", "spicy", "south", "indian", "food",
        "iphone", "android", "pixel", "mac", "windows", "linux", "vim", "vscode",
        // health
        "gym", "yoga", "run", "marathon", "weight", "diet", "sleep", "meditation",
        "allergy", "peanut", "lactose", "gluten",
        // actions & commitments
        "call", "email", "book", "schedule", "remind", "action", "task", "pending",
        "promise", "send", "document", "report", "submit", "pay", "bill", "renew",
        "passport", "visa", "insurance",
        // products & purchases
        "buy", "price", "expensive", "discount", "sale", "amazon", "flipkart",
        "laptop", "macbook", "headphone", "sony", "airpods", "camera", "kindle",
        // emotions & life
        "happy", "stressed", "worried", "excited", "anxious", "frustrated",
        "deferred", "postponed", "decided", "changed", "switched", "moved",
        // learning
        "rust", "golang", "python", "kubernetes", "aws", "course", "learn",
        "certification", "exam", "study",
        // time
        "monday", "friday", "december", "january", "march", "june", "october",
        "today", "tomorrow", "next", "week", "month", "year", "upcoming",
        "morning", "evening", "night", "daily", "weekly",
        // misc
        "important", "urgent", "briefing", "family", "personal", "professional"
    );

    private SqliteMemoryStore memStore;
    private SqliteGraphStore graphStore;
    private InMemoryVectorIndex index;
    private GraphService graph;
    private MemoryService memory;
    private ExtractionService extraction;

    @TempDir Path tmpDir;

    @BeforeEach
    void setup() {
        Path dbPath = tmpDir.resolve("interactive.db");
        memStore = new SqliteMemoryStore(dbPath);
        graphStore = new SqliteGraphStore(dbPath);
        index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(VOCAB);
        graph = new KnowledgeGraph(graphStore);

        ModelClient extractionModel = new ModelClient() {
            public ModelCapability capability() {
                return new ModelCapability("fake", "fake",
                    Set.of(Modality.TEXT), Set.of(Modality.TEXT), 8192, false, true, 0, 0, 10);
            }
            public ModelResponse complete(ModelRequest request) {
                String userMsg = request.messages().isEmpty() ? "" :
                        request.messages().get(request.messages().size() - 1).text();
                return ModelResponse.text(respond(userMsg), ModelResponse.Usage.zero());
            }
        };

        memory = new LayeredMemoryService(memStore, index, embedder, null,
                graph, null, 100, 0.1);
        extraction = new ExtractionPipeline(extractionModel, new GraphFactRouter(graph), memory);
    }

    @AfterEach
    void teardown() { memStore.close(); graphStore.close(); }

    @Test
    void buildComplexMemoryAndQuery() {
        String u = "kislay";
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  BUILDING COMPLEX MEMORY — everything stored, then you query");
        System.out.println("=".repeat(80));

        // ===== IDENTITY =====
        sim(u, "I'm Kislay Sinha, 32 years old, I work at Stripe as a principal engineer on the payments API team. I live in Bangalore with my wife Priya and our 4-year-old son Arjun.");
        sim(u, "My mother Asha lives in Mumbai. Dad passed away 3 years ago.");
        sim(u, "My best friend Amit from IIT works at Google in Singapore. We talk every week.");
        sim(u, "Neha is my colleague at Stripe, she leads the Atlas project. Vikram is my engineering manager.");

        // ===== HEALTH =====
        sim(u, "I have a peanut allergy — serious, carry an EpiPen. Also lactose intolerant.");
        sim(u, "I run 5km every morning at 6am. Training for the Bangalore marathon in March.");
        sim(u, "Dr. Sharma is my family doctor in Koramangala. Need to schedule Arjun's vaccination.");
        sim(u, "Been sleeping badly — averaging 5 hours. Priya says I should try meditation.");

        // ===== WORK =====
        sim(u, "The Atlas project has a hard deadline January 15th. We're behind by 2 sprints.");
        sim(u, "I have a standup every day at 10am and a sprint review every Friday at 3pm.");
        sim(u, "Vikram hinted at a promotion to Staff Engineer if Atlas ships on time.");
        sim(u, "I need to submit my self-review by December 20th for the performance cycle.");

        // ===== FINANCES =====
        sim(u, "I have a SIP of ₹25,000/month in Nifty 50 ETF through Zerodha.");
        sim(u, "Holding 500 shares of Tata Motors, bought at ₹420. Current price around ₹680.");
        sim(u, "Also watching Infosys — want to start a position if it drops below ₹1,400.");
        sim(u, "Home loan EMI is ₹52,000/month. 18 years remaining. HDFC Bank.");
        sim(u, "Considering putting some money into Bitcoin but Priya is against it. Will think about it.");

        // ===== PREFERENCES =====
        sim(u, "I use Vim for coding and Linux on my work machine. Mac for personal stuff.");
        sim(u, "Strict vegetarian. Love South Indian food, especially dosa and filter coffee.");
        sim(u, "I prefer dark mode everywhere. iPhone user, won't switch to Android.");
        sim(u, "Morning person — most productive before noon. Useless after 9pm.");

        // ===== PERSONAL / UPCOMING =====
        sim(u, "Priya's birthday is January 8th. Need to plan something special this time.");
        sim(u, "Our wedding anniversary is March 22nd. Thinking about a trip to Bali.");
        sim(u, "Mom's 60th birthday is in October. Arjun and I are planning a surprise party in Mumbai.");
        sim(u, "Promised Amit I'd visit him in Singapore in February. Need to renew my passport first.");

        // ===== DEFERRED DECISIONS =====
        sim(u, "Was looking at the Sony WH-1000XM5 headphones. ₹28,000 is steep. Will wait for a sale.");
        sim(u, "Thinking about doing an AWS Solutions Architect certification but not sure if it's worth the time.");
        sim(u, "Priya wants us to move to a bigger apartment. I think we should wait till the promotion is confirmed.");

        // ===== EMOTIONAL CONTEXT =====
        sim(u, "Feeling really stressed about Atlas deadline. Haven't taken a day off in 3 months.");
        sim(u, "Excited about Arjun starting school next year. He got into DPS Bangalore.");

        // ===== LIFE UPDATE — JOB CHANGE =====
        sim(u, "Update: I got the promotion! Now Staff Engineer at Stripe. ₹15L raise.");
        sim(u, "With the promotion confirmed, Priya and I decided to go ahead with the apartment move.");

        // ===== PREFERENCE CHANGE =====
        sim(u, "Switched from dark mode to light mode. The eye strain was getting bad.");

        // ===== FINAL SUMMARY =====
        System.out.println("\n" + "=".repeat(80));
        System.out.println("  MEMORY BUILT — Here's what was stored:");
        System.out.println("=".repeat(80));

        showStats(u);
        showGraph(u);

        System.out.println("\n" + "-".repeat(80));
        System.out.println("  READY FOR QUERIES — ask me anything about Kislay's life");
        System.out.println("-".repeat(80));

        // Pre-run some queries so user can see the format
        query(u, "who is Kislay");
        query(u, "family");
        query(u, "where does he work");
        query(u, "health allergies");
        query(u, "pending tasks actions");
        query(u, "investment portfolio stocks");
        query(u, "things he deferred or postponed");
        query(u, "upcoming dates birthdays anniversaries");
        query(u, "what is he stressed about");
        query(u, "Sony headphones");
        query(u, "Priya");
        query(u, "apartment move decision");
        query(u, "morning routine");
        query(u, "passport Singapore Amit");

        System.out.println("\n" + "=".repeat(80));
    }

    // ===== Helpers =====

    private void sim(String userId, String message) {
        ExtractionResult r = extraction.extract(userId, message);
        String types = r.hasExtractions() ? r.types().toString() : "none";
        System.out.printf("  ✓ [%s] %s%n", types, message.length() > 90 ? message.substring(0, 90) + "..." : message);
    }

    private void query(String userId, String q) {
        System.out.println("\n  🔍 \"" + q + "\"");
        List<MemoryItem> results = memory.recall(RecallRequest.of(userId, q, 1500));
        if (results.isEmpty()) {
            System.out.println("     → (nothing found)");
        } else {
            for (int i = 0; i < Math.min(results.size(), 5); i++) {
                MemoryItem m = results.get(i);
                String c = m.content().length() > 110 ? m.content().substring(0, 110) + "..." : m.content();
                System.out.printf("     %d. [%s|%.2f] %s%n", i+1, m.tier().name().charAt(0), m.confidence(), c);
            }
            if (results.size() > 5) System.out.println("     ... +" + (results.size()-5) + " more");
        }
    }

    private void showStats(String u) {
        System.out.println("\n  📈 Memory: " + memory.stats(u));
    }

    private void showGraph(String u) {
        List<Node> all = new ArrayList<>();
        for (String cat : List.of("person","organization","location","preference","goal","topic","event")) {
            all.addAll(graph.getNodesByCategory(u, cat));
        }
        System.out.println("  📊 Graph: " + all.size() + " nodes");
        for (Node n : all) {
            System.out.printf("     [%s] %s (%.2f, %dx)%n", n.category(), n.label(), n.confidence(), n.mentionCount());
        }
        Set<String> seen = new HashSet<>();
        int edgeCount = 0;
        for (Node n : all) {
            for (Edge e : graph.getEdgesForNode(u, n.id())) {
                if (e.isCurrent() && seen.add(e.id())) edgeCount++;
            }
        }
        System.out.println("  📊 Edges: " + edgeCount);
    }

    // ===== Extraction responses =====

    private String respond(String msg) {
        String m = msg.toLowerCase();

        if (m.contains("kislay sinha") && m.contains("stripe") && m.contains("priya") && m.contains("arjun")) {
            return """
                {"PROFILE": {"name": "Kislay Sinha", "occupation": "principal engineer", "company": "Stripe", "location_city": "Bangalore", "location_country": "India", "industry": "fintech"},
                 "RELATIONSHIP": [
                   {"person_name": "Priya", "relationship_type": "spouse", "context": "lives together in Bangalore"},
                   {"person_name": "Arjun", "relationship_type": "child", "context": "4-year-old son"}
                 ]}""";
        }
        if (m.contains("mother asha") && m.contains("mumbai") && m.contains("dad passed")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Asha", "relationship_type": "parent", "context": "mother, lives in Mumbai"}],
                 "EVENT": [{"summary": "Father passed away 3 years ago", "event_type": "milestone", "emotional_valence": "negative"}]}""";
        }
        if (m.contains("best friend amit") && m.contains("iit") && m.contains("google") && m.contains("singapore")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Amit", "relationship_type": "friend", "context": "best friend from IIT, works at Google in Singapore, talks weekly"}]}""";
        }
        if (m.contains("neha") && m.contains("colleague") && m.contains("vikram") && m.contains("manager")) {
            return """
                {"RELATIONSHIP": [
                   {"person_name": "Neha", "relationship_type": "colleague", "context": "leads the Atlas project at Stripe"},
                   {"person_name": "Vikram", "relationship_type": "manager", "context": "engineering manager at Stripe"}
                 ]}""";
        }
        if (m.contains("peanut allergy") && m.contains("epipen") && m.contains("lactose")) {
            return """
                {"PREFERENCE": [
                   {"subject": "allergy", "assertion": "severe peanut allergy, carries EpiPen", "domain": "health"},
                   {"subject": "dietary restriction", "assertion": "lactose intolerant", "domain": "health"}
                 ]}""";
        }
        if (m.contains("run 5km") && m.contains("6am") && m.contains("marathon") && m.contains("march")) {
            return """
                {"PREFERENCE": [{"subject": "exercise routine", "assertion": "runs 5km every morning at 6am", "domain": "health"}],
                 "GOAL": [{"title": "Run Bangalore marathon in March", "category": "health", "priority": "high"}]}""";
        }
        if (m.contains("dr. sharma") && m.contains("arjun") && m.contains("vaccination")) {
            return """
                {"RELATIONSHIP": [{"person_name": "Dr. Sharma", "relationship_type": "acquaintance", "context": "family doctor in Koramangala"}],
                 "FOLLOW_UP": [{"description": "Schedule Arjun's vaccination with Dr. Sharma", "follow_up_days": 7}]}""";
        }
        if (m.contains("sleeping badly") && m.contains("5 hours") && m.contains("meditation")) {
            return """
                {"PREFERENCE": [{"subject": "sleep", "assertion": "averaging only 5 hours, poor sleep quality", "domain": "health"}],
                 "IMPLICIT_COMMITMENT": [{"description": "Try meditation for better sleep as Priya suggested", "commitment_type": "planning_to", "related_person": "Priya", "estimated_timeframe": 14}]}""";
        }
        if (m.contains("atlas") && m.contains("deadline") && m.contains("january 15") && m.contains("2 sprints")) {
            return """
                {"EVENT": [{"summary": "Atlas project deadline January 15th — behind by 2 sprints", "event_type": "deadline", "date": "2027-01-15", "emotional_valence": "negative"}]}""";
        }
        if (m.contains("standup") && m.contains("10am") && m.contains("sprint review") && m.contains("friday")) {
            return """
                {"PREFERENCE": [
                   {"subject": "daily standup", "assertion": "every day at 10am", "domain": "work"},
                   {"subject": "sprint review", "assertion": "every Friday at 3pm", "domain": "work"}
                 ]}""";
        }
        if (m.contains("vikram") && m.contains("promotion") && m.contains("staff engineer") && m.contains("atlas")) {
            return """
                {"EVENT": [{"summary": "Vikram hinted at promotion to Staff Engineer if Atlas ships on time", "event_type": "milestone", "emotional_valence": "positive"}]}""";
        }
        if (m.contains("self-review") && m.contains("december 20") && m.contains("performance")) {
            return """
                {"FOLLOW_UP": [{"description": "Submit self-review by December 20th for performance cycle", "follow_up_days": 5}],
                 "IMPLICIT_COMMITMENT": [{"description": "Write and submit self-review before December 20th", "commitment_type": "need_to", "estimated_timeframe": 5}]}""";
        }
        if (m.contains("sip") && m.contains("25,000") && m.contains("nifty") && m.contains("zerodha")) {
            return """
                {"INVESTMENT_INTEREST": {"tickers": ["NIFTY50ETF"], "style": "index", "risk_appetite": "moderate"},
                 "PREFERENCE": [{"subject": "investment platform", "assertion": "uses Zerodha for SIP of ₹25,000/month in Nifty 50 ETF", "domain": "finance"}]}""";
        }
        if (m.contains("tata motors") && m.contains("500 shares") && m.contains("420")) {
            return """
                {"INVESTMENT_INTEREST": {"tickers": ["TATAMOTORS"], "style": "value"},
                 "PREFERENCE": [{"subject": "stock holding", "assertion": "holds 500 shares of Tata Motors bought at ₹420, current ~₹680", "domain": "finance"}]}""";
        }
        if (m.contains("infosys") && m.contains("below") && m.contains("1,400")) {
            return """
                {"DEFERRAL": [{"decision": "Start position in Infosys stock", "context": "Waiting for price to drop below ₹1,400", "days_until_followup": 30}]}""";
        }
        if (m.contains("home loan") && m.contains("emi") && m.contains("52,000") && m.contains("hdfc")) {
            return """
                {"PREFERENCE": [{"subject": "home loan", "assertion": "EMI ₹52,000/month with HDFC Bank, 18 years remaining", "domain": "finance"}]}""";
        }
        if (m.contains("bitcoin") && m.contains("priya") && m.contains("against") && m.contains("think about")) {
            return """
                {"DEFERRAL": [{"decision": "Investing in Bitcoin", "context": "Considering it but wife Priya is against the idea", "days_until_followup": 30}]}""";
        }
        if (m.contains("vim") && m.contains("linux") && m.contains("mac")) {
            return """
                {"PREFERENCE": [
                   {"subject": "code editor", "assertion": "uses Vim for coding", "domain": "technical"},
                   {"subject": "work OS", "assertion": "Linux on work machine, Mac for personal", "domain": "technical"}
                 ]}""";
        }
        if (m.contains("vegetarian") && m.contains("south indian") && m.contains("dosa") && m.contains("filter coffee")) {
            return """
                {"PREFERENCE": [
                   {"subject": "diet", "assertion": "strict vegetarian", "domain": "lifestyle"},
                   {"subject": "food", "assertion": "loves South Indian food, especially dosa and filter coffee", "domain": "lifestyle"}
                 ]}""";
        }
        if (m.contains("dark mode everywhere") && m.contains("iphone") && m.contains("android")) {
            return """
                {"PREFERENCE": [
                   {"subject": "UI theme", "assertion": "prefers dark mode everywhere", "domain": "technical"},
                   {"subject": "phone", "assertion": "iPhone user, won't switch to Android", "domain": "technical"}
                 ]}""";
        }
        if (m.contains("morning person") && m.contains("productive") && m.contains("before noon")) {
            return """
                {"PREFERENCE": [{"subject": "productivity", "assertion": "morning person, most productive before noon, useless after 9pm", "domain": "lifestyle"}]}""";
        }
        if (m.contains("priya") && m.contains("birthday") && m.contains("january 8")) {
            return """
                {"EVENT": [{"summary": "Priya's birthday on January 8th", "event_type": "milestone", "date": "2027-01-08"}],
                 "IMPLICIT_COMMITMENT": [{"description": "Plan something special for Priya's birthday January 8th", "commitment_type": "need_to", "related_person": "Priya", "estimated_timeframe": 21}]}""";
        }
        if (m.contains("anniversary") && m.contains("march 22") && m.contains("bali")) {
            return """
                {"EVENT": [{"summary": "Wedding anniversary March 22nd", "event_type": "milestone", "date": "2027-03-22"}],
                 "GOAL": [{"title": "Trip to Bali for wedding anniversary", "category": "personal", "priority": "medium"}]}""";
        }
        if (m.contains("mom") && m.contains("60th") && m.contains("october") && m.contains("surprise party")) {
            return """
                {"EVENT": [{"summary": "Mom Asha's 60th birthday in October — planning surprise party in Mumbai", "event_type": "milestone"}],
                 "IMPLICIT_COMMITMENT": [{"description": "Plan surprise party in Mumbai for mom's 60th birthday in October", "commitment_type": "planning_to", "related_person": "Asha", "estimated_timeframe": 90}]}""";
        }
        if (m.contains("promised amit") && m.contains("singapore") && m.contains("february") && m.contains("passport")) {
            return """
                {"IMPLICIT_COMMITMENT": [
                   {"description": "Visit Amit in Singapore in February", "commitment_type": "will_do", "related_person": "Amit", "estimated_timeframe": 60},
                   {"description": "Renew passport before Singapore trip", "commitment_type": "need_to", "estimated_timeframe": 30}
                 ]}""";
        }
        if (m.contains("sony") && m.contains("wh-1000xm5") && m.contains("28,000") && m.contains("sale")) {
            return """
                {"DEFERRAL": [{"decision": "Buying Sony WH-1000XM5 headphones", "context": "₹28,000 is too steep, waiting for a sale", "days_until_followup": 60}]}""";
        }
        if (m.contains("aws") && m.contains("certification") && m.contains("not sure") && m.contains("worth")) {
            return """
                {"DEFERRAL": [{"decision": "AWS Solutions Architect certification", "context": "Unsure if worth the time investment", "days_until_followup": 30}]}""";
        }
        if (m.contains("bigger apartment") && m.contains("priya wants") && m.contains("wait") && m.contains("promotion")) {
            return """
                {"DEFERRAL": [{"decision": "Moving to a bigger apartment", "context": "Priya wants to move but waiting for promotion confirmation first", "days_until_followup": 30}]}""";
        }
        if (m.contains("stressed") && m.contains("atlas") && m.contains("3 months") && m.contains("day off")) {
            return """
                {"EVENT": [{"summary": "Feeling stressed about Atlas deadline, no day off in 3 months", "event_type": "problem", "emotional_valence": "negative"}],
                 "PREFERENCE": [{"subject": "work-life balance", "assertion": "stressed, hasn't taken a day off in 3 months due to Atlas deadline", "domain": "work"}]}""";
        }
        if (m.contains("arjun") && m.contains("school") && m.contains("dps") && m.contains("excited")) {
            return """
                {"EVENT": [{"summary": "Arjun got into DPS Bangalore, starting school next year", "event_type": "achievement", "emotional_valence": "positive"}]}""";
        }
        if (m.contains("got the promotion") && m.contains("staff engineer") && m.contains("15l raise")) {
            return """
                {"PROFILE": {"name": "Kislay Sinha", "occupation": "Staff Engineer", "company": "Stripe"},
                 "EVENT": [{"summary": "Got promoted to Staff Engineer at Stripe with ₹15L raise", "event_type": "achievement", "emotional_valence": "positive"}],
                 "RESOLUTION": [{"decision": "Career progression", "choice": "Promoted to Staff Engineer at Stripe"}]}""";
        }
        if (m.contains("promotion confirmed") && m.contains("apartment move") && m.contains("decided")) {
            return """
                {"RESOLUTION": [{"decision": "Moving to a bigger apartment", "choice": "Going ahead with apartment move now that promotion is confirmed"}]}""";
        }
        if (m.contains("switched") && m.contains("dark mode") && m.contains("light mode") && m.contains("eye strain")) {
            return """
                {"PREFERENCE": [{"subject": "UI theme", "assertion": "switched to light mode due to eye strain", "domain": "technical"}],
                 "RESOLUTION": [{"decision": "UI theme preference", "choice": "Switched from dark mode to light mode due to eye strain"}]}""";
        }

        return "{}";
    }
}
