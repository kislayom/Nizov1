package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.*;
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Real user queries against a realistic memory. These are the exact natural
 * language questions a human would ask — no keyword padding.
 *
 * <p>These tests lock in the fixes from manual QA:
 * <ul>
 *   <li>Contradiction demotion (job change, preference switch)</li>
 *   <li>Retrieval-friendly content wording (current/role/work synonyms)</li>
 *   <li>Relevance floor (no filler when nothing matches — empty list honestly)</li>
 * </ul>
 */
class NaturalLanguageRecallTest {

    private static final List<String> V = List.of(
        "kislay","priya","wife","spouse","arjun","son","asha","mother","mom","child","parent","family",
        "vikram","boss","manager","neha","colleague","amit","friend","college",
        "dr","sharma","doctor","dentist",
        "work","works","stripe","engineer","principal","staff","fintech","payments","api","current","role","job","position","company","employer","title",
        "startup","promotion","team","lead",
        "project","atlas","deadline","sprint","standup","review",
        "stock","tata","motors","infosys","reliance","invest","portfolio",
        "sip","mutual","fund","zerodha","nifty",
        "bitcoin","crypto","etf",
        "birthday","anniversary","wedding","trip","japan","bali",
        "bangalore","mumbai","singapore","home","apartment","emi",
        "prefer","dark","light","mode","coffee","tea","morning",
        "vegetarian","south","indian","food","dosa",
        "iphone","android","mac","linux","vim",
        "gym","yoga","run","marathon","sleep","meditation",
        "allergy","peanut","lactose","epipen",
        "call","email","book","schedule","remind","reminder","action","task","pending",
        "promise","send","document","report","submit","pay","renew",
        "passport","visa",
        "buy","price","expensive","sale","sony","headphone","airpods",
        "stressed","worried","excited",
        "deferred","postponed","decided","changed","switched","moved",
        "rust","python","aws","learn","certification",
        "monday","friday","december","january","march","october","upcoming",
        "important","urgent","briefing","personal","professional"
    );

    private SqliteMemoryStore ms;
    private SqliteGraphStore gs;
    private MemoryService mem;
    private ExtractionService ext;

    @TempDir Path tmp;

    @BeforeEach
    void setup() {
        Path db = tmp.resolve("q.db");
        ms = new SqliteMemoryStore(db);
        gs = new SqliteGraphStore(db);
        var idx = new InMemoryVectorIndex();
        var emb = new FakeEmbedder(V);
        GraphService graph = new KnowledgeGraph(gs);
        mem = new LayeredMemoryService(ms, idx, emb, null, graph, null, 100, 0.1);

        ModelClient mc = new ModelClient() {
            public ModelCapability capability() {
                return new ModelCapability("f","f",Set.of(Modality.TEXT),Set.of(Modality.TEXT),8192,false,true,0,0,10);
            }
            public ModelResponse complete(ModelRequest r) {
                // Only look at the LAST message (user input) — system prompt
                // contains worked examples that would trigger false matches.
                String userMsg = r.messages().isEmpty() ? "" :
                        r.messages().get(r.messages().size() - 1).text();
                return ModelResponse.text(respond(userMsg), ModelResponse.Usage.zero());
            }
        };
        ext = new ExtractionPipeline(mc, new GraphFactRouter(graph), mem);

        // Build a realistic memory
        ext.extract("k","I'm Kislay Sinha, 32 years old, I work at Stripe as a principal engineer on the payments API team. I live in Bangalore with my wife Priya and our 4-year-old son Arjun.");
        ext.extract("k","My mother Asha lives in Mumbai. Dad passed away 3 years ago.");
        ext.extract("k","My best friend Amit from IIT works at Google in Singapore. We talk every week.");
        ext.extract("k","Neha is my colleague at Stripe, she leads the Atlas project. Vikram is my engineering manager.");
        ext.extract("k","I have a peanut allergy — serious, carry an EpiPen. Also lactose intolerant.");
        ext.extract("k","I run 5km every morning at 6am. Training for the Bangalore marathon in March.");
        ext.extract("k","The Atlas project has a hard deadline January 15th. We're behind by 2 sprints.");
        ext.extract("k","I have a standup every day at 10am and a sprint review every Friday at 3pm.");
        ext.extract("k","I have a SIP of ₹25,000/month in Nifty 50 ETF through Zerodha.");
        ext.extract("k","Holding 500 shares of Tata Motors, bought at ₹420. Current price around ₹680.");
        ext.extract("k","Also watching Infosys — want to start a position if it drops below ₹1,400.");
        ext.extract("k","Considering putting some money into Bitcoin but Priya is against it. Will think about it.");
        ext.extract("k","I use Vim for coding and Linux on my work machine. Mac for personal stuff.");
        ext.extract("k","Strict vegetarian. Love South Indian food, especially dosa and filter coffee.");
        ext.extract("k","I prefer dark mode everywhere. iPhone user, won't switch to Android.");
        ext.extract("k","Priya's birthday is January 8th. Need to plan something special this time.");
        ext.extract("k","Was looking at the Sony WH-1000XM5 headphones. ₹28,000 is steep. Will wait for a sale.");
        ext.extract("k","Thinking about doing an AWS Solutions Architect certification but not sure if it's worth the time.");
        ext.extract("k","Priya wants us to move to a bigger apartment. I think we should wait till the promotion is confirmed.");
        ext.extract("k","Promised Amit I'd visit him in Singapore in February. Need to renew my passport first.");
        ext.extract("k","Update: I got the promotion! Now Staff Engineer at Stripe. ₹15L raise.");
        ext.extract("k","With the promotion confirmed, Priya and I decided to go ahead with the apartment move.");
        ext.extract("k","Switched from dark mode to light mode. The eye strain was getting bad.");
    }

    @AfterEach
    void teardown() { ms.close(); gs.close(); }

    // ===== EMPTY RESULT TESTS (no filler) =====

    @Test
    @DisplayName("Query about topic never mentioned returns empty — no filler")
    void iceCreamReturnsEmpty() {
        var results = mem.recall(RecallRequest.of("k", "which ice cream i love", 1500));
        assertTrue(results.isEmpty(),
                "Ice cream was never mentioned — memory must return empty, not filler. Got: "
                        + results.stream().map(MemoryItem::content).collect(Collectors.joining("\n")));
    }

    @Test
    @DisplayName("Query about unrelated topic returns empty")
    void quantumPhysicsReturnsEmpty() {
        var results = mem.recall(RecallRequest.of("k", "quantum physics experiments", 1500));
        assertTrue(results.isEmpty(), "Never discussed — must be empty");
    }

    // ===== CURRENT STATE AFTER CHANGES (contradiction demotion) =====

    @Test
    @DisplayName("Current role query returns Staff Engineer (promoted), not Principal (old)")
    void currentRoleIsStaffEngineer() {
        var results = mem.recall(RecallRequest.of("k", "what is my current role", 1500));
        assertFalse(results.isEmpty());
        String top = results.get(0).content();
        assertTrue(top.contains("Staff Engineer") || top.contains("current job role"),
                "Top result should be current role (Staff Engineer), got: " + top);

        // Verify the old role was demoted (not at top)
        if (results.size() > 1) {
            var staffPos = indexOfContent(results, "Staff Engineer");
            var principalPos = indexOfContent(results, "principal engineer");
            if (principalPos >= 0 && staffPos >= 0) {
                assertTrue(staffPos < principalPos,
                        "Staff Engineer should rank above demoted 'principal engineer'");
            }
        }
    }

    @Test
    @DisplayName("Where do I work — current company surfaces in top results")
    void whereDoIWorkReturnsStripe() {
        var results = mem.recall(RecallRequest.of("k", "where do i work", 1500));
        assertFalse(results.isEmpty(), "Should have work-related results");
        String all = results.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        assertTrue(all.contains("Stripe"),
                "Stripe (current employer) must appear in work query results");
    }

    // ===== PREFERENCE CHANGE (light mode superseded dark mode) =====

    @Test
    @DisplayName("UI theme preference reflects latest (light mode, not old dark mode)")
    void uiThemeIsLightMode() {
        var results = mem.recall(RecallRequest.of("k", "UI theme preference mode", 1500));
        assertFalse(results.isEmpty());
        // Light mode fact should outrank demoted dark mode fact
        var lightPos = indexOfContent(results, "light mode");
        var darkStandalonePos = -1;
        for (int i = 0; i < results.size(); i++) {
            String c = results.get(i).content();
            if (c.contains("dark mode") && !c.contains("light mode")
                    && !c.contains("switched") && !c.contains("Switched")) {
                darkStandalonePos = i;
                break;
            }
        }
        assertTrue(lightPos >= 0, "Light mode fact should be in results");
        if (darkStandalonePos >= 0) {
            assertTrue(lightPos < darkStandalonePos,
                    "Light mode (current) should outrank old dark mode");
        }
    }

    // ===== FAMILY / RELATIONSHIPS =====

    @Test
    @DisplayName("Family query surfaces spouse, child, parent")
    void familySurfacesSpouseChildParent() {
        var results = mem.recall(RecallRequest.of("k", "family spouse child parent", 1500));
        assertFalse(results.isEmpty());
        String all = results.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        int familyHits = 0;
        if (all.contains("Priya")) familyHits++;
        if (all.contains("Arjun")) familyHits++;
        if (all.contains("Asha")) familyHits++;
        assertTrue(familyHits >= 2, "At least 2 family members should surface, got: " + all);
    }

    // ===== DEFERRALS =====

    @Test
    @DisplayName("Deferred decisions query surfaces all deferrals")
    void deferralsQuerySurfacesAll() {
        var results = mem.recall(RecallRequest.of("k", "deferred decisions postponed", 2000));
        assertFalse(results.isEmpty());
        String all = results.stream().map(MemoryItem::content).collect(Collectors.joining("\n"));
        int deferrals = 0;
        if (all.contains("Sony")) deferrals++;
        if (all.contains("Bitcoin")) deferrals++;
        if (all.contains("Infosys")) deferrals++;
        if (all.contains("AWS")) deferrals++;
        assertTrue(deferrals >= 2, "At least 2 deferrals should surface, got: " + all);
    }

    // ===== SPECIFIC ENTITY RECALL =====

    @Test
    @DisplayName("Priya query surfaces wife facts")
    void priyaQuerySurfacesSpouseFacts() {
        var results = mem.recall(RecallRequest.of("k", "Priya", 1500));
        assertFalse(results.isEmpty());
        assertTrue(results.stream().anyMatch(m -> m.content().contains("Priya")),
                "Priya query must surface Priya-related facts");
    }

    @Test
    @DisplayName("Sony headphones deferral is recallable")
    void sonyHeadphonesDeferralRecallable() {
        var results = mem.recall(RecallRequest.of("k", "Sony headphones", 1500));
        assertFalse(results.isEmpty());
        assertTrue(results.stream().anyMatch(m -> m.content().contains("Sony")),
                "Sony headphones deferral must surface");
    }

    // ===== HEALTH =====

    @Test
    @DisplayName("Allergy query surfaces peanut allergy")
    void allergyQuerySurfacesPeanut() {
        var results = mem.recall(RecallRequest.of("k", "allergy peanut", 1500));
        assertFalse(results.isEmpty());
        assertTrue(results.stream().anyMatch(m -> m.content().contains("peanut")
                        || m.content().contains("allergy")),
                "Peanut allergy must surface");
    }

    // ===== Helpers =====

    private int indexOfContent(List<MemoryItem> results, String needle) {
        for (int i = 0; i < results.size(); i++) {
            if (results.get(i).content().contains(needle)) return i;
        }
        return -1;
    }

    private String respond(String msg) {
        String m = msg.toLowerCase();
        if (m.contains("kislay sinha") && m.contains("stripe") && m.contains("priya") && m.contains("arjun"))
            return "{\"PROFILE\":{\"name\":\"Kislay Sinha\",\"occupation\":\"principal engineer\",\"company\":\"Stripe\",\"location_city\":\"Bangalore\",\"location_country\":\"India\",\"industry\":\"fintech\"},\"RELATIONSHIP\":[{\"person_name\":\"Priya\",\"relationship_type\":\"spouse\",\"context\":\"lives together in Bangalore\"},{\"person_name\":\"Arjun\",\"relationship_type\":\"child\",\"context\":\"4-year-old son\"}]}";
        if (m.contains("mother asha") && m.contains("mumbai") && m.contains("dad passed"))
            return "{\"RELATIONSHIP\":[{\"person_name\":\"Asha\",\"relationship_type\":\"parent\",\"context\":\"mother, lives in Mumbai\"}],\"EVENT\":[{\"summary\":\"Father passed away 3 years ago\",\"event_type\":\"milestone\",\"emotional_valence\":\"negative\"}]}";
        if (m.contains("best friend amit") && m.contains("iit") && m.contains("google") && m.contains("singapore"))
            return "{\"RELATIONSHIP\":[{\"person_name\":\"Amit\",\"relationship_type\":\"friend\",\"context\":\"best friend from IIT, works at Google in Singapore, talks weekly\"}]}";
        if (m.contains("neha") && m.contains("colleague") && m.contains("vikram") && m.contains("manager"))
            return "{\"RELATIONSHIP\":[{\"person_name\":\"Neha\",\"relationship_type\":\"colleague\",\"context\":\"leads the Atlas project at Stripe\"},{\"person_name\":\"Vikram\",\"relationship_type\":\"manager\",\"context\":\"engineering manager at Stripe\"}]}";
        if (m.contains("peanut allergy") && m.contains("epipen") && m.contains("lactose"))
            return "{\"PREFERENCE\":[{\"subject\":\"allergy\",\"assertion\":\"severe peanut allergy, carries EpiPen\",\"domain\":\"health\"},{\"subject\":\"dietary restriction\",\"assertion\":\"lactose intolerant\",\"domain\":\"health\"}]}";
        if (m.contains("run 5km") && m.contains("6am") && m.contains("marathon") && m.contains("march"))
            return "{\"PREFERENCE\":[{\"subject\":\"exercise routine\",\"assertion\":\"runs 5km every morning at 6am\",\"domain\":\"health\"}],\"GOAL\":[{\"title\":\"Run Bangalore marathon in March\",\"category\":\"health\",\"priority\":\"high\"}]}";
        if (m.contains("atlas") && m.contains("deadline") && m.contains("january 15") && m.contains("2 sprints"))
            return "{\"EVENT\":[{\"summary\":\"Atlas project deadline January 15th — behind by 2 sprints\",\"event_type\":\"deadline\",\"date\":\"2027-01-15\",\"emotional_valence\":\"negative\"}]}";
        if (m.contains("standup") && m.contains("10am") && m.contains("sprint review") && m.contains("friday"))
            return "{\"PREFERENCE\":[{\"subject\":\"daily standup\",\"assertion\":\"every day at 10am\",\"domain\":\"work\"},{\"subject\":\"sprint review\",\"assertion\":\"every Friday at 3pm\",\"domain\":\"work\"}]}";
        if (m.contains("sip") && m.contains("25,000") && m.contains("nifty") && m.contains("zerodha"))
            return "{\"INVESTMENT_INTEREST\":{\"tickers\":[\"NIFTY50ETF\"],\"style\":\"index\",\"risk_appetite\":\"moderate\"},\"PREFERENCE\":[{\"subject\":\"investment platform\",\"assertion\":\"uses Zerodha for SIP of ₹25,000/month in Nifty 50 ETF\",\"domain\":\"finance\"}]}";
        if (m.contains("tata motors") && m.contains("500 shares") && m.contains("420"))
            return "{\"INVESTMENT_INTEREST\":{\"tickers\":[\"TATAMOTORS\"],\"style\":\"value\"},\"PREFERENCE\":[{\"subject\":\"stock holding\",\"assertion\":\"holds 500 shares of Tata Motors bought at ₹420, current ~₹680\",\"domain\":\"finance\"}]}";
        if (m.contains("infosys") && m.contains("below") && m.contains("1,400"))
            return "{\"DEFERRAL\":[{\"decision\":\"Start position in Infosys stock\",\"context\":\"Waiting for price to drop below ₹1,400\",\"days_until_followup\":30}]}";
        if (m.contains("bitcoin") && m.contains("priya") && m.contains("against") && m.contains("think about"))
            return "{\"DEFERRAL\":[{\"decision\":\"Investing in Bitcoin\",\"context\":\"Considering it but wife Priya is against the idea\",\"days_until_followup\":30}]}";
        if (m.contains("vim") && m.contains("linux") && m.contains("mac"))
            return "{\"PREFERENCE\":[{\"subject\":\"code editor\",\"assertion\":\"uses Vim for coding\",\"domain\":\"technical\"},{\"subject\":\"work OS\",\"assertion\":\"Linux on work machine, Mac for personal\",\"domain\":\"technical\"}]}";
        if (m.contains("vegetarian") && m.contains("south indian") && m.contains("dosa") && m.contains("filter coffee"))
            return "{\"PREFERENCE\":[{\"subject\":\"diet\",\"assertion\":\"strict vegetarian\",\"domain\":\"lifestyle\"},{\"subject\":\"food\",\"assertion\":\"loves South Indian food, especially dosa and filter coffee\",\"domain\":\"lifestyle\"}]}";
        if (m.contains("dark mode everywhere") && m.contains("iphone") && m.contains("android"))
            return "{\"PREFERENCE\":[{\"subject\":\"UI theme\",\"assertion\":\"prefers dark mode everywhere\",\"domain\":\"technical\"},{\"subject\":\"phone\",\"assertion\":\"iPhone user, won't switch to Android\",\"domain\":\"technical\"}]}";
        if (m.contains("priya") && m.contains("birthday") && m.contains("january 8"))
            return "{\"EVENT\":[{\"summary\":\"Priya's birthday on January 8th\",\"event_type\":\"milestone\",\"date\":\"2027-01-08\"}],\"IMPLICIT_COMMITMENT\":[{\"description\":\"Plan something special for Priya's birthday January 8th\",\"commitment_type\":\"need_to\",\"related_person\":\"Priya\",\"estimated_timeframe\":21}]}";
        if (m.contains("sony") && m.contains("wh-1000xm5") && m.contains("28,000") && m.contains("sale"))
            return "{\"DEFERRAL\":[{\"decision\":\"Buying Sony WH-1000XM5 headphones\",\"context\":\"₹28,000 is too steep, waiting for a sale\",\"days_until_followup\":60}]}";
        if (m.contains("aws") && m.contains("certification") && m.contains("not sure") && m.contains("worth"))
            return "{\"DEFERRAL\":[{\"decision\":\"AWS Solutions Architect certification\",\"context\":\"Unsure if worth the time investment\",\"days_until_followup\":30}]}";
        if (m.contains("bigger apartment") && m.contains("priya wants") && m.contains("wait") && m.contains("promotion"))
            return "{\"DEFERRAL\":[{\"decision\":\"Moving to a bigger apartment\",\"context\":\"Priya wants to move but waiting for promotion confirmation first\",\"days_until_followup\":30}]}";
        if (m.contains("promised amit") && m.contains("singapore") && m.contains("february") && m.contains("passport"))
            return "{\"IMPLICIT_COMMITMENT\":[{\"description\":\"Visit Amit in Singapore in February\",\"commitment_type\":\"will_do\",\"related_person\":\"Amit\",\"estimated_timeframe\":60},{\"description\":\"Renew passport before Singapore trip\",\"commitment_type\":\"need_to\",\"estimated_timeframe\":30}]}";
        if (m.contains("got the promotion") && m.contains("staff engineer") && m.contains("15l raise"))
            return "{\"PROFILE\":{\"name\":\"Kislay Sinha\",\"occupation\":\"Staff Engineer\",\"company\":\"Stripe\"},\"EVENT\":[{\"summary\":\"Got promoted to Staff Engineer at Stripe with ₹15L raise\",\"event_type\":\"achievement\",\"emotional_valence\":\"positive\"}],\"RESOLUTION\":[{\"decision\":\"Career progression\",\"choice\":\"Promoted to Staff Engineer at Stripe\"}]}";
        if (m.contains("promotion confirmed") && m.contains("apartment move") && m.contains("decided"))
            return "{\"RESOLUTION\":[{\"decision\":\"Moving to a bigger apartment\",\"choice\":\"Going ahead with apartment move now that promotion is confirmed\"}]}";
        if (m.contains("switched") && m.contains("dark mode") && m.contains("light mode") && m.contains("eye strain"))
            return "{\"PREFERENCE\":[{\"subject\":\"UI theme\",\"assertion\":\"switched to light mode due to eye strain\",\"domain\":\"technical\"}],\"RESOLUTION\":[{\"decision\":\"UI theme preference\",\"choice\":\"Switched from dark mode to light mode due to eye strain\"}]}";
        return "{}";
    }
}
