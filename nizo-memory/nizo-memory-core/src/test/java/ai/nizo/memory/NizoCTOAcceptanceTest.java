package ai.nizo.memory;

import ai.nizo.memory.api.extract.ExtractionCategory;
import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.extract.ExtractionPipeline;
import ai.nizo.memory.extract.GraphFactRouter;
import ai.nizo.memory.graph.KnowledgeGraph;
import ai.nizo.memory.graph.SqliteGraphStore;
import ai.nizo.memory.llm.OllamaEmbeddingClient;
import ai.nizo.memory.llm.OllamaModelClient;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * <h1>Nizo Memory — CTO Acceptance Test</h1>
 *
 * <p>Ten hard-assertion contracts that collectively answer the question:
 * <em>"is this memory system safe to ship to production?"</em>.
 *
 * <p>Each test is self-documenting. Running it prints, for every contract:
 * <ol>
 *   <li><b>Why it matters</b> — what production failure this prevents</li>
 *   <li><b>Input</b> — the exact user message(s) fed in</li>
 *   <li><b>Expected</b> — the behavior we require, phrased as a contract</li>
 *   <li><b>Actual</b> — what the live system stored / recalled</li>
 *   <li><b>Result</b> — PASS or the hard assertion that failed</li>
 * </ol>
 *
 * <p>Backed by live Ollama (qwen2.5:14b extraction + nomic-embed-text embeddings).
 * Auto-skips if Ollama isn't reachable. Each test uses a fresh user-id + SQLite
 * file, so contracts are independent.
 */
class NizoCTOAcceptanceTest {

    private static final String OLLAMA_URL = "http://localhost:11434";

    /** Production-grade LLM. 14b is meaningfully more reliable at JSON
     *  output than 7b for multi-category extractions. Override via
     *  -Dnizo.test.llm=qwen2.5:7b if you need the smaller model. */
    private static final String LLM_MODEL = System.getProperty("nizo.test.llm", "qwen2.5:14b");

    @TempDir Path tmp;

    // ====================================================================
    // CONTRACT 1 — No fabricated facts in extraction output
    // ====================================================================

    @Test
    void extraction_neverStoresPlaceholderNamesOrCompanies() throws Exception {
        contract("C1", "No fabricated facts in extraction",
                "LLMs will emit 'You' or 'user' as a name when the message doesn't "
                        + "state one. And they'll invent placeholder companies like "
                        + "'Your Company' or 'the company'. Both poison the DB — every "
                        + "future recall about identity or employer returns the "
                        + "hallucination. Hard rule: no placeholder names or "
                        + "companies may land in storage.");

        Fixture fx = fixture("c1-user-pronoun");
        String input = "We closed our pre-seed round at $1M from Y Combinator alums. Building a fintech app for Mexico.";
        showInput(input, "(message deliberately omits the speaker's name — bait for LLM hallucination)");

        expect("No stored fact contains \"User's name is You / I / user / person / me\"");
        expect("No stored fact contains \"at Your Company / Our Company / the company\"");
        expect("No stored fact contains substring placeholders "
                + "(\"not provided\" / \"name withheld\" / \"undisclosed\" / etc.)");

        fx.ext.extract(fx.user, input);
        Thread.sleep(1500);
        List<String> contents = contentsOf(fx.recallAll());
        showStored(contents);

        List<String> badNames = List.of(
                "User's name is You", "User's name is I", "User's name is user",
                "User's name is person", "User's name is me");
        List<String> badCompanies = List.of(
                "as Product Manager at Your Company", "as Product Manager at Our Company",
                "at the company", "at Your Company", "at Our Company", "at The Company");
        for (String bad : badNames) assertFalse(containsContent(contents, bad),
                "Leaked placeholder name: '" + bad + "'");
        for (String bad : badCompanies) assertFalse(containsContent(contents, bad),
                "Leaked placeholder company: '" + bad + "'");
        // Substring placeholders the LLM emits under pressure
        List<String> badSubstrings = List.of(
                "name not provided", "name not specified", "name withheld",
                "company name not provided", "company name not specified",
                "not disclosed", "undisclosed", "to be determined",
                "placeholder", "redacted");
        for (String bad : badSubstrings) {
            for (String c : contents) {
                assertFalse(c.toLowerCase(Locale.ROOT).contains(bad),
                        "Leaked substring placeholder '" + bad + "' in: " + c);
            }
        }

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 2 — Festivals / places never classified as people
    // ====================================================================

    @Test
    void extraction_doesNotClassifyFestivalsOrPlacesAsPeople() throws Exception {
        contract("C2", "Festivals and places are not people",
                "Before this fix, the LLM stored 'Tirupati is the user's visit', "
                        + "'Pongal participants is the user's friend', and 'Arjun's wife' "
                        + "as person names. These garbled rows break recall — a query "
                        + "'who is my friend' returns 'Pongal participants'. The pipeline "
                        + "must reject festivals, places, possessive references, and "
                        + "generic role words as person_name before storage.");

        Fixture fx = fixture("c2-cultural");
        String in1 = "We're Tam Brahm. Visit Tirupati every Diwali and do kolam during Pongal with the family.";
        String in2 = "Attended Eid celebration at my friend's place last week.";
        showInputs(List.of(in1, in2),
                "(mixes festivals Pongal/Diwali/Eid, place Tirupati, activity kolam "
                        + "— all previously mis-extracted as person_name)");

        expect("No RELATIONSHIP fact with person_name = Tirupati, Pongal, Diwali, Eid, or Kolam");

        fx.ext.extract(fx.user, in1);
        fx.ext.extract(fx.user, in2);
        Thread.sleep(1500);
        List<String> contents = contentsOf(fx.recallAll());
        showStored(contents);

        for (String bad : List.of(
                "Tirupati is the user's visit",
                "Pongal participants is the user's friend",
                "Pongal is the user's", "Diwali is the user's",
                "Eid is the user's", "Kolam is the user's")) {
            assertFalse(containsContent(contents, bad),
                    "Festival/place classified as person: '" + bad + "'");
        }

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 3 — Investment narrative is preserved (no silent drop)
    // ====================================================================

    @Test
    void extraction_preservesInvestmentNarrative() throws Exception {
        contract("C3", "Investment narrative never silently vanishes",
                "The LLM sometimes classifies 'SIP of ₹80K in Nifty 50 via Zerodha + "
                        + "PPF + EPF + NPS' into INVESTMENT_INTEREST with an EMPTY "
                        + "tickers list and no summary — in which case the original "
                        + "financial context disappears. The system must guarantee the "
                        + "raw financial narrative is always recallable even when the "
                        + "LLM's structured output is incomplete.");

        Fixture fx = fixture("c3-investing");
        String input = "I do an SIP of ₹80,000 a month — Nifty 50 and Next 50 index funds via Zerodha. Small PPF, plus EPF + NPS tier-1.";
        showInput(input, "(heavy investment vocabulary — SIP / PPF / EPF / NPS / Zerodha — "
                + "that historically vanished)");

        expect("At least one stored fact mentions SIP / Nifty / PPF / EPF / NPS / index-fund / Zerodha");
        expect("Query 'how am i investing for retirement' returns ≥ 1 hit containing any of those terms");

        fx.ext.extract(fx.user, input);
        Thread.sleep(1500);
        List<String> contents = contentsOf(fx.recallAll());
        showStored(contents);

        String joined = String.join("\n", contents).toLowerCase(Locale.ROOT);
        boolean stored = joined.contains("sip") || joined.contains("nifty") || joined.contains("ppf")
                || joined.contains("epf") || joined.contains("nps")
                || joined.contains("index fund") || joined.contains("zerodha");
        assertTrue(stored, "Investment narrative vanished entirely. Stored:\n" + joined);

        var hits = fx.mem.recall(RecallRequest.of(fx.user, "how am i investing for retirement", 1500));
        showRecall("how am i investing for retirement", hits);
        assertFalse(hits.isEmpty(), "Investment recall returned empty");
        assertTrue(hits.stream().anyMatch(h -> {
            String c = h.content().toLowerCase(Locale.ROOT);
            return c.contains("sip") || c.contains("nifty") || c.contains("ppf")
                    || c.contains("epf") || c.contains("nps") || c.contains("index");
        }), "Recall didn't surface any SIP/PPF/EPF/NPS fact");

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 4 — Ticker field is validated
    // ====================================================================

    @Test
    void extraction_filtersNonTickersOutOfTickersField() throws Exception {
        contract("C4", "Ticker field only contains valid ticker symbols",
                "The LLM routinely dumps investor names and country codes into the "
                        + "tickers array: ['YC alums', 'Mexican angels', 'SG']. If we "
                        + "store these as tickers, any agent using them for price lookups "
                        + "or portfolio analysis will lookup nonsense. Tickers must be "
                        + "1-5 uppercase letters, single-word, not a country code.");

        Fixture fx = fixture("c4-tickers");
        String input = "Raised pre-seed from Y Combinator alums and Mexican angels. Moved to Singapore (SG).";
        showInput(input, "(bait — the LLM wants to put 'YC alums', 'Mexican angels', 'SG' into tickers)");

        expect("No stored 'tickers: [...]' fact contains 'YC alums', 'Mexican angels', or 'Y Combinator'");
        expect("No stored 'tickers: [...]' fact contains country codes (SG / US / UK / IN / CN / JP / KR)");

        fx.ext.extract(fx.user, input);
        Thread.sleep(1500);
        List<String> contents = contentsOf(fx.recallAll());
        showStored(contents);

        for (String c : contents) {
            if (c.contains("tickers: [")) {
                assertFalse(c.contains("YC alums"), "Junk 'YC alums' in tickers: " + c);
                assertFalse(c.contains("Mexican angels"), "Junk 'Mexican angels' in tickers: " + c);
                assertFalse(c.contains("Y Combinator"), "Junk 'Y Combinator' in tickers: " + c);
                for (String cc : List.of("SG", "US", "UK", "IN", "CN", "JP", "KR")) {
                    assertFalse(c.matches(".*tickers: \\[.*\\b" + cc + "\\b.*\\].*"),
                            "Country code '" + cc + "' in tickers: " + c);
                }
            }
        }

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 5 — Unknown queries return honest-empty (no noise)
    // ====================================================================

    @Test
    void recall_unknownQueriesReturnEmpty() throws Exception {
        contract("C5", "Unknown queries return empty",
                "If the user never mentioned their blood type, the memory system "
                        + "MUST NOT return a nearby-but-wrong fact like 'Arjun is my "
                        + "cardiologist'. Honest-empty beats confident-wrong for every "
                        + "downstream agent. nomic-embed-text has a soft noise floor "
                        + "(~0.50-0.58 sim on unrelated pairs), so a dedicated "
                        + "lexical-overlap guard is needed to close this class of leak. "
                        + "Contract: ≥ 6 of 7 unknown queries return empty.");

        Fixture fx = fixture("c5-unknowns");
        List<String> setup = List.of(
                "I'm Priya Reddy, VP Product at PhonePe in Bangalore.",
                "Husband Arjun is a cardiologist at Apollo. Daughter Ira is 3.",
                "SIP of ₹80K in Nifty 50 via Zerodha. Also PPF and EPF.",
                "Peanut allergy, always carry an EpiPen.",
                "Love Succession, Severance, The Bear. Filter kaapi on weekends.");
        showInputs(setup, "(rich context — 5 turns — so the memory has plenty of decoys)");

        String[] unknowns = {
                "what's my blood type",
                "do i have a pet",
                "what's my favourite colour",
                "what's my annual bonus",
                "what's my LinkedIn URL",
                "how many siblings do i have",
                "what car do i currently drive"
        };
        expect("Each of these 7 queries — none of them taught — returns empty. Require ≥ 6/7.");

        for (String s : setup) fx.ext.extract(fx.user, s);
        Thread.sleep(2000);

        int empties = 0;
        StringBuilder actualReport = new StringBuilder();
        for (String q : unknowns) {
            var hits = fx.mem.recall(RecallRequest.of(fx.user, q, 1500));
            if (hits.isEmpty()) {
                empties++;
                actualReport.append(String.format("  ✓ \"%s\" → (empty)%n", q));
            } else {
                actualReport.append(String.format("  ✗ \"%s\" → LEAKED %d item(s):%n", q, hits.size()));
                for (int i = 0; i < Math.min(3, hits.size()); i++) {
                    actualReport.append(String.format("        [%.2f] %s%n",
                            hits.get(i).confidence(), hits.get(i).content()));
                }
            }
        }
        System.out.println();
        System.out.println("  ACTUAL");
        System.out.print(actualReport);
        System.out.printf("%n  Empty: %d / 7 (need ≥ 6)%n", empties);

        assertTrue(empties >= 6,
                "Too many unknown queries leaked. Empty: " + empties + "/7 (need ≥ 6)");

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 6 — Contradiction demotion (newer > older)
    // ====================================================================

    @Test
    void recall_newerRoleRanksAboveOlderRole() throws Exception {
        contract("C6", "Newer facts rank above superseded ones",
                "When a user says 'I was Senior MLE, now Staff MLE', the current "
                        + "role must rank FIRST in any 'what's my current role' query. "
                        + "If an agent shows the stale role to the user, trust collapses. "
                        + "The pipeline uses demoteContradicted(); this contract "
                        + "verifies the effect at the recall layer.");

        Fixture fx = fixture("c6-promo");
        List<String> in = List.of(
                "I'm Rohan. Working as a Senior MLE at Grab.",
                "Got promoted! Now Staff MLE at Grab.");
        showInputs(in, "(the second turn contradicts the first — Senior → Staff)");

        String q = "what is my current role";
        expect("Query \"" + q + "\" returns the Staff (newer) role first");
        expect("If the old 'Senior MLE' fact still exists, it ranks strictly below the new one");

        for (String s : in) fx.ext.extract(fx.user, s);
        Thread.sleep(1500);
        var hits = fx.mem.recall(RecallRequest.of(fx.user, q, 1500));
        showRecall(q, hits);

        assertFalse(hits.isEmpty(), "Role recall returned empty");
        String top = hits.get(0).content();
        assertTrue(top.contains("Staff") || top.contains("current job role"),
                "Top result should surface newer (Staff) role, got: " + top);
        int staffIdx = -1, seniorIdx = -1;
        for (int i = 0; i < hits.size(); i++) {
            String c = hits.get(i).content();
            if (staffIdx < 0 && c.contains("Staff")) staffIdx = i;
            if (seniorIdx < 0 && c.contains("Senior MLE")) seniorIdx = i;
        }
        if (seniorIdx >= 0) {
            assertTrue(staffIdx >= 0 && staffIdx < seniorIdx,
                    "Older 'Senior MLE' ranked above newer 'Staff'");
        }

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 7 — World-corpus disambiguation (iPhone user ≠ job)
    // ====================================================================

    @Test
    void extraction_iPhoneUserIsPreferenceNotJob() throws Exception {
        contract("C7", "World-corpus guardrails work — 'iPhone user' is a preference",
                "Base-model LLMs will classify 'iPhone user' as an occupation if not "
                        + "corrected. The bundled extraction-guards heuristic says "
                        + "'iPhone user' is a phone PREFERENCE, not a job. This contract "
                        + "verifies the guardrail is actually being injected and honored.");

        Fixture fx = fixture("c7-iphone");
        String input = "I prefer dark mode and I'm an iPhone user.";
        showInput(input, "(the phrase 'iPhone user' is the exact bait the corpus guards against)");

        expect("No stored fact classifies 'iPhone user' as occupation");
        expect("At least one stored fact surfaces iPhone/iOS as a preference");

        fx.ext.extract(fx.user, input);
        Thread.sleep(1500);
        List<String> contents = contentsOf(fx.recallAll());
        showStored(contents);

        for (String c : contents) {
            String low = c.toLowerCase(Locale.ROOT);
            assertFalse(low.contains("as iphone user") || low.contains("occupation: iphone user"),
                    "iPhone user got classified as job: " + c);
        }
        boolean hasPhonePref = contents.stream()
                .anyMatch(c -> c.toLowerCase().contains("iphone") || c.toLowerCase().contains("ios"));
        assertTrue(hasPhonePref, "iPhone preference lost entirely");

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 8 — Multi-category extraction
    // ====================================================================

    @Test
    void extraction_richMessageFiresMultipleCategories() throws Exception {
        contract("C8", "Rich messages produce multiple fact types",
                "A single real-world sentence like 'Got promoted, Ira started preschool, "
                        + "planning Kyoto in April' has 3 distinct life facts. If "
                        + "extraction only captures one, 2/3 of the signal is lost. The "
                        + "pipeline must produce ≥ 2 facts and span at least one "
                        + "career/event category.");

        Fixture fx = fixture("c8-multi");
        String input = "Got promoted to Senior Director last week — Ira started preschool, and Priya and I are planning a trip to Kyoto in April.";
        showInput(input, "(3 distinct life events in one turn: promotion, schooling, travel plan)");

        expect("Extraction returns ≥ 2 distinct facts");
        expect("Categories include at least one of: PROFILE, RESOLUTION, EVENT");

        var r = fx.ext.extract(fx.user, input);
        System.out.printf("%n  ACTUAL%n    categories: %s%n    count: %d%n", r.types(), r.count());

        assertTrue(r.hasExtractions(), "Rich message extracted nothing");
        boolean hasCareer = r.types().contains(ExtractionCategory.PROFILE)
                || r.types().contains(ExtractionCategory.RESOLUTION)
                || r.types().contains(ExtractionCategory.EVENT);
        assertTrue(hasCareer, "No career/event category. Got: " + r.types());
        assertTrue(r.count() >= 2, "Expected ≥ 2 facts, got " + r.count());

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 9 — Warm-extraction latency budget
    // ====================================================================

    @Test
    void latency_warmExtractionUnder8Seconds() throws Exception {
        contract("C9", "Warm extraction stays under 8 seconds",
                "Nizo calls extract() on every message the user sends. A 15s warm "
                        + "latency means every reply feels broken. We showed earlier "
                        + "that dumping the whole 326-entry corpus into the prompt "
                        + "invalidated the KV cache and spiked warm latency to 15s. "
                        + "Static system prompt + always-on guards only keeps the "
                        + "cache hot and the warm call under 8s on reference hardware.");

        Fixture fx = fixture("c9-latency");
        showInput("warmup: \"I'm Kavya, software engineer at Swiggy.\"", "");
        fx.ext.extract(fx.user, "I'm Kavya, software engineer at Swiggy.");
        Thread.sleep(500);

        String measured = "Working on the delivery-partner onboarding flow.";
        showInput("measured: \"" + measured + "\"", "(first call after warmup — should be KV-cache-hot)");
        expect("Warm extraction completes in < 8,000 ms");

        long t0 = System.currentTimeMillis();
        fx.ext.extract(fx.user, measured);
        long dur = System.currentTimeMillis() - t0;
        System.out.printf("%n  ACTUAL%n    latency: %dms (budget: 8000ms)%n", dur);

        assertTrue(dur < 8_000, "Warm extraction too slow: " + dur + "ms (budget: 8000ms)");

        pass();
        fx.close();
    }

    // ====================================================================
    // CONTRACT 10 — 10-turn chained conversation, every key fact survives
    // ====================================================================

    @Test
    void chainedConversation_keyFactsSurviveTenTurns() throws Exception {
        contract("C10", "Key facts survive a 10-turn conversation",
                "A real user builds up context over many messages. Job, spouse, "
                        + "health, investments, travel plans, car — all come across "
                        + "different turns. Several weeks later, the agent must still "
                        + "recall every one of these on a natural-language query. "
                        + "This contract builds a 10-turn Meera conversation and "
                        + "asserts each key fact is recallable by its topic-keyword "
                        + "(Swiggy / Ashwin / allergy / SIP / Dubai / Hyundai).");

        Fixture fx = fixture("c10-chain");
        List<String> turns = List.of(
                "I'm Meera Nair, Principal PM at Swiggy in Bangalore.",
                "Married to Ashwin, he's at Razorpay. Daughter Anya is 5.",
                "We do Onam every year with family in Kochi. Strict vegetarian household.",
                "SIP of ₹1 lakh in Parag Parikh Flexi Cap + Mirae Emerging Bluechip. Also max out 80C.",
                "Tree-nut allergy — carry Allerject. No EpiPen in India, have to import.",
                "Training for Goa Half Marathon in December.",
                "Currently driving a 2019 Hyundai Creta. It's been reliable.",
                "Planning to move to Dubai on Swiggy's international expansion team in Q3.",
                "Considering a Rivian R1S if they launch in UAE.",
                "Got the Dubai offer confirmed today. Moving in September."
        );
        showInputs(turns, "(10-turn buildup about Meera Nair — Bangalore PM moving to Dubai)");

        expect("'where do i work'       surfaces fact containing 'Swiggy'");
        expect("'who is my spouse'      surfaces fact containing 'Ashwin'");
        expect("'any health concerns'   surfaces fact containing 'allergy'");
        expect("'my investments'        surfaces fact containing 'SIP'");
        expect("'my move to Dubai'      surfaces fact containing 'Dubai'");
        expect("'my current car'        surfaces fact containing 'Hyundai'");
        expect("'what is my blood type' (never taught) → empty OR ≤ 3 loose hits");

        for (String t : turns) fx.ext.extract(fx.user, t);
        Thread.sleep(2000);

        System.out.println();
        System.out.println("  ACTUAL");
        assertRecallContains(fx, "where do i work", "Swiggy");
        assertRecallContains(fx, "who is my spouse", "Ashwin");
        assertRecallContains(fx, "any health concerns", "allergy");
        assertRecallContains(fx, "my investments", "SIP");
        assertRecallContainsAny(fx, "my move to Dubai", List.of("Dubai"));
        assertRecallContains(fx, "my current car", "Hyundai");

        var bloodType = fx.mem.recall(RecallRequest.of(fx.user, "what is my blood type", 800));
        System.out.printf("    ? \"what is my blood type\" (never taught) → %d hit(s)%n", bloodType.size());
        if (!bloodType.isEmpty()) {
            assertTrue(bloodType.size() <= 3,
                    "Blood-type leaked " + bloodType.size() + " items: "
                            + String.join(" | ", contentsOf(bloodType)));
        }

        pass();
        fx.close();
    }

    // ====================================================================
    // Descriptive-output helpers
    // ====================================================================

    /** Print a contract header so running a single test tells the full story. */
    private static void contract(String id, String title, String whyItMatters) {
        System.out.println();
        System.out.println("================================================================================");
        System.out.println("  " + id + " — " + title);
        System.out.println("================================================================================");
        System.out.println();
        System.out.println("  WHY IT MATTERS");
        for (String line : wrap(whyItMatters, 74)) System.out.println("    " + line);
    }

    private static void showInput(String input, String note) {
        System.out.println();
        System.out.println("  INPUT");
        System.out.println("    " + quoted(input));
        if (note != null && !note.isEmpty()) {
            for (String line : wrap(note, 74)) System.out.println("    " + line);
        }
    }

    private static void showInputs(List<String> inputs, String note) {
        System.out.println();
        System.out.println("  INPUT (" + inputs.size() + " turn" + (inputs.size() == 1 ? "" : "s") + ")");
        int i = 1;
        for (String input : inputs) {
            System.out.printf("    %2d. %s%n", i++, truncate(input, 120));
        }
        if (note != null && !note.isEmpty()) {
            for (String line : wrap(note, 74)) System.out.println("    " + line);
        }
    }

    private static void expect(String description) {
        System.out.println("  EXPECTED");
        for (String line : wrap(description, 74)) System.out.println("    " + line);
    }

    private static void showStored(List<String> contents) {
        System.out.println();
        System.out.println("  STORED FACTS (" + contents.size() + ")");
        if (contents.isEmpty()) {
            System.out.println("    (none)");
            return;
        }
        int i = 1;
        for (String c : contents) {
            if (i > 10) {
                System.out.println("    ... +" + (contents.size() - 10) + " more");
                break;
            }
            System.out.printf("    %2d. %s%n", i++, truncate(c, 120));
        }
    }

    private static void showRecall(String query, List<MemoryItem> hits) {
        System.out.println();
        System.out.printf("  RECALL \"%s\" → %d hit(s)%n", query, hits.size());
        int i = 1;
        for (MemoryItem m : hits) {
            if (i > 5) { System.out.println("    ... +" + (hits.size() - 5) + " more"); break; }
            System.out.printf("    %d. [%.2f] %s%n", i++, m.confidence(), truncate(m.content(), 120));
        }
    }

    private static void pass() {
        System.out.println();
        System.out.println("  RESULT: PASS");
        System.out.println("--------------------------------------------------------------------------------");
    }

    private static String quoted(String s) {
        return "\"" + truncate(s, 180) + "\"";
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        String x = s.replace('\n', ' ');
        return x.length() > max ? x.substring(0, max - 3) + "..." : x;
    }

    private static List<String> wrap(String text, int width) {
        if (text == null || text.isEmpty()) return List.of();
        java.util.List<String> out = new java.util.ArrayList<>();
        StringBuilder line = new StringBuilder();
        for (String word : text.split("\\s+")) {
            if (line.length() + 1 + word.length() > width) {
                out.add(line.toString());
                line.setLength(0);
            }
            if (line.length() > 0) line.append(' ');
            line.append(word);
        }
        if (line.length() > 0) out.add(line.toString());
        return out;
    }

    // ====================================================================
    // Test fixture helpers
    // ====================================================================

    private Fixture fixture(String userId) throws Exception {
        assumeTrue(ollamaReachable(), "Ollama not reachable — skipping");
        Path db = tmp.resolve(userId + ".db");
        var ms = new SqliteMemoryStore(db);
        var gs = new SqliteGraphStore(db);
        var idx = new InMemoryVectorIndex();
        EmbeddingClient embedder = new OllamaEmbeddingClient(
                OLLAMA_URL, "nomic-embed-text", Duration.ofSeconds(30));
        GraphService graph = new KnowledgeGraph(gs);
        MemoryService mem = new LayeredMemoryService(
                ms, idx, embedder, null, graph, null, 100, 0.1, 0.55, 0.55);
        ModelClient extractor = new OllamaModelClient(
                OLLAMA_URL, LLM_MODEL, 0.1, Duration.ofSeconds(180));
        ExtractionService ext = new ExtractionPipeline(extractor, new GraphFactRouter(graph), mem);
        return new Fixture(userId, ms, gs, mem, ext);
    }

    private static boolean ollamaReachable() {
        try {
            var http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(2)).build();
            return http.send(HttpRequest.newBuilder(URI.create(OLLAMA_URL + "/api/tags")).GET().build(),
                    HttpResponse.BodyHandlers.discarding()).statusCode() == 200;
        } catch (Exception e) { return false; }
    }

    private static List<String> contentsOf(List<MemoryItem> items) {
        return items.stream().map(MemoryItem::content).toList();
    }

    private static boolean containsContent(List<String> contents, String needle) {
        return contents.stream().anyMatch(c -> c.contains(needle));
    }

    private static void assertRecallContains(Fixture fx, String query, String token) {
        var hits = fx.mem.recall(RecallRequest.of(fx.user, query, 1500));
        String icon = hits.stream().anyMatch(h ->
                h.content().toLowerCase(Locale.ROOT).contains(token.toLowerCase(Locale.ROOT))) ? "✓" : "✗";
        System.out.printf("    %s \"%s\" → surfaced '%s'? (hits=%d)%n", icon, query, token, hits.size());
        assertFalse(hits.isEmpty(), "'" + query + "' returned empty");
        boolean found = hits.stream().anyMatch(h -> h.content().toLowerCase(Locale.ROOT)
                .contains(token.toLowerCase(Locale.ROOT)));
        assertTrue(found, "'" + query + "' should surface '" + token + "'. Got:\n"
                + String.join("\n", contentsOf(hits)));
    }

    private static void assertRecallContainsAny(Fixture fx, String query, List<String> anyOf) {
        var hits = fx.mem.recall(RecallRequest.of(fx.user, query, 1500));
        boolean ok = hits.stream().anyMatch(h -> {
            String c = h.content().toLowerCase(Locale.ROOT);
            for (String t : anyOf) if (c.contains(t.toLowerCase(Locale.ROOT))) return true;
            return false;
        });
        System.out.printf("    %s \"%s\" → surfaced any of %s? (hits=%d)%n",
                ok ? "✓" : "✗", query, anyOf, hits.size());
        assertFalse(hits.isEmpty(), "'" + query + "' returned empty");
        assertTrue(ok, "'" + query + "' should surface any of " + anyOf);
    }

    private static final class Fixture {
        final String user;
        final SqliteMemoryStore ms;
        final SqliteGraphStore gs;
        final MemoryService mem;
        final ExtractionService ext;
        Fixture(String user, SqliteMemoryStore ms, SqliteGraphStore gs,
                MemoryService mem, ExtractionService ext) {
            this.user = user; this.ms = ms; this.gs = gs; this.mem = mem; this.ext = ext;
        }
        List<MemoryItem> recallAll() {
            return mem.recall(RecallRequest.of(user,
                    "user name work company family health preferences investments events", 4000));
        }
        void close() { ms.close(); gs.close(); }
    }
}
