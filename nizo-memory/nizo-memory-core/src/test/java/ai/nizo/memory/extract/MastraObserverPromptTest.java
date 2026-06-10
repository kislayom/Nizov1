package ai.nizo.memory.extract;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryService;
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
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pins down the contract that {@link ExtractionPipeline}'s system prompt
 * actually carries the Mastra-Observer extraction-discipline rules. These
 * rules are the half-day-of-work win from the assessment of Mastra's
 * Observational Memory: assertions vs questions, state-change framing,
 * temporal anchoring, multi-event splitting, exact-phrase preservation.
 *
 * <p>We assert by capturing the system prompt the pipeline sends to the
 * model. Each section name (and the canonical anti-example pair) must be
 * present — if a future refactor accidentally drops a section the test fails
 * loudly so the regression is caught at build time, not at benchmark time.
 *
 * <p>Note: we cannot unit-test that an LLM <em>follows</em> these rules — that
 * requires a benchmark run. This test guarantees the rules are <em>delivered</em>
 * to the model, which is the necessary precondition.
 */
class MastraObserverPromptTest {

    private SqliteMemoryStore store;
    private SqliteGraphStore graphStore;
    private InMemoryVectorIndex index;
    private FakeModelClient model;
    private MemoryService memory;
    private ExtractionPipeline pipeline;

    @BeforeEach
    void setUp(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("observer-prompt.db"));
        graphStore = new SqliteGraphStore(tmp.resolve("observer-prompt-graph.db"));
        index = new InMemoryVectorIndex();
        // Canned LLM response — empty extraction is fine; we only care that
        // the system prompt fired through the model client so we can inspect it.
        model = new FakeModelClient("{}");
        memory = new LayeredMemoryService(store, index, new FakeEmbedder(List.of("test")),
                new FakeModelClient("not used"), 999, 0.0);
        KnowledgeGraph kg = new KnowledgeGraph(graphStore);
        GraphFactRouter router = new GraphFactRouter(kg);
        pipeline = new ExtractionPipeline(model, router, memory);
    }

    @AfterEach
    void tearDown() {
        if (graphStore != null) graphStore.close();
    }

    @Test
    void promptCarriesExtractionDisciplineHeader() {
        pipeline.extract("user-1", "I have two kids");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Extraction discipline"),
                "prompt must carry the 'Extraction discipline' section header that names the Mastra port");
    }

    @Test
    void promptDistinguishesAssertionsFromQuestions() {
        pipeline.extract("user-1", "I have two kids");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Assertions vs questions"),
                "prompt must include the assertions-vs-questions section");
        assertTrue(prompt.contains("How many kids do I have?"),
                "prompt must include the canonical question example so the model can disambiguate");
        assertTrue(prompt.contains("AUTHORITATIVE"),
                "prompt must explicitly mark user assertions as authoritative — questions don't invalidate them");
    }

    @Test
    void promptInstructsStateChangeFraming() {
        pipeline.extract("user-1", "I have two kids and I work at Acme Corp.");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("State changes"),
                "prompt must include the state-changes section");
        assertTrue(prompt.contains("switched from"),
                "prompt must show the canonical 'switched from <prior>' example");
    }

    @Test
    void promptCarriesTemporalAnchoringRules() {
        pipeline.extract("user-1", "I have two kids and I work at Acme Corp.");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Temporal anchoring"),
                "prompt must include the temporal-anchoring section");
        assertTrue(prompt.contains("recently") && prompt.contains("soon"),
                "prompt must enumerate the vague time references that should NOT be anchored to a date");
    }

    @Test
    void promptInstructsMultiEventSplitting() {
        pipeline.extract("user-1", "I have two kids and I work at Acme Corp.");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Multi-event splitting"),
                "prompt must include the multi-event-splitting section");
        assertTrue(prompt.contains("SEPARATE entry"),
                "prompt must instruct the model to emit one entry per distinct event");
    }

    @Test
    void promptInstructsExactPhrasePreservation() {
        pipeline.extract("user-1", "I have two kids and I work at Acme Corp.");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Preserve unusual phrasing"),
                "prompt must include the preserve-unusual-phrasing section");
        assertTrue(prompt.contains("movement session"),
                "prompt must include the canonical 'movement session' anti-example pair");
    }

    @Test
    void promptStillEnforcesJsonOnlyOutputContract() {
        // Sanity: extending the prompt must NOT have weakened the existing
        // contract that the model returns ONE JSON object.
        pipeline.extract("user-1", "I have two kids and I work at Acme Corp.");

        String prompt = lastPrompt();
        assertTrue(prompt.contains("Output ONE valid JSON object"),
                "JSON-only output contract must still be present");
        assertTrue(prompt.contains("No prose, no markdown fences"),
                "no-prose / no-markdown rule must still be present");
    }

    /** Returns the most recent system prompt the pipeline sent to the LLM. */
    private String lastPrompt() {
        // The pipeline calls model.complete() with a system + user message;
        // FakeModelClient captures only the first message text, which is the
        // system prompt (per OllamaModelClient ordering).
        if (model.capturedPrompts.isEmpty()) {
            fail("pipeline did not call the LLM — cannot inspect prompt");
        }
        return model.capturedPrompts.get(model.capturedPrompts.size() - 1);
    }

}
