package ai.nizo.memory.compact;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeModelClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link CompactionService#autoCompact} — the proactive compaction
 * that fires when a conversation approaches the context window limit.
 */
class AutoCompactTest {

    private SqliteMemoryStore store;
    private MemoryService memory;

    @BeforeEach
    void setup(@TempDir Path tmp) {
        store = new SqliteMemoryStore(tmp.resolve("mem.db"));
        memory = new LayeredMemoryService(store, new InMemoryVectorIndex(),
                null, null, 100, 0.0);
    }

    @AfterEach
    void tearDown() { store.close(); }

    // -------------------- under budget --------------------

    @Test
    void underBudgetReturnsSkippedWithReason() {
        FakeModelClient model = new FakeModelClient("should not be called");
        CompactionService svc = new CompactionService(model, memory);

        List<Message> msgs = List.of(
                Message.user("hi"),
                Message.assistant("hello there"));

        // Use a very large context window so the messages are well under budget.
        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 100_000);

        assertFalse(result.compacted(), "should not compact when under budget");
        assertNotNull(result.skipReason(), "skipped result must include a reason");
        assertTrue(result.skipReason().contains("within budget"),
                "skip reason should mention being within budget, got: " + result.skipReason());
        assertEquals(0, model.invocations.get(),
                "model should not be invoked when under budget");
    }

    // -------------------- over budget --------------------

    @Test
    void overBudgetTriggersCompaction() {
        FakeModelClient model = new FakeModelClient(prompt ->
                "## Summary\n- User discussed GPU setup\n- Decided on 48GB VRAM config");
        CompactionService svc = new CompactionService(model, memory);

        // Build a conversation whose token count exceeds contextWindow - RESERVE_BUFFER.
        // RESERVE_BUFFER is 13_000, so with a 14_000-token window the budget is only ~1000.
        List<Message> msgs = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            msgs.add(Message.user("Tell me about item " + i + " " + "details ".repeat(30)));
            msgs.add(Message.assistant("Answer for item " + i + " " + "explanation ".repeat(30)));
        }

        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 14_000);

        assertTrue(result.compacted(), "should compact when over budget");
        assertNotNull(result.summary(), "compacted result must contain a summary");
        assertTrue(result.summary().contains("GPU"),
                "summary should reflect model output");
        assertTrue(result.messagesCompacted() > 0,
                "should report number of messages compacted");
        assertEquals(1, model.invocations.get(),
                "model should be invoked exactly once for compaction");
    }

    // -------------------- exactly at threshold --------------------

    @Test
    void exactlyAtThresholdShouldNotCompact() {
        FakeModelClient model = new FakeModelClient("should not be called");
        CompactionService svc = new CompactionService(model, memory);

        // autoCompact compacts when: used >= contextWindowTokens - RESERVE_BUFFER (13_000).
        // So if used < contextWindow - 13_000, it skips.
        // With a tiny conversation and a context window set so that
        // contextWindow - 13_000 == exact used tokens + 1, it should still skip.
        // A single short message produces very few tokens.
        List<Message> msgs = List.of(Message.user("short"));

        // "short" is ~2 tokens. Recalled memory adds ~0 in a fresh store.
        // Set context window so that contextWindow - 13_000 > 2 (i.e., contextWindow > 13_002).
        // At 13_003, budget is 3 tokens, and used is ~2 → under budget → skip.
        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 13_003);

        assertFalse(result.compacted(),
                "should not compact when used tokens are just below threshold");
        assertEquals(0, model.invocations.get());
    }

    @Test
    void exactlyAtThresholdBoundaryTriggersCompaction() {
        FakeModelClient model = new FakeModelClient(prompt -> "compacted summary");
        CompactionService svc = new CompactionService(model, memory);

        // autoCompact triggers when: used >= contextWindowTokens - RESERVE_BUFFER (13_000).
        // After that, it calls compact(messages, max(2000, target)) where
        // target = (contextWindow - 13_000) / 2. So the compact() call also
        // requires inputTokens >= max(2000, target).
        //
        // Strategy: use enough messages to generate ~3000+ tokens (exceeds the
        // 2000 floor in compact), and set contextWindow so the autoCompact
        // threshold is well below that.
        List<Message> msgs = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            msgs.add(Message.user("Message " + i + " " + "padding ".repeat(25)));
        }

        // contextWindow = 14_000 → threshold = 14_000 - 13_000 = 1_000.
        // 60 msgs at ~50 tokens each = ~3000 tokens > 1_000 → autoCompact triggers.
        // compact target = max(2000, (14_000-13_000)/2) = max(2000, 500) = 2000.
        // input tokens in compact (with role: prefix) will be ~3500 > 2000 → compacts.
        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 14_000);

        assertTrue(result.compacted(),
                "should compact when used tokens meet or exceed threshold");
        assertEquals(1, model.invocations.get());
    }

    // -------------------- very large conversation --------------------

    @Test
    void veryLargeConversationCompactionReducesTokenCount() {
        FakeModelClient model = new FakeModelClient(prompt ->
                "Key decisions: use 48GB VRAM. Next step: benchmark.");
        CompactionService svc = new CompactionService(model, memory);

        // Build a very large conversation.
        List<Message> msgs = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            msgs.add(Message.user("Question " + i + ": " + "elaborate content ".repeat(40)));
            msgs.add(Message.assistant("Answer " + i + ": " + "detailed response ".repeat(40)));
        }

        // Estimate input token count (rough: each msg ~160+ tokens, 400 msgs total).
        int estimatedInputTokens = 0;
        for (Message m : msgs) {
            estimatedInputTokens += m.text().length() / 4; // rough heuristic like Tokens.count
        }

        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 15_000);

        assertTrue(result.compacted(), "large conversation should trigger compaction");
        assertNotNull(result.summary());

        // The summary produced by the model is much shorter than the original.
        int summaryLength = result.summary().length();
        assertTrue(summaryLength < estimatedInputTokens,
                "summary (" + summaryLength + " chars) should be significantly smaller " +
                "than input (" + estimatedInputTokens + " estimated tokens)");
        assertTrue(result.inputTokens() > result.outputTokens(),
                "output tokens (" + result.outputTokens() + ") should be less than " +
                "input tokens (" + result.inputTokens() + ")");
        assertEquals(400, result.messagesCompacted(),
                "all 400 messages should have been compacted");
    }

    // -------------------- null model --------------------

    @Test
    void autoCompactWithNullModelReturnsSkipped() {
        CompactionService svc = new CompactionService(null, memory);

        List<Message> msgs = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            msgs.add(Message.user("message " + i + " " + "padding ".repeat(20)));
        }

        CompactionService.CompactionResult result = svc.autoCompact("default", msgs, 14_000);

        assertFalse(result.compacted(), "autoCompact with null model should return skipped");
        assertNotNull(result.skipReason());
        assertTrue(result.skipReason().contains("no summariser"),
                "skip reason should mention no summariser, got: " + result.skipReason());
    }
}
