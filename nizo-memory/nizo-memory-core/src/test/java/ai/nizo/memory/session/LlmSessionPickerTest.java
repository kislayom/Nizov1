package ai.nizo.memory.session;

import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeModelClient;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the contract of {@link LlmSessionPicker}: prompt shape, parsing
 * tolerance, and abstention semantics. These are the pieces most likely to
 * regress under prompt drift or model sloppiness.
 */
class LlmSessionPickerTest {

    private static SqliteMemoryStore.SessionInfo sess(String id, String preview) {
        return new SqliteMemoryStore.SessionInfo(id, Instant.EPOCH, Instant.EPOCH, 10, preview);
    }

    @Test
    void parsesCommaSeparatedPositions() {
        FakeModelClient llm = new FakeModelClient("2, 4, 5");
        var picker = new LlmSessionPicker(llm);

        List<SqliteMemoryStore.SessionInfo> manifest = List.of(
                sess("a", "trip planning"),
                sess("b", "weekend cooking"),
                sess("c", "work deadlines"),
                sess("d", "kids birthday"),
                sess("e", "annual review"));

        Set<String> picks = picker.pick("what did I book last week?", manifest, 5);

        assertEquals(Set.of("b", "d", "e"), picks, "positions 2, 4, 5 should map to ids b, d, e");
        assertTrue(llm.capturedPrompts.get(0).contains("[1]"),
                "prompt should include 1-indexed session positions");
    }

    @Test
    void toleratesProseSurroundingPickedNumbers() {
        FakeModelClient llm = new FakeModelClient("I'd pick sessions 1 and 3.");
        var picker = new LlmSessionPicker(llm);

        List<SqliteMemoryStore.SessionInfo> manifest = List.of(
                sess("alpha", "morning run"),
                sess("beta", "evening walk"),
                sess("gamma", "weekend hike"));

        Set<String> picks = picker.pick("when do I hike?", manifest, 3);

        assertEquals(Set.of("alpha", "gamma"), picks);
    }

    @Test
    void returnsEmptyWhenModelAbstainsWithNone() {
        FakeModelClient llm = new FakeModelClient("NONE");
        var picker = new LlmSessionPicker(llm);

        Set<String> picks = picker.pick("anything about quantum computing?",
                List.of(sess("a", "coffee"), sess("b", "tea")), 3);

        assertTrue(picks.isEmpty(), "NONE response should yield an empty (abstain) set");
    }

    @Test
    void dropsOutOfRangePositions() {
        FakeModelClient llm = new FakeModelClient("1, 99, 3");
        var picker = new LlmSessionPicker(llm);

        List<SqliteMemoryStore.SessionInfo> manifest = List.of(
                sess("a", "x"), sess("b", "y"), sess("c", "z"));

        Set<String> picks = picker.pick("q", manifest, 5);

        assertEquals(Set.of("a", "c"), picks, "99 is out of range and must be ignored");
    }

    @Test
    void capsAtTopN() {
        FakeModelClient llm = new FakeModelClient("1, 2, 3, 4, 5");
        var picker = new LlmSessionPicker(llm);

        List<SqliteMemoryStore.SessionInfo> manifest = List.of(
                sess("a", "x"), sess("b", "y"), sess("c", "z"),
                sess("d", "w"), sess("e", "v"));

        Set<String> picks = picker.pick("q", manifest, 2);

        assertEquals(2, picks.size(), "topN of 2 should clamp the pick count");
        assertEquals(Set.of("a", "b"), picks, "should keep earliest picks from response order");
    }

    @Test
    void abstainsOnLlmFailure() {
        FakeModelClient llm = new FakeModelClient(prompt -> {
            throw new RuntimeException("boom");
        });
        var picker = new LlmSessionPicker(llm);

        Set<String> picks = picker.pick("q",
                List.of(sess("a", "x"), sess("b", "y")), 3);

        assertTrue(picks.isEmpty(), "llm exception should yield empty (abstain), not propagate");
    }

    @Test
    void singleSessionManifestShortCircuits() {
        FakeModelClient llm = new FakeModelClient("this should never be read");
        var picker = new LlmSessionPicker(llm);

        Set<String> picks = picker.pick("q", List.of(sess("only", "x")), 3);

        assertEquals(Set.of("only"), picks);
        assertEquals(0, llm.invocations.get(),
                "single-session manifest must not invoke the LLM — it's a trivial pick");
    }

    @Test
    void emptyManifestReturnsEmpty() {
        FakeModelClient llm = new FakeModelClient("1");
        var picker = new LlmSessionPicker(llm);

        Set<String> picks = picker.pick("q", List.of(), 3);

        assertTrue(picks.isEmpty());
        assertEquals(0, llm.invocations.get(), "empty manifest short-circuits");
    }
}
