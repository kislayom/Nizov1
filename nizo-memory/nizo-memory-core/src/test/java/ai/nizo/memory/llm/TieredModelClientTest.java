package ai.nizo.memory.llm;

import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.testsupport.FakeModelClient;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Behavioural tests for {@link TieredModelClient}.
 *
 * <p>Pins down the routing contract:
 * <ol>
 *   <li>Short input → small backend.</li>
 *   <li>Long input → big backend.</li>
 *   <li>Threshold is inclusive of "small" — exact match counts as small.</li>
 *   <li>System-property override forces every call to one tier.</li>
 *   <li>Constructor rejects null backends and non-positive thresholds.</li>
 * </ol>
 */
class TieredModelClientTest {

    private FakeModelClient small;
    private FakeModelClient big;

    @BeforeEach
    void setUp() {
        small = new FakeModelClient("small-response");
        big = new FakeModelClient("big-response");
    }

    @AfterEach
    void tearDown() {
        System.clearProperty(TieredModelClient.ROUTING_BYPASS_PROPERTY);
    }

    @Test
    void routesShortInputToSmall() {
        TieredModelClient tiered = new TieredModelClient(small, big, 100);
        ModelRequest req = ModelRequest.of(List.of(Message.user("hi")));

        var resp = tiered.complete(req);

        assertEquals("small-response", resp.text());
        assertEquals(1, small.invocations.get(), "small backend must be invoked once");
        assertEquals(0, big.invocations.get(), "big backend must NOT be invoked for short input");
    }

    @Test
    void routesLongInputToBig() {
        TieredModelClient tiered = new TieredModelClient(small, big, 20);
        // Build a long enough message to clear the 20-token threshold.
        String longText = "this is a long message that contains many words and clauses ".repeat(10);
        ModelRequest req = ModelRequest.of(List.of(Message.user(longText)));

        var resp = tiered.complete(req);

        assertEquals("big-response", resp.text());
        assertEquals(0, small.invocations.get(), "small must NOT see long input");
        assertEquals(1, big.invocations.get(), "big backend must be invoked once");
    }

    @Test
    void boundaryAtThresholdIsSmall() {
        // The contract: route to BIG only when tokens > threshold.
        // Construct a message whose token count is equal to the threshold.
        String text = "alpha beta gamma delta";  // ≈ 4 whitespace-tokens
        int tokens = TieredModelClient.inputTokens(
                ModelRequest.of(List.of(Message.user(text))));
        TieredModelClient tiered = new TieredModelClient(small, big, tokens);

        tiered.complete(ModelRequest.of(List.of(Message.user(text))));

        assertEquals(1, small.invocations.get(),
                "input exactly at threshold must route to small, not big");
        assertEquals(0, big.invocations.get());
    }

    @Test
    void forceSmallBypassRoutesEverythingToSmall() {
        System.setProperty(TieredModelClient.ROUTING_BYPASS_PROPERTY, "small");
        TieredModelClient tiered = new TieredModelClient(small, big, 5);

        // Long input that would normally go to big.
        String longText = "alpha beta gamma delta epsilon zeta eta theta ".repeat(5);
        tiered.complete(ModelRequest.of(List.of(Message.user(longText))));

        assertEquals(1, small.invocations.get(),
                "force=small must override token-based routing");
        assertEquals(0, big.invocations.get());
    }

    @Test
    void forceBigBypassRoutesEverythingToBig() {
        System.setProperty(TieredModelClient.ROUTING_BYPASS_PROPERTY, "big");
        TieredModelClient tiered = new TieredModelClient(small, big, 1000);

        // Tiny input that would normally go to small.
        tiered.complete(ModelRequest.of(List.of(Message.user("hi"))));

        assertEquals(0, small.invocations.get());
        assertEquals(1, big.invocations.get(),
                "force=big must override token-based routing");
    }

    @Test
    void rejectsNullBackends() {
        assertThrows(IllegalArgumentException.class,
                () -> new TieredModelClient(null, big, 100));
        assertThrows(IllegalArgumentException.class,
                () -> new TieredModelClient(small, null, 100));
    }

    @Test
    void rejectsNonPositiveThreshold() {
        assertThrows(IllegalArgumentException.class,
                () -> new TieredModelClient(small, big, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new TieredModelClient(small, big, -5));
    }

    @Test
    void capabilityReportsTieredModelId() {
        TieredModelClient tiered = new TieredModelClient(small, big, 100);
        var cap = tiered.capability();
        assertTrue(cap.id().startsWith("tiered:"),
                "synthesised model id must indicate this is a tiered client; got " + cap.id());
        assertEquals("tiered", cap.provider(),
                "provider must be 'tiered' so observability can group");
    }
}
