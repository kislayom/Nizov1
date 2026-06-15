package ai.nizo.llm;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.ChatStreamHandler;
import ai.nizo.api.llm.LlmClient;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests {@link FailoverLlmClient}: transport blips retry, exhausted providers fail over, model/4xx
 * errors propagate without retry, total exhaustion yields a friendly {@link LlmUnavailableException},
 * and streaming only retries BEFORE the first token (never double-emits).
 */
class FailoverLlmClientTest {

    private static final long[] NO_SLEEP = {0};
    private static final ChatRequest REQ = ChatRequest.of("m", List.of(ChatMessage.user("hi")));

    private static ChatResponse ok(String c) { return new ChatResponse(c, List.of(), "stop", ChatResponse.Usage.EMPTY); }
    private static RuntimeException transport() { return new RuntimeException("LLM request failed: Connection refused"); }
    private static RuntimeException http400() { return new RuntimeException("LLM HTTP 400: bad request"); }
    private static RuntimeException http503() { return new RuntimeException("LLM HTTP 503: overloaded"); }

    private static final class StubLlm implements LlmClient {
        final Deque<Object> chatScript = new ArrayDeque<>();        // ChatResponse or RuntimeException
        final Deque<Consumer<ChatStreamHandler>> streamScript = new ArrayDeque<>();
        int chatCalls = 0, streamCalls = 0;
        @Override public ChatResponse chat(ChatRequest r) {
            chatCalls++;
            Object o = chatScript.poll();
            if (o instanceof RuntimeException re) throw re;
            return (ChatResponse) o;
        }
        @Override public void streamChat(ChatRequest r, ChatStreamHandler h) {
            streamCalls++;
            Consumer<ChatStreamHandler> a = streamScript.poll();
            if (a != null) a.accept(h);
        }
    }

    private static final class RecHandler implements ChatStreamHandler {
        final List<String> tokens = new ArrayList<>();
        ChatResponse completed;
        Throwable error;
        @Override public void onToken(String t) { tokens.add(t); }
        @Override public void onComplete(ChatResponse r) { completed = r; }
        @Override public void onError(Throwable t) { error = t; }
    }

    @Test
    void retriesTransportThenSucceeds() {
        StubLlm p = new StubLlm();
        p.chatScript.add(transport());
        p.chatScript.add(ok("recovered"));
        ChatResponse r = new FailoverLlmClient(List.of(p), 2, NO_SLEEP).chat(REQ);
        assertEquals("recovered", r.content());
        assertEquals(2, p.chatCalls, "should have retried once after the blip");
    }

    @Test
    void failsOverToSecondProvider() {
        StubLlm p1 = new StubLlm();
        p1.chatScript.add(transport()); p1.chatScript.add(transport()); p1.chatScript.add(transport());
        StubLlm p2 = new StubLlm();
        p2.chatScript.add(ok("from-fallback"));
        ChatResponse r = new FailoverLlmClient(List.of(p1, p2), 2, NO_SLEEP).chat(REQ);
        assertEquals("from-fallback", r.content());
        assertEquals(3, p1.chatCalls, "primary tried 1 + 2 retries");
        assertEquals(1, p2.chatCalls, "then failed over to the second provider");
    }

    @Test
    void serverError5xxIsRetryable() {
        StubLlm p = new StubLlm();
        p.chatScript.add(http503());
        p.chatScript.add(ok("ok"));
        assertEquals("ok", new FailoverLlmClient(List.of(p), 2, NO_SLEEP).chat(REQ).content());
        assertEquals(2, p.chatCalls);
    }

    @Test
    void modelError4xxPropagatesWithoutRetry() {
        StubLlm p = new StubLlm();
        p.chatScript.add(http400());
        p.chatScript.add(ok("never-reached"));
        RuntimeException thrown = assertThrows(RuntimeException.class,
                () -> new FailoverLlmClient(List.of(p), 2, NO_SLEEP).chat(REQ));
        assertFalse(thrown instanceof LlmUnavailableException, "4xx is not an availability problem");
        assertTrue(thrown.getMessage().contains("400"));
        assertEquals(1, p.chatCalls, "client errors must not be retried");
    }

    @Test
    void allExhaustedThrowsUnavailable() {
        StubLlm p = new StubLlm();
        p.chatScript.add(transport()); p.chatScript.add(transport()); p.chatScript.add(transport());
        LlmUnavailableException ex = assertThrows(LlmUnavailableException.class,
                () -> new FailoverLlmClient(List.of(p), 2, NO_SLEEP).chat(REQ));
        assertTrue(ex.getMessage().toLowerCase().contains("unavailable"));
        assertEquals(3, p.chatCalls);
    }

    @Test
    void streamRetriesBeforeFirstToken() {
        StubLlm p = new StubLlm();
        p.streamScript.add(h -> h.onError(transport()));                       // pre-token failure
        p.streamScript.add(h -> { h.onToken("hi"); h.onComplete(ok("hi")); }); // succeeds on retry
        RecHandler rec = new RecHandler();
        new FailoverLlmClient(List.of(p), 2, NO_SLEEP).streamChat(REQ, rec);
        assertEquals(List.of("hi"), rec.tokens);
        assertNotNull(rec.completed);
        assertNull(rec.error, "the swallowed pre-token error must not reach the caller");
        assertEquals(2, p.streamCalls);
    }

    @Test
    void streamDoesNotRetryAfterTokensEmitted() {
        StubLlm p = new StubLlm();
        p.streamScript.add(h -> { h.onToken("partial"); h.onError(transport()); }); // mid-stream death
        RecHandler rec = new RecHandler();
        new FailoverLlmClient(List.of(p), 2, NO_SLEEP).streamChat(REQ, rec);
        assertEquals(List.of("partial"), rec.tokens);
        assertNotNull(rec.error, "a mid-stream error must be forwarded, not swallowed");
        assertEquals(1, p.streamCalls, "must NOT retry once output has been emitted");
    }
}
