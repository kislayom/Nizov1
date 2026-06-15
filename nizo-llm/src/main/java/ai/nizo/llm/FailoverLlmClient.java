package ai.nizo.llm;

import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.ChatStreamHandler;
import ai.nizo.api.llm.LlmClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * An {@link LlmClient} decorator that makes the model layer resilient: it retries TRANSPORT
 * failures with backoff and, if more than one provider is configured, fails over to the next in
 * order. A model/request error (HTTP 4xx, bad prompt) is NOT retried — failover can't fix it — so
 * it propagates immediately.
 *
 * <p><b>Local-first by default.</b> Configure it with a single provider (the local llama endpoint)
 * and it is purely a resilience wrapper: brief blips, a cold-start tail, or a momentary drop
 * self-heal instead of hard-erroring on the first hiccup, and nothing ever leaves the box. Add a
 * second provider (e.g. {@code NIZO_LLM_FALLBACK_URL}) only if you explicitly want cross-provider
 * failover — that one trades the local-first posture for availability, so it is opt-in.
 *
 * <p>When every provider is exhausted, throws (or, for streaming, surfaces via {@code onError}) an
 * {@link LlmUnavailableException} whose message is friendly enough to show the user directly —
 * "temporarily unavailable… may be busy generating music…retry shortly".
 *
 * <h2>Streaming failover is first-token-safe</h2>
 * Once any token/thinking/tool-delta has been emitted to the caller's handler, we CANNOT retry —
 * doing so would double-emit. So a stream that dies after partial output propagates the error; only
 * a failure BEFORE the first emission (the common "connection refused at the start" case) is
 * retried/failed-over.
 *
 * <p>Tunables: retries default 2 per provider; backoff is short ({@code {1s,3s,6s}}) so interactive
 * chat never blocks for minutes — a 13-minute YuE pause won't be waited out, the user just gets the
 * clear "busy" message and retries. Background jobs (Deep Work) keep their own longer backoff.
 */
public final class FailoverLlmClient implements LlmClient {

    private static final Logger LOG = LoggerFactory.getLogger(FailoverLlmClient.class);
    private static final Pattern HTTP_5XX = Pattern.compile("HTTP\\s*5\\d\\d");
    private static final Pattern HTTP_4XX = Pattern.compile("HTTP\\s*4\\d\\d");

    private final List<LlmClient> providers;
    private final int retriesPerProvider;
    private final long[] backoffMs;

    public FailoverLlmClient(List<LlmClient> providers, int retriesPerProvider, long[] backoffMs) {
        if (providers == null || providers.isEmpty())
            throw new IllegalArgumentException("at least one provider required");
        this.providers = List.copyOf(providers);
        this.retriesPerProvider = Math.max(0, retriesPerProvider);
        this.backoffMs = (backoffMs == null || backoffMs.length == 0) ? new long[]{1000, 3000, 6000} : backoffMs;
    }

    @Override
    public ChatResponse chat(ChatRequest request) {
        Throwable last = null;
        for (int p = 0; p < providers.size(); p++) {
            LlmClient provider = providers.get(p);
            for (int attempt = 0; attempt <= retriesPerProvider; attempt++) {
                try {
                    return provider.chat(request);
                } catch (RuntimeException e) {
                    last = e;
                    if (!isRetryable(e)) throw e;   // model / 4xx error — failover won't help
                    boolean moreHere = attempt < retriesPerProvider;
                    boolean moreProviders = p < providers.size() - 1;
                    if (!moreHere && !moreProviders) break;
                    LOG.warn("LLM chat transport failure (provider {}/{}, attempt {}): {} — {}",
                            p + 1, providers.size(), attempt + 1, brief(e),
                            moreHere ? "retrying" : "failing over");
                    sleep(backoffFor(attempt));
                    if (!moreHere) break;          // exhausted this provider → advance to next
                }
            }
        }
        throw new LlmUnavailableException(UNAVAILABLE_MSG, last);
    }

    @Override
    public void streamChat(ChatRequest request, ChatStreamHandler handler) {
        Throwable last = null;
        for (int p = 0; p < providers.size(); p++) {
            LlmClient provider = providers.get(p);
            for (int attempt = 0; attempt <= retriesPerProvider; attempt++) {
                AtomicBoolean emitted = new AtomicBoolean(false);
                AtomicBoolean completed = new AtomicBoolean(false);
                AtomicBoolean forwardedErr = new AtomicBoolean(false);
                AtomicReference<Throwable> err = new AtomicReference<>();

                ChatStreamHandler wrapped = new ChatStreamHandler() {
                    @Override public void onToken(String t) { emitted.set(true); handler.onToken(t); }
                    @Override public void onThinking(String t) { emitted.set(true); handler.onThinking(t); }
                    @Override public void onToolCallDelta(String id, String n, String a) { emitted.set(true); handler.onToolCallDelta(id, n, a); }
                    @Override public void onComplete(ChatResponse r) { completed.set(true); handler.onComplete(r); }
                    @Override public void onError(Throwable t) {
                        err.set(t);
                        if (emitted.get()) { forwardedErr.set(true); handler.onError(t); }  // mid-stream: forward, can't retry
                    }
                };

                try {
                    provider.streamChat(request, wrapped);
                } catch (RuntimeException e) {
                    err.compareAndSet(null, e);
                }

                if (completed.get()) return;                          // success
                if (emitted.get()) {                                  // partial output then died — can't retry
                    Throwable e = err.get();
                    if (e != null && !forwardedErr.get()) handler.onError(e);
                    else if (e == null) handler.onError(new RuntimeException("stream ended without completion"));
                    return;
                }
                Throwable e = err.get();
                last = e;
                if (e != null && !isRetryable(e)) { handler.onError(e); return; }  // model error before any output
                boolean moreHere = attempt < retriesPerProvider;
                boolean moreProviders = p < providers.size() - 1;
                if (!moreHere && !moreProviders) break;
                LOG.warn("LLM stream transport failure (provider {}/{}, attempt {}): {} — {}",
                        p + 1, providers.size(), attempt + 1, brief(e), moreHere ? "retrying" : "failing over");
                sleep(backoffFor(attempt));
                if (!moreHere) break;
            }
        }
        handler.onError(new LlmUnavailableException(UNAVAILABLE_MSG, last));
    }

    /** Transport / server failures are worth retrying or failing over; client/model errors are not. */
    static boolean isRetryable(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof java.net.ConnectException
                    || t instanceof java.net.http.HttpConnectTimeoutException
                    || t instanceof java.net.http.HttpTimeoutException
                    || t instanceof java.io.IOException) return true;
            String m = t.getMessage();
            if (m != null) {
                if (m.contains("LLM request failed") || m.contains("Connection refused")
                        || m.contains("connection was closed") || m.contains("GOAWAY")) return true;
                if (HTTP_5XX.matcher(m).find()) return true;   // server error — try again / next provider
                if (HTTP_4XX.matcher(m).find()) return false;  // client error — won't change on retry
            }
        }
        return false;
    }

    private long backoffFor(int attempt) {
        return backoffMs[Math.min(attempt, backoffMs.length - 1)];
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
    }

    private static String brief(Throwable e) {
        if (e == null) return "?";
        String m = e.getMessage();
        return e.getClass().getSimpleName() + (m == null ? "" : ": " + (m.length() > 100 ? m.substring(0, 100) : m));
    }

    static final String UNAVAILABLE_MSG =
            "The local model is temporarily unavailable — it may be starting up, or busy generating "
          + "music (which pauses chat for ~13 min). Please retry shortly.";
}
