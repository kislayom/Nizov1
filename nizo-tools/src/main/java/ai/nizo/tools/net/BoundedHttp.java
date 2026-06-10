package ai.nizo.tools.net;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;

/**
 * Bounded HTTP body reading.
 *
 * <p>{@link HttpResponse.BodyHandlers#ofString()} happily slurps an unbounded stream into the
 * heap. A misbehaving (or malicious) server can OOM us by serving an infinite stream. Every
 * outbound HTTP call inside nizo-tools should use {@link #ofString(int)} or
 * {@link #readAll(InputStream, int)} so a hard byte cap aborts the read with a clean
 * {@link IOException}.
 *
 * <h2>How the cap is enforced</h2>
 * {@link #ofString(int)} returns a {@link HttpResponse.BodyHandler} backed by a custom
 * {@link Flow.Subscriber} that tracks running byte count. As soon as the total exceeds the cap
 * we cancel the subscription and complete the future exceptionally with
 * {@link IOException}{@code ("response exceeded N bytes")}. This means we don't hold a
 * full copy of an oversized response in memory — we abort mid-stream.
 *
 * <p>Charset is taken from the {@code Content-Type} header when available, falling back to
 * UTF-8 (HTTP/1.1 RFC 7231 says UTF-8 is the safe default for unknown text).
 *
 * <h2>Defaults</h2>
 * {@link #DEFAULT_MAX_BYTES} is 5 MiB — generous for HTML pages and JSON payloads, refuses
 * downloads of binaries, video, or runaway responses.
 */
public final class BoundedHttp {

    /** 5 MiB — cap for HTML/JSON bodies the agent reads. */
    public static final int DEFAULT_MAX_BYTES = 5 * 1024 * 1024;

    private BoundedHttp() { /* no instances */ }

    /**
     * A {@link HttpResponse.BodyHandler} that reads up to {@code maxBytes} into a string and
     * raises {@link IOException} if the response is larger. Aborts mid-stream — does not
     * buffer the oversize body before failing.
     *
     * @param maxBytes hard cap; pass {@link #DEFAULT_MAX_BYTES} unless you have a reason to override
     * @return body handler suitable for {@link java.net.http.HttpClient#send}
     */
    public static HttpResponse.BodyHandler<String> ofString(int maxBytes) {
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be positive, got " + maxBytes);
        }
        return responseInfo -> new BoundedStringSubscriber(maxBytes, charsetFor(responseInfo));
    }

    /** Convenience overload using {@link #DEFAULT_MAX_BYTES}. */
    public static HttpResponse.BodyHandler<String> ofString() {
        return ofString(DEFAULT_MAX_BYTES);
    }

    /**
     * Read an {@link InputStream} into a UTF-8 string with a hard byte cap. Throws
     * {@link IOException} the moment the cumulative byte count exceeds {@code maxBytes}.
     *
     * <p>Useful for non-{@code HttpClient} call sites (raw socket, third-party SDK etc.).
     *
     * @param in       source stream
     * @param maxBytes hard cap
     * @return decoded string
     * @throws IOException if the stream reads more than {@code maxBytes} or any I/O error occurs
     */
    public static String readAll(InputStream in, int maxBytes) throws IOException {
        if (in == null) throw new IOException("null input stream");
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be positive, got " + maxBytes);
        }
        ByteArrayOutputStream buf = new ByteArrayOutputStream(Math.min(maxBytes, 64 * 1024));
        byte[] chunk = new byte[8 * 1024];
        long total = 0;
        int n;
        while ((n = in.read(chunk)) >= 0) {
            total += n;
            if (total > maxBytes) {
                throw new IOException("response exceeded " + maxBytes + " bytes");
            }
            buf.write(chunk, 0, n);
        }
        return buf.toString(StandardCharsets.UTF_8);
    }

    private static Charset charsetFor(HttpResponse.ResponseInfo info) {
        return info.headers().firstValue("Content-Type")
                .map(BoundedHttp::charsetFromContentType)
                .orElse(StandardCharsets.UTF_8);
    }

    /**
     * Parse {@code charset=...} out of a Content-Type header. Defaults to UTF-8 on missing or
     * unsupported charset names — safer than throwing during a body decode.
     */
    static Charset charsetFromContentType(String contentType) {
        if (contentType == null) return StandardCharsets.UTF_8;
        String lower = contentType.toLowerCase(Locale.ROOT);
        int idx = lower.indexOf("charset=");
        if (idx < 0) return StandardCharsets.UTF_8;
        String name = contentType.substring(idx + "charset=".length()).trim();
        // Strip trailing params or quotes
        int semi = name.indexOf(';');
        if (semi >= 0) name = name.substring(0, semi).trim();
        if (name.startsWith("\"") && name.endsWith("\"") && name.length() >= 2) {
            name = name.substring(1, name.length() - 1);
        }
        try {
            return Charset.forName(name);
        } catch (Exception e) {
            return StandardCharsets.UTF_8;
        }
    }

    /**
     * A {@link HttpResponse.BodySubscriber} that accumulates body bytes up to a cap, then
     * fails the future with {@link IOException}. Cancels the upstream subscription as soon as
     * the cap is exceeded so we don't keep buffering.
     */
    static final class BoundedStringSubscriber implements HttpResponse.BodySubscriber<String> {
        private final int maxBytes;
        private final Charset charset;
        private final ByteArrayOutputStream buf;
        private final CompletableFuture<String> future = new CompletableFuture<>();
        private Flow.Subscription subscription;
        private long total;
        private boolean failed;

        BoundedStringSubscriber(int maxBytes, Charset charset) {
            this.maxBytes = maxBytes;
            this.charset = charset;
            this.buf = new ByteArrayOutputStream(Math.min(maxBytes, 64 * 1024));
        }

        @Override
        public CompletionStage<String> getBody() {
            return future;
        }

        @Override
        public void onSubscribe(Flow.Subscription s) {
            this.subscription = s;
            s.request(Long.MAX_VALUE);
        }

        @Override
        public void onNext(List<ByteBuffer> items) {
            if (failed) return;
            for (ByteBuffer bb : items) {
                int remaining = bb.remaining();
                if (total + remaining > maxBytes) {
                    failed = true;
                    if (subscription != null) subscription.cancel();
                    future.completeExceptionally(new IOException(
                            "response exceeded " + maxBytes + " bytes"));
                    return;
                }
                byte[] chunk = new byte[remaining];
                bb.get(chunk);
                buf.write(chunk, 0, remaining);
                total += remaining;
            }
        }

        @Override
        public void onError(Throwable t) {
            if (!failed) future.completeExceptionally(t);
        }

        @Override
        public void onComplete() {
            if (!failed) future.complete(buf.toString(charset));
        }
    }
}
