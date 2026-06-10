package ai.nizo.channels.web;

import com.sun.net.httpserver.HttpExchange;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Body-size guard for the embedded HTTP server.
 *
 * <p>Why: {@code HttpExchange#getRequestBody().readAllBytes()} happily slurps a multi-gigabyte
 * upload into the JVM heap and the process either OOMs or freezes for tens of seconds while the
 * GC thrashes. The fix is a hard limit (5 MB by default) enforced two ways:
 *
 * <ol>
 *   <li>Pre-flight {@code Content-Length} check — if the header is present and {@code > max} we
 *       respond 413 immediately without reading a byte.</li>
 *   <li>Streaming counter — a {@link FilterInputStream} that throws
 *       {@link BodyTooLargeException} the moment cumulative bytes exceed {@code max}. This catches
 *       chunked transfers and clients that lie about Content-Length.</li>
 * </ol>
 *
 * <p>A 5 MB ceiling is generous for chat — even with images base64'd into JSON, individual
 * snapshots on a phone are ~2 MB. The voice transcribe/music compose endpoints are the only
 * candidates for a higher per-route ceiling, but 5 MB is enough for short clips so we use the
 * same number everywhere for simplicity.
 */
public final class BoundedBodyReader {

    /** Default per-request body limit. Configurable via {@code NIZO_WEB_MAX_BODY_BYTES}. */
    public static final long DEFAULT_MAX_BYTES = 5L * 1024 * 1024;

    private BoundedBodyReader() {}

    public static long configuredMaxBytes() {
        String env = System.getenv("NIZO_WEB_MAX_BODY_BYTES");
        if (env == null || env.isBlank()) return DEFAULT_MAX_BYTES;
        try {
            long v = Long.parseLong(env.trim());
            if (v < 1024) return DEFAULT_MAX_BYTES;     // refuse to lower below 1 KB
            return v;
        } catch (NumberFormatException e) {
            return DEFAULT_MAX_BYTES;
        }
    }

    /**
     * Inspect the {@code Content-Length} header. If it's set and exceeds {@code max}, write a
     * 413 response and return {@code false}. Returns {@code true} when the caller may proceed.
     *
     * <p>A missing or unparseable Content-Length is permissive here — chunked transfers omit it
     * and we still want to support them. The streaming layer ({@link #wrap}) catches those.
     */
    public static boolean enforceContentLength(HttpExchange ex, long max) throws IOException {
        String cl = ex.getRequestHeaders().getFirst("Content-Length");
        if (cl == null) return true;
        try {
            long len = Long.parseLong(cl.trim());
            if (len > max) {
                writeTooLarge(ex, max, len);
                return false;
            }
            return true;
        } catch (NumberFormatException e) {
            // Malformed header — let the stream layer handle it.
            return true;
        }
    }

    /** Wrap an input stream so it throws {@link BodyTooLargeException} after {@code max} bytes. */
    public static InputStream wrap(InputStream raw, long max) {
        return new LimitedInputStream(raw, max);
    }

    /** Read the entire body but never more than {@code max} bytes. Throws on overflow. */
    public static byte[] readAllBounded(HttpExchange ex, long max) throws IOException {
        try (InputStream is = wrap(ex.getRequestBody(), max)) {
            return is.readAllBytes();
        }
    }

    /** Convenience: handle the 413 response. Public so handlers that catch the streaming
     *  exception can render the error in the same shape. */
    public static void writeTooLarge(HttpExchange ex, long max, long observed) throws IOException {
        String body = "{\"error\":\"payload too large\",\"limitBytes\":" + max
                + ",\"observedBytes\":" + observed
                + ",\"hint\":\"reduce the request body or set NIZO_WEB_MAX_BODY_BYTES to a higher value\"}";
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(413, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    public static final class BodyTooLargeException extends IOException {
        private final long limit;
        public BodyTooLargeException(long limit) {
            super("body exceeds " + limit + " bytes");
            this.limit = limit;
        }
        public long limit() { return limit; }
    }

    /** Counting wrapper that aborts as soon as the limit is breached. */
    private static final class LimitedInputStream extends FilterInputStream {
        private final long max;
        private long count;
        LimitedInputStream(InputStream in, long max) {
            super(in);
            this.max = max;
        }

        @Override
        public int read() throws IOException {
            int b = super.read();
            if (b == -1) return -1;
            count++;
            if (count > max) throw new BodyTooLargeException(max);
            return b;
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int n = super.read(b, off, len);
            if (n == -1) return -1;
            count += n;
            if (count > max) throw new BodyTooLargeException(max);
            return n;
        }
    }
}
