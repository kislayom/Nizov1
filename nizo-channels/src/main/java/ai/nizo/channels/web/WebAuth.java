package ai.nizo.channels.web;

import com.sun.net.httpserver.HttpExchange;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.Set;

/**
 * Token-based authentication for the local web channel.
 *
 * <p>Why this exists: state-changing endpoints (notably {@code POST /api/mcp} which spawns
 * subprocesses, {@code DELETE /api/sessions/*} which wipes data, {@code POST /api/chat/messages}
 * which costs LLM tokens) need a real defense beyond "we bind to 127.0.0.1, mostly". Anything on
 * the loopback interface — including any other process on the box — could otherwise drive the
 * agent. With a per-machine token in {@code ~/.nizo/web-token} (mode 600), only callers who can
 * read the user's home dir get to call mutating endpoints.
 *
 * <p>Auth is presented in either of two ways:
 * <ul>
 *   <li>HTTP header {@code X-Nizo-Token: <token>}</li>
 *   <li>Cookie {@code nizo_token=<token>}</li>
 * </ul>
 * The browser UI uses the cookie (set automatically on the first {@code GET /} from a loopback
 * caller). The iOS app and CLI clients use the header. EventSource cannot set custom headers, so
 * the cookie path is what enables SSE auth for browsers.
 *
 * <p>Origin check: cross-origin browser POSTs always carry a non-loopback {@code Origin} header.
 * We reject those even with a valid token — defense-in-depth against a malicious page on another
 * tab that has stolen the cookie via some other vulnerability.
 */
public final class WebAuth {

    private static final Logger LOG = LoggerFactory.getLogger(WebAuth.class);
    private static final String TOKEN_HEADER = "X-Nizo-Token";
    private static final String TOKEN_COOKIE = "nizo_token";
    /** Comparison helper that blocks timing oracle on token equality. */
    private static boolean constantTimeEquals(String a, String b) {
        if (a == null || b == null) return false;
        byte[] x = a.getBytes(StandardCharsets.UTF_8);
        byte[] y = b.getBytes(StandardCharsets.UTF_8);
        if (x.length != y.length) return false;
        int diff = 0;
        for (int i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
        return diff == 0;
    }

    private final String token;
    private final Path tokenPath;

    public WebAuth(String token, Path tokenPath) {
        this.token = token;
        this.tokenPath = tokenPath;
    }

    /** The current session token (32-byte URL-safe base64). */
    public String token() { return token; }

    /** Where the token is persisted on disk. */
    public Path tokenPath() { return tokenPath; }

    /**
     * Load the token from {@code ~/.nizo/web-token}, generating a fresh one if absent.
     * On POSIX systems the file is created with {@code rw-------} (mode 600).
     */
    public static WebAuth loadOrCreate() throws IOException {
        Path home = Paths.get(System.getProperty("user.home", ".")).resolve(".nizo");
        return loadOrCreate(home);
    }

    /** Variant that allows specifying the parent directory. Used by tests. */
    public static WebAuth loadOrCreate(Path nizoDir) throws IOException {
        Files.createDirectories(nizoDir);
        Path tokenFile = nizoDir.resolve("web-token");
        String token;
        if (Files.exists(tokenFile)) {
            String s = Files.readString(tokenFile, StandardCharsets.UTF_8).trim();
            if (s.isEmpty()) {
                token = generateAndPersist(tokenFile);
            } else {
                token = s;
            }
        } else {
            token = generateAndPersist(tokenFile);
        }
        LOG.info("Web token at {} — required for state-changing requests.", tokenFile);
        return new WebAuth(token, tokenFile);
    }

    private static String generateAndPersist(Path tokenFile) throws IOException {
        byte[] raw = new byte[32];
        new SecureRandom().nextBytes(raw);
        String token = Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
        // Write the file first, then tighten perms. Order matters: setPosixFilePermissions on a
        // non-existent file would throw.
        Files.writeString(tokenFile, token + "\n", StandardCharsets.UTF_8);
        try {
            Set<PosixFilePermission> mode600 = PosixFilePermissions.fromString("rw-------");
            Files.setPosixFilePermissions(tokenFile, mode600);
        } catch (UnsupportedOperationException ignored) {
            // Non-POSIX (Windows). Best-effort restriction below.
            try { tokenFile.toFile().setReadable(false, false); } catch (Exception ignore) {}
            try { tokenFile.toFile().setReadable(true, true); } catch (Exception ignore) {}
            try { tokenFile.toFile().setWritable(false, false); } catch (Exception ignore) {}
            try { tokenFile.toFile().setWritable(true, true); } catch (Exception ignore) {}
        }
        return token;
    }

    // ─────────────────────────── per-request checks ───────────────────────────

    /**
     * Pull the presented token from the request, if any. Header wins over cookie.
     * Returns {@code null} when no token was supplied.
     */
    public static String presentedToken(HttpExchange ex) {
        String h = ex.getRequestHeaders().getFirst(TOKEN_HEADER);
        if (h != null && !h.isBlank()) return h.trim();
        List<String> cookies = ex.getRequestHeaders().get("Cookie");
        if (cookies == null) return null;
        for (String header : cookies) {
            // A single Cookie header can contain multiple name=value pairs separated by ';'.
            for (String pair : header.split(";")) {
                String t = pair.trim();
                int eq = t.indexOf('=');
                if (eq <= 0) continue;
                if (TOKEN_COOKIE.equals(t.substring(0, eq))) {
                    return t.substring(eq + 1).trim();
                }
            }
        }
        return null;
    }

    /** {@code true} iff the supplied request carries the correct token. */
    public boolean isAuthenticated(HttpExchange ex) {
        return constantTimeEquals(token, presentedToken(ex));
    }

    /**
     * Origin check for state-changing requests.
     *
     * <p>Browsers ALWAYS send {@code Origin} on cross-origin POSTs, so the rule is:
     * "if there's an Origin and it isn't loopback, reject". Same-origin and non-browser clients
     * (curl, the iOS app's URLSession, native channels) typically don't set Origin at all — they
     * pass cleanly. The result: a malicious site at evil.com loading the user's localhost URL via
     * fetch() can never POST through, even if the user already authenticated this session.
     */
    public static boolean originIsAcceptable(HttpExchange ex) {
        String origin = ex.getRequestHeaders().getFirst("Origin");
        if (origin == null || origin.isBlank() || "null".equalsIgnoreCase(origin)) return true;
        String lower = origin.toLowerCase();
        if (lower.startsWith("http://localhost") || lower.startsWith("https://localhost")) return true;
        if (lower.startsWith("http://127.") || lower.startsWith("https://127.")) return true;
        if (lower.startsWith("http://[::1]") || lower.startsWith("https://[::1]")) return true;
        return false;
    }

    /** True iff the request came in over the loopback interface. Used by the index handler to
     *  decide whether to seed the cookie. */
    public static boolean isLoopback(HttpExchange ex) {
        InetSocketAddress sa = ex.getRemoteAddress();
        if (sa == null) return false;
        InetAddress addr = sa.getAddress();
        return addr != null && addr.isLoopbackAddress();
    }

    /** Cookie attributes string emitted by {@link #setCookieValue(String)}. */
    public static String setCookieValue(String token) {
        // HttpOnly: blocks document.cookie reads from any in-page script.
        // SameSite=Strict: cookie is never sent on cross-origin navigations / requests.
        // Path=/: scoped to all routes (including SSE under /api/chat/events).
        // Max-Age: 30 days, then user re-authenticates by reloading. Token in file doesn't change
        //          unless the user deletes it; cookie just expires so a stolen value gets stale.
        return TOKEN_COOKIE + "=" + token + "; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000";
    }
}
