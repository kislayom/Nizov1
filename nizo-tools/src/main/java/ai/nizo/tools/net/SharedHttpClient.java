package ai.nizo.tools.net;

import java.net.http.HttpClient;
import java.time.Duration;

/**
 * One {@link HttpClient} for the whole tool suite.
 *
 * <p>Each {@code new HttpClient()} call spins up its own selector thread and connection pool. We
 * had four clients before — one inside each tool. Sharing one instance halves cold-start cost
 * and lets the JDK's pool reuse keep-alive connections across {@code web_fetch}, {@code web_search},
 * {@code wikipedia}, etc.
 *
 * <p>{@link #INSTANCE} is built lazily-eagerly (on classload) with reasonable defaults:
 * <ul>
 *   <li>10s connect timeout</li>
 *   <li>{@link HttpClient.Redirect#NORMAL} (follows redirects but not http→https downgrade)</li>
 *   <li>HTTP/2 preferred (falls back to HTTP/1.1 transparently)</li>
 * </ul>
 *
 * <p>Per-request timeouts (read/wait) still go on the {@code HttpRequest.Builder} — those are
 * call-specific. Tools that want a different per-request connect timeout should still use this
 * client; the per-request timeout dominates anyway.
 */
public final class SharedHttpClient {

    /** The shared client. Thread-safe — {@link HttpClient} is documented as such. */
    public static final HttpClient INSTANCE = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .version(HttpClient.Version.HTTP_2)
            .build();

    private SharedHttpClient() { /* no instances */ }
}
