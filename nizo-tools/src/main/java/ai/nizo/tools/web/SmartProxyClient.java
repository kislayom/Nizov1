package ai.nizo.tools.web;

import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.Optional;

/**
 * SmartProxy "Universal Scraping API" — paid, anti-bot-bypassing fetch layer.
 *
 * <p>Used as the LAST RESORT when our direct scrapers (Bing, DuckDuckGo, web_fetch) get
 * blocked, rate-limited, or return non-200 responses. SmartProxy renders the page through
 * their fingerprint-rotation infrastructure and returns clean HTML.
 *
 * <p><b>Configuration</b> — env vars (the client is a no-op if either is missing):
 * <ul>
 *   <li>{@code SMARTPROXY_USERNAME} — sub-account name, e.g. {@code smart-XXXXXXXX}</li>
 *   <li>{@code SMARTPROXY_PASSWORD} — sub-account password</li>
 *   <li>{@code SMARTPROXY_GEO} — optional geo (default {@code US})</li>
 *   <li>{@code SMARTPROXY_LOCALE} — optional locale (default {@code en-US})</li>
 * </ul>
 *
 * <p><b>Endpoint</b>: {@code https://scraper.smartproxy.org/v1/query}, source
 * {@code uni_scraper}, JSON body {@code {context:{url}, source, format:[html], geo, locale}}.
 *
 * <p><b>Cost note</b>: each call burns one paid request. Use only after free providers
 * have failed.
 */
public final class SmartProxyClient {

    private static final Logger LOG = LoggerFactory.getLogger(SmartProxyClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String ENDPOINT = "https://scraper.smartproxy.org/v1/query";

    private final HttpClient http = SharedHttpClient.INSTANCE;

    private final String basicAuth;     // null → disabled
    private final String geo;
    private final String locale;
    private final boolean jsRenderDefault;

    /**
     * Tri-state result of the startup self-check. {@link #UNKNOWN} = never probed;
     * {@link #VERIFIED} = at least one successful auth round-trip; {@link #FAILED} = the
     * probe got a 4xx/5xx that names auth specifically (so we know it's misconfigured, not
     * just a transient blip). Per-fetch failures don't flip this back to FAILED — they're
     * usually target-site bot-blocks, not credential rejection.
     */
    public enum Health { UNKNOWN, VERIFIED, FAILED }
    private volatile Health health = Health.UNKNOWN;
    private volatile String healthDetail = "";

    /** Fires the probe at most once per JVM, no matter how many client instances are made. */
    private static final java.util.concurrent.atomic.AtomicBoolean PROBE_FIRED = new java.util.concurrent.atomic.AtomicBoolean(false);

    public SmartProxyClient() {
        String user = System.getenv("SMARTPROXY_USERNAME");
        String pass = System.getenv("SMARTPROXY_PASSWORD");
        if (user != null && !user.isBlank() && pass != null && !pass.isBlank()) {
            String creds = user + ":" + pass;
            this.basicAuth = "Basic " + Base64.getEncoder().encodeToString(creds.getBytes(StandardCharsets.UTF_8));
            LOG.info("SmartProxy enabled (sub-account: {})", user);
        } else {
            this.basicAuth = null;
        }
        this.geo    = Optional.ofNullable(System.getenv("SMARTPROXY_GEO")).orElse("US");
        this.locale = Optional.ofNullable(System.getenv("SMARTPROXY_LOCALE")).orElse("en-US");
        // js_render=true is REQUIRED for the universal scraper to actually fetch pages —
        // js_render=false returns "Find item by ID failed" 500s for this account tier.
        // It does cost more (paid render time, ~30-45s per call), so use sparingly.
        this.jsRenderDefault = !"false".equalsIgnoreCase(
                Optional.ofNullable(System.getenv("SMARTPROXY_JS_RENDER")).orElse("true"));

        // First constructed client fires the auth probe in the background. Can be skipped
        // by setting NIZO_SMARTPROXY_SKIP_PROBE=1 (useful for tests + local dev where the
        // probe burns a credit on every JVM start).
        if (basicAuth != null && PROBE_FIRED.compareAndSet(false, true)
                && !"1".equals(System.getenv("NIZO_SMARTPROXY_SKIP_PROBE"))) {
            selfCheckAsync();
        }
    }

    /** True if SMARTPROXY_USERNAME + SMARTPROXY_PASSWORD are set. */
    public boolean isEnabled() { return basicAuth != null; }

    /** Current health state. {@link Health#UNKNOWN} until {@link #selfCheck()} completes. */
    public Health health() { return health; }

    /** Human-readable detail for the current health state (last error, or "ok"). */
    public String healthDetail() { return healthDetail; }

    /**
     * Fire one cheap probe to confirm credentials work. Non-blocking by default — runs on a
     * daemon thread so {@code Bootstrap} startup isn't gated on a 30-second SmartProxy round
     * trip. If creds are wrong, the WARN log lands within ~5 seconds and the {@link #health}
     * flag flips so callers can render an actionable error to the LLM ("SmartProxy is
     * misconfigured — no fallback for bot-blocked sites") instead of silently degrading.
     */
    public void selfCheckAsync() {
        if (!isEnabled()) {
            health = Health.UNKNOWN;
            healthDetail = "no credentials configured";
            return;
        }
        Thread t = new Thread(this::selfCheck, "smartproxy-selfcheck");
        t.setDaemon(true);
        t.start();
    }

    /** Synchronous version of {@link #selfCheckAsync()} — used by tests. */
    public Health selfCheck() {
        if (!isEnabled()) { health = Health.UNKNOWN; return health; }
        try {
            // example.com is the cheapest possible probe — small body, no anti-bot, available
            // worldwide. We pass js_render=false to keep the credit cost minimal.
            ObjectNode root = MAPPER.createObjectNode();
            root.put("source", "uni_scraper");
            root.put("geo", geo);
            root.put("locale", locale);
            root.put("js_render", false);
            ArrayNode format = root.putArray("format");
            format.add("html");
            ObjectNode context = root.putObject("context");
            context.put("url", "https://example.com/");
            context.put("screenshot_type", 1);
            HttpResponse<String> resp = http.send(
                    HttpRequest.newBuilder(URI.create(ENDPOINT))
                            .timeout(Duration.ofSeconds(15))
                            .header("Authorization", basicAuth)
                            .header("Content-Type", "application/json")
                            .header("Accept", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(root)))
                            .build(),
                    BoundedHttp.ofString());
            int sc = resp.statusCode();
            if (sc == 401 || sc == 403) {
                health = Health.FAILED;
                healthDetail = "auth rejected: HTTP " + sc;
                LOG.error("SmartProxy auth FAILED — credentials rejected (HTTP {}). Bot-blocked sites won't have a fallback.", sc);
                return health;
            }
            if (sc / 100 == 2) {
                health = Health.VERIFIED;
                healthDetail = "ok";
                LOG.info("SmartProxy auth OK (probe HTTP {})", sc);
                return health;
            }
            // 5xx, 429, etc. — not a credential issue, leave UNKNOWN so we keep trying.
            health = Health.UNKNOWN;
            healthDetail = "probe inconclusive: HTTP " + sc;
            LOG.warn("SmartProxy probe inconclusive (HTTP {}); will keep trying on real fetches", sc);
            return health;
        } catch (Exception e) {
            // Network/timeout — could be transient, leave UNKNOWN.
            health = Health.UNKNOWN;
            healthDetail = "probe failed: " + e.getClass().getSimpleName() + ": " + e.getMessage();
            LOG.warn("SmartProxy probe error: {}", e.toString());
            return health;
        }
    }

    /**
     * Fetch a URL through SmartProxy's universal scraper.
     *
     * @param url target URL (http or https)
     * @return HTML body, or empty if SP fails (caller decides whether to fall back further)
     */
    public Optional<String> fetchHtml(String url) {
        return fetchHtml(url, jsRenderDefault);
    }

    /** As {@link #fetchHtml(String)} but lets the caller force {@code js_render} on/off.
     *  Most sites need true (verified working). Static HTML can pass false but that path
     *  currently returns 500 from the SP API for this account — leave it true unless we
     *  later enable the no-JS profile in the dashboard. */
    public Optional<String> fetchHtml(String url, boolean jsRender) {
        if (!isEnabled()) return Optional.empty();
        // Even though SmartProxy is doing the actual fetching, the target URL is the
        // attacker-controlled input. Refuse internal/metadata targets the same as elsewhere.
        try {
            URI target = URI.create(url);
            SsrfGuard.assertSafe(target);
        } catch (SecurityException se) {
            LOG.warn("SsrfGuard refused SmartProxy fetch for {}: {}", url, se.getMessage());
            return Optional.empty();
        } catch (IllegalArgumentException iae) {
            LOG.warn("invalid URL passed to SmartProxy: {}", url);
            return Optional.empty();
        }
        try {
            ObjectNode root = MAPPER.createObjectNode();
            root.put("source", "uni_scraper");
            root.put("geo", geo);
            root.put("locale", locale);
            root.put("js_render", jsRender);
            ArrayNode format = root.putArray("format");
            format.add("html");
            ObjectNode context = root.putObject("context");
            context.put("url", url);
            // screenshot_type:1 is required by the dashboard template for uni_scraper
            // (per the user's curl). Server tolerates it on html-only fetches.
            context.put("screenshot_type", 1);

            String body = MAPPER.writeValueAsString(root);

            HttpResponse<String> resp = http.send(
                    HttpRequest.newBuilder(URI.create(ENDPOINT))
                            .timeout(Duration.ofSeconds(90))   // js_render scrapes can take 30-45s
                            .header("Authorization", basicAuth)
                            .header("Content-Type", "application/json")
                            .header("Accept", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body))
                            .build(),
                    BoundedHttp.ofString());

            int sc = resp.statusCode();
            if (sc / 100 != 2) {
                LOG.warn("SmartProxy fetch failed for {}: HTTP {} body={}", url, sc, abbreviate(resp.body(), 200));
                return Optional.empty();
            }
            // Universal scraper response shapes: depending on tier, body might be:
            //   - direct HTML string (not JSON)
            //   - JSON with results[0].content
            //   - JSON with content
            String respBody = resp.body() == null ? "" : resp.body();
            // Heuristic: if it starts with "<" it's already HTML
            if (respBody.startsWith("<")) {
                return Optional.of(respBody);
            }
            try {
                JsonNode parsed = MAPPER.readTree(respBody);
                JsonNode r = parsed.path("results").path(0);
                if (!r.isMissingNode()) {
                    String html = r.path("content").asText("");
                    if (!html.isEmpty()) return Optional.of(html);
                }
                String alt = parsed.path("content").asText("");
                if (!alt.isEmpty()) return Optional.of(alt);
            } catch (Exception jsonEx) {
                // wasn't JSON — return raw body
                if (!respBody.isEmpty()) return Optional.of(respBody);
            }
            LOG.warn("SmartProxy result empty / unrecognized for {}: {}", url, abbreviate(respBody, 200));
            return Optional.empty();
        } catch (Exception e) {
            LOG.warn("SmartProxy fetch threw for {}: {}", url, e.toString());
            return Optional.empty();
        }
    }

    private static String abbreviate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }
}
