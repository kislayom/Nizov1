package ai.nizo.tools.finance;

import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.CookieManager;
import java.net.CookiePolicy;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Shared Yahoo Finance {@code v10/finance/quoteSummary} client.
 *
 * <p>Yahoo's structured fundamentals/analyst/insider/earnings data lives behind a crumb-cookie
 * dance — anonymous GETs return {@code 401 Invalid Cookie}. The flow this class implements
 * mirrors yfinance and is the same dance most public Yahoo wrappers use:
 *
 * <ol>
 *   <li>{@code GET https://fc.yahoo.com} (just to seed the cookie jar with consent + session
 *       cookies). Response body is discarded.</li>
 *   <li>{@code GET https://query{1,2}.finance.yahoo.com/v1/test/getcrumb} with the cookies from
 *       step 1. Returns a plain-text crumb string.</li>
 *   <li>{@code GET https://query{1,2}.finance.yahoo.com/v10/finance/quoteSummary/{ticker}
 *       ?modules=...&crumb=<crumb>} with the same cookies. Returns the structured JSON.</li>
 * </ol>
 *
 * <p>The crumb is cached for {@value #CRUMB_TTL_MS}ms (~6h). On 401/403 we transparently
 * refresh the crumb once and retry; persistent failure surfaces as a runtime exception.
 *
 * <p><b>Single-flight crumb refresh.</b> When the agent kicks off the stock-analyst-estimates
 * skill, three tools (analyst-ratings, earnings-history, insider-activity) call {@code fetch()}
 * back-to-back within a few hundred ms. Without coordination each thread independently calls
 * {@code refreshCrumb()}, hammering Yahoo's getcrumb endpoint and reliably tripping its
 * per-IP 429 rate limit (observed in production). {@code refreshCrumb()} is now serialized
 * with a {@code synchronized} block — concurrent callers wait for the first refresh to land,
 * then read the freshly-cached crumb without re-fetching.
 *
 * <p><b>Host rotation + backoff.</b> Yahoo serves both {@code query1} and {@code query2}; when
 * one IP-rate-limits, the other usually still works. {@code refreshCrumb()} rotates between
 * them on 429 and waits (300ms, 1s, 3s) before each retry, with up to 3 attempts per host.
 *
 * <p>Each instance owns its own {@link HttpClient} with its own {@link CookieManager} — we
 * don't pollute the shared global client with Yahoo's session cookies.
 */
public final class YahooQuoteSummary {

    private static final Logger LOG = LoggerFactory.getLogger(YahooQuoteSummary.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Browser-ish UA. Yahoo rejects the simple "Nizo/1.0" UA on /v10/. */
    private static final String UA =
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0";

    /** Crumb is rotated infrequently by Yahoo; 6h is well within the safe window. */
    private static final long CRUMB_TTL_MS = 6L * 3600_000L;

    /** Hosts to rotate through for both crumb fetch and quoteSummary requests. */
    private static final String[] HOSTS = { "query2.finance.yahoo.com", "query1.finance.yahoo.com" };

    /** Backoff schedule when we hit 429 — first retry is short, subsequent attempts wait longer. */
    private static final long[] RETRY_DELAYS_MS = { 300L, 1000L, 3000L };

    private final HttpClient http;
    private final AtomicReference<CrumbState> cached = new AtomicReference<>();
    /** Host index for quoteSummary requests — rotates on 429 to spread load. */
    private final AtomicReference<Integer> hostIdx = new AtomicReference<>(0);

    /**
     * Tier-2 fallback HTML scraper used when both FMP and the direct API path fail. Optional —
     * null is acceptable. The scraper self-disables if SmartProxy isn't configured.
     */
    private final YahooHtmlScraper fallback;

    /**
     * Tier-0 PRIMARY source — Financial Modeling Prep. When configured (FMP_API_KEY env vars
     * set) it serves data directly from FMP's structured API, which is more reliable than the
     * Yahoo direct path and doesn't suffer the crumb-429 problem at all.
     */
    private final FmpClient fmp;

    public YahooQuoteSummary() {
        this(null, null);
    }

    public YahooQuoteSummary(YahooHtmlScraper fallback) {
        this(null, fallback);
    }

    public YahooQuoteSummary(FmpClient fmp, YahooHtmlScraper fallback) {
        CookieManager cookies = new CookieManager();
        cookies.setCookiePolicy(CookiePolicy.ACCEPT_ALL);
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .cookieHandler(cookies)
                .version(HttpClient.Version.HTTP_2)
                .build();
        this.fallback = fallback;
        this.fmp = fmp;
    }

    /**
     * Fetch the named quoteSummary modules for a ticker.
     *
     * <p>Returns the {@code result[0]} JsonNode — that's the per-ticker payload. Each module
     * appears as a top-level key (e.g. {@code incomeStatementHistory}, {@code earningsTrend},
     * {@code recommendationTrend}, …).
     *
     * @param ticker  symbol — accepts US (AAPL) or exchange-suffixed (HDFCBANK.NS, SAP.DE)
     * @param modules one or more module names, see Yahoo's quoteSummary docs
     * @return {@code quoteSummary.result[0]} JSON node
     * @throws Exception network/parse/auth failure (after retries); message is short enough
     *                   to surface verbatim in tool output
     */
    public JsonNode fetch(String ticker, String... modules) throws Exception {
        if (ticker == null || ticker.isBlank()) {
            throw new IllegalArgumentException("ticker is required");
        }
        if (modules == null || modules.length == 0) {
            throw new IllegalArgumentException("at least one module is required");
        }
        String mods = String.join(",", modules);

        // ───── Tier 0: FMP (Financial Modeling Prep) ─────
        // When configured, FMP is the most reliable source — clean structured API, no crumb
        // dance, no bot-blocks. We try it FIRST. If it returns data for every requested module,
        // we're done. If it returns partial data (e.g. some modules unsupported by FMP), we
        // still take what we can and fall through for the rest. If it fails entirely, fall
        // through to Yahoo paths.
        if (fmp != null && fmp.isEnabled() && fmp.canServe(java.util.Set.of(modules))) {
            try {
                JsonNode fmpResult = fmp.fetch(ticker, modules);
                if (fmpResult != null && !fmpResult.isEmpty()) {
                    // Check whether FMP covered every requested module. If yes, done.
                    boolean allCovered = true;
                    for (String mod : modules) {
                        if (fmpResult.path(mod).isMissingNode() || fmpResult.path(mod).isNull()) {
                            allCovered = false;
                            break;
                        }
                    }
                    if (allCovered) {
                        LOG.info("FMP served all {} module(s) for {}", modules.length, ticker);
                        return fmpResult;
                    }
                    // Partial — keep what we got and fill gaps from Yahoo below.
                    LOG.info("FMP served partial data for {}, continuing with Yahoo for missing modules", ticker);
                }
            } catch (Exception fmpErr) {
                LOG.info("FMP failed for {} ({}), falling through to Yahoo", ticker, fmpErr.getMessage());
            }
        }

        // Try the direct API path next. On any failure (crumb 429, network, etc.) fall through
        // to the HTML scraper if available — the user gets data either way.
        try {
            String crumb = ensureFreshCrumb();
            for (int attempt = 0; attempt < HOSTS.length * 2; attempt++) {
                String host = HOSTS[(hostIdx.get() + attempt) % HOSTS.length];
                DoFetchResult r = doFetch(host, ticker, mods, crumb);
                if (r.body != null) return parse(r.body, ticker);
                if (r.authFailed) {
                    crumb = forceRefreshCrumb();
                    continue;
                }
                if (r.rateLimited) {
                    hostIdx.set((hostIdx.get() + 1) % HOSTS.length);
                    continue;
                }
                throw new RuntimeException("Yahoo quoteSummary failed: " + r.error);
            }
            throw new RuntimeException("Yahoo quoteSummary giving up after host rotation");
        } catch (Exception apiErr) {
            if (fallback == null || !fallback.isAvailable()) throw apiErr;
            LOG.info("Yahoo API path failed ({}), trying HTML scrape fallback for {}",
                    apiErr.getMessage(), ticker);
            java.util.Optional<JsonNode> scraped = fallback.fetch(ticker, modules);
            if (scraped.isPresent()) {
                LOG.info("YahooHtmlScraper succeeded for {} ({} module(s) recovered)", ticker, modules.length);
                return scraped.get();
            }
            // Fallback also empty — propagate the original API error so the LLM sees it.
            throw apiErr;
        }
    }

    /** Get a valid crumb, fetching one only if the cache is empty or stale. Single-flight. */
    private synchronized String ensureFreshCrumb() throws Exception {
        CrumbState st = cached.get();
        long now = Instant.now().toEpochMilli();
        if (st != null && now - st.fetchedAtMs < CRUMB_TTL_MS) {
            return st.crumb;
        }
        return forceRefreshCrumb();
    }

    /** Force a fresh crumb fetch (called after a 401/403). Single-flight via {@code synchronized}. */
    private synchronized String forceRefreshCrumb() throws Exception {
        // 1. Seed cookies. Body discarded; what we want is the Set-Cookie response.
        try {
            URI consent = URI.create("https://fc.yahoo.com/");
            SsrfGuard.assertSafe(consent);
            HttpRequest seed = HttpRequest.newBuilder(consent)
                    .timeout(Duration.ofSeconds(10))
                    .header("User-Agent", UA)
                    .header("Accept", "*/*")
                    .GET().build();
            http.send(seed, HttpResponse.BodyHandlers.discarding());
        } catch (Exception e) {
            // Yahoo's consent endpoint is flaky — but cookies are sometimes set on getcrumb
            // directly. Keep going.
            LOG.debug("Yahoo consent seeding failed (non-fatal): {}", e.toString());
        }

        // 2. Try getcrumb across both hosts, with backoff on 429. Yahoo aggressively rate-limits
        //    individual IPs on getcrumb when called from datacenter ranges; rotating hosts +
        //    a few seconds of backoff usually clears it.
        Exception lastEx = null;
        int attempts = HOSTS.length * RETRY_DELAYS_MS.length;
        for (int i = 0; i < attempts; i++) {
            String host = HOSTS[i % HOSTS.length];
            long delay = RETRY_DELAYS_MS[i / HOSTS.length];
            try {
                String crumb = doGetCrumb(host);
                if (!crumb.isEmpty()) {
                    cached.set(new CrumbState(crumb, Instant.now().toEpochMilli()));
                    LOG.debug("Yahoo crumb refreshed via {}", host);
                    return crumb;
                }
            } catch (Exception e) {
                lastEx = e;
                String msg = e.getMessage() == null ? "" : e.getMessage();
                if (msg.contains("HTTP 429")) {
                    LOG.debug("Yahoo getcrumb 429 on {}, waiting {}ms before next attempt", host, delay);
                    try { Thread.sleep(delay); }
                    catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                    continue;
                }
                // Non-429 failure — propagate immediately, don't waste retries on permanent errors.
                throw e;
            }
        }
        throw new RuntimeException("Yahoo getcrumb exhausted retries on all hosts: "
                + (lastEx == null ? "unknown" : lastEx.getMessage()));
    }

    private String doGetCrumb(String host) throws Exception {
        URI uri = URI.create("https://" + host + "/v1/test/getcrumb");
        SsrfGuard.assertSafe(uri);
        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(10))
                .header("User-Agent", UA)
                .header("Accept", "text/plain,*/*")
                .header("Accept-Language", "en-US,en;q=0.9")
                .GET().build();
        HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
        if (resp.statusCode() != 200) {
            throw new RuntimeException("Yahoo getcrumb HTTP " + resp.statusCode() + " on " + host);
        }
        return resp.body() == null ? "" : resp.body().trim();
    }

    /** Single-host quoteSummary fetch. Returns body on 200; signals auth-failure / 429 / other. */
    private DoFetchResult doFetch(String host, String ticker, String modules, String crumb)
            throws Exception {
        String url = "https://" + host + "/v10/finance/quoteSummary/"
                + URLEncoder.encode(ticker, StandardCharsets.UTF_8)
                + "?modules=" + URLEncoder.encode(modules, StandardCharsets.UTF_8)
                + "&crumb=" + URLEncoder.encode(crumb == null ? "" : crumb, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);

        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(15))
                .header("User-Agent", UA)
                .header("Accept", "application/json")
                .header("Accept-Language", "en-US,en;q=0.9")
                .GET().build();
        HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
        int sc = resp.statusCode();
        if (sc == 200) return DoFetchResult.ofBody(resp.body());
        if (sc == 401 || sc == 403) {
            LOG.debug("Yahoo quoteSummary {} on {} returned {} (crumb expired)", ticker, host, sc);
            return DoFetchResult.auth();
        }
        if (sc == 429) {
            LOG.debug("Yahoo quoteSummary {} on {} returned 429", ticker, host);
            return DoFetchResult.rate();
        }
        if (sc == 404) {
            return DoFetchResult.ofError("ticker not found: " + ticker);
        }
        return DoFetchResult.ofError("HTTP " + sc + " on " + host);
    }

    private static JsonNode parse(String body, String ticker) throws Exception {
        JsonNode root = MAPPER.readTree(body);
        JsonNode err = root.path("quoteSummary").path("error");
        if (err.isObject() && !err.isEmpty() && !err.isNull()) {
            String desc = err.path("description").asText("unknown");
            throw new RuntimeException("Yahoo quoteSummary error for " + ticker + ": " + desc);
        }
        JsonNode result = root.path("quoteSummary").path("result").path(0);
        if (result.isMissingNode() || result.isNull()) {
            throw new RuntimeException("Yahoo quoteSummary returned no data for " + ticker);
        }
        return result;
    }

    /** Helper — Yahoo wraps numeric fields as {@code {raw: 123, fmt: "$123", longFmt:...}}. */
    public static double rawDouble(JsonNode wrapped) {
        if (wrapped == null || wrapped.isMissingNode() || wrapped.isNull()) return Double.NaN;
        JsonNode raw = wrapped.path("raw");
        if (raw.isMissingNode() || raw.isNull()) {
            return wrapped.isNumber() ? wrapped.asDouble() : Double.NaN;
        }
        return raw.asDouble();
    }

    /** Helper — long variant of {@link #rawDouble(JsonNode)}. */
    public static long rawLong(JsonNode wrapped) {
        if (wrapped == null || wrapped.isMissingNode() || wrapped.isNull()) return 0L;
        JsonNode raw = wrapped.path("raw");
        if (raw.isMissingNode() || raw.isNull()) {
            return wrapped.isNumber() ? wrapped.asLong() : 0L;
        }
        return raw.asLong();
    }

    /** Helper — extract {@code fmt} or fall back to raw stringification. */
    public static String fmtString(JsonNode wrapped) {
        if (wrapped == null || wrapped.isMissingNode() || wrapped.isNull()) return "";
        JsonNode fmt = wrapped.path("fmt");
        if (!fmt.isMissingNode() && !fmt.isNull()) return fmt.asText("");
        return wrapped.isTextual() ? wrapped.asText("") : "";
    }

    private record CrumbState(String crumb, long fetchedAtMs) {}

    /** Disambiguate the 4 outcomes of a single-host fetch. */
    private record DoFetchResult(String body, boolean authFailed, boolean rateLimited, String error) {
        static DoFetchResult ofBody(String b)   { return new DoFetchResult(b, false, false, null); }
        static DoFetchResult auth()             { return new DoFetchResult(null, true, false, null); }
        static DoFetchResult rate()             { return new DoFetchResult(null, false, true, null); }
        static DoFetchResult ofError(String m)  { return new DoFetchResult(null, false, false, m); }
    }
}
