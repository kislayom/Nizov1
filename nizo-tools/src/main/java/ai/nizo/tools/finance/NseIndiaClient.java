package ai.nizo.tools.finance;

import ai.nizo.tools.net.BoundedHttp;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.CookieHandler;
import java.net.CookieManager;
import java.net.CookiePolicy;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Locale;

/**
 * NSE India scraper — fills gaps that no other source covers cheaply:
 *
 * <ul>
 *   <li><b>Insider trading</b> ({@code /api/corporates-pit}) — SEBI Form-D/F filings,
 *       the only authoritative source. Yahoo / Finnhub / FMP don't have this for Indian
 *       names on their free tiers.</li>
 *   <li><b>Equity quote</b> ({@code /api/quote-equity}) — same data Yahoo serves but
 *       directly from the exchange, useful when Yahoo throttles.</li>
 * </ul>
 *
 * <p>NSE's endpoints require a session cookie obtained from a public HTML page first.
 * We do that warmup once per-instance, then reuse the cookies across calls. The
 * client uses its OWN {@link HttpClient} with a {@link CookieManager} since the
 * shared one elsewhere doesn't keep cookies (we'd pollute its jar with NSE state).
 *
 * <p>Disabled silently when {@code NIZO_NSEINDIA_DISABLED=1} is set, or when the
 * warmup fails (e.g. NSE geo-blocks the server's IP — they do this for some
 * regions; in that case set {@code NIZO_NSEINDIA_DISABLED=1} to avoid noise).
 */
public final class NseIndiaClient {

    private static final Logger LOG = LoggerFactory.getLogger(NseIndiaClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String BASE = "https://www.nseindia.com";
    private static final String WARMUP_URL = BASE + "/companies-listing/corporate-filings-insider-trading";
    private static final Duration TIMEOUT = Duration.ofSeconds(10);
    // NSE's WAF is allergic to obvious bot UAs — masquerade as a recent Firefox build.
    private static final String UA =
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0";

    private final HttpClient http;
    private final boolean enabled;
    private volatile boolean warmedUp = false;

    public NseIndiaClient() {
        this.enabled = !"1".equals(System.getenv("NIZO_NSEINDIA_DISABLED"));
        // Per-instance CookieManager so we don't pollute the global shared client.
        CookieManager cm = new CookieManager();
        cm.setCookiePolicy(CookiePolicy.ACCEPT_ALL);
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .cookieHandler(cm)
                .build();
        if (enabled) LOG.info("NSE India enabled (insider trading + equity quote fallback for .NS / .BO)");
        else LOG.info("NSE India disabled via NIZO_NSEINDIA_DISABLED=1");
    }

    public boolean isEnabled() { return enabled; }

    /** True if the ticker is an NSE/BSE symbol (.NS or .BO suffix). */
    public static boolean isIndianTicker(String ticker) {
        if (ticker == null) return false;
        String upper = ticker.toUpperCase(Locale.ROOT);
        return upper.endsWith(".NS") || upper.endsWith(".BO");
    }

    /** Strip .NS/.BO → bare NSE symbol. */
    public static String toNseSymbol(String ticker) {
        if (ticker == null) return "";
        String upper = ticker.toUpperCase(Locale.ROOT);
        if (upper.endsWith(".NS") || upper.endsWith(".BO")) return upper.substring(0, upper.length() - 3);
        return upper;
    }

    /**
     * Fetch raw insider-trading data from NSE for a ticker. Returns the parsed JSON
     * (shape: {@code {data: [{acqName, anex, date, secAcq, secVal, befAcqSharesNo, ...}]}})
     * or {@code null} on any failure. Call signs:
     *
     * <pre>
     *   anex                    "7(3)" = transaction filing, "7(4)" = pre-clearance
     *   acqName                 person or entity name
     *   acqMode                 "Market Purchase" / "Market Sale" / "Allotment" / ...
     *   secAcq                  shares transacted (string with commas, parse with toLong)
     *   secVal                  total value in INR
     *   date                    "10-Dec-2024 20:01" — filing timestamp
     *   tdpTransactionType      "Buy" / "Sell"
     * </pre>
     */
    public JsonNode insiderTrading(String ticker) {
        if (!isEnabled() || !isIndianTicker(ticker)) return null;
        if (!ensureWarmedUp()) return null;
        String symbol = toNseSymbol(ticker);
        String url = BASE + "/api/corporates-pit?index=equities&symbol="
                + URLEncoder.encode(symbol, StandardCharsets.UTF_8);
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "application/json")
                    .header("Referer", WARMUP_URL)
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("NSE insider HTTP {} for {}", resp.statusCode(), symbol);
                return null;
            }
            return MAPPER.readTree(resp.body());
        } catch (Exception e) {
            LOG.warn("NSE insider fetch failed for {}: {}", symbol, e.toString());
            return null;
        }
    }

    /**
     * Fetch constituents of an NSE-published index. Returns parsed JSON (shape:
     * {@code {data: [{symbol, identifier, lastPrice, ..., meta: {isin, companyName,
     * industry, ...}}], advance: {...}, declines: {...}, ...}}) or {@code null}.
     *
     * <p>Used by IndiaUniverseTool to build the "Top 10 Indian stocks" universe.
     * Authoritative source — NSE publishes constituent updates same-day after
     * monthly/semi-annual rebalances.
     *
     * @param indexName e.g. {@code "NIFTY 500"}, {@code "NIFTY 50"}, {@code "NIFTY BANK"},
     *                  {@code "NIFTY MIDCAP 150"}. URL-encoded internally; pass the
     *                  human-readable label as it appears on nseindia.com/market-data.
     */
    public JsonNode indexConstituents(String indexName) {
        if (!isEnabled()) return null;
        if (!ensureWarmedUp()) return null;
        String url = BASE + "/api/equity-stockIndices?index="
                + URLEncoder.encode(indexName, StandardCharsets.UTF_8);
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "application/json")
                    .header("Referer", BASE + "/market-data/live-equity-market")
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("NSE index-constituents HTTP {} for '{}'", resp.statusCode(), indexName);
                return null;
            }
            return MAPPER.readTree(resp.body());
        } catch (Exception e) {
            LOG.warn("NSE index-constituents fetch failed for '{}': {}", indexName, e.toString());
            return null;
        }
    }

    /**
     * One-time warmup — NSE's API endpoints reject requests without session cookies.
     * Hits a public HTML page first, lets the CookieManager pick up {@code nseappid} etc.,
     * then future API calls can ride those cookies. Idempotent: skipped if already done.
     * Returns false on failure so callers can short-circuit gracefully.
     */
    private boolean ensureWarmedUp() {
        if (warmedUp) return true;
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(WARMUP_URL))
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "text/html,application/xhtml+xml")
                    .GET().build();
            HttpResponse<Void> resp = http.send(req, java.net.http.HttpResponse.BodyHandlers.discarding());
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("NSE warmup HTTP {} — disabling client for this run", resp.statusCode());
                return false;
            }
            warmedUp = true;
            return true;
        } catch (Exception e) {
            LOG.warn("NSE warmup failed (geo-blocked? throttled?): {}", e.toString());
            return false;
        }
    }
}
