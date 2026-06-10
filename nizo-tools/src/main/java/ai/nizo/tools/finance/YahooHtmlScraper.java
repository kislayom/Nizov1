package ai.nizo.tools.finance;

import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SsrfGuard;
import ai.nizo.tools.web.SmartProxyClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
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
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fallback path for Yahoo Finance structured data when the v10 quoteSummary API is blocked
 * (typical when {@code getcrumb} is rate-limiting our IP, which happens regularly).
 *
 * <h2>Strategy: harvest the SvelteKit hydration payload</h2>
 *
 * <p>Yahoo Finance's modern UI is a SvelteKit app. The server-side renderer prefetches the
 * v10 quoteSummary API (using its OWN internal crumb), then embeds the entire response into
 * the HTML page as a hydration payload, like:
 *
 * <pre>{@code
 * <script type="application/json" data-sveltekit-fetched
 *         data-url="https://query1.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=...&crumb=NeuCwGiYaqL"
 *         data-ttl="-1">
 *   {"status":200,"statusText":"OK","headers":{},"body":"{\"quoteSummary\":{\"result\":[{...all the data...}]}}"}
 * </script>
 * }</pre>
 *
 * <p>By fetching the public {@code /quote/{ticker}/...} page and parsing those script blocks,
 * we get the same data the v10 API returns — without any crumb auth on our side. The page
 * itself is bot-checked but accepts a regular User-Agent + the {@code Sec-Fetch-*} header set
 * (verified working from datacenter IPs in May 2026).
 *
 * <h2>Tier strategy</h2>
 * <ol>
 *   <li><b>Direct fetch</b> with browser headers — free, fast (~500ms), reliable from
 *       datacenter IPs.</li>
 *   <li><b>SmartProxy js-render</b> — paid, slow (30-45s), reliable when (1) above gets a 403
 *       from a transient bot-block.</li>
 * </ol>
 *
 * <p>Each section result is cached in memory for {@value #CACHE_TTL_MS}ms — same data shape
 * the v10 quoteSummary API returns, so callers ({@code StockFundamentalsTool} et al.) read it
 * with the existing {@code result.path("incomeStatementHistory").path(...)} navigation.
 */
public final class YahooHtmlScraper {

    private static final Logger LOG = LoggerFactory.getLogger(YahooHtmlScraper.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 6h cache — financial-statement data changes per-quarter, not per-minute. */
    private static final long CACHE_TTL_MS = 6L * 3600_000L;

    /** Module → page-section that contains it. Determines which URL we fetch. */
    private static final Map<String, String> MODULE_TO_SECTION = Map.ofEntries(
            // /quote/AAPL/ — the root page already covers most "quick stats" modules
            Map.entry("price",                       ""),
            Map.entry("summaryDetail",               ""),
            Map.entry("summaryProfile",              ""),
            Map.entry("assetProfile",                ""),
            // /quote/AAPL/key-statistics — adds key stats + valuation ratios
            Map.entry("defaultKeyStatistics",        "key-statistics"),
            Map.entry("financialData",               "key-statistics"),
            // /quote/AAPL/financials — financial-statements pages
            Map.entry("incomeStatementHistory",      "financials"),
            Map.entry("incomeStatementHistoryQuarterly", "financials"),
            Map.entry("balanceSheetHistory",         "balance-sheet"),
            Map.entry("balanceSheetHistoryQuarterly", "balance-sheet"),
            Map.entry("cashflowStatementHistory",    "cash-flow"),
            Map.entry("cashflowStatementHistoryQuarterly", "cash-flow"),
            // /quote/AAPL/analysis — analyst data
            Map.entry("recommendationTrend",         "analysis"),
            Map.entry("upgradeDowngradeHistory",     "analysis"),
            Map.entry("earningsHistory",             "analysis"),
            Map.entry("earningsTrend",               "analysis"),
            Map.entry("earnings",                    "analysis"),
            Map.entry("calendarEvents",              "analysis"),
            // /quote/AAPL/insider-transactions — insider data
            Map.entry("insiderTransactions",         "insider-transactions"),
            Map.entry("netSharePurchaseActivity",    "insider-transactions"),
            // /quote/AAPL/holders — institutional + insider holders
            Map.entry("insiderHolders",              "holders"),
            Map.entry("institutionOwnership",        "holders"),
            Map.entry("fundOwnership",               "holders"),
            Map.entry("majorHoldersBreakdown",       "holders"),
            Map.entry("majorDirectHolders",          "holders")
    );

    /**
     * Locates SvelteKit-hydration script blocks whose {@code data-url} contains a
     * quoteSummary URL. Capture group 1 = the inner JSON payload (which itself has a
     * stringified body field).
     */
    private static final Pattern SVELTE_HYDRATION_BLOCK = Pattern.compile(
            "<script[^>]*type=\"application/json\"[^>]*data-sveltekit-fetched[^>]*"
                    + "data-url=\"([^\"]*v10/finance/quoteSummary[^\"]*)\"[^>]*>([\\s\\S]+?)</script>",
            Pattern.CASE_INSENSITIVE);

    /** Browser-equivalent headers — required to clear Yahoo's Cloudflare check from datacenter IPs. */
    private static final String UA_CHROME =
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    + "Chrome/127.0.0.0 Safari/537.36";

    /** Cache key = TICKER:section. Value = synthesized {@code result[0]} JsonNode. */
    private final Map<String, CachedSection> cache = new ConcurrentHashMap<>();

    private final HttpClient http;
    private final SmartProxyClient smartProxy;     // tier 2 — paid, JS-render, last resort

    public YahooHtmlScraper(SmartProxyClient smartProxy) {
        // Per-instance HttpClient with cookie jar (Yahoo sets consent + session cookies on
        // first hit; we keep them so subsequent fetches in the same JVM look like one user).
        CookieManager cookies = new CookieManager();
        cookies.setCookiePolicy(CookiePolicy.ACCEPT_ALL);
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .cookieHandler(cookies)
                .version(HttpClient.Version.HTTP_2)
                .build();
        this.smartProxy = smartProxy;
    }

    /** Always available now — direct fetch needs no external service. */
    public boolean isAvailable() {
        return true;
    }

    /**
     * Best-effort fetch of the named modules for a ticker. Returns a synthesized {@code
     * result[0]}-shaped JsonNode (same structure the v10 API returns) merging modules pulled
     * from one or more page sections, or empty if every section failed.
     */
    public Optional<JsonNode> fetch(String ticker, String... modules) {
        if (ticker == null || ticker.isBlank() || modules == null || modules.length == 0) {
            return Optional.empty();
        }

        // Group requested modules by section page so we make at most one fetch per section.
        java.util.Set<String> sections = new java.util.LinkedHashSet<>();
        for (String m : modules) {
            sections.add(MODULE_TO_SECTION.getOrDefault(m, ""));
        }

        ObjectNode synth = MAPPER.createObjectNode();
        boolean anySuccess = false;

        for (String section : sections) {
            JsonNode store = fetchSectionStore(ticker, section);
            if (store == null || store.isMissingNode() || store.isNull()) continue;
            anySuccess = true;
            for (String mod : modules) {
                if (!section.equals(MODULE_TO_SECTION.getOrDefault(mod, ""))) continue;
                JsonNode found = store.path(mod);
                if (!found.isMissingNode() && !found.isNull()) {
                    synth.set(mod, found.deepCopy());
                }
            }
        }

        return anySuccess ? Optional.of(synth) : Optional.empty();
    }

    /** Fetch + parse one Yahoo section page. Returns the embedded {@code result[0]} node. */
    private JsonNode fetchSectionStore(String ticker, String section) {
        String cacheKey = ticker.toUpperCase() + ":" + section;
        CachedSection cs = cache.get(cacheKey);
        long now = System.currentTimeMillis();
        if (cs != null && now - cs.fetchedAtMs < CACHE_TTL_MS) {
            return cs.store;
        }

        String url = buildPageUrl(ticker, section);

        // Tier 1: direct fetch with browser headers (free, fast, ~500ms — works for Yahoo
        // pages because the right Sec-Fetch-* header set clears Yahoo's Cloudflare check
        // even from datacenter IPs).
        String html = directFetch(url);

        // Tier 2: SmartProxy (paid, JS-render capable, last resort for the rare case
        // direct fetch trips a transient bot-block).
        if (html == null && smartProxy != null && smartProxy.isEnabled()) {
            LOG.info("Direct fetch failed for {}, trying SmartProxy", url);
            html = smartProxy.fetchHtml(url).orElse(null);
        }

        if (html == null) {
            LOG.warn("YahooHtmlScraper: both direct + SmartProxy failed for {}", url);
            return null;
        }

        JsonNode store = extractQuoteSummaryFromHtml(html);
        if (store == null) {
            LOG.warn("YahooHtmlScraper: hydration block not found in HTML for {} (len={})", url, html.length());
            return null;
        }
        cache.put(cacheKey, new CachedSection(store, now));
        return store;
    }

    /** Direct HTTP fetch with full browser-emulating header set. Returns null on any error. */
    private String directFetch(String url) {
        try {
            URI uri = URI.create(url);
            SsrfGuard.assertSafe(uri);
            HttpRequest req = HttpRequest.newBuilder(uri)
                    .timeout(Duration.ofSeconds(15))
                    .header("User-Agent", UA_CHROME)
                    .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
                    .header("Accept-Language", "en-US,en;q=0.9")
                    // Sec-Fetch-* are REQUIRED — without them Yahoo's Cloudflare config returns 404.
                    .header("Sec-Fetch-Dest", "document")
                    .header("Sec-Fetch-Mode", "navigate")
                    .header("Sec-Fetch-Site", "none")
                    .header("Sec-Fetch-User", "?1")
                    .header("Upgrade-Insecure-Requests", "1")
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString(8 * 1024 * 1024));   // pages are ~2.5MB
            int sc = resp.statusCode();
            if (sc == 200) return resp.body();
            LOG.debug("YahooHtmlScraper direct fetch {}: HTTP {}", url, sc);
            return null;
        } catch (Exception e) {
            LOG.debug("YahooHtmlScraper direct fetch {} threw: {}", url, e.toString());
            return null;
        }
    }

    private static String buildPageUrl(String ticker, String section) {
        String enc = URLEncoder.encode(ticker, StandardCharsets.UTF_8);
        if (section == null || section.isEmpty()) {
            return "https://finance.yahoo.com/quote/" + enc + "/";
        }
        return "https://finance.yahoo.com/quote/" + enc + "/" + section + "/";
    }

    /**
     * Parse Yahoo's hydration block out of an HTML page and dig down to the {@code result[0]}
     * subtree. The block looks like:
     * <pre>
     * &lt;script ... data-url=".../quoteSummary/..."&gt;
     *   {"status":200,...,"body":"{\"quoteSummary\":{\"result\":[{...}]}}"}
     * &lt;/script&gt;
     * </pre>
     * We unwrap the outer JSON, take the {@code body} string, parse it, and return the
     * inner {@code quoteSummary.result[0]}.
     *
     * <p>Multiple hydration blocks may match (different module sets); we accept the first
     * non-empty result and merge subsequent ones if they bring additional modules.
     */
    static JsonNode extractQuoteSummaryFromHtml(String html) {
        if (html == null || html.isEmpty()) return null;
        Matcher m = SVELTE_HYDRATION_BLOCK.matcher(html);
        ObjectNode merged = null;
        while (m.find()) {
            String inner = m.group(2);
            try {
                JsonNode outer = MAPPER.readTree(inner);
                String body = outer.path("body").asText("");
                if (body.isEmpty()) continue;
                JsonNode bodyJson = MAPPER.readTree(body);
                JsonNode result = bodyJson.path("quoteSummary").path("result").path(0);
                if (result.isMissingNode() || result.isNull()) continue;
                if (merged == null) {
                    merged = MAPPER.createObjectNode();
                }
                // Merge each top-level field. Last one wins on conflict (rare in practice).
                java.util.Iterator<Map.Entry<String, JsonNode>> it = result.fields();
                while (it.hasNext()) {
                    Map.Entry<String, JsonNode> e = it.next();
                    merged.set(e.getKey(), e.getValue().deepCopy());
                }
            } catch (Exception parseEx) {
                LOG.debug("hydration block parse failed: {}", parseEx.toString());
            }
        }
        return merged;
    }

    private record CachedSection(JsonNode store, long fetchedAtMs) {}
}
