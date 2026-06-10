package ai.nizo.tools.finance;

import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Finnhub data provider — Tier-2 fallback below Yahoo + FMP for the finance tools that
 * have no other source today (ticker search, analyst recommendations, earnings, insider).
 *
 * <p>Free tier: 60 requests/minute, no daily cap. Covers NASDAQ, NYSE, NSE (e.g.
 * {@code RELIANCE.NS}), BSE ({@code RELIANCE.BO}), ASX ({@code BHP.AX}), LSE, Tokyo,
 * and most other major exchanges Yahoo indexes. Single API key read from
 * {@code FINNHUB_TOKEN} env var; disabled silently if not present so the deploy still
 * works without one.
 *
 * <p>Endpoints we use (all GET, all return JSON):
 * <pre>
 *   /search?q=…&token=…
 *   /stock/recommendation?symbol=…&token=…
 *   /stock/earnings?symbol=…&token=…
 *   /stock/insider-transactions?symbol=…&token=…
 * </pre>
 *
 * <p>Each method returns either a {@link JsonNode} on success or {@code null} on any
 * failure (no key, HTTP error, parse error, timeout) — callers decide whether to surface
 * a "data unavailable" placeholder or just skip. We don't throw because callers are
 * supposed to be in a fallback path where the previous tier already failed.
 */
public final class FinnhubClient {

    private static final Logger LOG = LoggerFactory.getLogger(FinnhubClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String BASE = "https://finnhub.io/api/v1";
    private static final Duration TIMEOUT = Duration.ofSeconds(6);
    private static final String UA = "nizo-finance/1.0";

    private final HttpClient http = SharedHttpClient.INSTANCE;
    private final String token;

    public FinnhubClient() { this(System.getenv("FINNHUB_TOKEN")); }

    public FinnhubClient(String token) {
        this.token = (token == null || token.isBlank()) ? null : token.trim();
        if (this.token == null) {
            LOG.info("Finnhub disabled (FINNHUB_TOKEN not set)");
        } else {
            LOG.info("Finnhub enabled (token length {}); free tier = 60 req/min", this.token.length());
        }
    }

    public boolean isEnabled() { return token != null; }

    /**
     * Ticker autocomplete — returns the parsed Finnhub {@code /search} response or null.
     * Useful shape: {@code {"count":…,"result":[{symbol,description,displaySymbol,type}, …]}}.
     */
    public JsonNode search(String query) {
        if (!isEnabled() || query == null || query.isBlank()) return null;
        return getJson("/search?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8));
    }

    /**
     * Analyst recommendation trend — array of monthly buckets, newest first.
     * {@code [{period,buy,hold,sell,strongBuy,strongSell,symbol}, …]}.
     */
    public JsonNode recommendations(String symbol) {
        if (!isEnabled() || symbol == null || symbol.isBlank()) return null;
        return getJson("/stock/recommendation?symbol=" + URLEncoder.encode(symbol, StandardCharsets.UTF_8));
    }

    /**
     * Earnings surprises — array of past earnings prints with actual vs estimate.
     * {@code [{period,actual,estimate,surprise,surprisePercent,symbol}, …]}.
     */
    public JsonNode earnings(String symbol) {
        if (!isEnabled() || symbol == null || symbol.isBlank()) return null;
        return getJson("/stock/earnings?symbol=" + URLEncoder.encode(symbol, StandardCharsets.UTF_8));
    }

    /**
     * Insider transactions for the last ~6 months. Wrapped:
     * {@code {"data":[{name,share,change,filingDate,transactionDate,transactionCode,transactionPrice}, …]}}.
     */
    public JsonNode insiderTransactions(String symbol) {
        if (!isEnabled() || symbol == null || symbol.isBlank()) return null;
        return getJson("/stock/insider-transactions?symbol=" + URLEncoder.encode(symbol, StandardCharsets.UTF_8));
    }

    /** Single HTTP GET → JsonNode, or null on any failure. Adds the auth token. */
    private JsonNode getJson(String pathWithQuery) {
        String url = BASE + pathWithQuery
                + (pathWithQuery.contains("?") ? "&" : "?") + "token=" + URLEncoder.encode(token, StandardCharsets.UTF_8);
        try {
            URI uri = URI.create(url);
            SsrfGuard.assertSafe(uri);
            HttpRequest req = HttpRequest.newBuilder(uri)
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "application/json")
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("Finnhub HTTP {} for {}", resp.statusCode(), redact(pathWithQuery));
                return null;
            }
            return MAPPER.readTree(resp.body());
        } catch (Exception e) {
            LOG.warn("Finnhub call failed for {}: {}", redact(pathWithQuery), e.toString());
            return null;
        }
    }

    /** Strip the token from a URL fragment before logging it. */
    private static String redact(String s) {
        return s == null ? "?" : s.replaceAll("token=[^&]+", "token=***");
    }
}
