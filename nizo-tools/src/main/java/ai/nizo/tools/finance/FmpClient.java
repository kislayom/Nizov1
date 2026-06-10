package ai.nizo.tools.finance;

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
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Financial Modeling Prep (FMP) data provider — Tier-0 source for structured financial data,
 * ported from Kimaya's {@code FMPProvider}.
 *
 * <p>FMP gives clean, well-tested fundamentals (income/balance/cashflow), key metrics, ratios,
 * analyst recommendations, earnings surprises, and insider transactions in a single REST API
 * with no auth dance. Free tier is 250 requests/day per API key — we read up to 5 keys from
 * env vars ({@code FMP_API_KEY}, {@code FMP_API_KEY1}…{@code FMP_API_KEY4}) and rotate
 * automatically when one key approaches its daily limit.
 *
 * <h2>Output shape</h2>
 * Methods return JSON in Yahoo's {@code quoteSummary.result[0]} shape so the existing
 * {@code stock_*} tools and adapter code work unchanged. FMP's bare numeric fields go
 * through as-is — {@link YahooQuoteSummary#rawDouble} accepts both {@code {raw, fmt}}-wrapped
 * Yahoo values AND bare numbers, so no wrapping is necessary.
 *
 * <h2>Module mapping</h2>
 * <pre>
 * Yahoo module name              ← FMP endpoint(s)
 * ─────────────────────────────────────────────────
 * incomeStatementHistory          /income-statement?period=annual
 * balanceSheetHistory             /balance-sheet-statement?period=annual
 * cashflowStatementHistory        /cash-flow-statement?period=annual
 * defaultKeyStatistics            /key-metrics + /ratios
 * summaryDetail                   /profile + /ratios
 * financialData                   /ratios + /key-metrics + /profile
 * assetProfile                    /profile (sector, industry, business summary)
 * price                           /profile (price, marketCap, currency, exchange)
 * recommendationTrend             /analyst-stock-recommendations (aggregated)
 * upgradeDowngradeHistory         /analyst-stock-recommendations (raw rows)
 * earningsHistory                 /earnings-surprises
 * insiderTransactions             /insider-trading
 * </pre>
 */
public final class FmpClient {

    private static final Logger LOG = LoggerFactory.getLogger(FmpClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP = SharedHttpClient.INSTANCE;

    private static final String BASE_URL = "https://financialmodelingprep.com/stable";
    private static final int REQUESTS_PER_KEY_PER_DAY = 250;

    /** Modules this client knows how to source. Anything outside this set is unsupported. */
    private static final Set<String> SUPPORTED_MODULES = Set.of(
            "incomeStatementHistory",
            "balanceSheetHistory",
            "cashflowStatementHistory",
            "defaultKeyStatistics",
            "summaryDetail",
            "financialData",
            "assetProfile",
            "price",
            "recommendationTrend",
            "upgradeDowngradeHistory",
            "earningsHistory",
            "insiderTransactions"
    );

    private final List<String> apiKeys;
    private final Map<String, Integer> keyRequestCounts = new HashMap<>();
    private int currentKeyIndex = 0;
    private LocalDate lastResetDate = LocalDate.now(ZoneOffset.UTC);

    public FmpClient() {
        this(loadKeysFromEnv());
    }

    public FmpClient(List<String> apiKeys) {
        this.apiKeys = apiKeys == null ? List.of() : List.copyOf(apiKeys);
        for (String k : this.apiKeys) keyRequestCounts.put(k, 0);
        if (this.apiKeys.isEmpty()) {
            LOG.info("FMP disabled (no FMP_API_KEY[1-4] env vars set)");
        } else {
            LOG.info("FMP enabled — {} key(s) in rotation, {} requests/day total cap",
                    this.apiKeys.size(), this.apiKeys.size() * REQUESTS_PER_KEY_PER_DAY);
        }
    }

    /** Read FMP_API_KEY + FMP_API_KEY1..FMP_API_KEY4 from env. Order matters for rotation. */
    private static List<String> loadKeysFromEnv() {
        List<String> keys = new ArrayList<>();
        for (String name : new String[]{"FMP_API_KEY", "FMP_API_KEY1", "FMP_API_KEY2", "FMP_API_KEY3", "FMP_API_KEY4"}) {
            String v = System.getenv(name);
            if (v != null && !v.isBlank()) keys.add(v.trim());
        }
        return keys;
    }

    public boolean isEnabled() { return !apiKeys.isEmpty(); }

    /** Per-day count of requests still available across all keys. */
    public int remainingDailyRequests() {
        int total = 0;
        for (String k : apiKeys) {
            total += Math.max(0, REQUESTS_PER_KEY_PER_DAY - keyRequestCounts.getOrDefault(k, 0));
        }
        return total;
    }

    /**
     * Fetch one or more modules and return them merged into a Yahoo {@code result[0]}-shaped
     * JsonNode.
     *
     * @return synthesized result, or {@code null} if FMP is disabled / nothing fetched
     */
    public JsonNode fetch(String ticker, String... modules) throws Exception {
        if (!isEnabled() || ticker == null || ticker.isBlank()
                || modules == null || modules.length == 0) {
            return null;
        }
        ObjectNode synth = MAPPER.createObjectNode();

        // We batch endpoint calls — multiple modules from the same endpoint share one fetch.
        Set<String> requested = Set.copyOf(Arrays.asList(modules));
        boolean anySuccess = false;

        // /profile is cheap and feeds multiple Yahoo modules, fetch once.
        boolean needsProfile = requested.stream().anyMatch(
                m -> Set.of("price", "summaryDetail", "assetProfile", "financialData", "defaultKeyStatistics").contains(m));
        JsonNode profile = needsProfile ? fetchProfile(ticker) : null;

        boolean needsRatios = requested.stream().anyMatch(
                m -> Set.of("financialData", "summaryDetail", "defaultKeyStatistics").contains(m));
        JsonNode ratios = needsRatios ? fetchRatios(ticker) : null;

        boolean needsKeyMetrics = requested.stream().anyMatch(
                m -> Set.of("defaultKeyStatistics", "financialData").contains(m));
        JsonNode keyMetrics = needsKeyMetrics ? fetchKeyMetrics(ticker) : null;

        for (String mod : modules) {
            try {
                JsonNode payload = switch (mod) {
                    case "incomeStatementHistory"    -> incomeStatementHistory(fetchAnnualIncome(ticker));
                    case "balanceSheetHistory"       -> balanceSheetHistory(fetchAnnualBalance(ticker));
                    case "cashflowStatementHistory"  -> cashflowStatementHistory(fetchAnnualCashflow(ticker));
                    case "price"                     -> priceModule(profile);
                    case "summaryDetail"             -> summaryDetailModule(profile, ratios);
                    case "assetProfile"              -> assetProfileModule(profile);
                    case "defaultKeyStatistics"      -> defaultKeyStatisticsModule(keyMetrics, ratios);
                    case "financialData"             -> financialDataModule(profile, keyMetrics, ratios, fetchPriceTargetConsensus(ticker));
                    case "recommendationTrend"       -> recommendationTrendModule(fetchAnalystRecs(ticker));
                    case "upgradeDowngradeHistory"   -> upgradeDowngradeHistoryModule(fetchAnalystRecs(ticker));
                    case "earningsHistory"           -> earningsHistoryModule(fetchEarningsSurprises(ticker));
                    case "insiderTransactions"       -> insiderTransactionsModule(fetchInsiderTrades(ticker));
                    default -> null;
                };
                if (payload != null && !payload.isNull() && !payload.isEmpty()) {
                    synth.set(mod, payload);
                    anySuccess = true;
                }
            } catch (Exception e) {
                LOG.warn("FMP module {} for {} failed: {}", mod, ticker, e.toString());
            }
        }

        return anySuccess ? synth : null;
    }

    /** Returns true if FMP can serve any of the requested modules. */
    public boolean canServe(Set<String> modules) {
        if (!isEnabled() || modules == null) return false;
        for (String m : modules) {
            if (SUPPORTED_MODULES.contains(m)) return true;
        }
        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Endpoint fetchers — return the FMP raw JSON (array root)
    // ─────────────────────────────────────────────────────────────────────────

    private JsonNode fetchProfile(String ticker) throws Exception {
        return fetchArrayFirst("/profile?symbol=" + enc(ticker));
    }

    /**
     * PUBLIC — fetch historical end-of-day OHLCV for a ticker over a date range. Used by
     * {@link HistoricalPriceTool} as a fallback when Yahoo's v8 chart endpoint rate-limits.
     *
     * <p>FMP returns rows newest-first; caller may need to reverse for chart consumers.
     * Each row has: {@code date} (YYYY-MM-DD), {@code open}, {@code high}, {@code low},
     * {@code close}, {@code adjClose}, {@code volume}.
     *
     * @param ticker  e.g. "AAPL"
     * @param fromIso start date "YYYY-MM-DD" (inclusive)
     * @param toIso   end date "YYYY-MM-DD" (inclusive)
     * @return array of bars (may be empty), or null on hard FMP failure
     */
    public JsonNode fetchHistoricalEod(String ticker, String fromIso, String toIso) throws Exception {
        String path = "/historical-price-eod/full?symbol=" + enc(ticker)
                + "&from=" + enc(fromIso)
                + "&to=" + enc(toIso);
        return fetchArray(path);
    }

    private JsonNode fetchRatios(String ticker) throws Exception {
        return fetchArrayFirst("/ratios?symbol=" + enc(ticker));
    }

    private JsonNode fetchKeyMetrics(String ticker) throws Exception {
        return fetchArrayFirst("/key-metrics?symbol=" + enc(ticker));
    }

    private JsonNode fetchAnnualIncome(String ticker) throws Exception {
        return fetchArray("/income-statement?symbol=" + enc(ticker) + "&period=annual");
    }

    private JsonNode fetchAnnualBalance(String ticker) throws Exception {
        return fetchArray("/balance-sheet-statement?symbol=" + enc(ticker) + "&period=annual");
    }

    private JsonNode fetchAnnualCashflow(String ticker) throws Exception {
        return fetchArray("/cash-flow-statement?symbol=" + enc(ticker) + "&period=annual");
    }

    /**
     * Analyst rating distribution per month — FMP renamed this in 2026 from
     * {@code /analyst-stock-recommendations} to {@code /grades-historical}.
     * Field names also changed: now {@code analystRatingsStrongBuy/Buy/Hold/Sell/StrongSell}.
     */
    private JsonNode fetchAnalystRecs(String ticker) throws Exception {
        return fetchArray("/grades-historical?symbol=" + enc(ticker));
    }

    /** Consensus price targets (low/mean/median/high) — augments {@code financialData}. */
    private JsonNode fetchPriceTargetConsensus(String ticker) throws Exception {
        return fetchArrayFirst("/price-target-consensus?symbol=" + enc(ticker));
    }

    /**
     * Earnings + insider — both endpoints were paywalled or 404'd on FMP free tier as of
     * May 2026. Return null so the caller falls through to Yahoo (which scrapes
     * /quote/{ticker}/analysis and /quote/{ticker}/insider-transactions for free).
     */
    private JsonNode fetchEarningsSurprises(String ticker) {
        return null;
    }

    private JsonNode fetchInsiderTrades(String ticker) {
        return null;
    }

    /** Run the request, rotate the key, return the parsed array (may be empty/null). */
    private JsonNode fetchArray(String pathQuery) throws Exception {
        String key = nextActiveKey();
        if (key == null) {
            throw new RuntimeException("FMP all keys exhausted for the day");
        }
        String url = BASE_URL + pathQuery + (pathQuery.contains("?") ? "&" : "?") + "apikey=" + key;
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        recordRequest(key);
        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(15))
                .header("Accept", "application/json")
                .header("User-Agent", "Nizo/1.0 (+local agent)")
                .GET().build();
        HttpResponse<String> resp = HTTP.send(req, BoundedHttp.ofString());
        int sc = resp.statusCode();
        if (sc == 429) {
            // Force-rotate this key by marking it exhausted, then retry once on the next.
            LOG.warn("FMP key (idx {}) returned 429 — marking exhausted and retrying", currentKeyIndex);
            keyRequestCounts.put(key, REQUESTS_PER_KEY_PER_DAY);
            currentKeyIndex = (currentKeyIndex + 1) % apiKeys.size();
            return fetchArray(pathQuery);  // recursive — bounded by exhaustion
        }
        if (sc / 100 != 2) {
            throw new RuntimeException("FMP HTTP " + sc + " for " + redactKey(url));
        }
        return MAPPER.readTree(resp.body());
    }

    private JsonNode fetchArrayFirst(String pathQuery) throws Exception {
        JsonNode arr = fetchArray(pathQuery);
        return (arr != null && arr.isArray() && arr.size() > 0) ? arr.get(0) : null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FMP → Yahoo module shape converters
    // ─────────────────────────────────────────────────────────────────────────

    /** Yahoo: {@code incomeStatementHistory: {incomeStatementHistory: [{...}, ...]}} */
    private JsonNode incomeStatementHistory(JsonNode arr) {
        if (arr == null || !arr.isArray()) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode out = wrap.putArray("incomeStatementHistory");
        for (JsonNode r : arr) {
            ObjectNode row = out.addObject();
            row.put("endDate", epochOf(r.path("date").asText("")));
            row.put("totalRevenue",          r.path("revenue").asLong(0));
            row.put("costOfRevenue",         r.path("costOfRevenue").asLong(0));
            row.put("grossProfit",           r.path("grossProfit").asLong(0));
            row.put("totalOperatingExpenses", r.path("operatingExpenses").asLong(0));
            row.put("operatingIncome",       r.path("operatingIncome").asLong(0));
            row.put("ebit",                  r.path("operatingIncome").asLong(0));
            row.put("interestExpense",       r.path("interestExpense").asLong(0));
            row.put("incomeTaxExpense",      r.path("incomeTaxExpense").asLong(0));
            row.put("netIncome",             r.path("netIncome").asLong(0));
            row.put("dilutedEPS",            r.path("epsdiluted").asDouble(0));
        }
        return wrap;
    }

    /** Yahoo: {@code balanceSheetHistory: {balanceSheetStatements: [{...}, ...]}} */
    private JsonNode balanceSheetHistory(JsonNode arr) {
        if (arr == null || !arr.isArray()) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode out = wrap.putArray("balanceSheetStatements");
        for (JsonNode r : arr) {
            ObjectNode row = out.addObject();
            row.put("endDate", epochOf(r.path("date").asText("")));
            row.put("totalAssets",            r.path("totalAssets").asLong(0));
            row.put("totalCurrentAssets",     r.path("totalCurrentAssets").asLong(0));
            row.put("cash",                   r.path("cashAndCashEquivalents").asLong(0));
            row.put("shortTermInvestments",   r.path("shortTermInvestments").asLong(0));
            row.put("totalLiab",              r.path("totalLiabilities").asLong(0));
            row.put("totalCurrentLiabilities", r.path("totalCurrentLiabilities").asLong(0));
            row.put("longTermDebt",           r.path("longTermDebt").asLong(0));
            row.put("totalStockholderEquity", r.path("totalStockholdersEquity").asLong(0));
            row.put("retainedEarnings",       r.path("retainedEarnings").asLong(0));
        }
        return wrap;
    }

    /** Yahoo: {@code cashflowStatementHistory: {cashflowStatements: [{...}, ...]}} */
    private JsonNode cashflowStatementHistory(JsonNode arr) {
        if (arr == null || !arr.isArray()) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode out = wrap.putArray("cashflowStatements");
        for (JsonNode r : arr) {
            ObjectNode row = out.addObject();
            row.put("endDate", epochOf(r.path("date").asText("")));
            row.put("totalCashFromOperatingActivities", r.path("operatingCashFlow").asLong(0));
            row.put("capitalExpenditures",              r.path("capitalExpenditure").asLong(0));   // FMP signs negative
            row.put("dividendsPaid",                    r.path("dividendsPaid").asLong(0));
            row.put("repurchaseOfStock",                r.path("commonStockRepurchased").asLong(0));
            row.put("issuanceOfStock",                  r.path("commonStockIssued").asLong(0));
            row.put("netBorrowings",                    r.path("debtIssuance").asLong(0));
            row.put("totalCashflowsFromInvestingActivities", r.path("netCashUsedForInvestingActivites").asLong(0));
            row.put("totalCashFromFinancingActivities", r.path("netCashUsedProvidedByFinancingActivities").asLong(0));
        }
        return wrap;
    }

    /** Yahoo: {@code price} module — top-level price/marketCap/currency/exchange. */
    private JsonNode priceModule(JsonNode profile) {
        if (profile == null) return null;
        ObjectNode out = MAPPER.createObjectNode();
        out.put("regularMarketPrice", profile.path("price").asDouble(0));
        out.put("marketCap",          profile.path("mktCap").asLong(0));
        out.put("currency",           profile.path("currency").asText(""));
        out.put("exchangeName",       profile.path("exchange").asText(""));
        out.put("longName",           profile.path("companyName").asText(""));
        out.put("symbol",             profile.path("symbol").asText(""));
        return out;
    }

    /** Yahoo: {@code summaryDetail} module — ratios + 52w range + dividend yield etc. */
    private JsonNode summaryDetailModule(JsonNode profile, JsonNode ratios) {
        ObjectNode out = MAPPER.createObjectNode();
        if (profile != null) {
            out.put("beta", profile.path("beta").asDouble(0));
            // Parse "169.21-288.62" 52w range
            String range = profile.path("range").asText("");
            if (range.contains("-")) {
                String[] p = range.split("-");
                if (p.length == 2) {
                    try {
                        out.put("fiftyTwoWeekLow",  Double.parseDouble(p[0].trim()));
                        out.put("fiftyTwoWeekHigh", Double.parseDouble(p[1].trim()));
                    } catch (NumberFormatException ignored) {}
                }
            }
            out.put("volume",         profile.path("volume").asLong(0));
            out.put("averageVolume",  profile.path("averageVolume").asLong(0));
        }
        if (ratios != null) {
            out.put("trailingPE",       ratios.path("priceToEarningsRatio").asDouble(0));
            out.put("priceToSalesTrailing12Months", ratios.path("priceToSalesRatio").asDouble(0));
            out.put("dividendYield",    ratios.path("dividendYield").asDouble(0));
            out.put("payoutRatio",      ratios.path("payoutRatio").asDouble(0));
        }
        return out;
    }

    /** Yahoo: {@code assetProfile} module — sector/industry/business summary. */
    private JsonNode assetProfileModule(JsonNode profile) {
        if (profile == null) return null;
        ObjectNode out = MAPPER.createObjectNode();
        out.put("sector",              profile.path("sector").asText(""));
        out.put("industry",            profile.path("industry").asText(""));
        out.put("country",             profile.path("country").asText(""));
        out.put("fullTimeEmployees",   profile.path("fullTimeEmployees").asLong(0));
        out.put("website",             profile.path("website").asText(""));
        out.put("longBusinessSummary", profile.path("description").asText(""));
        return out;
    }

    /** Yahoo: {@code defaultKeyStatistics} — valuation multiples, beta, shares outstanding etc. */
    private JsonNode defaultKeyStatisticsModule(JsonNode keyMetrics, JsonNode ratios) {
        ObjectNode out = MAPPER.createObjectNode();
        if (keyMetrics != null) {
            out.put("enterpriseValue",     keyMetrics.path("enterpriseValue").asLong(0));
            out.put("enterpriseToEbitda",  keyMetrics.path("enterpriseValueOverEBITDA").asDouble(0));
            out.put("enterpriseToRevenue", keyMetrics.path("enterpriseValueMultiple").asDouble(0));
            out.put("sharesOutstanding",   (long) keyMetrics.path("marketCap").asDouble(0));   // approx; profile has direct
            out.put("forwardPE",           keyMetrics.path("forwardPE").asDouble(0));
        }
        if (ratios != null) {
            out.put("priceToBook",         ratios.path("priceToBookRatio").asDouble(0));
            out.put("pegRatio",            ratios.path("priceToEarningsGrowthRatio").asDouble(0));
        }
        return out.size() > 0 ? out : null;
    }

    /** Yahoo: {@code financialData} — current ratio, quick ratio, ROE, margins, target prices. */
    private JsonNode financialDataModule(JsonNode profile, JsonNode keyMetrics, JsonNode ratios, JsonNode targetConsensus) {
        ObjectNode out = MAPPER.createObjectNode();
        if (ratios != null) {
            out.put("currentRatio",     ratios.path("currentRatio").asDouble(0));
            out.put("quickRatio",       ratios.path("quickRatio").asDouble(0));
            out.put("debtToEquity",     ratios.path("debtToEquityRatio").asDouble(0));
            out.put("returnOnEquity",   ratios.path("returnOnEquity").asDouble(0));
            out.put("returnOnAssets",   ratios.path("returnOnAssets").asDouble(0));
            out.put("grossMargins",     ratios.path("grossProfitMargin").asDouble(0));
            out.put("operatingMargins", ratios.path("operatingProfitMargin").asDouble(0));
            out.put("profitMargins",    ratios.path("netProfitMargin").asDouble(0));
            out.put("ebitdaMargins",    ratios.path("ebitdaMargin").asDouble(0));
        }
        if (keyMetrics != null) {
            out.put("freeCashflow",      keyMetrics.path("freeCashFlow").asLong(0));
            out.put("operatingCashflow", keyMetrics.path("operatingCashFlow").asLong(0));
            out.put("totalCash",         keyMetrics.path("cashAndCashEquivalents").asLong(0));
            out.put("totalDebt",         keyMetrics.path("totalDebt").asLong(0));
        }
        if (profile != null) {
            out.put("currentPrice", profile.path("price").asDouble(0));
        }
        // FMP /price-target-consensus → Yahoo's targetLow/Mean/Median/High fields. Only set
        // if the consensus row is real; otherwise our caller (AnalystRatingsTool) will treat
        // these as missing and skip the price-target band rendering.
        if (targetConsensus != null && !targetConsensus.isMissingNode() && !targetConsensus.isNull()) {
            double tl = targetConsensus.path("targetLow").asDouble(0);
            double tc = targetConsensus.path("targetConsensus").asDouble(0);
            double tm = targetConsensus.path("targetMedian").asDouble(0);
            double th = targetConsensus.path("targetHigh").asDouble(0);
            if (tl > 0) out.put("targetLowPrice",    tl);
            if (tc > 0) out.put("targetMeanPrice",   tc);
            if (tm > 0) out.put("targetMedianPrice", tm);
            if (th > 0) out.put("targetHighPrice",   th);
        }
        return out.size() > 0 ? out : null;
    }

    /** Yahoo: {@code recommendationTrend.trend[]} — aggregated rating distribution per period. */
    private JsonNode recommendationTrendModule(JsonNode arr) {
        if (arr == null || !arr.isArray() || arr.size() == 0) return null;
        // FMP /grades-historical rows look like:
        //   {"date":"2026-05-01","analystRatingsStrongBuy":7,"analystRatingsBuy":25,
        //    "analystRatingsHold":15,"analystRatingsSell":1,"analystRatingsStrongSell":1}
        // Yahoo's shape:
        //   trend: [{period, strongBuy, buy, hold, sell, strongSell}]
        // Strip the "analystRatings" prefix and relabel period to "0m"/"-1m"/...
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode trend = wrap.putArray("trend");
        // FMP returns newest first; map first 4 to "0m", "-1m", "-2m", "-3m".
        String[] periods = {"0m", "-1m", "-2m", "-3m"};
        for (int i = 0; i < Math.min(4, arr.size()); i++) {
            JsonNode r = arr.get(i);
            ObjectNode p = trend.addObject();
            p.put("period", periods[i]);
            p.put("strongBuy",  r.path("analystRatingsStrongBuy").asInt(0));
            p.put("buy",        r.path("analystRatingsBuy").asInt(0));
            p.put("hold",       r.path("analystRatingsHold").asInt(0));
            p.put("sell",       r.path("analystRatingsSell").asInt(0));
            p.put("strongSell", r.path("analystRatingsStrongSell").asInt(0));
        }
        return wrap;
    }

    /**
     * Yahoo: {@code upgradeDowngradeHistory.history[]} — firm-by-firm rating actions.
     * FMP's analyst-stock-recommendations is per-month aggregate, NOT per-firm. So we synthesize
     * "actions" from the diff between consecutive months — when buy count rises, infer
     * upgrades. Limited fidelity but better than nothing.
     */
    private JsonNode upgradeDowngradeHistoryModule(JsonNode arr) {
        if (arr == null || !arr.isArray() || arr.size() < 2) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode hist = wrap.putArray("history");
        // FMP /grades-historical is per-month aggregate, not per-firm. Synthesize "actions" by
        // diffing consecutive months — if (strongBuy+buy) went up, infer an upgrade. Limited
        // fidelity but better than empty (Yahoo's per-firm endpoint is more granular but blocked).
        for (int i = 0; i < Math.min(8, arr.size() - 1); i++) {
            JsonNode cur = arr.get(i);
            JsonNode prev = arr.get(i + 1);
            int curBuy = cur.path("analystRatingsStrongBuy").asInt(0)
                       + cur.path("analystRatingsBuy").asInt(0);
            int prevBuy = prev.path("analystRatingsStrongBuy").asInt(0)
                        + prev.path("analystRatingsBuy").asInt(0);
            int curSell = cur.path("analystRatingsStrongSell").asInt(0)
                        + cur.path("analystRatingsSell").asInt(0);
            int prevSell = prev.path("analystRatingsStrongSell").asInt(0)
                         + prev.path("analystRatingsSell").asInt(0);
            String action = curBuy > prevBuy ? "up" : curSell > prevSell ? "down" : "main";
            ObjectNode ev = hist.addObject();
            ev.put("epochGradeDate", epochOf(cur.path("date").asText("")));
            ev.put("firm",      "FMP aggregate");
            ev.put("toGrade",   curBuy > curSell ? "Buy" : curSell > curBuy ? "Sell" : "Hold");
            ev.put("fromGrade", prevBuy > prevSell ? "Buy" : prevSell > prevBuy ? "Sell" : "Hold");
            ev.put("action",    action);
        }
        return wrap;
    }

    /** Yahoo: {@code earningsHistory.history[]} — per-quarter EPS estimate vs actual. */
    private JsonNode earningsHistoryModule(JsonNode arr) {
        if (arr == null || !arr.isArray() || arr.size() == 0) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode hist = wrap.putArray("history");
        int count = 0;
        for (JsonNode r : arr) {
            if (count >= 8) break;
            double est = r.path("estimatedEarning").asDouble(Double.NaN);
            double act = r.path("actualEarningResult").asDouble(Double.NaN);
            if (Double.isNaN(est) && Double.isNaN(act)) continue;
            ObjectNode row = hist.addObject();
            row.put("quarter", epochOf(r.path("date").asText("")));
            row.put("epsEstimate", est);
            row.put("epsActual",   act);
            double surprisePct = 0.0;
            if (!Double.isNaN(act) && !Double.isNaN(est) && est != 0) {
                surprisePct = (act - est) / Math.abs(est) * 100;
            }
            row.put("surprisePercent", surprisePct);
            row.put("period", "-" + count + "q");
            count++;
        }
        return wrap;
    }

    /** Yahoo: {@code insiderTransactions.transactions[]} — Form-4 filings. */
    private JsonNode insiderTransactionsModule(JsonNode arr) {
        if (arr == null || !arr.isArray() || arr.size() == 0) return null;
        ObjectNode wrap = MAPPER.createObjectNode();
        ArrayNode tx = wrap.putArray("transactions");
        for (JsonNode r : arr) {
            if (tx.size() >= 50) break;
            ObjectNode row = tx.addObject();
            row.put("startDate",  epochOf(r.path("transactionDate").asText(r.path("filingDate").asText(""))));
            row.put("filerName",  r.path("reportingName").asText(""));
            row.put("filerRelation", r.path("typeOfOwner").asText(""));
            String txType = r.path("transactionType").asText(r.path("acquistionOrDisposition").asText(""));
            row.put("transactionText", txType);
            // Compute value from price * shares if we have both
            long shares = r.path("securitiesTransacted").asLong(r.path("shares").asLong(0));
            double price = r.path("price").asDouble(0);
            row.put("shares", shares);
            row.put("value",  (long) (shares * price));
            row.put("ownership", r.path("acquistionOrDisposition").asText(""));
            row.put("moneyText", price > 0 ? "$" + price : "");
        }
        return wrap;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Key rotation
    // ─────────────────────────────────────────────────────────────────────────

    private synchronized String nextActiveKey() {
        if (apiKeys.isEmpty()) return null;
        // Reset counts at UTC day boundary
        LocalDate today = LocalDate.now(ZoneOffset.UTC);
        if (!today.equals(lastResetDate)) {
            LOG.info("FMP: resetting daily request counts ({} keys)", apiKeys.size());
            keyRequestCounts.replaceAll((k, v) -> 0);
            lastResetDate = today;
            currentKeyIndex = 0;
        }
        int start = currentKeyIndex;
        do {
            String k = apiKeys.get(currentKeyIndex);
            int n = keyRequestCounts.getOrDefault(k, 0);
            if (n < REQUESTS_PER_KEY_PER_DAY) return k;
            currentKeyIndex = (currentKeyIndex + 1) % apiKeys.size();
            if (currentKeyIndex == start) return null;     // all exhausted
        } while (true);
    }

    private synchronized void recordRequest(String key) {
        if (key == null) return;
        int n = keyRequestCounts.getOrDefault(key, 0) + 1;
        keyRequestCounts.put(key, n);
        if (n >= REQUESTS_PER_KEY_PER_DAY) {
            int idx = apiKeys.indexOf(key) + 1;
            LOG.info("FMP: key #{} hit daily limit ({}), rotating", idx, REQUESTS_PER_KEY_PER_DAY);
            currentKeyIndex = (currentKeyIndex + 1) % apiKeys.size();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static String enc(String s) {
        return URLEncoder.encode(s == null ? "" : s.toUpperCase(), StandardCharsets.UTF_8);
    }

    /** Convert a "yyyy-MM-dd" string to epoch seconds. Returns 0 on failure. */
    private static long epochOf(String dateStr) {
        if (dateStr == null || dateStr.isBlank()) return 0L;
        try {
            // FMP usually emits "2024-09-28" but sometimes "2024-09-28 00:00:00"
            String d = dateStr.length() >= 10 ? dateStr.substring(0, 10) : dateStr;
            return LocalDate.parse(d).atStartOfDay(ZoneOffset.UTC).toEpochSecond();
        } catch (Exception e) {
            return 0L;
        }
    }

    /** Strip the apikey query param before logging URLs. */
    private static String redactKey(String url) {
        return url.replaceAll("apikey=[^&]+", "apikey=<REDACTED>");
    }
}
