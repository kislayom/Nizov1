package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.finance.buffett.BuffettAnalysisEngine;
import ai.nizo.tools.finance.buffett.BuffettMetrics;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Buffett-Munger investment scorecard for a public stock.
 *
 * <p>Purely deterministic compute on top of the same FMP / Yahoo data that
 * {@link StockFundamentalsTool} already pulls — no LLM in the calculation, no extra HTTP
 * round-trips. Output goes verbatim into a {@code chart-buffett} fenced block on the
 * front-end which renders score bars, moat gauge, margin-of-safety pill, the Munger
 * checklist, and a Buy/Strong-Buy/Watch/Pass verdict.
 *
 * <p>What the LLM should call this tool for:
 * <ul>
 *   <li>"Is X a Buffett-style buy?"</li>
 *   <li>"Compute the moat score for X"</li>
 *   <li>"Run a Buffett-Munger analysis on X"</li>
 *   <li>As section E of the {@code stock_fundamentals_analyst} sub-skill report</li>
 * </ul>
 *
 * <p>This tool reuses the {@link YahooQuoteSummary} client (which itself routes through
 * FMP first, falling back to Yahoo direct + HTML scraper). All inputs come from the same
 * v10-shaped JSON the existing tools already understand.
 */
public final class StockBuffettScoreTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(StockBuffettScoreTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String[] MODULES = {
            "price",
            "summaryDetail",
            "financialData",
            "defaultKeyStatistics",
            "incomeStatementHistory",
            "balanceSheetHistory",
            "cashflowStatementHistory",
            "assetProfile"
    };

    private final YahooQuoteSummary yahoo;
    private final ScreenerInClient screener;

    public StockBuffettScoreTool(YahooQuoteSummary yahoo) {
        this(yahoo, new ScreenerInClient());
    }

    public StockBuffettScoreTool(YahooQuoteSummary yahoo, ScreenerInClient screener) {
        this.yahoo = yahoo;
        this.screener = screener;
    }

    @Override public String name() { return "stock_buffett_score"; }

    @Override
    public String description() {
        return "Buffett-Munger investment scorecard for a public stock — deterministic, no-LLM compute. "
                + "Returns: 0-100 Buffett score (5 component breakdowns), moat score 0-10 with type+trend, "
                + "weighted intrinsic value (DCF + Growth-PE + 10-Cap + Graham), margin of safety, capital "
                + "allocation grade, owner earnings, FCF yield, Munger checklist (5 boolean gates), red/green "
                + "flags, verdict (Strong Buy/Buy/Watch/Pass) + buy + strong-buy prices + position sizing. "
                + "Output goes verbatim into a `chart-buffett` fenced block on the front-end.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "ticker": { "type": "string", "description": "Ticker like AAPL, MSFT, HDFCBANK.NS" }
              },
              "required": ["ticker"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            String ticker = args.path("ticker").asText("").trim().toUpperCase();
            if (ticker.isEmpty()) return ToolResult.error("ticker is required");

            JsonNode result = yahoo.fetch(ticker, MODULES);

            // For Indian stocks Yahoo often returns sparse IS/BS/CF arrays (only the price /
            // summaryDetail modules survive), so the Buffett engine ends up with no statements
            // to score on. Fall through to Screener.in (which returns Yahoo-shaped JSON with
            // 11 years of IS/BS/CF) when we detect that emptiness. Same fallback ladder as
            // StockFundamentalsTool.
            if (screener != null && NseIndiaClient.isIndianTicker(ticker)
                    && (result == null || isSparse(result))) {
                JsonNode scr = screener.fetch(ticker);
                if (scr != null && !isSparse(scr)) {
                    // Merge Yahoo's price + summaryDetail (which usually survive) with Screener's
                    // statement arrays so the engine has both market data AND historicals.
                    result = mergeYahooScreener(result, scr);
                    LOG.info("stock_buffett_score: using Screener.in fallback for {} (Yahoo sparse)", ticker);
                }
            }

            // Build the inputs the engine expects. Each is a flat JsonNode of fields.
            // The engine reads fields with Yahoo's {raw, fmt} wrapper OR bare numbers, so we can
            // pass tool output through largely as-is.
            JsonNode price = result.path("price");
            JsonNode summary = result.path("summaryDetail");
            JsonNode fd = result.path("financialData");
            JsonNode dks = result.path("defaultKeyStatistics");
            JsonNode profile = result.path("assetProfile");

            // Merge price+summary+fd+dks into one "metrics" view since the engine reads them all
            // by field name. ObjectNode.setAll lets us union them.
            ObjectNode metrics = MAPPER.createObjectNode();
            mergeInto(metrics, price);
            mergeInto(metrics, summary);
            mergeInto(metrics, fd);
            mergeInto(metrics, dks);

            JsonNode incomeArr = result.path("incomeStatementHistory").path("incomeStatementHistory");
            JsonNode balanceArr = result.path("balanceSheetHistory").path("balanceSheetStatements");
            JsonNode cashFlowArr = result.path("cashflowStatementHistory").path("cashflowStatements");

            // External DCF — we don't have one yet from FMP free tier, so pass NaN.
            // (TODO: add FMP /discounted-cash-flow if available.)
            double dcfPerShare = Double.NaN;

            // Build a "quote"-shaped node that the engine can read currentPrice + marketCap from.
            ObjectNode quoteIn = MAPPER.createObjectNode();
            quoteIn.put("currentPrice",
                    extractDouble(price, "regularMarketPrice", extractDouble(metrics, "currentPrice", 0)));
            quoteIn.put("marketCap", extractDouble(price, "marketCap", extractDouble(metrics, "marketCap", 0)));

            BuffettMetrics m = BuffettAnalysisEngine.analyze(
                    quoteIn, metrics, incomeArr, cashFlowArr, balanceArr, profile, dcfPerShare);

            String currency = price.path("currency").asText("USD");
            ObjectNode out = m.toJson(ticker, currency);
            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("stock_buffett_score failed: {}", e.toString());
            return ToolResult.error("stock_buffett_score failed: " + e.getMessage());
        }
    }

    /**
     * Copy every top-level field from {@code src} into {@code target}, preferring
     * existing-target values on conflict (so earlier-merged sources win for ambiguous keys
     * like "currency"). The engine reads by simple field path so a flat union is enough.
     */
    private static void mergeInto(ObjectNode target, JsonNode src) {
        if (src == null || src.isMissingNode() || src.isNull() || !src.isObject()) return;
        src.fields().forEachRemaining(e -> {
            if (!target.has(e.getKey())) target.set(e.getKey(), e.getValue());
        });
    }

    private static double extractDouble(JsonNode parent, String field, double fallback) {
        if (parent == null || parent.isMissingNode() || parent.isNull()) return fallback;
        JsonNode n = parent.path(field);
        if (n.isMissingNode() || n.isNull()) return fallback;
        if (n.isObject()) {
            JsonNode raw = n.path("raw");
            return (raw.isMissingNode() || raw.isNull()) ? fallback : raw.asDouble(fallback);
        }
        return n.asDouble(fallback);
    }

    /**
     * "Sparse" = no usable IS/BS/CF statement arrays. This matches what we observe for
     * Indian stocks on Yahoo's free tier — they return {@code price + summaryDetail + some
     * defaultKeyStatistics} but the three statement arrays are empty.
     */
    private static boolean isSparse(JsonNode result) {
        if (result == null) return true;
        JsonNode inc = result.path("incomeStatementHistory").path("incomeStatementHistory");
        return !inc.isArray() || inc.isEmpty();
    }

    /**
     * Merge Yahoo's surviving modules (price + summaryDetail are usually fine even for
     * Indian stocks) with Screener.in's rich statement arrays. Returns a Yahoo-shaped
     * JsonNode the engine can read as-is.
     */
    private static JsonNode mergeYahooScreener(JsonNode yahooResult, JsonNode screenerResult) {
        if (yahooResult == null || yahooResult.isMissingNode()) return screenerResult;
        if (screenerResult == null || screenerResult.isMissingNode()) return yahooResult;
        ObjectNode out = MAPPER.createObjectNode();
        // Start with Screener (has full statement arrays + Indian-flavored ratios)
        screenerResult.fields().forEachRemaining(e -> out.set(e.getKey(), e.getValue()));
        // Overlay Yahoo's price/summaryDetail (live market price + 52w range)
        for (String mod : new String[]{"price", "summaryDetail", "defaultKeyStatistics", "assetProfile"}) {
            JsonNode m = yahooResult.path(mod);
            if (m != null && !m.isMissingNode() && !m.isNull() && m.isObject() && m.size() > 0) {
                out.set(mod, m);
            }
        }
        return out;
    }
}
