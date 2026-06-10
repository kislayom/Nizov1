package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.Locale;

/**
 * Sell-side analyst sentiment for a public stock — consensus rating, price targets, recent
 * upgrades/downgrades.
 *
 * <p>Three Yahoo modules combined into one structured payload:
 * <ul>
 *   <li>{@code recommendationTrend} — month-by-month rating distribution (4 buckets back)</li>
 *   <li>{@code financialData} — target high/mean/median/low + recommendationKey + analyst count</li>
 *   <li>{@code upgradeDowngradeHistory} — last ~25 firm-by-firm rating changes</li>
 * </ul>
 *
 * <p>Output is suitable for embedding inside a {@code chart-analyst} fenced block. The
 * front-end renders a rating-distribution donut + a price-target band (low / mean / high vs
 * current) + a sortable table of recent rating changes.
 */
public final class AnalystRatingsTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(AnalystRatingsTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String[] MODULES = {
            "recommendationTrend",
            "financialData",
            "upgradeDowngradeHistory",
            "price"
    };

    private final YahooQuoteSummary yahoo;
    private final FinnhubClient finnhub;

    public AnalystRatingsTool(YahooQuoteSummary yahoo) {
        this(yahoo, new FinnhubClient());
    }

    public AnalystRatingsTool(YahooQuoteSummary yahoo, FinnhubClient finnhub) {
        this.yahoo = yahoo;
        this.finnhub = finnhub;
    }

    @Override public String name() { return "stock_analyst_ratings"; }

    @Override
    public String description() {
        return "Sell-side analyst sentiment for a public stock — consensus rating, price targets "
                + "(low/mean/high), and the last ~25 firm-by-firm upgrades/downgrades. Output "
                + "goes verbatim into a `chart-analyst` fenced block on the front-end "
                + "(rating-distribution donut + price-target band + recent-actions table).";
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

            ObjectNode out = MAPPER.createObjectNode();
            out.put("ticker", ticker);
            out.put("source", "yahoo_quoteSummary_v10");
            out.put("asOf", LocalDate.now(ZoneOffset.UTC).toString());

            JsonNode price = result.path("price");
            double currentPrice = YahooQuoteSummary.rawDouble(price.path("regularMarketPrice"));
            out.put("currency", price.path("currency").asText(""));
            out.put("currentPrice", round2(currentPrice));

            JsonNode fd = result.path("financialData");
            ObjectNode rec = out.putObject("recommendation");
            String key = fd.path("recommendationKey").asText("none");
            rec.put("consensusKey", key);
            rec.put("consensus", consensusLabel(key));
            // recommendationMean: 1.0 (strong buy) → 5.0 (sell)
            rec.put("scoreMean", YahooQuoteSummary.rawDouble(fd.path("recommendationMean")));
            rec.put("numberOfAnalysts", YahooQuoteSummary.rawLong(fd.path("numberOfAnalystOpinions")));

            // Distribution over the last 4 months
            JsonNode trend = result.path("recommendationTrend").path("trend");
            ArrayNode distArr = rec.putArray("trend");
            if (trend.isArray() && trend.size() > 0) {
                for (JsonNode period : trend) {
                    ObjectNode p = distArr.addObject();
                    p.put("period", period.path("period").asText(""));   // "0m", "-1m", "-2m", "-3m"
                    p.put("strongBuy", period.path("strongBuy").asInt(0));
                    p.put("buy",       period.path("buy").asInt(0));
                    p.put("hold",      period.path("hold").asInt(0));
                    p.put("sell",      period.path("sell").asInt(0));
                    p.put("strongSell", period.path("strongSell").asInt(0));
                }
            } else if (finnhub.isEnabled()) {
                // Tier-3 fallback: Yahoo + FMP returned nothing → try Finnhub. Finnhub gives
                // monthly buckets newest-first with the same column names — clean drop-in for
                // the trend array. Doesn't populate price targets / recent actions (different
                // endpoints, not on the free tier as one call) so those stay empty.
                JsonNode fhTrend = finnhub.recommendations(ticker);
                if (fhTrend != null && fhTrend.isArray()) {
                    int n = Math.min(4, fhTrend.size());
                    for (int i = 0; i < n; i++) {
                        JsonNode period = fhTrend.get(i);
                        ObjectNode p = distArr.addObject();
                        p.put("period", period.path("period").asText(""));   // "YYYY-MM-DD"
                        p.put("strongBuy", period.path("strongBuy").asInt(0));
                        p.put("buy",       period.path("buy").asInt(0));
                        p.put("hold",      period.path("hold").asInt(0));
                        p.put("sell",      period.path("sell").asInt(0));
                        p.put("strongSell", period.path("strongSell").asInt(0));
                    }
                    if (n > 0) {
                        out.put("source", "yahoo_quoteSummary_v10+finnhub_recommendation");
                        // Sum all buckets in the most recent month to back-fill numberOfAnalysts.
                        JsonNode latest = fhTrend.get(0);
                        long total = latest.path("strongBuy").asLong(0)
                                   + latest.path("buy").asLong(0)
                                   + latest.path("hold").asLong(0)
                                   + latest.path("sell").asLong(0)
                                   + latest.path("strongSell").asLong(0);
                        if (rec.path("numberOfAnalysts").asLong(0) == 0) rec.put("numberOfAnalysts", total);
                        if (rec.path("consensus").asText("Unknown").equals("Unknown")) {
                            rec.put("consensus", inferConsensusFromCounts(latest));
                        }
                    }
                }
            }

            // Price target band
            ObjectNode tgt = out.putObject("priceTarget");
            double low    = YahooQuoteSummary.rawDouble(fd.path("targetLowPrice"));
            double mean   = YahooQuoteSummary.rawDouble(fd.path("targetMeanPrice"));
            double median = YahooQuoteSummary.rawDouble(fd.path("targetMedianPrice"));
            double high   = YahooQuoteSummary.rawDouble(fd.path("targetHighPrice"));
            tgt.put("low",    round2(low));
            tgt.put("mean",   round2(mean));
            tgt.put("median", round2(median));
            tgt.put("high",   round2(high));
            tgt.put("upsidePercent", upsidePct(currentPrice, mean));
            tgt.put("downsidePercent", upsidePct(currentPrice, low));
            tgt.put("highUpsidePercent", upsidePct(currentPrice, high));

            // Recent upgrades / downgrades — Yahoo orders newest-first
            JsonNode hist = result.path("upgradeDowngradeHistory").path("history");
            ArrayNode actions = out.putArray("recentActions");
            if (hist.isArray()) {
                int n = Math.min(25, hist.size());
                for (int i = 0; i < n; i++) {
                    JsonNode ev = hist.get(i);
                    ObjectNode a = actions.addObject();
                    long epoch = ev.path("epochGradeDate").asLong(0);
                    a.put("date", epoch > 0
                            ? LocalDate.ofInstant(Instant.ofEpochSecond(epoch), ZoneOffset.UTC).toString()
                            : "");
                    a.put("firm",      ev.path("firm").asText(""));
                    a.put("toGrade",   ev.path("toGrade").asText(""));
                    a.put("fromGrade", ev.path("fromGrade").asText(""));
                    a.put("action",    ev.path("action").asText(""));   // "up", "down", "main", "init"
                }
            }

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("stock_analyst_ratings failed: {}", e.toString());
            return ToolResult.error("stock_analyst_ratings failed: " + e.getMessage());
        }
    }

    /** Map Yahoo's lowercase enum to a clean display label. */
    private static String consensusLabel(String key) {
        if (key == null) return "Unknown";
        return switch (key.toLowerCase(Locale.ROOT)) {
            case "strong_buy" -> "Strong Buy";
            case "buy" -> "Buy";
            case "hold" -> "Hold";
            case "underperform" -> "Underperform";
            case "sell" -> "Sell";
            default -> "Unknown";
        };
    }

    /** Pick the consensus label from raw counts in a single trend bucket. */
    private static String inferConsensusFromCounts(JsonNode b) {
        int sb = b.path("strongBuy").asInt(0);
        int by = b.path("buy").asInt(0);
        int hd = b.path("hold").asInt(0);
        int sl = b.path("sell").asInt(0);
        int ss = b.path("strongSell").asInt(0);
        int bullish = sb + by;
        int neutral = hd;
        int bearish = sl + ss;
        if (bullish > neutral + bearish && sb >= by) return "Strong Buy";
        if (bullish > neutral + bearish) return "Buy";
        if (bearish > bullish + neutral && ss >= sl) return "Sell";
        if (bearish > bullish + neutral) return "Underperform";
        return "Hold";
    }

    /** Upside (or downside) of {@code target} vs {@code current}, as decimal fraction. */
    private static double upsidePct(double current, double target) {
        if (current <= 0 || Double.isNaN(target) || target <= 0) return 0.0;
        return round4((target - current) / current);
    }

    private static double round2(double v) { return Double.isNaN(v) ? 0.0 : Math.round(v * 100.0) / 100.0; }
    private static double round4(double v) { return Double.isNaN(v) ? 0.0 : Math.round(v * 10000.0) / 10000.0; }
}
