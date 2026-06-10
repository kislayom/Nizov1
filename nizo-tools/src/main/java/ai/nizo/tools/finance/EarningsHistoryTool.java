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

/**
 * Earnings beat/miss history + forward estimates + next reporting date.
 *
 * <p>Yahoo modules combined:
 * <ul>
 *   <li>{@code earningsHistory} — quarterly EPS estimate vs actual, last ~4-8 quarters</li>
 *   <li>{@code earnings} — earnings chart (quarterly EPS + revenue history)</li>
 *   <li>{@code earningsTrend} — forward estimates for the next 4 periods (current Q, next Q,
 *       current FY, next FY) with growth percentages</li>
 *   <li>{@code calendarEvents} — next earnings reporting window + ex-dividend date</li>
 * </ul>
 *
 * <p>Computes a simple beat-rate summary (e.g. "8 of 8 beats over the last 8 quarters,
 * average surprise +6.2%") that's useful as a one-line quality signal.
 *
 * <p>Output goes verbatim into a {@code chart-earnings} fenced block — front-end renders a
 * paired bar chart (estimate vs actual per quarter), beat-rate pill, and a "next earnings"
 * countdown card.
 */
public final class EarningsHistoryTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(EarningsHistoryTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String[] MODULES = {
            "earningsHistory",
            "earnings",
            "earningsTrend",
            "calendarEvents"
    };

    private final YahooQuoteSummary yahoo;
    private final FinnhubClient finnhub;

    public EarningsHistoryTool(YahooQuoteSummary yahoo) {
        this(yahoo, new FinnhubClient());
    }

    public EarningsHistoryTool(YahooQuoteSummary yahoo, FinnhubClient finnhub) {
        this.yahoo = yahoo;
        this.finnhub = finnhub;
    }

    @Override public String name() { return "stock_earnings_history"; }

    @Override
    public String description() {
        return "Earnings beat/miss history + forward estimates + next reporting date for a "
                + "public stock. Returns last ~8 quarters of EPS estimate vs actual, beat rate, "
                + "average surprise, current and next-fiscal-year estimates, and the next earnings "
                + "report date. Output goes verbatim into a `chart-earnings` fenced block on the "
                + "front-end (estimate-vs-actual bars + beat-rate pill + next-earnings countdown).";
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

            // ========== History (estimate vs actual per quarter) ==========
            JsonNode histArr = result.path("earningsHistory").path("history");
            // Tier-3 fallback: if Yahoo + FMP returned nothing, try Finnhub. Finnhub's
            // /stock/earnings returns an array of {period, actual, estimate, surprise,
            // surprisePercent, symbol} — re-shape into Yahoo's history layout so the rest
            // of the code path below is unchanged.
            if ((!histArr.isArray() || histArr.size() == 0) && finnhub.isEnabled()) {
                JsonNode fhEarnings = finnhub.earnings(ticker);
                if (fhEarnings != null && fhEarnings.isArray() && fhEarnings.size() > 0) {
                    ArrayNode synth = MAPPER.createArrayNode();
                    for (JsonNode r : fhEarnings) {
                        ObjectNode row = synth.addObject();
                        String period = r.path("period").asText("");   // "YYYY-MM-DD"
                        long epoch = 0;
                        try {
                            if (!period.isBlank()) epoch = LocalDate.parse(period).atStartOfDay(ZoneOffset.UTC).toEpochSecond();
                        } catch (Exception ignored) {}
                        ObjectNode q = row.putObject("quarter");
                        q.put("raw", epoch);
                        row.put("period", "");
                        ObjectNode est = row.putObject("epsEstimate");
                        est.put("raw", r.path("estimate").asDouble(Double.NaN));
                        ObjectNode act = row.putObject("epsActual");
                        act.put("raw", r.path("actual").asDouble(Double.NaN));
                        ObjectNode surp = row.putObject("surprisePercent");
                        surp.put("raw", r.path("surprisePercent").asDouble(Double.NaN));
                    }
                    histArr = synth;
                    out.put("source", "yahoo_quoteSummary_v10+finnhub_earnings");
                }
            }
            ArrayNode history = out.putArray("history");
            int beats = 0, misses = 0;
            double cumulativeSurprise = 0.0;
            int countedSurprise = 0;
            int currentStreak = 0;
            String currentStreakKind = "";

            if (histArr.isArray()) {
                // Yahoo orders newest-first; flip for chart left→right reading.
                for (int i = histArr.size() - 1; i >= 0; i--) {
                    JsonNode r = histArr.get(i);
                    ObjectNode row = history.addObject();
                    long epoch = YahooQuoteSummary.rawLong(r.path("quarter"));
                    String date = epoch > 0
                            ? LocalDate.ofInstant(Instant.ofEpochSecond(epoch), ZoneOffset.UTC).toString()
                            : "";
                    row.put("date", date);
                    row.put("period", r.path("period").asText(""));        // "-1q", "-2q", etc.
                    double estimate = YahooQuoteSummary.rawDouble(r.path("epsEstimate"));
                    double actual   = YahooQuoteSummary.rawDouble(r.path("epsActual"));
                    double surprisePct = YahooQuoteSummary.rawDouble(r.path("surprisePercent"));
                    row.put("epsEstimate", estimate);
                    row.put("epsActual",   actual);
                    row.put("surprisePercent", surprisePct);
                    boolean beat = !Double.isNaN(actual) && !Double.isNaN(estimate)
                            && actual > estimate;
                    boolean missed = !Double.isNaN(actual) && !Double.isNaN(estimate)
                            && actual < estimate;
                    row.put("beat", beat);
                    if (beat)   beats++;
                    if (missed) misses++;
                    if (!Double.isNaN(surprisePct) && surprisePct != 0.0) {
                        cumulativeSurprise += surprisePct;
                        countedSurprise++;
                    }
                }

                // Streak counts from the most recent quarter backward (history is now oldest-first).
                for (int i = history.size() - 1; i >= 0; i--) {
                    JsonNode r = history.get(i);
                    boolean beat = r.path("beat").asBoolean(false);
                    boolean missed = !beat
                            && !Double.isNaN(r.path("epsActual").asDouble(Double.NaN))
                            && r.path("epsActual").asDouble() < r.path("epsEstimate").asDouble();
                    String kind = beat ? "beat" : missed ? "miss" : "";
                    if (kind.isEmpty()) break;
                    if (currentStreakKind.isEmpty()) {
                        currentStreakKind = kind;
                        currentStreak = 1;
                    } else if (currentStreakKind.equals(kind)) {
                        currentStreak++;
                    } else {
                        break;
                    }
                }
            }

            // ========== Beat-rate summary ==========
            ObjectNode summary = out.putObject("summary");
            int totalCounted = beats + misses;
            summary.put("beatCount", beats);
            summary.put("missCount", misses);
            summary.put("totalCounted", totalCounted);
            summary.put("beatRate", totalCounted > 0 ? round4((double) beats / totalCounted) : 0.0);
            summary.put("averageSurprisePercent", countedSurprise > 0
                    ? round4(cumulativeSurprise / countedSurprise) : 0.0);
            summary.put("currentStreak", currentStreak);
            summary.put("currentStreakKind", currentStreakKind);

            // ========== Forward estimates ==========
            JsonNode trends = result.path("earningsTrend").path("trend");
            ArrayNode forwardArr = out.putArray("forward");
            if (trends.isArray()) {
                for (JsonNode t : trends) {
                    ObjectNode fw = forwardArr.addObject();
                    fw.put("period",      t.path("period").asText(""));   // "0q","+1q","0y","+1y"
                    fw.put("endDate",     t.path("endDate").asText(""));
                    fw.put("epsEstimateAvg", YahooQuoteSummary.rawDouble(t.path("earningsEstimate").path("avg")));
                    fw.put("epsEstimateLow", YahooQuoteSummary.rawDouble(t.path("earningsEstimate").path("low")));
                    fw.put("epsEstimateHigh", YahooQuoteSummary.rawDouble(t.path("earningsEstimate").path("high")));
                    fw.put("epsAnalystCount", YahooQuoteSummary.rawLong(t.path("earningsEstimate").path("numberOfAnalysts")));
                    fw.put("epsGrowth",   YahooQuoteSummary.rawDouble(t.path("earningsEstimate").path("growth")));
                    fw.put("revenueEstimateAvg", YahooQuoteSummary.rawLong(t.path("revenueEstimate").path("avg")));
                    fw.put("revenueEstimateLow", YahooQuoteSummary.rawLong(t.path("revenueEstimate").path("low")));
                    fw.put("revenueEstimateHigh", YahooQuoteSummary.rawLong(t.path("revenueEstimate").path("high")));
                    fw.put("revenueGrowth", YahooQuoteSummary.rawDouble(t.path("revenueEstimate").path("growth")));
                }
            }

            // ========== Next earnings date ==========
            JsonNode cal = result.path("calendarEvents");
            ObjectNode next = out.putObject("next");
            JsonNode earningsCal = cal.path("earnings");
            // Yahoo gives a min/max date range when the company hasn't pinned the day.
            JsonNode earningsDate = earningsCal.path("earningsDate");
            String nextDate = "";
            if (earningsDate.isArray() && earningsDate.size() > 0) {
                long epoch = YahooQuoteSummary.rawLong(earningsDate.get(0));
                if (epoch > 0) {
                    nextDate = LocalDate.ofInstant(Instant.ofEpochSecond(epoch), ZoneOffset.UTC).toString();
                }
            }
            next.put("date", nextDate);
            if (earningsDate.isArray() && earningsDate.size() > 1) {
                long epoch2 = YahooQuoteSummary.rawLong(earningsDate.get(1));
                if (epoch2 > 0) {
                    next.put("dateMax", LocalDate.ofInstant(Instant.ofEpochSecond(epoch2), ZoneOffset.UTC).toString());
                }
            }
            next.put("epsEstimateAvg",  YahooQuoteSummary.rawDouble(earningsCal.path("earningsAverage")));
            next.put("epsEstimateLow",  YahooQuoteSummary.rawDouble(earningsCal.path("earningsLow")));
            next.put("epsEstimateHigh", YahooQuoteSummary.rawDouble(earningsCal.path("earningsHigh")));
            next.put("revenueEstimateAvg",  YahooQuoteSummary.rawLong(earningsCal.path("revenueAverage")));
            next.put("revenueEstimateLow",  YahooQuoteSummary.rawLong(earningsCal.path("revenueLow")));
            next.put("revenueEstimateHigh", YahooQuoteSummary.rawLong(earningsCal.path("revenueHigh")));

            // Days until reporting (negative if past)
            if (!nextDate.isEmpty()) {
                LocalDate today = LocalDate.now(ZoneOffset.UTC);
                next.put("daysUntil", java.time.temporal.ChronoUnit.DAYS.between(today, LocalDate.parse(nextDate)));
            }

            // Ex-dividend (handy bonus from same module)
            JsonNode exDiv = cal.path("exDividendDate");
            long exDivEpoch = YahooQuoteSummary.rawLong(exDiv);
            if (exDivEpoch > 0) {
                out.put("exDividendDate", LocalDate.ofInstant(Instant.ofEpochSecond(exDivEpoch), ZoneOffset.UTC).toString());
            }

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("stock_earnings_history failed: {}", e.toString());
            return ToolResult.error("stock_earnings_history failed: " + e.getMessage());
        }
    }

    private static double round4(double v) { return Double.isNaN(v) ? 0.0 : Math.round(v * 10000.0) / 10000.0; }
}
