package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * India picks backtest — Phase 5 of the "India Picks" build.
 *
 * <p>Replays the picks algorithm against historical OHLCV data and reports
 * standard performance metrics: cumulative return vs. benchmark, Sharpe ratio,
 * max drawdown, hit rate, win-to-loss ratio.
 *
 * <p><b>Methodology (simplified — survivorship-aware where possible):</b>
 * <ol>
 *   <li>Universe: takes a list of tickers (from the caller, typically the current
 *       NIFTY 500). Survivorship bias is acknowledged in the disclaimer — a
 *       proper point-in-time NIFTY 500 reconstruction needs a paid data feed.</li>
 *   <li>For each rebalance date (default monthly over the lookback window):
 *       <ul>
 *         <li>Pull each ticker's price as of that date from {@link HistoricalPriceTool}.</li>
 *         <li>Score using PRICE-BASED factors only (1y momentum, 1m momentum,
 *             52w-range position) — these are the signals available historically
 *             without paid point-in-time fundamentals. Future enhancement: plug in
 *             a fundamentals provider that supports point-in-time queries.</li>
 *         <li>Pick top-N. Hold for the rebalance interval.</li>
 *       </ul></li>
 *   <li>Track each pick's forward return until the next rebalance.</li>
 *   <li>Compute metrics: cumulative return, annualized return, annualized vol,
 *       Sharpe (risk-free 7% repo rate), max drawdown, hit rate.</li>
 *   <li>Compare against the {@code benchmark} ticker (default {@code ^NSEI} for
 *       NIFTY 50, or {@code ^NSEBANK} etc. when scoped to sectors).</li>
 * </ol>
 *
 * <p><b>Honest caveats:</b>
 * <ul>
 *   <li><b>Survivorship bias</b>: NIFTY 500 today excludes delisted / acquired names.
 *       Backtest IS slightly biased upward.</li>
 *   <li><b>Look-ahead-free</b>: scores computed using ONLY data dated &le; rebalance date.</li>
 *   <li><b>Transaction costs</b>: not modeled (Indian retail is ~0.10% all-in via discount
 *       brokers; institutional even lower). Subtract ~25 bps per rebalance for a fair compare.</li>
 *   <li><b>Slippage</b>: not modeled. Less material for liquid NIFTY 500 names.</li>
 *   <li><b>Tax</b>: not modeled. STCG / LTCG treatment varies by holding period.</li>
 * </ul>
 *
 * <p>Output is a {@code chart-india-backtest} fenced JSON the front-end renderer
 * displays as: header card with key metrics + equity curve overlay vs benchmark
 * + table of monthly picks + monthly returns.
 */
public final class IndiaPicksBacktestTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaPicksBacktestTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HistoricalPriceTool historical;

    public IndiaPicksBacktestTool(HistoricalPriceTool historical) {
        this.historical = historical;
    }

    @Override public String name() { return "india_picks_backtest"; }

    @Override
    public String description() {
        return "Backtest the India Picks ranking algorithm vs a benchmark over a historical "
                + "window. Picks top-N each rebalance using price-based factors (momentum + 52w "
                + "position), holds until the next rebalance. Returns Sharpe, max drawdown, "
                + "annualized return, hit rate, equity curve. Acknowledged limitations: "
                + "survivorship bias (uses CURRENT universe), no fundamental factors historically "
                + "(needs paid point-in-time feed), no txn costs / slippage / tax modeling.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "tickers":       { "type": "array", "items": { "type": "string" }, "description": "Yahoo-style tickers to use as the universe. Typically the current NIFTY 500." },
                "benchmark":     { "type": "string", "description": "Benchmark ticker for comparison, default '^NSEI'." },
                "topN":          { "type": "integer", "description": "Top-N picks per rebalance (default 10)." },
                "rebalanceDays": { "type": "integer", "description": "Days between rebalances (default 30 = monthly)." },
                "lookbackDays":  { "type": "integer", "description": "Total backtest horizon in days (default 1825 = 5y, max 3650)." },
                "riskFreePct":   { "type": "number",  "description": "Annual risk-free rate for Sharpe (default 7.0% — Indian 10y G-Sec proxy)." }
              },
              "required": ["tickers"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        long t0 = System.currentTimeMillis();
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            JsonNode tickersA = args.path("tickers");
            if (!tickersA.isArray() || tickersA.isEmpty()) return ToolResult.error("tickers (array) is required");
            String benchmark    = args.path("benchmark").asText("^NSEI").trim();
            int topN            = Math.max(1, Math.min(50,  args.path("topN").asInt(10)));
            int rebalanceDays   = Math.max(7, Math.min(180, args.path("rebalanceDays").asInt(30)));
            int lookbackDays    = Math.max(180, Math.min(3650, args.path("lookbackDays").asInt(1825)));
            double riskFreePct  = args.path("riskFreePct").asDouble(7.0);

            List<String> tickers = new ArrayList<>();
            tickersA.forEach(t -> { String s = t.asText("").trim().toUpperCase(); if (!s.isEmpty()) tickers.add(s); });
            LOG.info("india_picks_backtest: starting — {} tickers, top {} / rebalance {}d / lookback {}d",
                    tickers.size(), topN, rebalanceDays, lookbackDays);

            // 1. Fetch historical bars for each ticker + the benchmark. Use HistoricalPriceTool
            //    with range matching the lookback. Bars come newest-first per Yahoo's convention.
            String range = lookbackDays >= 1825 ? "5y" : (lookbackDays >= 365 ? "2y" : "1y");
            Map<String, double[][]> bars = new HashMap<>();   // ticker -> [[epochSec, close], ...]
            for (String t : tickers) {
                double[][] arr = fetchClosesSorted(t, range);
                if (arr != null && arr.length > 30) bars.put(t, arr);
            }
            double[][] benchBars = fetchClosesSorted(benchmark, range);
            if (benchBars == null || benchBars.length < 30) {
                return ToolResult.error("benchmark " + benchmark + " has no usable bars");
            }
            LOG.info("india_picks_backtest: fetched bars for {}/{} tickers + benchmark {} ({} bars)",
                    bars.size(), tickers.size(), benchmark, benchBars.length);

            // 2. Generate rebalance dates — every N days within the lookback window.
            long now = System.currentTimeMillis() / 1000;
            long horizon = now - (long) lookbackDays * 86400L;
            List<Long> rebalanceTs = new ArrayList<>();
            for (long t = horizon; t < now; t += (long) rebalanceDays * 86400L) rebalanceTs.add(t);
            LOG.info("india_picks_backtest: {} rebalance dates", rebalanceTs.size());

            // 3. Walk each rebalance: score, pick top N, compute next-period return.
            List<RebalancePeriod> history = new ArrayList<>();
            double equity = 1.0;     // strategy
            double benchEq = 1.0;
            List<Double> equityCurve = new ArrayList<>();
            List<Double> benchCurve  = new ArrayList<>();
            equityCurve.add(equity);
            benchCurve.add(benchEq);
            for (int i = 0; i < rebalanceTs.size() - 1; i++) {
                long tNow  = rebalanceTs.get(i);
                long tNext = rebalanceTs.get(i + 1);
                // Score each ticker AS OF tNow using only data <= tNow
                List<Scored> scored = new ArrayList<>();
                for (Map.Entry<String, double[][]> e : bars.entrySet()) {
                    Scored s = scoreAt(e.getKey(), e.getValue(), tNow);
                    if (s != null) scored.add(s);
                }
                if (scored.size() < topN) continue;
                scored.sort(Comparator.comparingDouble((Scored s) -> s.composite).reversed());
                List<Scored> picks = scored.subList(0, topN);
                // Forward return: equal-weight basket from tNow to tNext.
                double basketRet = 0;
                int valid = 0;
                List<String> tickersPicked = new ArrayList<>();
                for (Scored s : picks) {
                    double r = forwardReturn(bars.get(s.ticker), tNow, tNext);
                    if (!Double.isNaN(r)) { basketRet += r; valid++; tickersPicked.add(s.ticker); }
                }
                if (valid == 0) continue;
                basketRet /= valid;
                double benchRet = forwardReturn(benchBars, tNow, tNext);
                if (Double.isNaN(benchRet)) benchRet = 0;
                equity  *= (1 + basketRet);
                benchEq *= (1 + benchRet);
                equityCurve.add(equity);
                benchCurve.add(benchEq);
                history.add(new RebalancePeriod(tNow, tNext, basketRet, benchRet, tickersPicked));
            }

            // 4. Compute metrics
            int periods = history.size();
            double years = (lookbackDays / 365.25);
            double cumStrat = equity - 1;
            double cumBench = benchEq - 1;
            double annStrat = years > 0 ? Math.pow(1 + cumStrat, 1.0 / years) - 1 : 0;
            double annBench = years > 0 ? Math.pow(1 + cumBench, 1.0 / years) - 1 : 0;
            double[] periodicStrat = history.stream().mapToDouble(p -> p.stratRet).toArray();
            double[] periodicBench = history.stream().mapToDouble(p -> p.benchRet).toArray();
            double volStrat = stdDev(periodicStrat) * Math.sqrt(365.25 / rebalanceDays);
            double volBench = stdDev(periodicBench) * Math.sqrt(365.25 / rebalanceDays);
            double rfPeriodic = (riskFreePct / 100.0) * (rebalanceDays / 365.25);
            double sharpeStrat = volStrat > 0 ? (annStrat - riskFreePct / 100.0) / volStrat : 0;
            double sharpeBench = volBench > 0 ? (annBench - riskFreePct / 100.0) / volBench : 0;
            double maxDDStrat = maxDrawdown(equityCurve);
            double maxDDBench = maxDrawdown(benchCurve);
            int wins = (int) java.util.stream.IntStream.range(0, periodicStrat.length)
                    .filter(i -> periodicStrat[i] > periodicBench[i]).count();
            double hitRate = periods > 0 ? (double) wins / periods : 0;
            double winLossRatio;
            {
                double avgWin = java.util.stream.DoubleStream.of(periodicStrat).filter(x -> x > 0).average().orElse(0);
                double avgLoss = Math.abs(java.util.stream.DoubleStream.of(periodicStrat).filter(x -> x < 0).average().orElse(0));
                winLossRatio = avgLoss > 0 ? avgWin / avgLoss : 0;
            }

            // 5. Assemble JSON output
            ObjectNode out = MAPPER.createObjectNode();
            out.put("asOf", java.time.Instant.now().toString());
            out.put("benchmark", benchmark);
            out.put("topN", topN);
            out.put("rebalanceDays", rebalanceDays);
            out.put("lookbackDays", lookbackDays);
            out.put("universeSize", tickers.size());
            out.put("universeWithBars", bars.size());
            out.put("periods", periods);
            out.put("years", round2(years));

            ObjectNode metrics = out.putObject("metrics");
            ObjectNode strat = metrics.putObject("strategy");
            strat.put("cumReturnPct",   round2(cumStrat * 100));
            strat.put("annReturnPct",   round2(annStrat * 100));
            strat.put("annVolPct",      round2(volStrat * 100));
            strat.put("sharpe",         round2(sharpeStrat));
            strat.put("maxDrawdownPct", round2(maxDDStrat * 100));
            strat.put("hitRatePct",     round2(hitRate * 100));
            strat.put("winLossRatio",   round2(winLossRatio));
            ObjectNode bench = metrics.putObject("benchmark");
            bench.put("cumReturnPct",   round2(cumBench * 100));
            bench.put("annReturnPct",   round2(annBench * 100));
            bench.put("annVolPct",      round2(volBench * 100));
            bench.put("sharpe",         round2(sharpeBench));
            bench.put("maxDrawdownPct", round2(maxDDBench * 100));
            ObjectNode alpha = metrics.putObject("alpha");
            alpha.put("cumExcessPct",  round2((cumStrat - cumBench) * 100));
            alpha.put("annExcessPct",  round2((annStrat - annBench) * 100));
            alpha.put("sharpeExcess",  round2(sharpeStrat - sharpeBench));

            // Equity curves (lightweight — keep small)
            ArrayNode eq = out.putArray("equityCurve");
            for (int i = 0; i < equityCurve.size(); i++) {
                ObjectNode pt = MAPPER.createObjectNode();
                long ts = (i < rebalanceTs.size()) ? rebalanceTs.get(i) : rebalanceTs.get(rebalanceTs.size() - 1);
                pt.put("date", LocalDate.ofEpochDay(ts / 86400L).toString());
                pt.put("strategy", round4(equityCurve.get(i)));
                pt.put("benchmark", round4(benchCurve.get(i)));
                eq.add(pt);
            }
            // Most-recent picks
            ArrayNode hist = out.putArray("recentPicks");
            int show = Math.min(history.size(), 6);
            for (int i = history.size() - show; i < history.size(); i++) {
                RebalancePeriod p = history.get(i);
                ObjectNode row = MAPPER.createObjectNode();
                row.put("rebalanceDate", LocalDate.ofEpochDay(p.start / 86400L).toString());
                row.put("stratRetPct",  round2(p.stratRet * 100));
                row.put("benchRetPct",  round2(p.benchRet * 100));
                ArrayNode tk = row.putArray("tickers");
                p.tickers.forEach(tk::add);
                hist.add(row);
            }
            out.put("elapsedMs", System.currentTimeMillis() - t0);
            out.put("disclaimer",
                    "Backtest uses CURRENT universe (survivorship bias). Price-based factors only "
                  + "(no point-in-time fundamentals). No transaction costs / slippage / tax modeled. "
                  + "Past performance does not predict future returns. Research output, not advice.");

            String md = renderMarkdown(out);
            String fence = "```chart-india-backtest\n" + MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out) + "\n```";
            return ToolResult.ok(md + "\n\n" + fence + "\n");
        } catch (Exception e) {
            LOG.warn("india_picks_backtest failed: {}", e.toString());
            return ToolResult.error("india_picks_backtest failed: " + e.getMessage());
        }
    }

    /* ─────────────────────────────────────────────────────────── scoring */

    /**
     * Compute a price-only composite score for {@code ticker} using bars dated &lt;= {@code asOfTs}.
     * Uses 1-year momentum + 1-month momentum + 52-week range position. Returns null if
     * insufficient data at the given point in time.
     */
    private static Scored scoreAt(String ticker, double[][] sorted, long asOfTs) {
        // sorted is ascending by ts. Find the last idx with ts <= asOfTs.
        int idx = lastBarIdxAtOrBefore(sorted, asOfTs);
        if (idx < 60) return null;  // need at least ~3 months of history
        double now    = sorted[idx][1];
        double idx1m  = Math.max(0, idx - 22);
        double idx1y  = Math.max(0, idx - 252);
        double price1m = sorted[(int) idx1m][1];
        double price1y = sorted[(int) idx1y][1];
        double ret1m = (now - price1m) / price1m;
        double ret1y = (now - price1y) / price1y;
        // 52w range position (0-1, 1 = at 52w high)
        double hi52 = Double.NEGATIVE_INFINITY, lo52 = Double.POSITIVE_INFINITY;
        int startScan = Math.max(0, idx - 252);
        for (int i = startScan; i <= idx; i++) {
            if (sorted[i][1] > hi52) hi52 = sorted[i][1];
            if (sorted[i][1] < lo52) lo52 = sorted[i][1];
        }
        double rangePos = hi52 > lo52 ? (now - lo52) / (hi52 - lo52) : 0.5;
        // Composite: 50% 1y momentum, 25% 1m momentum, 25% range position
        double mom1y  = clamp((ret1y + 0.25) / 0.75, 0, 1) * 100;
        double mom1m  = clamp((ret1m + 0.10) / 0.30, 0, 1) * 100;
        double rngPos = rangePos * 100;
        double composite = 0.5 * mom1y + 0.25 * mom1m + 0.25 * rngPos;
        Scored s = new Scored();
        s.ticker = ticker;
        s.composite = composite;
        s.ret1y = ret1y;
        s.ret1m = ret1m;
        s.rangePos = rangePos;
        return s;
    }

    private static double forwardReturn(double[][] sorted, long fromTs, long toTs) {
        if (sorted == null) return Double.NaN;
        int iFrom = lastBarIdxAtOrBefore(sorted, fromTs);
        int iTo   = lastBarIdxAtOrBefore(sorted, toTs);
        if (iFrom < 0 || iTo <= iFrom) return Double.NaN;
        double pFrom = sorted[iFrom][1];
        double pTo   = sorted[iTo][1];
        if (pFrom <= 0 || pTo <= 0) return Double.NaN;
        return (pTo - pFrom) / pFrom;
    }

    /** Binary search for last idx whose ts <= asOfTs. sorted is ascending by ts. */
    private static int lastBarIdxAtOrBefore(double[][] sorted, long asOfTs) {
        if (sorted == null || sorted.length == 0) return -1;
        int lo = 0, hi = sorted.length - 1, ans = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if ((long) sorted[mid][0] <= asOfTs) { ans = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return ans;
    }

    /* ─────────────────────────────────────────────────────── data fetch */

    /**
     * Pull bars via {@link HistoricalPriceTool} and return as an ascending-by-time
     * {@code [[epochSec, close], ...]} 2D array. Returns null on failure.
     */
    private double[][] fetchClosesSorted(String ticker, String range) {
        try {
            String args = MAPPER.writeValueAsString(Map.of("ticker", ticker, "range", range, "interval", "1d"));
            ToolResult r = historical.execute(args);
            if (r == null || !r.ok() || r.content() == null) return null;
            JsonNode root = MAPPER.readTree(r.content());
            JsonNode barsNode = root.path("bars");
            if (!barsNode.isArray() || barsNode.isEmpty()) return null;
            List<double[]> rows = new ArrayList<>();
            for (JsonNode bar : barsNode) {
                String dateStr = bar.path("date").asText("");
                double close = bar.path("close").asDouble(Double.NaN);
                if (dateStr.isEmpty() || Double.isNaN(close) || close <= 0) continue;
                try {
                    LocalDate d = LocalDate.parse(dateStr);
                    long ts = d.toEpochDay() * 86400L;
                    rows.add(new double[]{ts, close});
                } catch (Exception ignored) { /* skip bad dates */ }
            }
            rows.sort(Comparator.comparingDouble(a -> a[0]));
            return rows.toArray(new double[0][]);
        } catch (Exception e) {
            LOG.debug("fetchClosesSorted failed for {}: {}", ticker, e.toString());
            return null;
        }
    }

    /* ─────────────────────────────────────────────────────────── helpers */

    private static double stdDev(double[] xs) {
        if (xs.length < 2) return 0;
        double sum = 0; for (double x : xs) sum += x; double mean = sum / xs.length;
        double v = 0; for (double x : xs) v += (x - mean) * (x - mean);
        return Math.sqrt(v / (xs.length - 1));
    }

    private static double maxDrawdown(List<Double> curve) {
        double peak = Double.NEGATIVE_INFINITY, mdd = 0;
        for (double v : curve) {
            if (v > peak) peak = v;
            double dd = peak > 0 ? (peak - v) / peak : 0;
            if (dd > mdd) mdd = dd;
        }
        return mdd;
    }

    private static double clamp(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
    private static double round2(double v) { return Math.round(v * 100.0) / 100.0; }
    private static double round4(double v) { return Math.round(v * 10000.0) / 10000.0; }

    private static String renderMarkdown(ObjectNode payload) {
        JsonNode m = payload.path("metrics");
        JsonNode s = m.path("strategy");
        JsonNode b = m.path("benchmark");
        JsonNode a = m.path("alpha");
        StringBuilder out = new StringBuilder();
        out.append("# India Picks Backtest\n\n");
        out.append("**Benchmark:** ").append(payload.path("benchmark").asText())
           .append(" · **Top:** ").append(payload.path("topN").asInt())
           .append(" · **Rebalance:** ").append(payload.path("rebalanceDays").asInt()).append("d")
           .append(" · **Lookback:** ").append(payload.path("years").asDouble()).append("y")
           .append(" · **Periods:** ").append(payload.path("periods").asInt()).append("\n\n");
        out.append("| Metric | Strategy | Benchmark | Excess |\n|---|---|---|---|\n");
        out.append("| Cum return | ").append(s.path("cumReturnPct").asDouble()).append("% | ")
           .append(b.path("cumReturnPct").asDouble()).append("% | ")
           .append(a.path("cumExcessPct").asDouble()).append("% |\n");
        out.append("| Ann return | ").append(s.path("annReturnPct").asDouble()).append("% | ")
           .append(b.path("annReturnPct").asDouble()).append("% | ")
           .append(a.path("annExcessPct").asDouble()).append("% |\n");
        out.append("| Ann vol | ").append(s.path("annVolPct").asDouble()).append("% | ")
           .append(b.path("annVolPct").asDouble()).append("% | — |\n");
        out.append("| Sharpe (7% rf) | ").append(s.path("sharpe").asDouble()).append(" | ")
           .append(b.path("sharpe").asDouble()).append(" | ")
           .append(a.path("sharpeExcess").asDouble()).append(" |\n");
        out.append("| Max drawdown | -").append(s.path("maxDrawdownPct").asDouble()).append("% | -")
           .append(b.path("maxDrawdownPct").asDouble()).append("% | — |\n");
        out.append("| Hit rate vs bench | ").append(s.path("hitRatePct").asDouble()).append("% | — | — |\n");
        out.append("\n").append("> ").append(payload.path("disclaimer").asText()).append("\n");
        return out.toString();
    }

    /* ─────────────────────────────────────────────────────────── types */

    private static final class Scored {
        String ticker;
        double composite;
        double ret1y;
        double ret1m;
        double rangePos;
    }

    private record RebalancePeriod(long start, long end, double stratRet, double benchRet, List<String> tickers) {}
}
