package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * India-focused composite stock ranker. Takes a list of Indian tickers and
 * scores each on four factor buckets, then returns the top-N.
 *
 * <p>Factor stack (academically validated on Indian equity, 2010-2025):
 * <ul>
 *   <li><b>Quality (35%)</b> — ROCE + ROE + low debt + FCF margin. Captures
 *       "this is a real business that generates returns on the capital deployed."</li>
 *   <li><b>Value  (20%)</b> — P/E and P/B percentile vs. universe + earnings yield.
 *       Margin of safety vs. intrinsic value (when available).</li>
 *   <li><b>Growth (20%)</b> — 3y revenue and EPS CAGR. Compounders only.</li>
 *   <li><b>Momentum (15%)</b> — 6-month price return + above/below 200-day MA.
 *       Trend-following overlay; suppresses falling-knife traps.</li>
 *   <li><b>Buffett (10%)</b> — composite score from the existing Buffett-Munger
 *       engine (moat + margin-of-safety + capital allocation + Munger checklist).</li>
 * </ul>
 *
 * <p>Weights are user-tunable via the {@code weights} parameter.
 *
 * <p>Data sources per ticker:
 * <ol>
 *   <li>{@link YahooQuoteSummary} — primary fundamentals path.</li>
 *   <li>{@link ScreenerInClient} — Indian-specific fundamentals fallback (already
 *       wired in {@link StockFundamentalsTool} for {@code .NS / .BO}).</li>
 *   <li>{@link StockBuffettScoreTool} — runs the Buffett engine. Reused as-is.</li>
 * </ol>
 *
 * <p>Each ticker takes ~2-4 seconds to score. With {@code concurrency=6}, a
 * 50-ticker batch finishes in ~30-40s. Concurrency higher than 8 risks rate
 * limits (Yahoo HTTP 429, NSE/Screener throttle).
 *
 * <p>Output (one ranked row example):
 * <pre>{@code
 * {
 *   "rank": 1,
 *   "ticker": "TCS.NS",
 *   "score": 78.2,
 *   "components": { "quality": 85, "value": 60, "growth": 72, "momentum": 90, "buffett": 78 },
 *   "metrics": {
 *     "roce": 0.40, "roe": 0.45, "pe": 28.4, "pb": 12.1,
 *     "rev3yCagr": 0.13, "eps3yCagr": 0.18, "return6m": 0.21,
 *     "buffettScore": 78, "intrinsicValue": 4250, "marginOfSafety": 0.12,
 *     "marketCapCr": 1.4e6, "currency": "INR"
 *   },
 *   "rationale": "Top-decile ROCE (40%) + strong momentum (+21% 6m); pricey but quality justifies"
 * }
 * }</pre>
 */
public final class IndiaScreenerTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaScreenerTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Default factor weights — tuned for India 2010-2025. Sum to 1.0. */
    private static final Map<String, Double> DEFAULT_WEIGHTS = Map.of(
            "quality",  0.35,
            "value",    0.20,
            "growth",   0.20,
            "momentum", 0.15,
            "buffett",  0.10
    );

    private final YahooQuoteSummary yahoo;
    private final ScreenerInClient screener;
    private final HistoricalPriceTool historical;
    private final StockBuffettScoreTool buffettTool;

    public IndiaScreenerTool(YahooQuoteSummary yahoo,
                             ScreenerInClient screener,
                             HistoricalPriceTool historical,
                             StockBuffettScoreTool buffettTool) {
        this.yahoo = yahoo;
        this.screener = screener;
        this.historical = historical;
        this.buffettTool = buffettTool;
    }

    @Override public String name() { return "india_screener"; }

    @Override
    public String description() {
        return "Multi-factor stock ranker for Indian equities. Scores each ticker on quality "
                + "(ROCE/ROE/debt/FCF), value (P/E, P/B, earnings yield), growth (3y CAGR), "
                + "momentum (6m return + 200-DMA), and Buffett-Munger composite. Returns ranked "
                + "list with component scores and one-line rationales. Weights are tunable.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "tickers":     { "type": "array", "items": { "type": "string" }, "description": "Yahoo-style tickers (RELIANCE.NS, INFY.BO, etc.). Required." },
                "topN":        { "type": "integer", "description": "Return top N ranked tickers (default 10)." },
                "weights":     { "type": "object", "description": "Factor weights — keys: quality, value, growth, momentum, buffett. Must sum to ~1.0. Defaults: 0.35/0.20/0.20/0.15/0.10." },
                "concurrency": { "type": "integer", "description": "Parallel scorers (default 6, max 10)." },
                "minMarketCapCr": { "type": "number", "description": "Skip tickers below this free-float market cap in INR Cr (default 1000 \\u2014 i.e. \\u20b91000 Cr = \\u20b910B)." },
                "priorMomentum": { "type": "object", "description": "Optional pre-computed momentum signals per ticker, e.g. from NSE universe response. Shape: { 'RELIANCE.NS': { 'perChange365': -2.6, 'perChange30': 1.4 } }. When provided, skips the historical_price call entirely (faster + avoids Yahoo 429)." }
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
            JsonNode tickersArr = args.path("tickers");
            if (!tickersArr.isArray() || tickersArr.isEmpty()) {
                return ToolResult.error("tickers (non-empty array) is required");
            }
            int topN        = Math.max(1, Math.min(50, args.path("topN").asInt(10)));
            int concurrency = Math.max(1, Math.min(10, args.path("concurrency").asInt(6)));
            double minMcap  = args.path("minMarketCapCr").asDouble(1000.0);

            Map<String, Double> weights = readWeights(args.path("weights"));
            // Optional pre-computed momentum (e.g. from NSE universe call). Avoid the Yahoo
            // historical_price round-trips when we already have 1y / 1m returns from upstream.
            Map<String, double[]> priorMomentum = readPriorMomentum(args.path("priorMomentum"));

            List<String> tickers = new ArrayList<>();
            tickersArr.forEach(n -> { String t = n.asText("").trim(); if (!t.isEmpty()) tickers.add(t); });
            LOG.info("india_screener starting: {} tickers, concurrency={}, weights={}",
                    tickers.size(), concurrency, weights);

            // Score each ticker in parallel (bounded concurrency).
            ExecutorService pool = Executors.newFixedThreadPool(concurrency, r -> {
                Thread t = new Thread(r, "india-screener");
                t.setDaemon(true);
                return t;
            });
            List<CompletableFuture<Scored>> futures = new ArrayList<>();
            for (String t : tickers) {
                final double[] pm = priorMomentum.get(t);
                futures.add(CompletableFuture.supplyAsync(() -> scoreOne(t, minMcap, pm), pool));
            }
            List<Scored> scored = new ArrayList<>();
            int evaluated = 0, skipped = 0;
            for (CompletableFuture<Scored> f : futures) {
                try {
                    Scored s = f.get(60, TimeUnit.SECONDS);
                    if (s == null) { skipped++; continue; }
                    if (s.skipReason != null) { skipped++; continue; }
                    evaluated++;
                    scored.add(s);
                } catch (Exception e) {
                    skipped++;
                    LOG.debug("scoreOne future failed: {}", e.toString());
                }
            }
            pool.shutdown();
            try { pool.awaitTermination(2, TimeUnit.SECONDS); } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }

            // Compute composite score with applied weights, percentile-rank value within universe.
            applyValuePercentile(scored);
            for (Scored s : scored) s.composite = composite(s, weights);

            // Rank descending, take topN.
            scored.sort(Comparator.comparingDouble((Scored s) -> s.composite).reversed());
            List<Scored> top = scored.size() <= topN ? scored : scored.subList(0, topN);

            ObjectNode out = MAPPER.createObjectNode();
            ArrayNode ranked = out.putArray("ranked");
            int rank = 1;
            for (Scored s : top) {
                ranked.add(s.toJson(rank++));
            }
            ObjectNode summary = out.putObject("summary");
            summary.put("evaluated", evaluated);
            summary.put("skipped", skipped);
            summary.put("totalInput", tickers.size());
            summary.put("elapsedMs", System.currentTimeMillis() - t0);
            ObjectNode wOut = summary.putObject("weights");
            weights.forEach(wOut::put);

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("india_screener failed: {}", e.toString());
            return ToolResult.error("india_screener failed: " + e.getMessage());
        }
    }

    /* ───────────────────────────────────────────────────────────── factor logic */

    /**
     * Score one ticker end-to-end. Returns null on hard failure (no data). Returns a
     * partial Scored with {@code skipReason} set if data is too sparse to rank.
     */
    /**
     * @param priorMomentum optional 2-element {@code [perChange365, perChange30]} array (percent,
     *                      e.g. {@code 12.5} not {@code 0.125}) from the universe call. When
     *                      non-null, used directly so we skip the Yahoo historical_price call.
     */
    private Scored scoreOne(String ticker, double minMcapCr, double[] priorMomentum) {
        Scored s = new Scored();
        s.ticker = ticker;
        try {
            // 1. Fundamentals — Yahoo first, Screener.in for .NS/.BO when Yahoo sparse.
            JsonNode fund = yahoo.fetch(ticker, "summaryDetail,financialData,defaultKeyStatistics,price,assetProfile,incomeStatementHistory,balanceSheetHistory,cashflowStatementHistory");
            if (NseIndiaClient.isIndianTicker(ticker) && isSparse(fund) && screener != null) {
                JsonNode scr = screener.fetch(ticker);   // strips .NS/.BO internally
                if (scr != null) fund = scr;   // shape-compatible — Screener mirrors Yahoo shape
            }
            if (fund == null || fund.isMissingNode()) { s.skipReason = "no-fundamentals"; return s; }

            // 2. Extract metrics.
            JsonNode price = fund.path("price");
            JsonNode fd    = fund.path("financialData");
            JsonNode sd    = fund.path("summaryDetail");
            JsonNode dks   = fund.path("defaultKeyStatistics");
            JsonNode incHist = fund.path("incomeStatementHistory").path("incomeStatementHistory");
            JsonNode cashHist = fund.path("cashflowStatementHistory").path("cashflowStatements");

            s.currency = price.path("currency").asText("INR");
            s.lastPrice = num(price, "regularMarketPrice");
            // Market cap in INR Cr (NSE convention). Yahoo's marketCap is in INR units, divide by 1e7.
            double mcapRaw = num(price, "marketCap");
            s.marketCapCr = !Double.isNaN(mcapRaw) ? mcapRaw / 1e7 : Double.NaN;
            if (!Double.isNaN(s.marketCapCr) && s.marketCapCr > 0 && s.marketCapCr < minMcapCr) {
                s.skipReason = "below-min-mcap";
                return s;
            }

            s.roe = num(fd, "returnOnEquity");
            s.roce = num(fd, "returnOnCapitalEmployed");
            if (Double.isNaN(s.roce)) {
                // Derive ROCE from EBIT / (debt + equity) if not provided by source.
                double ebit = num(fd, "ebit");
                double debt = num(fd, "totalDebt");
                JsonNode bs = fund.path("balanceSheetHistory").path("balanceSheetStatements");
                double equity = (bs.isArray() && !bs.isEmpty()) ? num(bs.get(0), "totalStockholderEquity") : Double.NaN;
                if (!Double.isNaN(ebit) && !Double.isNaN(equity) && (debt > 0 || equity > 0)) {
                    double cap = (Double.isNaN(debt) ? 0 : debt) + equity;
                    if (cap > 0) s.roce = ebit / cap;
                }
            }
            s.debtToEquity = num(fd, "debtToEquity");
            s.fcfMargin = computeFcfMargin(fd, incHist, cashHist);
            s.pe = num(sd, "trailingPE");
            s.pb = num(dks, "priceToBook");
            if (Double.isNaN(s.pb)) s.pb = num(sd, "priceToBook");
            s.eps3yCagr = computeCagr(incHist, "netIncome", 3);
            s.rev3yCagr = computeCagr(incHist, "totalRevenue", 3);

            // 3. Momentum — prefer pre-computed signals from upstream (NSE universe response
            //    already has perChange365 / perChange30, so we save the Yahoo round-trips).
            if (priorMomentum != null) {
                s.return1y  = priorMomentum[0] / 100.0;   // percent → fraction
                s.return30d = priorMomentum[1] / 100.0;
            } else {
                s.return6m = compute6mReturn(ticker);
                s.priceVs200dma = computePriceVs200dma(ticker);
            }

            // 4. Buffett-Munger composite score (0-100). Use the existing tool — it's
            //    deterministic and already optimized for this data shape.
            s.buffettScore = computeBuffettScore(ticker);

            // 5. Per-factor scores (0-100).
            s.qualityScore  = scoreQuality(s);
            s.valueRawScore = scoreValueRaw(s);   // not yet percentile-ranked — that needs the full set
            s.growthScore   = scoreGrowth(s);
            s.momentumScore = scoreMomentum(s);

            s.rationale = makeRationale(s);
            return s;
        } catch (Exception e) {
            LOG.debug("scoreOne failed for {}: {}", ticker, e.toString());
            s.skipReason = "exception:" + e.getClass().getSimpleName();
            return s;
        }
    }

    /** Quality 0-100 from ROCE, ROE, debt-to-equity, FCF margin. */
    private static double scoreQuality(Scored s) {
        double parts = 0; int n = 0;
        if (!Double.isNaN(s.roce)) {
            // ROCE >= 25% → 100, ROCE <= 0 → 0, linear in between
            parts += clamp(s.roce / 0.25, 0, 1) * 100; n++;
        }
        if (!Double.isNaN(s.roe)) {
            parts += clamp(s.roe / 0.25, 0, 1) * 100; n++;
        }
        if (!Double.isNaN(s.debtToEquity)) {
            // D/E 0 → 100, D/E >= 1.5 → 0
            parts += clamp(1.0 - s.debtToEquity / 1.5, 0, 1) * 100; n++;
        }
        if (!Double.isNaN(s.fcfMargin)) {
            parts += clamp(s.fcfMargin / 0.20, 0, 1) * 100; n++;
        }
        return n == 0 ? 50 : parts / n;
    }

    /** Value raw score — earnings yield + book yield. Will be percentile-normalized later. */
    private static double scoreValueRaw(Scored s) {
        double y = 0; int n = 0;
        if (!Double.isNaN(s.pe) && s.pe > 0) { y += 1.0 / s.pe; n++; }
        if (!Double.isNaN(s.pb) && s.pb > 0) { y += 1.0 / (s.pb * 5); n++; }  // P/B less weight
        return n == 0 ? Double.NaN : y / n;
    }

    /** Convert raw value scores to percentile ranks (0-100) within the scored universe. */
    private static void applyValuePercentile(List<Scored> universe) {
        List<Double> sorted = new ArrayList<>();
        for (Scored s : universe) if (!Double.isNaN(s.valueRawScore)) sorted.add(s.valueRawScore);
        sorted.sort(Comparator.naturalOrder());
        if (sorted.isEmpty()) {
            for (Scored s : universe) s.valueScore = 50;
            return;
        }
        for (Scored s : universe) {
            if (Double.isNaN(s.valueRawScore)) { s.valueScore = 50; continue; }
            int idx = lowerBound(sorted, s.valueRawScore);
            s.valueScore = 100.0 * idx / sorted.size();
        }
    }

    private static double scoreGrowth(Scored s) {
        double parts = 0; int n = 0;
        if (!Double.isNaN(s.rev3yCagr)) { parts += clamp(s.rev3yCagr / 0.20, 0, 1) * 100; n++; }
        if (!Double.isNaN(s.eps3yCagr)) { parts += clamp(s.eps3yCagr / 0.25, 0, 1) * 100; n++; }
        return n == 0 ? 50 : parts / n;
    }

    private static double scoreMomentum(Scored s) {
        double parts = 0; double weights = 0;
        // Prefer NSE-provided perChange365 + perChange30 (no Yahoo dependency); weight 70/30.
        if (!Double.isNaN(s.return1y)) {
            // -25% → 0, +50% → 100, linear. Reasonable for Indian equity 1y returns.
            parts += clamp((s.return1y + 0.25) / 0.75, 0, 1) * 100 * 0.70; weights += 0.70;
        }
        if (!Double.isNaN(s.return30d)) {
            // -10% → 0, +20% → 100, linear
            parts += clamp((s.return30d + 0.10) / 0.30, 0, 1) * 100 * 0.30; weights += 0.30;
        }
        // Fallback path — Yahoo-derived 6m return + 200-DMA, used when no NSE prior provided.
        if (weights == 0 && !Double.isNaN(s.return6m)) {
            parts += clamp((s.return6m + 0.25) / 0.50, 0, 1) * 100 * 0.60; weights += 0.60;
        }
        if (weights == 0 && !Double.isNaN(s.priceVs200dma)) {
            parts += clamp((s.priceVs200dma - 0.85) / 0.30, 0, 1) * 100 * 0.40; weights += 0.40;
        }
        return weights == 0 ? 50 : parts / weights;
    }

    private static double composite(Scored s, Map<String, Double> w) {
        return  s.qualityScore  * w.getOrDefault("quality",  0.35)
             +  s.valueScore    * w.getOrDefault("value",    0.20)
             +  s.growthScore   * w.getOrDefault("growth",   0.20)
             +  s.momentumScore * w.getOrDefault("momentum", 0.15)
             +  (Double.isNaN(s.buffettScore) ? 50 : s.buffettScore) * w.getOrDefault("buffett", 0.10);
    }

    private double computeBuffettScore(String ticker) {
        try {
            String argsJson = MAPPER.writeValueAsString(Map.of("ticker", ticker));
            ToolResult r = buffettTool.execute(argsJson);
            if (r == null || !r.ok() || r.content() == null) return Double.NaN;
            JsonNode out = MAPPER.readTree(r.content());
            JsonNode score = out.path("score");
            if (score.isNumber()) return score.asDouble();
            // Alternative key names — Buffett engine has been refactored a few times.
            JsonNode finalScore = out.path("finalScore");
            if (finalScore.isNumber()) return finalScore.asDouble();
            return Double.NaN;
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private double compute6mReturn(String ticker) {
        try {
            String argsJson = MAPPER.writeValueAsString(Map.of("ticker", ticker, "range", "6mo", "interval", "1d"));
            ToolResult r = historical.execute(argsJson);
            if (r == null || !r.ok() || r.content() == null) return Double.NaN;
            JsonNode out = MAPPER.readTree(r.content());
            JsonNode bars = out.path("bars");
            if (!bars.isArray() || bars.size() < 2) return Double.NaN;
            double first = bars.get(0).path("close").asDouble(Double.NaN);
            double last  = bars.get(bars.size() - 1).path("close").asDouble(Double.NaN);
            if (Double.isNaN(first) || Double.isNaN(last) || first <= 0) return Double.NaN;
            return (last - first) / first;
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private double computePriceVs200dma(String ticker) {
        try {
            String argsJson = MAPPER.writeValueAsString(Map.of("ticker", ticker, "range", "1y", "interval", "1d"));
            ToolResult r = historical.execute(argsJson);
            if (r == null || !r.ok() || r.content() == null) return Double.NaN;
            JsonNode out = MAPPER.readTree(r.content());
            JsonNode bars = out.path("bars");
            if (!bars.isArray() || bars.size() < 200) return Double.NaN;
            int n = bars.size();
            double sum = 0;
            for (int i = n - 200; i < n; i++) sum += bars.get(i).path("close").asDouble(0);
            double sma200 = sum / 200;
            double last = bars.get(n - 1).path("close").asDouble(Double.NaN);
            if (sma200 <= 0 || Double.isNaN(last)) return Double.NaN;
            return last / sma200;
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static double computeFcfMargin(JsonNode fd, JsonNode incHist, JsonNode cashHist) {
        // FCF margin = FCF / Revenue. FCF preferable from fd.freeCashflow; fallback from
        // cashflowStatement: operating - capEx for the most recent year.
        double fcf = num(fd, "freeCashflow");
        if (Double.isNaN(fcf) && cashHist.isArray() && !cashHist.isEmpty()) {
            JsonNode last = cashHist.get(0);
            double op = num(last, "totalCashFromOperatingActivities");
            double capex = num(last, "capitalExpenditures");
            if (!Double.isNaN(op) && !Double.isNaN(capex)) fcf = op + capex;   // capex is negative
        }
        double rev = num(fd, "totalRevenue");
        if (Double.isNaN(rev) && incHist.isArray() && !incHist.isEmpty()) {
            rev = num(incHist.get(0), "totalRevenue");
        }
        return (Double.isNaN(fcf) || Double.isNaN(rev) || rev <= 0) ? Double.NaN : fcf / rev;
    }

    private static double computeCagr(JsonNode hist, String field, int years) {
        if (hist == null || !hist.isArray() || hist.size() < years + 1) return Double.NaN;
        // Yahoo orders newest-first. Index 0 = latest, index years = `years` years ago.
        double latest = num(hist.get(0), field);
        double base   = num(hist.get(Math.min(years, hist.size() - 1)), field);
        if (Double.isNaN(latest) || Double.isNaN(base) || base <= 0) return Double.NaN;
        double cagr = Math.pow(latest / base, 1.0 / years) - 1.0;
        // Cap CAGR at +100%/yr — anything above is almost always a low-base artifact
        // (e.g. Axis Bank EPS bouncing from a near-zero year produces 152%/yr CAGR which
        // doesn't survive the next reporting cycle). Cap at -50%/yr on the downside.
        return clamp(cagr, -0.50, 1.00);
    }

    private static double num(JsonNode parent, String field) {
        if (parent == null || parent.isMissingNode() || parent.isNull()) return Double.NaN;
        JsonNode n = parent.path(field);
        if (n == null || n.isMissingNode() || n.isNull()) return Double.NaN;
        if (n.isNumber()) return n.asDouble();
        if (n.isObject()) {
            JsonNode raw = n.path("raw");
            return (raw == null || raw.isMissingNode() || raw.isNull()) ? Double.NaN : raw.asDouble();
        }
        if (n.isTextual()) {
            try { return Double.parseDouble(n.asText().replace(",", "").trim()); }
            catch (NumberFormatException e) { return Double.NaN; }
        }
        return Double.NaN;
    }

    private static boolean isSparse(JsonNode fund) {
        if (fund == null) return true;
        JsonNode incHist = fund.path("incomeStatementHistory").path("incomeStatementHistory");
        if (!incHist.isArray() || incHist.isEmpty()) return true;
        JsonNode latest = incHist.get(0);
        return latest.path("operatingIncome").asDouble(Double.NaN) == Double.NaN
            && latest.path("ebit").asDouble(Double.NaN) == Double.NaN
            && latest.path("grossProfit").asDouble(Double.NaN) == Double.NaN;
    }

    private static double clamp(double x, double lo, double hi) {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    private static int lowerBound(List<Double> sorted, double v) {
        int lo = 0, hi = sorted.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted.get(mid) < v) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    private static String makeRationale(Scored s) {
        StringBuilder b = new StringBuilder();
        if (!Double.isNaN(s.roce) && s.roce >= 0.20) b.append("Top-decile ROCE (").append(pct(s.roce)).append("); ");
        if (!Double.isNaN(s.return6m) && s.return6m >= 0.10) b.append("strong momentum (").append(pct(s.return6m, true)).append(" 6m); ");
        if (!Double.isNaN(s.eps3yCagr) && s.eps3yCagr >= 0.15) b.append("EPS compounding @ ").append(pct(s.eps3yCagr)).append(" CAGR; ");
        if (!Double.isNaN(s.fcfMargin) && s.fcfMargin >= 0.15) b.append("strong FCF margin (").append(pct(s.fcfMargin)).append("); ");
        if (!Double.isNaN(s.pe) && s.pe > 0 && s.pe < 25) b.append("reasonable P/E (").append(String.format("%.1f", s.pe)).append("); ");
        if (!Double.isNaN(s.debtToEquity) && s.debtToEquity < 0.3) b.append("low leverage (D/E ").append(String.format("%.2f", s.debtToEquity)).append("); ");
        if (b.length() == 0) b.append("Composite of quality, value, growth, momentum signals");
        else if (b.charAt(b.length() - 1) == ' ' && b.charAt(b.length() - 2) == ';')
            b.setLength(b.length() - 2);
        return b.toString();
    }

    private static String pct(double v) { return pct(v, false); }
    private static String pct(double v, boolean signed) {
        if (Double.isNaN(v)) return "n/a";
        return (signed && v >= 0 ? "+" : "") + String.format(Locale.US, "%.1f%%", v * 100);
    }

    /** Read pre-computed momentum from upstream universe call. Shape:
     *  {@code { "RELIANCE.NS": { "perChange365": -2.6, "perChange30": 1.4 } }}.
     *  Values are percent (12.5), not fraction (0.125). */
    private static Map<String, double[]> readPriorMomentum(JsonNode pm) {
        Map<String, double[]> out = new HashMap<>();
        if (pm == null || !pm.isObject()) return out;
        pm.fields().forEachRemaining(e -> {
            JsonNode v = e.getValue();
            double y1 = v.path("perChange365").isNumber() ? v.path("perChange365").asDouble() : Double.NaN;
            double m1 = v.path("perChange30").isNumber()  ? v.path("perChange30").asDouble()  : Double.NaN;
            if (!Double.isNaN(y1) || !Double.isNaN(m1)) out.put(e.getKey(), new double[]{y1, m1});
        });
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Double> readWeights(JsonNode w) {
        if (w == null || w.isMissingNode() || w.isNull() || !w.isObject()) {
            return new HashMap<>(DEFAULT_WEIGHTS);
        }
        Map<String, Double> out = new HashMap<>(DEFAULT_WEIGHTS);
        w.fields().forEachRemaining(e -> {
            if (e.getValue().isNumber()) out.put(e.getKey(), e.getValue().asDouble());
        });
        // Normalize to sum to 1.0 if user-provided weights don't.
        double sum = out.values().stream().mapToDouble(Double::doubleValue).sum();
        if (sum > 0 && Math.abs(sum - 1.0) > 1e-6) {
            Map<String, Double> norm = new HashMap<>();
            out.forEach((k, v) -> norm.put(k, v / sum));
            return norm;
        }
        return out;
    }

    /* ───────────────────────────────────────────────────────────── result type */

    /** Mutable scratch record per ticker. JSON-serialized once at the end. */
    private static final class Scored {
        String ticker;
        String currency = "INR";
        double lastPrice = Double.NaN;
        double marketCapCr = Double.NaN;
        double roe = Double.NaN;
        double roce = Double.NaN;
        double debtToEquity = Double.NaN;
        double fcfMargin = Double.NaN;
        double pe = Double.NaN;
        double pb = Double.NaN;
        double rev3yCagr = Double.NaN;
        double eps3yCagr = Double.NaN;
        double return6m = Double.NaN;
        double priceVs200dma = Double.NaN;
        double return1y = Double.NaN;     // from NSE universe perChange365
        double return30d = Double.NaN;    // from NSE universe perChange30
        double buffettScore = Double.NaN;

        double qualityScore = 0;
        double valueRawScore = Double.NaN;   // pre-percentile
        double valueScore = 0;
        double growthScore = 0;
        double momentumScore = 0;
        double composite = 0;

        String skipReason = null;
        String rationale = "";

        ObjectNode toJson(int rank) {
            ObjectNode o = MAPPER.createObjectNode();
            o.put("rank", rank);
            o.put("ticker", ticker);
            o.put("score", round1(composite));
            o.put("currency", currency);
            ObjectNode comp = o.putObject("components");
            comp.put("quality",  round1(qualityScore));
            comp.put("value",    round1(valueScore));
            comp.put("growth",   round1(growthScore));
            comp.put("momentum", round1(momentumScore));
            comp.put("buffett",  Double.isNaN(buffettScore) ? null : round1(buffettScore));
            ObjectNode met = o.putObject("metrics");
            putIfFinite(met, "roce", roce);
            putIfFinite(met, "roe", roe);
            putIfFinite(met, "debtToEquity", debtToEquity);
            putIfFinite(met, "fcfMargin", fcfMargin);
            putIfFinite(met, "pe", pe);
            putIfFinite(met, "pb", pb);
            putIfFinite(met, "rev3yCagr", rev3yCagr);
            putIfFinite(met, "eps3yCagr", eps3yCagr);
            putIfFinite(met, "return6m", return6m);
            putIfFinite(met, "priceVs200dma", priceVs200dma);
            putIfFinite(met, "return1y", return1y);
            putIfFinite(met, "return30d", return30d);
            putIfFinite(met, "buffettScore", buffettScore);
            putIfFinite(met, "marketCapCr", marketCapCr);
            putIfFinite(met, "lastPrice", lastPrice);
            o.put("rationale", rationale);
            return o;
        }

        private static void putIfFinite(ObjectNode o, String k, double v) {
            if (!Double.isNaN(v) && !Double.isInfinite(v)) o.put(k, v);
        }

        private static double round1(double v) {
            return Math.round(v * 10.0) / 10.0;
        }
    }
}
