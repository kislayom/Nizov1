package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.finance.model.HistoricalPrice;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Computes the standard technical-analysis indicator suite from a bar series:
 * SMA20/50/200, EMA12/26, RSI14, MACD (12-26-9), Bollinger Bands (20, 2σ), ATR14, OBV, VWAP.
 *
 * <p>Pure-compute — no external network. The caller passes a ticker and we fetch the underlying
 * 1-year daily history via {@link HistoricalPriceTool}, then run the formulas. This is faster
 * and more robust than asking the LLM to compute indicators (it can't — and asking a tool API
 * costs an extra round-trip).
 *
 * <p>Output JSON includes both the latest snapshot ({@code latest}) AND the full series
 * ({@code series}) so the chart can overlay any indicator on demand without re-fetching.
 *
 * <p>Pushes Kimaya's TechnicalIndicatorsCard forward by:
 * <ol>
 *   <li>Returning the FULL series, not just the latest value — enables the chart to render
 *       SMA50/200 as overlays and Bollinger as a band.</li>
 *   <li>Including a {@code trend} for each metric (UP/DOWN/FLAT) computed from the slope of
 *       the last 5 values, so the front-end shows mini trend arrows next to each number.</li>
 *   <li>Producing an {@code overallSignal} (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL) from a
 *       weighted vote across MACD, RSI, SMA-cross, and price-vs-Bollinger.</li>
 * </ol>
 */
public final class TechnicalIndicatorsTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(TechnicalIndicatorsTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HistoricalPriceTool prices;

    public TechnicalIndicatorsTool(HistoricalPriceTool prices) {
        this.prices = prices;
    }

    @Override public String name() { return "technical_indicators"; }

    @Override
    public String description() {
        return "Compute SMA/EMA/RSI/MACD/Bollinger/ATR/OBV/VWAP for a ticker over the last 1 year "
                + "(daily bars). Returns latest snapshot + full series + per-metric trend + an overall "
                + "BUY/HOLD/SELL signal. Front-end renders a chart-tech fenced block from the output.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "ticker": { "type": "string", "description": "Ticker symbol, e.g. AAPL." }
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

            List<HistoricalPrice> bars = prices.fetchOne(ticker, "1y", "1d");
            if (bars.size() < 30) {
                return ToolResult.error("not enough history for " + ticker + " — got " + bars.size() + " bars");
            }
            return ToolResult.ok(computeJson(ticker, bars));
        } catch (Exception e) {
            LOG.warn("technical_indicators failed: {}", e.toString());
            return ToolResult.error("technical_indicators failed: " + e.getMessage());
        }
    }

    private String computeJson(String ticker, List<HistoricalPrice> bars) throws Exception {
        int n = bars.size();
        double[] close = new double[n], high = new double[n], low = new double[n];
        long[] vol = new long[n];
        for (int i = 0; i < n; i++) {
            close[i] = bars.get(i).close();
            high[i] = bars.get(i).high();
            low[i] = bars.get(i).low();
            vol[i] = bars.get(i).volume();
        }

        double[] sma20 = sma(close, 20);
        double[] sma50 = sma(close, 50);
        double[] sma200 = sma(close, 200);
        double[] ema12 = ema(close, 12);
        double[] ema26 = ema(close, 26);
        double[] rsi14 = rsi(close, 14);
        double[] macd = sub(ema12, ema26);
        double[] macdSignal = ema(macd, 9);
        double[] macdHist = sub(macd, macdSignal);
        double[][] bb = bollinger(close, 20, 2.0);
        double[] atr14 = atr(high, low, close, 14);
        double[] obv = obv(close, vol);
        double[] vwap = vwap(high, low, close, vol);

        ObjectNode root = MAPPER.createObjectNode();
        root.put("ticker", ticker);
        root.put("source", "computed_from_yahoo_v8");
        root.put("bars", n);
        root.put("currentPrice", close[n - 1]);
        root.put("timestamp", bars.get(n - 1).date().toString());

        ObjectNode latest = root.putObject("latest");
        ObjectNode latestMA = latest.putObject("movingAverages");
        latestMA.put("sma20", val(sma20, n - 1));
        latestMA.put("sma50", val(sma50, n - 1));
        latestMA.put("sma200", val(sma200, n - 1));
        latestMA.put("ema12", val(ema12, n - 1));
        latestMA.put("ema26", val(ema26, n - 1));
        ObjectNode latestMo = latest.putObject("momentum");
        latestMo.put("rsi14", val(rsi14, n - 1));
        latestMo.put("rsiSignal", rsiSignal(val(rsi14, n - 1)));
        latestMo.put("macd", val(macd, n - 1));
        latestMo.put("macdSignal", val(macdSignal, n - 1));
        latestMo.put("macdHistogram", val(macdHist, n - 1));
        // Null-safe trend: when the series has NaN / not enough bars, val() returns null —
        // unboxing into `> 0` blows up with NPE (May 2026 regression after FMP fallback
        // sometimes returned shorter series than Yahoo's). Surface "UNKNOWN" explicitly.
        Double macdH = val(macdHist, n - 1);
        latestMo.put("macdTrend", macdH == null ? "UNKNOWN" : (macdH > 0 ? "BULLISH" : "BEARISH"));
        ObjectNode latestVol = latest.putObject("volatility");
        latestVol.put("bollingerUpper", val(bb[0], n - 1));
        latestVol.put("bollingerMiddle", val(bb[1], n - 1));
        latestVol.put("bollingerLower", val(bb[2], n - 1));
        latestVol.put("atr14", val(atr14, n - 1));
        ObjectNode latestVolu = latest.putObject("volume");
        latestVolu.put("obv", val(obv, n - 1));
        latestVolu.put("vwap", val(vwap, n - 1));

        ObjectNode trends = root.putObject("trends");
        trends.put("sma20", trendOf(sma20, 5));
        trends.put("sma50", trendOf(sma50, 5));
        trends.put("sma200", trendOf(sma200, 5));
        trends.put("rsi14", trendOf(rsi14, 5));
        trends.put("macd", trendOf(macd, 5));
        trends.put("atr14", trendOf(atr14, 5));
        trends.put("obv", trendOf(obv, 5));

        // Overall signal — weighted vote across the four major checks. dval() returns NaN
        // instead of null so the primitive-double signature works even when an indicator
        // can't be computed (e.g. sma200 with <200 bars). overallSignal already guards
        // each input with !Double.isNaN().
        root.put("overallSignal", overallSignal(close[n - 1],
                dval(sma20, n - 1), dval(sma50, n - 1), dval(sma200, n - 1),
                dval(rsi14, n - 1), dval(macdHist, n - 1),
                dval(bb[0], n - 1), dval(bb[2], n - 1)));

        // Full series so the front-end can render any overlay without an extra fetch.
        ObjectNode series = root.putObject("series");
        // Sample every 4th bar for chart smoothness; keeps payload reasonable for 252-day year.
        seriesArray(series, "dates", bars);
        arr(series, "sma20", sma20);
        arr(series, "sma50", sma50);
        arr(series, "sma200", sma200);
        arr(series, "ema12", ema12);
        arr(series, "ema26", ema26);
        arr(series, "rsi14", rsi14);
        arr(series, "macd", macd);
        arr(series, "macdSignal", macdSignal);
        arr(series, "macdHistogram", macdHist);
        arr(series, "bollingerUpper", bb[0]);
        arr(series, "bollingerMiddle", bb[1]);
        arr(series, "bollingerLower", bb[2]);
        arr(series, "atr14", atr14);
        arr(series, "obv", obv);
        arr(series, "vwap", vwap);

        return MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(root);
    }

    // ===== Indicator math =====

    private static double[] sma(double[] x, int p) {
        double[] out = new double[x.length];
        java.util.Arrays.fill(out, Double.NaN);
        if (x.length < p) return out;
        double sum = 0;
        for (int i = 0; i < p; i++) sum += x[i];
        out[p - 1] = sum / p;
        for (int i = p; i < x.length; i++) {
            sum += x[i] - x[i - p];
            out[i] = sum / p;
        }
        return out;
    }

    private static double[] ema(double[] x, int p) {
        double[] out = new double[x.length];
        java.util.Arrays.fill(out, Double.NaN);
        if (x.length < p) return out;
        double k = 2.0 / (p + 1);
        // Seed with SMA(p)
        double seed = 0;
        for (int i = 0; i < p; i++) seed += x[i];
        seed /= p;
        out[p - 1] = seed;
        for (int i = p; i < x.length; i++) {
            out[i] = (x[i] - out[i - 1]) * k + out[i - 1];
        }
        return out;
    }

    private static double[] rsi(double[] x, int p) {
        double[] out = new double[x.length];
        java.util.Arrays.fill(out, Double.NaN);
        if (x.length <= p) return out;
        double gain = 0, loss = 0;
        for (int i = 1; i <= p; i++) {
            double d = x[i] - x[i - 1];
            if (d >= 0) gain += d; else loss += -d;
        }
        double avgGain = gain / p, avgLoss = loss / p;
        out[p] = 100 - 100 / (1 + (avgLoss == 0 ? 1e9 : avgGain / avgLoss));
        for (int i = p + 1; i < x.length; i++) {
            double d = x[i] - x[i - 1];
            double g = d > 0 ? d : 0;
            double l = d < 0 ? -d : 0;
            avgGain = (avgGain * (p - 1) + g) / p;
            avgLoss = (avgLoss * (p - 1) + l) / p;
            out[i] = 100 - 100 / (1 + (avgLoss == 0 ? 1e9 : avgGain / avgLoss));
        }
        return out;
    }

    private static double[][] bollinger(double[] x, int p, double k) {
        double[] mid = sma(x, p);
        double[] upper = new double[x.length];
        double[] lower = new double[x.length];
        java.util.Arrays.fill(upper, Double.NaN);
        java.util.Arrays.fill(lower, Double.NaN);
        for (int i = p - 1; i < x.length; i++) {
            double mean = mid[i];
            double sumSq = 0;
            for (int j = 0; j < p; j++) {
                double d = x[i - j] - mean;
                sumSq += d * d;
            }
            double sd = Math.sqrt(sumSq / p);
            upper[i] = mean + k * sd;
            lower[i] = mean - k * sd;
        }
        return new double[][]{upper, mid, lower};
    }

    private static double[] atr(double[] h, double[] l, double[] c, int p) {
        double[] tr = new double[h.length];
        tr[0] = h[0] - l[0];
        for (int i = 1; i < h.length; i++) {
            tr[i] = Math.max(h[i] - l[i], Math.max(Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1])));
        }
        // Wilder's smoothing
        double[] out = new double[h.length];
        java.util.Arrays.fill(out, Double.NaN);
        if (h.length < p) return out;
        double seed = 0;
        for (int i = 0; i < p; i++) seed += tr[i];
        out[p - 1] = seed / p;
        for (int i = p; i < h.length; i++) {
            out[i] = (out[i - 1] * (p - 1) + tr[i]) / p;
        }
        return out;
    }

    private static double[] obv(double[] c, long[] v) {
        double[] out = new double[c.length];
        out[0] = 0;
        for (int i = 1; i < c.length; i++) {
            if (c[i] > c[i - 1]) out[i] = out[i - 1] + v[i];
            else if (c[i] < c[i - 1]) out[i] = out[i - 1] - v[i];
            else out[i] = out[i - 1];
        }
        return out;
    }

    private static double[] vwap(double[] h, double[] l, double[] c, long[] v) {
        double[] out = new double[c.length];
        double cumPV = 0, cumV = 0;
        for (int i = 0; i < c.length; i++) {
            double tp = (h[i] + l[i] + c[i]) / 3.0;
            cumPV += tp * v[i];
            cumV += v[i];
            out[i] = cumV == 0 ? Double.NaN : cumPV / cumV;
        }
        return out;
    }

    private static double[] sub(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (Double.isNaN(a[i]) || Double.isNaN(b[i])) ? Double.NaN : a[i] - b[i];
        }
        return out;
    }

    // ===== Helpers =====

    private static String rsiSignal(Double rsi) {
        if (rsi == null || rsi.isNaN()) return "UNKNOWN";
        if (rsi > 70) return "OVERBOUGHT";
        if (rsi < 30) return "OVERSOLD";
        return "NEUTRAL";
    }

    private static String trendOf(double[] x, int look) {
        if (x.length < look + 1) return "FLAT";
        // Find the last `look+1` non-NaN values
        int found = 0;
        double first = Double.NaN, last = Double.NaN;
        for (int i = x.length - 1; i >= 0 && found < look + 1; i--) {
            if (!Double.isNaN(x[i])) {
                if (Double.isNaN(last)) last = x[i];
                first = x[i];
                found++;
            }
        }
        if (Double.isNaN(first) || Double.isNaN(last) || first == 0) return "FLAT";
        double change = (last - first) / Math.abs(first);
        if (change > 0.005) return "UP";
        if (change < -0.005) return "DOWN";
        return "FLAT";
    }

    private static String overallSignal(double price, double sma20, double sma50, double sma200,
                                        double rsi, double macdHist, double bbUpper, double bbLower) {
        int score = 0;
        // SMA cross — short above long is bullish
        if (!Double.isNaN(sma20) && !Double.isNaN(sma50) && sma20 > sma50) score++;
        if (!Double.isNaN(sma50) && !Double.isNaN(sma200) && sma50 > sma200) score++;
        if (!Double.isNaN(sma20) && !Double.isNaN(sma50) && sma20 < sma50) score--;
        if (!Double.isNaN(sma50) && !Double.isNaN(sma200) && sma50 < sma200) score--;
        // MACD histogram positive = momentum bullish
        if (!Double.isNaN(macdHist) && macdHist > 0) score++;
        if (!Double.isNaN(macdHist) && macdHist < 0) score--;
        // RSI extremes — overbought = sell pressure
        if (!Double.isNaN(rsi) && rsi > 70) score--;
        if (!Double.isNaN(rsi) && rsi < 30) score++;
        // Bollinger position
        if (!Double.isNaN(bbUpper) && price >= bbUpper) score--;
        if (!Double.isNaN(bbLower) && price <= bbLower) score++;

        if (score >= 4) return "STRONG_BUY";
        if (score >= 2) return "BUY";
        if (score <= -4) return "STRONG_SELL";
        if (score <= -2) return "SELL";
        return "HOLD";
    }

    private static Double val(double[] x, int i) {
        if (i < 0 || i >= x.length || Double.isNaN(x[i])) return null;
        return x[i];
    }

    /** Primitive-double variant of {@link #val} — returns {@code NaN} for missing data
     *  instead of null. Used by helpers (overallSignal) whose signatures take primitives
     *  and already guard each input with {@link Double#isNaN}. */
    private static double dval(double[] x, int i) {
        if (i < 0 || i >= x.length) return Double.NaN;
        return x[i];
    }

    private static void arr(ObjectNode parent, String key, double[] x) {
        var a = parent.putArray(key);
        for (double v : x) {
            if (Double.isNaN(v)) a.addNull();
            else a.add(v);
        }
    }

    private static void seriesArray(ObjectNode parent, String key, List<HistoricalPrice> bars) {
        var a = parent.putArray(key);
        for (HistoricalPrice p : bars) a.add(p.date().toString());
    }
}
