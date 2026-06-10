package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.net.BoundedHttp;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * India macro dashboard — pulls the key top-down signals that feed the
 * regime classifier used by {@code india_top_picks} (Phase 2).
 *
 * <p>Signals covered (all from free public sources, mostly cached 4-12h):
 * <ul>
 *   <li><b>FX</b> — USD/INR + 30d change (Yahoo {@code INR=X}).</li>
 *   <li><b>Oil</b> — WTI + Brent (Yahoo {@code CL=F}, {@code BZ=F}). India imports
 *       ~85% of its crude — oil up means refining margin squeeze + CAD widening.</li>
 *   <li><b>Gold</b> — USD/oz (Yahoo {@code GC=F}). Risk-off / inflation proxy.</li>
 *   <li><b>India 10y G-Sec yield</b> — Yahoo {@code IN10YT=RR} — bond market signal.</li>
 *   <li><b>India GDP growth</b> — World Bank API (annual {@code NY.GDP.MKTP.KD.ZG}).</li>
 *   <li><b>India CPI</b> — World Bank API (annual {@code FP.CPI.TOTL.ZG}).</li>
 *   <li><b>NIFTY 50 + Bank index PE</b> — placeholder (NSE has these in their
 *       daily indices file; integration deferred to Phase 3 sector view).</li>
 *   <li><b>RBI repo rate</b> — currently a curated lookup table (RBI doesn't expose
 *       a clean JSON endpoint; the policy schedule + last rate is stable enough that
 *       a hardcoded list with quarterly review is a reasonable Phase-2 trade-off).</li>
 * </ul>
 *
 * <p>Outputs a {@code regime} object that classifies each axis as
 * {@code cutting | hiking | neutral} (rates), {@code strong | weak | neutral} (INR),
 * {@code bullish | bearish | neutral} (commodities), {@code expansive | tight}
 * (fiscal). The macro analyst sub-skill consumes this regime to bias sector picks
 * (e.g. INR weak → tilt IT exporters / pharma; rates cutting → tilt financials).
 *
 * <p>Cache: 4h on disk at {@code ~/.nizo/cache/india-macro.json}. Forecast: even
 * with a fresh fetch every 4h, the total bandwidth is &lt;200 KB/day.
 */
public final class IndiaMacroDashboardTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaMacroDashboardTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final long CACHE_TTL_MS = 4L * 60L * 60L * 1000L;   // 4h

    /** Curated RBI repo-rate history. Update quarterly (or on policy day). */
    private static final double RBI_REPO_RATE = 6.25;              // as of 2026-05 (placeholder)
    private static final String RBI_REPO_STANCE = "neutral";       // "cutting" / "hiking" / "neutral"

    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .version(HttpClient.Version.HTTP_2)
            .build();

    private final Path cacheDir;

    public IndiaMacroDashboardTool() {
        String home = System.getProperty("user.home", ".");
        this.cacheDir = Paths.get(home, ".nizo", "cache");
        try { Files.createDirectories(cacheDir); } catch (Exception e) { /* best-effort */ }
    }

    @Override public String name() { return "india_macro_dashboard"; }

    @Override
    public String description() {
        return "India macro dashboard — FX (USD/INR), commodities (WTI/Brent/gold), India 10y "
                + "yield, GDP growth, CPI inflation, RBI repo rate, plus a regime classification "
                + "({rate, inr, commodity, fiscal}). Feeds the india_macro_analyst sub-skill which "
                + "tilts sector picks based on top-down conditions. Cached 4h on disk.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "refresh": { "type": "boolean", "description": "If true, bypass the 4h cache and re-fetch." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            boolean force = args.path("refresh").asBoolean(false);

            if (!force) {
                ObjectNode cached = readCache();
                if (cached != null) {
                    return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(cached));
                }
            }

            // Fan out parallel fetches — each one is independent. Pool sized to the
            // longest critical path so we wait only ~timeout for the slowest.
            ExecutorService pool = Executors.newFixedThreadPool(8, r -> {
                Thread t = new Thread(r, "macro-fetch");
                t.setDaemon(true);
                return t;
            });
            CompletableFuture<double[]> usdInrF = supply(pool, () -> yahooQuote("INR=X"));
            CompletableFuture<double[]> wtiF    = supply(pool, () -> yahooQuote("CL=F"));
            CompletableFuture<double[]> brentF  = supply(pool, () -> yahooQuote("BZ=F"));
            CompletableFuture<double[]> goldF   = supply(pool, () -> yahooQuote("GC=F"));
            CompletableFuture<double[]> in10yF  = supply(pool, () -> yahooQuote("IN10YT=RR"));
            CompletableFuture<Double>   gdpF    = supply(pool, () -> worldBank("NY.GDP.MKTP.KD.ZG"));
            CompletableFuture<Double>   cpiF    = supply(pool, () -> worldBank("FP.CPI.TOTL.ZG"));

            double[] usdInr = await(usdInrF, new double[]{Double.NaN, Double.NaN});
            double[] wti    = await(wtiF,    new double[]{Double.NaN, Double.NaN});
            double[] brent  = await(brentF,  new double[]{Double.NaN, Double.NaN});
            double[] gold   = await(goldF,   new double[]{Double.NaN, Double.NaN});
            double[] in10y  = await(in10yF,  new double[]{Double.NaN, Double.NaN});
            double gdp = await(gdpF, Double.NaN);
            double cpi = await(cpiF, Double.NaN);

            pool.shutdown();
            try { pool.awaitTermination(2, TimeUnit.SECONDS); }
            catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

            ObjectNode out = MAPPER.createObjectNode();
            out.put("asOf", Instant.now().toString());
            out.put("source", "yahoo finance + world bank (cached 4h)");

            ObjectNode rates = out.putObject("rates");
            rates.put("rbiRepo", RBI_REPO_RATE);
            rates.put("rbiStance", RBI_REPO_STANCE);
            putIfFinite(rates, "tenYrYield", in10y[0]);
            putIfFinite(rates, "tenYrYieldChange30d", in10y[1]);

            ObjectNode fx = out.putObject("fx");
            putIfFinite(fx, "usdInr", usdInr[0]);
            putIfFinite(fx, "usdInrChange30d", usdInr[1]);

            ObjectNode commodities = out.putObject("commodities");
            putIfFinite(commodities, "wtiUsd",        wti[0]);
            putIfFinite(commodities, "wtiChange30d",  wti[1]);
            putIfFinite(commodities, "brentUsd",      brent[0]);
            putIfFinite(commodities, "brentChange30d",brent[1]);
            putIfFinite(commodities, "goldUsd",       gold[0]);
            putIfFinite(commodities, "goldChange30d", gold[1]);

            ObjectNode macro = out.putObject("macro");
            putIfFinite(macro, "gdpGrowthYoY", gdp);
            putIfFinite(macro, "cpiInflationYoY", cpi);

            // Regime classification — interprets the raw numbers into actionable axes.
            ObjectNode regime = out.putObject("regime");
            regime.put("rate", RBI_REPO_STANCE);   // "cutting" / "hiking" / "neutral"
            regime.put("inr", classifyInr(usdInr[1]));
            regime.put("commodity", classifyOil(wti[1], brent[1]));
            regime.put("growth", classifyGrowth(gdp));
            regime.put("inflation", classifyInflation(cpi));
            // Composite tilt — single bias signal for the picks engine
            regime.put("equityTilt", composeTilt(usdInr[1], wti[1], gdp, cpi));

            ObjectNode interp = out.putObject("interpretation");
            interp.put("inrSignal", inrSignal(usdInr[1]));
            interp.put("oilSignal", oilSignal(wti[1], brent[1]));
            interp.put("growthSignal", growthSignal(gdp));
            interp.put("inflationSignal", inflationSignal(cpi));

            writeCache(out);
            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("india_macro_dashboard failed: {}", e.toString());
            return ToolResult.error("india_macro_dashboard failed: " + e.getMessage());
        }
    }

    /* ─────────────────────────────────────────────────────────── fetchers */

    /**
     * Yahoo v8 chart endpoint — returns price + 30-day % change. Used uniformly for
     * FX, commodities, indices. Output: {@code [latest, change30d]}.
     */
    private static double[] yahooQuote(String symbol) throws Exception {
        // 1-month range, 1-day interval → ~22 closes. Enough to compute change30d.
        String url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
                   + "?range=1mo&interval=1d";
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(8))
                .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0")
                .header("Accept", "application/json")
                .GET().build();
        HttpResponse<String> resp = HTTP.send(req, BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("yahoo HTTP " + resp.statusCode() + " for " + symbol);
        }
        JsonNode root = MAPPER.readTree(resp.body());
        JsonNode result = root.path("chart").path("result").path(0);
        JsonNode closes = result.path("indicators").path("quote").path(0).path("close");
        if (!closes.isArray() || closes.size() < 2) throw new RuntimeException("no closes for " + symbol);
        double latest = closes.get(closes.size() - 1).asDouble();
        double first  = closes.get(0).asDouble();
        if (first == 0 || Double.isNaN(first) || Double.isNaN(latest)) {
            return new double[]{latest, Double.NaN};
        }
        return new double[]{latest, (latest - first) / first * 100.0};   // percent
    }

    /**
     * World Bank API — returns the most recent annual indicator value for India.
     * Format: {@code https://api.worldbank.org/v2/country/IN/indicator/<CODE>?format=json}.
     */
    private static double worldBank(String indicator) throws Exception {
        String url = "https://api.worldbank.org/v2/country/IN/indicator/"
                   + indicator + "?format=json&per_page=10";
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(8))
                .header("Accept", "application/json")
                .GET().build();
        HttpResponse<String> resp = HTTP.send(req, BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("worldbank HTTP " + resp.statusCode() + " for " + indicator);
        }
        JsonNode root = MAPPER.readTree(resp.body());
        // World Bank returns [metadata, [rows]]
        JsonNode rows = root.isArray() && root.size() >= 2 ? root.get(1) : null;
        if (rows == null || !rows.isArray()) throw new RuntimeException("worldbank empty for " + indicator);
        // Find the most recent non-null reading.
        for (JsonNode row : rows) {
            JsonNode v = row.path("value");
            if (v.isNumber()) return v.asDouble();
        }
        throw new RuntimeException("worldbank all-null for " + indicator);
    }

    /* ─────────────────────────────────────────────────────── classifiers */

    /** USD/INR change 30d — INR weakening means USD/INR up. */
    private static String classifyInr(double usdInrChg30d) {
        if (Double.isNaN(usdInrChg30d)) return "unknown";
        if (usdInrChg30d > 1.0)  return "weak";       // INR depreciated >1% — tilt IT/pharma
        if (usdInrChg30d < -1.0) return "strong";     // INR appreciated >1% — caution on exporters
        return "neutral";
    }

    /** Oil up = bad for India macro (CAD widens, refining margin squeezed). */
    private static String classifyOil(double wtiChg30d, double brentChg30d) {
        double avg = avgFinite(wtiChg30d, brentChg30d);
        if (Double.isNaN(avg)) return "unknown";
        if (avg > 5.0)  return "bearish";     // oil up >5% — defensive
        if (avg < -5.0) return "bullish";     // oil down >5% — risk-on
        return "neutral";
    }

    private static String classifyGrowth(double gdpYoy) {
        if (Double.isNaN(gdpYoy)) return "unknown";
        if (gdpYoy > 6.5) return "expansive";   // India trend ~6.5% — anything above is hot
        if (gdpYoy < 4.5) return "tight";       // below 4.5% is slowdown territory
        return "neutral";
    }

    private static String classifyInflation(double cpiYoy) {
        if (Double.isNaN(cpiYoy)) return "unknown";
        if (cpiYoy > 6.0) return "hot";          // outside RBI 2-6% band
        if (cpiYoy < 3.0) return "cool";
        return "in-band";
    }

    /** Composite tilt — single signal for picks engine. */
    private static String composeTilt(double usdInrChg, double oilChg, double gdp, double cpi) {
        // Negative scores = bearish; positive = bullish.
        double score = 0;
        if (!Double.isNaN(usdInrChg)) score += -usdInrChg * 0.5;          // INR weakening = INR bearish
        if (!Double.isNaN(oilChg))    score += -oilChg * 0.5;             // oil up = bearish for India
        if (!Double.isNaN(gdp))       score += (gdp - 6.0) * 2;           // above-trend growth
        if (!Double.isNaN(cpi))       score += (4.5 - cpi) * 2;           // inflation below midpoint
        if (score >  4) return "bullish";
        if (score < -4) return "bearish";
        return "neutral";
    }

    private static String inrSignal(double chg) {
        if (Double.isNaN(chg)) return "no data";
        if (chg > 1.0)  return String.format("INR weakened %.1f%% over 30d — tilt to IT exporters + pharma", chg);
        if (chg < -1.0) return String.format("INR strengthened %.1f%% over 30d — caution on exporters", -chg);
        return "INR range-bound";
    }

    private static String oilSignal(double wti, double brent) {
        double avg = avgFinite(wti, brent);
        if (Double.isNaN(avg)) return "no data";
        if (avg > 5.0)  return String.format("Crude up %.1f%% over 30d — refining margin squeeze, CAD widens", avg);
        if (avg < -5.0) return String.format("Crude down %.1f%% over 30d — supportive of refining + autos", -avg);
        return "Crude range-bound";
    }

    private static String growthSignal(double gdp) {
        if (Double.isNaN(gdp)) return "no data";
        if (gdp > 6.5) return String.format("GDP growing %.1f%% YoY — above India trend, tilt cyclicals", gdp);
        if (gdp < 4.5) return String.format("GDP growing only %.1f%% YoY — slowdown, tilt defensives", gdp);
        return String.format("GDP growing %.1f%% YoY — near trend", gdp);
    }

    private static String inflationSignal(double cpi) {
        if (Double.isNaN(cpi)) return "no data";
        if (cpi > 6.0) return String.format("CPI at %.1f%% YoY — above RBI band, expect tightness", cpi);
        if (cpi < 3.0) return String.format("CPI at %.1f%% YoY — below RBI floor, room to cut", cpi);
        return String.format("CPI at %.1f%% YoY — within RBI 2-6%% band", cpi);
    }

    /* ──────────────────────────────────────────────────────────── helpers */

    private static <T> CompletableFuture<T> supply(ExecutorService pool, ThrowingSupplier<T> s) {
        return CompletableFuture.supplyAsync(() -> {
            try { return s.get(); }
            catch (Exception e) { throw new RuntimeException(e); }
        }, pool);
    }

    private static <T> T await(CompletableFuture<T> f, T fallback) {
        try { return f.get(10, TimeUnit.SECONDS); }
        catch (Exception e) { LOG.debug("macro fetch fallback: {}", e.toString()); return fallback; }
    }

    private interface ThrowingSupplier<T> { T get() throws Exception; }

    private static void putIfFinite(ObjectNode o, String k, double v) {
        if (!Double.isNaN(v) && !Double.isInfinite(v)) o.put(k, v);
    }

    private static double avgFinite(double a, double b) {
        boolean fa = !Double.isNaN(a), fb = !Double.isNaN(b);
        if (fa && fb) return (a + b) / 2;
        if (fa) return a;
        if (fb) return b;
        return Double.NaN;
    }

    /* ──────────────────────────────────────────────────────── disk cache */

    private Path cachePath() { return cacheDir.resolve("india-macro.json"); }

    private ObjectNode readCache() {
        Path p = cachePath();
        if (!Files.isRegularFile(p)) return null;
        try {
            long age = System.currentTimeMillis() - Files.getLastModifiedTime(p).toMillis();
            if (age > CACHE_TTL_MS) return null;
            JsonNode node = MAPPER.readTree(Files.readAllBytes(p));
            if (!(node instanceof ObjectNode obj)) return null;
            obj.put("source", "yahoo+worldbank (cached " + (age / 60_000L) + "m ago)");
            return obj;
        } catch (Exception e) {
            LOG.debug("macro cache read failed: {}", e.toString());
            return null;
        }
    }

    private void writeCache(ObjectNode payload) {
        try {
            Files.write(cachePath(), MAPPER.writeValueAsBytes(payload),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (Exception e) { LOG.debug("macro cache write failed: {}", e.toString()); }
    }
}
