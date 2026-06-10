package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.finance.model.HistoricalPrice;
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
import java.util.List;
import java.util.Map;

/**
 * Fetches OHLCV historical price data via Yahoo Finance's public v8 chart endpoint.
 * Used by the technical analyst to feed the interactive chart + indicators on the front-end.
 *
 * <p>Yahoo's v8 endpoint takes:
 * <ul>
 *   <li>{@code range} — e.g. {@code 1d / 5d / 1mo / 3mo / 6mo / 1y / 2y / 5y / 10y / max}</li>
 *   <li>{@code interval} — e.g. {@code 1m / 5m / 15m / 30m / 60m / 1d / 1wk / 1mo}</li>
 *   <li>{@code includePrePost=false}</li>
 * </ul>
 *
 * <p>Returns JSON with arrays for timestamp + open + high + low + close + volume + adjclose
 * — we flatten them into a list of {@link HistoricalPrice} records and return a JSON array
 * the LLM (or the front-end) can render.
 *
 * <p>Forward of Kimaya's YahooFinanceProvider in two ways:
 * <ol>
 *   <li>Returns ALL standard timeframes (1D/5D/1M/3M/6M/1Y/2Y/5Y/10Y/MAX) in one call when
 *       the caller asks for {@code range="all_timeframes"} — useful for the chart's
 *       timeframe-pill switcher to avoid N round-trips.</li>
 *   <li>Streams compact JSON suitable for embedding in a {@code chart-interactive} fenced
 *       block — front-end hydrates without an extra fetch.</li>
 * </ol>
 */
public final class HistoricalPriceTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(HistoricalPriceTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP = SharedHttpClient.INSTANCE;
    private static final String UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0";

    /** Standard timeframes for the chart pill bar. */
    private static final List<TimeframeSpec> ALL_TIMEFRAMES = List.of(
            new TimeframeSpec("1D", "1d", "5m"),
            new TimeframeSpec("5D", "5d", "30m"),
            new TimeframeSpec("1M", "1mo", "1d"),
            new TimeframeSpec("3M", "3mo", "1d"),
            new TimeframeSpec("6M", "6mo", "1d"),
            new TimeframeSpec("1Y", "1y", "1d"),
            new TimeframeSpec("2Y", "2y", "1wk"),
            new TimeframeSpec("5Y", "5y", "1wk"),
            new TimeframeSpec("10Y", "10y", "1mo"),
            new TimeframeSpec("ALL", "max", "1mo")
    );

    /**
     * Optional FMP fallback. When provided AND Yahoo's v8 chart endpoint returns 429
     * (observed when 5+ sub-agents fan out concurrently — thundering herd), we fall back
     * to FMP's {@code /historical-price-eod/full} endpoint which is a paid API not subject
     * to Yahoo's rate-limit. Daily resolution only — intraday timeframes (1D, 5D) aren't
     * fallback-eligible (FMP free tier doesn't expose intraday).
     */
    private final FmpClient fmpFallback;

    /**
     * Optional 4th-tier fallback for Indian tickers (.NS / .BO). Used when Yahoo 429s,
     * FMP exhausts its quota (HTTP 402), AND Stooq returns the API-key signup page
     * (which it does after the 2026 free-CSV lockdown). Screener.in publishes a clean
     * undocumented JSON chart endpoint that has no rate limit on free access.
     * Close-only (no O/H/L) but sufficient for technical indicators.
     */
    private final ScreenerInClient screenerFallback;

    public HistoricalPriceTool() { this(null, null); }

    public HistoricalPriceTool(FmpClient fmpFallback) { this(fmpFallback, null); }

    public HistoricalPriceTool(FmpClient fmpFallback, ScreenerInClient screenerFallback) {
        this.fmpFallback = fmpFallback;
        this.screenerFallback = screenerFallback;
    }

    @Override public String name() { return "historical_price"; }

    @Override
    public String description() {
        return "Fetch historical OHLCV price data for a ticker. "
                + "Use 'all_timeframes' to get every standard window (1D/5D/1M/3M/6M/1Y/2Y/5Y/10Y/ALL) "
                + "in one call — ideal for the interactive chart. Or pass a specific {range, interval} "
                + "for one window. Output is JSON the front-end's chart-interactive fenced block can render.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "ticker":   { "type": "string", "description": "Ticker symbol, e.g. AAPL, MSFT, RELIANCE.NS." },
                "range":    { "type": "string", "description": "Time range. 'all_timeframes' returns every window in one call. Otherwise: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max." },
                "interval": { "type": "string", "description": "Bar interval. Ignored when range='all_timeframes'. Otherwise: 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo." }
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
            String range = args.path("range").asText("all_timeframes").trim();
            String interval = args.path("interval").asText("").trim();

            if ("all_timeframes".equalsIgnoreCase(range)) {
                return ToolResult.ok(fetchAllTimeframesJson(ticker));
            }
            if (interval.isEmpty()) {
                interval = inferInterval(range);
            }
            List<HistoricalPrice> bars = fetchOne(ticker, range, interval);
            return ToolResult.ok(barsToJson(ticker, range, interval, bars));
        } catch (Exception e) {
            LOG.warn("historical_price failed: {}", e.toString());
            return ToolResult.error("historical_price failed: " + e.getMessage());
        }
    }

    /** Fetch all standard windows. Sequential per-timeframe so we don't amplify the
     *  Yahoo v8 thundering-herd already created by parallel sub-agents. Each timeframe
     *  tries Yahoo first; on Yahoo 429 (or any failure) we fall back to FMP for daily-or-
     *  coarser timeframes. Intraday (1D, 5D) Yahoo-only — those go empty if Yahoo is hot.
     */
    private String fetchAllTimeframesJson(String ticker) throws Exception {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("ticker", ticker);
        root.put("source", "yahoo_v8_chart+fmp_fallback");
        ObjectNode timeframes = root.putObject("timeframes");

        boolean fmpAvailable = fmpFallback != null && fmpFallback.isEnabled();
        java.util.Map<String, java.util.List<HistoricalPrice>> fmpByLabel = null;

        for (TimeframeSpec t : ALL_TIMEFRAMES) {
            ArrayNode arr = timeframes.putArray(t.label);
            try {
                List<HistoricalPrice> bars = fetchOne(ticker, t.range, t.interval);
                writeBars(arr, bars);
            } catch (Exception yahooErr) {
                LOG.warn("Yahoo v8 chart {} failed for {}: {}", t.label, ticker, yahooErr.toString());
                // Daily-or-coarser: try FMP fallback. 1D / 5D need intraday so we skip.
                if (fmpAvailable && isFmpEligible(t)) {
                    if (fmpByLabel == null) {
                        try {
                            fmpByLabel = buildFmpHistoricalByLabel(ticker);
                            LOG.info("FMP historical fallback fetched for {} ({} bars total)",
                                    ticker, fmpByLabel.values().stream().mapToInt(java.util.List::size).sum());
                        } catch (Exception fmpErr) {
                            LOG.warn("FMP historical fallback ALSO failed for {}: {}", ticker, fmpErr.toString());
                            fmpByLabel = java.util.Collections.emptyMap();
                        }
                    }
                    java.util.List<HistoricalPrice> fb = fmpByLabel.get(t.label);
                    if (fb != null && !fb.isEmpty()) {
                        writeBars(arr, fb);
                        // Populate the bars cache from the FMP fallback so downstream tools
                        // (technical_indicators) can reuse these bars without hitting Yahoo
                        // again. Without this, a Yahoo 429 on this timeframe leaks through
                        // to downstream callers who do their own (also-throttled) fetchOne.
                        BARS_CACHE.put(ticker + "|" + t.range + "|" + t.interval,
                                new CachedBars(fb, System.nanoTime()));
                        persistDiskCacheAsync();
                        continue;   // FMP served this timeframe; skip Screener
                    }
                }
                // 4th-tier: Screener.in for Indian tickers. Reached when Yahoo failed AND
                // (FMP unavailable OR FMP failed OR FMP empty for this label). Only kicks
                // in for .NS / .BO and only for daily-or-coarser timeframes.
                if (screenerFallback != null && isFmpEligible(t)
                        && ScreenerInClient.isIndianTicker(ticker)) {
                    java.util.List<HistoricalPrice> sb = tryScreener(ticker, t.range, t.interval);
                    if (sb != null && !sb.isEmpty()) {
                        writeBars(arr, sb);
                        BARS_CACHE.put(ticker + "|" + t.range + "|" + t.interval,
                                new CachedBars(sb, System.nanoTime()));
                        persistDiskCacheAsync();
                        LOG.info("Screener.in fallback served {} for {} ({} bars)",
                                t.label, ticker, sb.size());
                    }
                }
            }
        }
        return MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(root);
    }

    /** Per-timeframe: is FMP eligible? FMP serves daily (or coarser via downsampling). */
    private static boolean isFmpEligible(TimeframeSpec t) {
        return !"1D".equals(t.label) && !"5D".equals(t.label);
    }

    /**
     * Pull a long-history daily-bar window from FMP and bucket it into our standard
     * {@code TimeframeSpec} labels via downsampling (1d→weekly via Mondays, 1d→monthly via
     * month-end). One FMP call covers all daily-or-coarser fallbacks for a ticker.
     */
    private java.util.Map<String, java.util.List<HistoricalPrice>> buildFmpHistoricalByLabel(String ticker) throws Exception {
        java.time.LocalDate today = java.time.LocalDate.now(java.time.ZoneOffset.UTC);
        java.time.LocalDate from = today.minusYears(20);   // FMP free tier supports this
        com.fasterxml.jackson.databind.JsonNode rows = fmpFallback.fetchHistoricalEod(ticker,
                from.toString(), today.toString());

        java.util.List<HistoricalPrice> daily = new java.util.ArrayList<>();
        if (rows != null && rows.isArray()) {
            // FMP returns newest-first; reverse to oldest-first to match Yahoo shape.
            for (int i = rows.size() - 1; i >= 0; i--) {
                com.fasterxml.jackson.databind.JsonNode r = rows.get(i);
                String dateStr = r.path("date").asText("");
                if (dateStr.length() < 10) continue;
                try {
                    java.time.LocalDate d = java.time.LocalDate.parse(dateStr.substring(0, 10));
                    daily.add(new HistoricalPrice(d,
                            r.path("open").asDouble(0),
                            r.path("high").asDouble(0),
                            r.path("low").asDouble(0),
                            r.path("close").asDouble(0),
                            r.path("adjClose").asDouble(r.path("close").asDouble(0)),
                            r.path("volume").asLong(0)));
                } catch (Exception parseErr) { /* skip malformed row */ }
            }
        }

        java.util.Map<String, java.util.List<HistoricalPrice>> byLabel = new java.util.HashMap<>();
        // Crop daily to the time-window each label expects, then downsample to match interval.
        byLabel.put("1M",  cropDays(daily, 30));
        byLabel.put("3M",  cropDays(daily, 90));
        byLabel.put("6M",  cropDays(daily, 180));
        byLabel.put("1Y",  cropDays(daily, 365));
        byLabel.put("2Y",  downsampleWeekly(cropDays(daily, 365 * 2)));
        byLabel.put("5Y",  downsampleWeekly(cropDays(daily, 365 * 5)));
        byLabel.put("10Y", downsampleMonthly(cropDays(daily, 365 * 10)));
        byLabel.put("ALL", downsampleMonthly(daily));
        return byLabel;
    }

    private static java.util.List<HistoricalPrice> cropDays(java.util.List<HistoricalPrice> bars, int days) {
        java.time.LocalDate cutoff = java.time.LocalDate.now(java.time.ZoneOffset.UTC).minusDays(days);
        java.util.List<HistoricalPrice> out = new java.util.ArrayList<>();
        for (HistoricalPrice p : bars) if (!p.date().isBefore(cutoff)) out.add(p);
        return out;
    }

    private static java.util.List<HistoricalPrice> downsampleWeekly(java.util.List<HistoricalPrice> bars) {
        // Take the FRIDAY (or last available day) of each week.
        java.util.Map<String, HistoricalPrice> byWeek = new java.util.LinkedHashMap<>();
        for (HistoricalPrice p : bars) {
            // ISO week-of-year + year as key
            java.time.LocalDate d = p.date();
            int wk = d.get(java.time.temporal.WeekFields.ISO.weekOfWeekBasedYear());
            int yr = d.get(java.time.temporal.WeekFields.ISO.weekBasedYear());
            byWeek.put(yr + "-W" + wk, p);   // last bar of each week wins
        }
        return new java.util.ArrayList<>(byWeek.values());
    }

    private static java.util.List<HistoricalPrice> downsampleMonthly(java.util.List<HistoricalPrice> bars) {
        java.util.Map<String, HistoricalPrice> byMonth = new java.util.LinkedHashMap<>();
        for (HistoricalPrice p : bars) {
            String key = p.date().getYear() + "-" + p.date().getMonthValue();
            byMonth.put(key, p);    // last bar of each month wins
        }
        return new java.util.ArrayList<>(byMonth.values());
    }

    private static void writeBars(ArrayNode arr, java.util.List<HistoricalPrice> bars) {
        for (HistoricalPrice p : bars) {
            ObjectNode row = arr.addObject();
            row.put("date", p.date().toString());
            row.put("open", p.open());
            row.put("high", p.high());
            row.put("low", p.low());
            row.put("close", p.close());
            row.put("adjustedClose", p.adjustedClose());
            row.put("volume", p.volume());
        }
    }

    private String barsToJson(String ticker, String range, String interval, List<HistoricalPrice> bars) throws Exception {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("ticker", ticker);
        root.put("source", "yahoo_v8_chart");
        root.put("range", range);
        root.put("interval", interval);
        ArrayNode arr = root.putArray("bars");
        for (HistoricalPrice p : bars) {
            ObjectNode row = arr.addObject();
            row.put("date", p.date().toString());
            row.put("open", p.open());
            row.put("high", p.high());
            row.put("low", p.low());
            row.put("close", p.close());
            row.put("adjustedClose", p.adjustedClose());
            row.put("volume", p.volume());
        }
        return MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(root);
    }

    /**
     * Tiny TTL cache for Yahoo v8 responses keyed by ticker|range|interval. Yahoo's chart
     * endpoint rate-limits aggressively (429 after 1-2 hits/sec). Stock-analysis pipelines
     * call fetchOne several times within seconds: prefetch → technical_indicators → maybe
     * historical_price tool — all wanting the same 1y/1d OHLCV. The 60s in-memory TTL
     * is plenty for one pipeline to finish and short enough that intraday data freshness
     * isn't compromised.
     *
     * <p>The cache also persists to {@code ~/.nizo/bars-cache.dat} so a JVM restart doesn't
     * wipe successful fetches — important when Yahoo + FMP are both exhausted (which they
     * will be after a developer hammers the pipeline). Disk-cached entries are accepted as
     * "stale-OK" only when both upstream sources are down (they're stamped on load with the
     * original fetch time, so a fresh hit will still refresh them).
     */
    private static final long CACHE_TTL_NS = 60_000_000_000L;
    private static final long DISK_CACHE_TTL_HOURS = 24;
    private static final java.nio.file.Path DISK_CACHE_PATH;
    private static final java.util.concurrent.ConcurrentMap<String, CachedBars> BARS_CACHE =
            new java.util.concurrent.ConcurrentHashMap<>();
    private record CachedBars(List<HistoricalPrice> bars, long fetchedAtNs) implements java.io.Serializable {}

    static {
        DISK_CACHE_PATH = java.nio.file.Paths.get(System.getProperty("user.home"), ".nizo", "bars-cache.dat");
        loadDiskCache();
    }

    @SuppressWarnings("unchecked")
    private static void loadDiskCache() {
        try {
            if (!java.nio.file.Files.exists(DISK_CACHE_PATH)) return;
            try (java.io.ObjectInputStream in = new java.io.ObjectInputStream(java.nio.file.Files.newInputStream(DISK_CACHE_PATH))) {
                Map<String, java.util.List<Object>> raw = (Map<String, java.util.List<Object>>) in.readObject();
                long now = System.nanoTime();
                long maxAgeNanos = DISK_CACHE_TTL_HOURS * 3600L * 1_000_000_000L;
                for (Map.Entry<String, java.util.List<Object>> e : raw.entrySet()) {
                    Long savedAtMillis = (Long) e.getValue().get(0);
                    java.util.List<HistoricalPrice> bars = (java.util.List<HistoricalPrice>) e.getValue().get(1);
                    long ageMillis = System.currentTimeMillis() - savedAtMillis;
                    if (ageMillis > maxAgeNanos / 1_000_000L) continue;     // too old
                    // Stamp with a fetch time that's already older than the in-memory TTL so
                    // any successful fresh fetch immediately overrides it. We're only here
                    // to act as the LAST RESORT when Yahoo + FMP are both exhausted.
                    BARS_CACHE.put(e.getKey(), new CachedBars(bars, now - CACHE_TTL_NS - 1L));
                }
                LOG.info("loaded {} bars-cache entries from disk", raw.size());
            }
        } catch (Throwable t) {
            LOG.warn("could not load disk bars cache (proceeding fresh): {}", t.toString());
        }
    }

    private static void persistDiskCacheAsync() {
        // Fire-and-forget on a daemon thread to keep fetchOne snappy.
        Thread.ofVirtual().start(() -> {
            try {
                java.nio.file.Files.createDirectories(DISK_CACHE_PATH.getParent());
                Map<String, java.util.List<Object>> raw = new java.util.HashMap<>();
                long nowMillis = System.currentTimeMillis();
                for (Map.Entry<String, CachedBars> e : BARS_CACHE.entrySet()) {
                    raw.put(e.getKey(), java.util.List.of(nowMillis, e.getValue().bars()));
                }
                try (java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(java.nio.file.Files.newOutputStream(DISK_CACHE_PATH,
                        java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING))) {
                    out.writeObject(raw);
                }
            } catch (Throwable t) {
                LOG.warn("could not persist disk bars cache: {}", t.toString());
            }
        });
    }

    /** Single Yahoo v8 chart call → OHLCV list. Public so caller code (other tools) can reuse.
     *  Cached for 60s to dodge Yahoo's rate limiter when the pipeline's prefetch + sub-skill
     *  + LLM each independently ask for the same bars within seconds.
     */
    public List<HistoricalPrice> fetchOne(String ticker, String range, String interval) throws Exception {
        String key = ticker + "|" + range + "|" + interval;
        CachedBars cached = BARS_CACHE.get(key);
        long now = System.nanoTime();
        if (cached != null && now - cached.fetchedAtNs < CACHE_TTL_NS) {
            return cached.bars;
        }
        String url = "https://query2.finance.yahoo.com/v8/finance/chart/"
                + URLEncoder.encode(ticker, StandardCharsets.UTF_8)
                + "?range=" + URLEncoder.encode(range, StandardCharsets.UTF_8)
                + "&interval=" + URLEncoder.encode(interval, StandardCharsets.UTF_8)
                + "&includePrePost=false";
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);

        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(15))
                .header("User-Agent", UA)
                .header("Accept", "application/json")
                .GET().build();
        HttpResponse<String> resp;
        try {
            resp = HTTP.send(req, BoundedHttp.ofString());
        } catch (Exception netErr) {
            // Network failure (DNS, timeout, connection reset, etc.) — last resort: stale
            // cache. Same fallback strategy as 429 / 5xx below.
            if (cached != null) {
                LOG.warn("Yahoo v8 chart network error for {}|{}|{}, returning stale cache: {}",
                        ticker, range, interval, netErr.toString());
                return cached.bars;
            }
            throw netErr;
        }
        if (resp.statusCode() / 100 != 2) {
            // On 429 / 5xx: try stale cache first, then Stooq fallback (no rate-limit),
            // then fail. Stooq is a free unofficial mirror with daily OHLCV — coverage is
            // best for US tickers but also includes major ASX (.au), LSE (.uk), Tokyo
            // (.jp), and some Indian (.in) names. We use it only for daily bars; intraday
            // (1D / 5D) timeframes have no Stooq fallback (they need true minute data).
            if ((resp.statusCode() == 429 || resp.statusCode() / 100 == 5) && cached != null) {
                LOG.warn("Yahoo v8 chart {} for {}|{}|{}, returning stale cache",
                        resp.statusCode(), ticker, range, interval);
                return cached.bars;
            }
            if (resp.statusCode() == 429 || resp.statusCode() / 100 == 5) {
                List<HistoricalPrice> stooq = tryStooq(ticker, range, interval);
                if (stooq != null && !stooq.isEmpty()) {
                    BARS_CACHE.put(key, new CachedBars(stooq, now));
                    persistDiskCacheAsync();
                    LOG.info("Yahoo {} for {}|{}|{}, served from Stooq fallback ({} bars)",
                            resp.statusCode(), ticker, range, interval, stooq.size());
                    return stooq;
                }
                // 4th-tier: Screener.in chart API for Indian tickers. Only path that works
                // after Yahoo 429 + FMP 402 + Stooq paywall on .NS / .BO names.
                List<HistoricalPrice> scr = tryScreener(ticker, range, interval);
                if (scr != null && !scr.isEmpty()) {
                    BARS_CACHE.put(key, new CachedBars(scr, now));
                    persistDiskCacheAsync();
                    LOG.info("Yahoo {} for {}|{}|{}, served from Screener.in fallback ({} bars)",
                            resp.statusCode(), ticker, range, interval, scr.size());
                    return scr;
                }
            }
            throw new RuntimeException("Yahoo v8 chart HTTP " + resp.statusCode());
        }
        List<HistoricalPrice> bars = parseChartJson(resp.body());
        BARS_CACHE.put(key, new CachedBars(bars, now));
        persistDiskCacheAsync();
        return bars;
    }

    /**
     * Stooq.com free OHLCV fallback (no API key, no rate-limit). Used when Yahoo's v8
     * chart endpoint is throttling us AND we have no recent stale cache. Returns null
     * if Stooq has no data for this ticker — caller should propagate the original Yahoo
     * failure in that case.
     *
     * <p>Stooq URL form: {@code https://stooq.com/q/d/l/?s=<symbol>&i=d}.
     * Ticker conventions:
     *  <ul>
     *    <li>AAPL → aapl.us (US)</li>
     *    <li>BHP.AX → bhp.au (Australian)</li>
     *    <li>VOD.L → vod.uk (London)</li>
     *    <li>RELIANCE.NS → reliance.in (Indian, partial coverage)</li>
     *    <li>7203.T → 7203.jp (Japanese)</li>
     *  </ul>
     * Stooq always returns daily granularity; the {@code interval} arg is ignored.
     * For 1wk / 1mo callers, the daily bars can be resampled by the caller.
     */
    /**
     * Screener.in chart-API fallback for Indian tickers. The 4th-tier safety net after
     * Yahoo (429), FMP (402), and Stooq (paywalled). Close-only — sufficient for
     * {@code chart-interactive} (which can render line) and {@code chart-tech}
     * indicators (RSI/MACD/SMA/EMA all operate on close).
     *
     * <p>Maps requested {@code range} to a days-back parameter for the Screener API:
     * <pre>
     *   1mo → 30   3mo → 90   6mo → 180   1y → 365
     *   2y  → 730  5y  → 1825 10y → 3650  max → 3650 (Screener caps near 10y)
     * </pre>
     * Returns {@code null} for intraday ranges (1d/5d) — Screener is daily-only.
     */
    private List<HistoricalPrice> tryScreener(String yahooTicker, String range, String interval) {
        if (screenerFallback == null) return null;
        if (!ScreenerInClient.isIndianTicker(yahooTicker)) return null;
        if ("1d".equals(range) || "5d".equals(range)) return null;   // Screener is daily-only
        int days = switch (range == null ? "" : range) {
            case "1mo" -> 30;
            case "3mo" -> 90;
            case "6mo" -> 180;
            case "1y"  -> 365;
            case "2y"  -> 730;
            case "5y"  -> 1825;
            case "10y", "max" -> 3650;
            default    -> 365;
        };
        return screenerFallback.historicalBars(yahooTicker, days);
    }

    private static List<HistoricalPrice> tryStooq(String yahooTicker, String range, String interval) {
        // Intraday ranges can't be served — Stooq is daily-only.
        if ("1d".equals(range) || "5d".equals(range)) return null;
        if (!"1d".equals(interval) && !"1wk".equals(interval) && !"1mo".equals(interval)) return null;
        String stooqSymbol = toStooqSymbol(yahooTicker);
        if (stooqSymbol == null) return null;
        String url = "https://stooq.com/q/d/l/?s=" + URLEncoder.encode(stooqSymbol, StandardCharsets.UTF_8) + "&i=d";
        try {
            URI uri = URI.create(url);
            SsrfGuard.assertSafe(uri);
            HttpRequest req = HttpRequest.newBuilder(uri)
                    .timeout(Duration.ofSeconds(10))
                    .header("User-Agent", UA)
                    .GET().build();
            HttpResponse<String> resp = HTTP.send(req, BoundedHttp.ofString());
            if (resp.statusCode() / 100 != 2) return null;
            String body = resp.body();
            if (body == null || body.isBlank() || body.contains("No data")) return null;
            // CSV: Date,Open,High,Low,Close,Volume
            List<HistoricalPrice> bars = new ArrayList<>();
            String[] lines = body.split("\\r?\\n");
            for (int i = 1; i < lines.length; i++) {       // skip header
                String[] cols = lines[i].split(",");
                if (cols.length < 6) continue;
                try {
                    LocalDate d = LocalDate.parse(cols[0]);
                    double open = Double.parseDouble(cols[1]);
                    double high = Double.parseDouble(cols[2]);
                    double low  = Double.parseDouble(cols[3]);
                    double close = Double.parseDouble(cols[4]);
                    long vol = (long) Math.max(0, Double.parseDouble(cols[5]));
                    bars.add(new HistoricalPrice(d, open, high, low, close, close, vol));
                } catch (Exception parseErr) { /* skip malformed row */ }
            }
            return bars;
        } catch (Exception e) {
            LOG.debug("Stooq fallback failed for {}: {}", stooqSymbol, e.toString());
            return null;
        }
    }

    /** Translate a Yahoo ticker to Stooq's symbol convention; null if unmapped. */
    private static String toStooqSymbol(String yahooTicker) {
        if (yahooTicker == null || yahooTicker.isBlank()) return null;
        if (yahooTicker.startsWith("^")) return null;            // indices not supported on Stooq's free CSV
        String t = yahooTicker.toLowerCase();
        if (t.endsWith(".ns") || t.endsWith(".bo")) return t.substring(0, t.length() - 3) + ".in";
        if (t.endsWith(".ax")) return t.substring(0, t.length() - 3) + ".au";
        if (t.endsWith(".l"))  return t.substring(0, t.length() - 2) + ".uk";
        if (t.endsWith(".t"))  return t.substring(0, t.length() - 2) + ".jp";
        if (t.endsWith(".to")) return t.substring(0, t.length() - 3) + ".ca";   // Toronto
        if (t.endsWith(".de") || t.endsWith(".pa") || t.endsWith(".hk")) return t; // close enough
        if (t.matches("[a-z0-9.-]+")) return t + ".us";          // default: assume US
        return null;
    }

    /** Parse Yahoo v8 chart response into OHLCV bars. */
    private static List<HistoricalPrice> parseChartJson(String json) throws Exception {
        JsonNode root = MAPPER.readTree(json);
        JsonNode result = root.path("chart").path("result").path(0);
        if (result.isMissingNode()) {
            String err = root.path("chart").path("error").path("description").asText("unknown");
            throw new RuntimeException("Yahoo chart error: " + err);
        }
        JsonNode tsNode = result.path("timestamp");
        JsonNode quote = result.path("indicators").path("quote").path(0);
        JsonNode adjclose = result.path("indicators").path("adjclose").path(0).path("adjclose");
        JsonNode opens = quote.path("open");
        JsonNode highs = quote.path("high");
        JsonNode lows = quote.path("low");
        JsonNode closes = quote.path("close");
        JsonNode volumes = quote.path("volume");

        List<HistoricalPrice> out = new ArrayList<>();
        int n = tsNode.size();
        for (int i = 0; i < n; i++) {
            long ts = tsNode.get(i).asLong();
            // Skip nulls (Yahoo emits null in arrays for missing/non-trading bars)
            if (opens.get(i) == null || opens.get(i).isNull()) continue;
            double o = opens.get(i).asDouble();
            double h = highs.get(i).asDouble();
            double l = lows.get(i).asDouble();
            double c = closes.get(i).asDouble();
            double ac = adjclose.size() > i && !adjclose.get(i).isNull() ? adjclose.get(i).asDouble() : c;
            long v = volumes.get(i) != null && !volumes.get(i).isNull() ? volumes.get(i).asLong() : 0L;
            LocalDate date = LocalDate.ofInstant(java.time.Instant.ofEpochSecond(ts), ZoneOffset.UTC);
            out.add(new HistoricalPrice(date, o, h, l, c, ac, v));
        }
        return out;
    }

    private static String inferInterval(String range) {
        return switch (range) {
            case "1d" -> "5m";
            case "5d" -> "30m";
            case "1mo", "3mo", "6mo", "1y" -> "1d";
            case "2y", "5y" -> "1wk";
            default -> "1mo";
        };
    }

    private record TimeframeSpec(String label, String range, String interval) {}
}
