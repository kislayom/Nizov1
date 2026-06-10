package ai.nizo.tools.finance;

import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.nizo.tools.finance.model.HistoricalPrice;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Screener.in scraper — Tier-2 fallback for Indian-listed companies whose fundamentals
 * Yahoo and FMP cover sparsely (anything outside Nifty-100, plus some mid-caps).
 *
 * <p>Screener.in publishes consolidated financials for NSE/BSE-listed companies as
 * structured HTML tables on URLs like {@code /company/<SYMBOL>/consolidated/}. Numbers
 * are in INR crores (₹1 cr = ₹10,000,000); we convert to absolute values to match
 * Yahoo's raw-number shape.
 *
 * <p>No API key, no rate limit (within reason — keep requests below 1/sec to be a good
 * citizen). The site is unofficial scraping but has remained stable for years.
 *
 * <h2>Output shape</h2>
 * Produces a Yahoo {@code quoteSummary.result[0]}-shaped JsonNode for the modules:
 * {@code incomeStatementHistory}, {@code balanceSheetHistory}, {@code cashflowStatementHistory},
 * {@code defaultKeyStatistics}, {@code summaryDetail}, {@code financialData}.
 *
 * <p>Disabled silently when {@code NIZO_SCREENERIN_DISABLED=1} is set so tests / dev
 * environments can avoid hitting Screener.
 */
public final class ScreenerInClient {

    private static final Logger LOG = LoggerFactory.getLogger(ScreenerInClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String BASE = "https://www.screener.in/company/";
    private static final Duration TIMEOUT = Duration.ofSeconds(10);
    private static final String UA = "Mozilla/5.0 nizo-finance/1.0";
    private static final long CRORE = 10_000_000L;   // 1 cr = 10^7

    private final HttpClient http = SharedHttpClient.INSTANCE;
    private final boolean enabled;

    public ScreenerInClient() {
        this.enabled = !"1".equals(System.getenv("NIZO_SCREENERIN_DISABLED"));
        if (enabled) LOG.info("Screener.in enabled (Indian-equity fundamentals fallback)");
        else LOG.info("Screener.in disabled via NIZO_SCREENERIN_DISABLED=1");
    }

    public boolean isEnabled() { return enabled; }

    /**
     * True if {@code ticker} looks like an Indian-listed equity that Screener.in covers.
     * Pattern: {@code SYMBOL.NS} (NSE) or {@code SYMBOL.BO} (BSE). Strips the suffix
     * because Screener.in URLs use the bare symbol.
     */
    public static boolean isIndianTicker(String ticker) {
        if (ticker == null) return false;
        String upper = ticker.toUpperCase(Locale.ROOT);
        return upper.endsWith(".NS") || upper.endsWith(".BO");
    }

    /** Strip {@code .NS}/{@code .BO} suffix → bare Screener.in symbol. */
    public static String toScreenerSymbol(String ticker) {
        if (ticker == null) return "";
        String upper = ticker.toUpperCase(Locale.ROOT);
        if (upper.endsWith(".NS") || upper.endsWith(".BO")) return upper.substring(0, upper.length() - 3);
        return upper;
    }

    /**
     * Fetch + parse the Screener.in page for a ticker. Returns a Yahoo quoteSummary-shaped
     * JsonNode (with {@code incomeStatementHistory.incomeStatementHistory}[], etc.) or
     * {@code null} on any failure.
     */
    public JsonNode fetch(String ticker) {
        if (!isEnabled() || !isIndianTicker(ticker)) return null;
        String symbol = toScreenerSymbol(ticker);
        String url = BASE + symbol + "/consolidated/";
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "text/html,application/xhtml+xml")
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
            if (resp.statusCode() == 404) {
                // Try non-consolidated URL — newly-listed companies don't always have a
                // consolidated view. Falls back to standalone financials.
                String alt = BASE + symbol + "/";
                req = HttpRequest.newBuilder(URI.create(alt))
                        .timeout(TIMEOUT)
                        .header("User-Agent", UA)
                        .header("Accept", "text/html,application/xhtml+xml")
                        .GET().build();
                resp = http.send(req, BoundedHttp.ofString());
            }
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("Screener.in HTTP {} for {}", resp.statusCode(), symbol);
                return null;
            }
            return parsePage(resp.body(), symbol);
        } catch (Exception e) {
            LOG.warn("Screener.in fetch failed for {}: {}", symbol, e.toString());
            return null;
        }
    }

    /** Parse the Screener.in HTML into the Yahoo quoteSummary result shape. */
    private static JsonNode parsePage(String html, String symbol) {
        Document doc = Jsoup.parse(html);
        ObjectNode result = MAPPER.createObjectNode();

        // ── Profit & Loss → incomeStatementHistory ─────────────────────────
        Map<String, Map<String, Double>> pl = parseSection(doc, "profit-loss");
        if (!pl.isEmpty()) {
            ObjectNode wrap = result.putObject("incomeStatementHistory");
            ArrayNode rows = wrap.putArray("incomeStatementHistory");
            for (Map.Entry<String, Map<String, Double>> e : pl.entrySet()) {
                ObjectNode row = rows.addObject();
                row.putObject("endDate").put("raw", parseDateEpoch(e.getKey())).put("fmt", e.getKey());
                Map<String, Double> cols = e.getValue();
                // Crores → absolute. Screener field names → Yahoo field names.
                // Screener's P&L summary doesn't expose Material Cost (it's collapsed under
                // Expenses). Skip costOfRevenue + grossProfit when missing so the front-end
                // shows "—" instead of "Gross Profit = Revenue" which is misleading.
                putRawCr(row, "totalRevenue",      cols.get("Sales"));
                if (cols.get("Material Cost") != null) {
                    putRawCr(row, "costOfRevenue", cols.get("Material Cost"));
                    putRawCr(row, "grossProfit",   diff(cols.get("Sales"), cols.get("Material Cost")));
                }
                putRawCr(row, "operatingExpenses", cols.get("Expenses"));
                putRawCr(row, "operatingIncome",   cols.get("Operating Profit"));
                putRawCr(row, "ebit",              cols.get("Operating Profit"));
                if (cols.get("Depreciation") != null) {
                    putRawCr(row, "ebitda",        sum(cols.get("Operating Profit"), cols.get("Depreciation")));
                }
                putRawCr(row, "interestExpense",   cols.get("Interest"));
                putRawCr(row, "netIncome",         cols.get("Net Profit"));
                // Screener gives EPS in rupees directly (no crore conversion)
                Double eps = cols.get("EPS in Rs");
                if (eps != null) row.putObject("dilutedEPS").put("raw", eps).put("fmt", String.valueOf(eps));
            }
        }

        // ── Balance Sheet → balanceSheetHistory ─────────────────────────────
        Map<String, Map<String, Double>> bs = parseSection(doc, "balance-sheet");
        if (!bs.isEmpty()) {
            ObjectNode wrap = result.putObject("balanceSheetHistory");
            ArrayNode rows = wrap.putArray("balanceSheetStatements");
            for (Map.Entry<String, Map<String, Double>> e : bs.entrySet()) {
                ObjectNode row = rows.addObject();
                row.putObject("endDate").put("raw", parseDateEpoch(e.getKey())).put("fmt", e.getKey());
                Map<String, Double> cols = e.getValue();
                putRawCr(row, "commonStock",       cols.get("Equity Capital"));
                putRawCr(row, "retainedEarnings",  cols.get("Reserves"));
                putRawCr(row, "totalStockholderEquity", sum(cols.get("Equity Capital"), cols.get("Reserves")));
                putRawCr(row, "longTermDebt",      cols.get("Borrowings"));
                putRawCr(row, "totalCurrentLiabilities", cols.get("Other Liabilities"));
                putRawCr(row, "totalAssets",       cols.get("Total Assets"));
                putRawCr(row, "totalLiab",         sum(cols.get("Borrowings"), cols.get("Other Liabilities")));
                putRawCr(row, "netTangibleAssets", cols.get("Fixed Assets"));
                putRawCr(row, "totalInvestments",  cols.get("Investments"));
                putRawCr(row, "otherAssets",       cols.get("Other Assets"));
            }
        }

        // ── Cash Flow → cashflowStatementHistory ────────────────────────────
        Map<String, Map<String, Double>> cf = parseSection(doc, "cash-flow");
        if (!cf.isEmpty()) {
            ObjectNode wrap = result.putObject("cashflowStatementHistory");
            ArrayNode rows = wrap.putArray("cashflowStatements");
            for (Map.Entry<String, Map<String, Double>> e : cf.entrySet()) {
                ObjectNode row = rows.addObject();
                row.putObject("endDate").put("raw", parseDateEpoch(e.getKey())).put("fmt", e.getKey());
                Map<String, Double> cols = e.getValue();
                putRawCr(row, "totalCashFromOperatingActivities",  cols.get("Cash from Operating Activity"));
                putRawCr(row, "totalCashflowsFromInvestingActivities", cols.get("Cash from Investing Activity"));
                putRawCr(row, "totalCashFromFinancingActivities",  cols.get("Cash from Financing Activity"));
                putRawCr(row, "changeInCash",                       cols.get("Net Cash Flow"));
                // Free cash flow approx: operating + investing (investing usually negative)
                putRawCr(row, "freeCashflow", sum(cols.get("Cash from Operating Activity"),
                                                  cols.get("Cash from Investing Activity")));
            }
        }

        // ── Key ratios → defaultKeyStatistics + summaryDetail + financialData ──
        Map<String, Double> ratios = parseTopRatios(doc);
        if (!ratios.isEmpty()) {
            ObjectNode kstats = result.putObject("defaultKeyStatistics");
            ObjectNode sd     = result.putObject("summaryDetail");
            ObjectNode fd     = result.putObject("financialData");
            putRaw(kstats, "trailingPE",       ratios.get("Stock P/E"));
            putRaw(kstats, "bookValue",        ratios.get("Book Value"));
            putRaw(kstats, "dividendYield",    pctToFraction(ratios.get("Dividend Yield")));
            // Derive P/B from current price / book value when Screener doesn't ship it
            // (the top-ratios block on most Indian pages doesn't include P/B explicitly).
            Double pbDerived = (ratios.get("Current Price") != null && ratios.get("Book Value") != null
                                && ratios.get("Book Value") != 0)
                    ? ratios.get("Current Price") / ratios.get("Book Value") : null;
            putRaw(kstats, "priceToBook",      pbDerived != null ? pbDerived : ratios.get("Price to book value"));
            putRaw(sd,     "trailingPE",       ratios.get("Stock P/E"));
            putRaw(sd,     "priceToBook",      pbDerived);
            putRaw(sd,     "marketCap",        crToRupees(ratios.get("Market Cap")));
            putRaw(sd,     "dividendYield",    pctToFraction(ratios.get("Dividend Yield")));
            putRaw(fd,     "returnOnEquity",   pctToFraction(ratios.get("ROE")));
            putRaw(fd,     "returnOnAssets",   pctToFraction(ratios.get("ROA")));
            // ROCE is the canonical Indian-equity profitability metric — surface it
            // separately so the front-end can render it where ROA used to sit for
            // Yahoo-sourced reports.
            putRaw(fd,     "returnOnCapitalEmployed", pctToFraction(ratios.get("ROCE")));
            putRaw(fd,     "debtToEquity",     ratios.get("Debt to equity"));
            putRaw(fd,     "currentPrice",     ratios.get("Current Price"));
            putRaw(fd,     "financialCurrency", null);     // INR implicit
            // Profile-shaped block so downstream code finding companyName works.
            ObjectNode price = result.putObject("price");
            price.put("symbol", symbol + ".NS");
            price.put("longName", symbol);
            price.put("currency", "INR");
            putRaw(price, "regularMarketPrice", ratios.get("Current Price"));
            putRaw(price, "marketCap", crToRupees(ratios.get("Market Cap")));
        }
        return result;
    }

    /**
     * Parse a Screener.in section table (e.g. profit-loss) into a {@code period → {label → value}}
     * map. Period keys are the {@code data-date-key} attribute on header cells; values are the
     * (Indian-numeric, comma-stripped) numbers in each row.
     */
    private static Map<String, Map<String, Double>> parseSection(Document doc, String sectionId) {
        Map<String, Map<String, Double>> out = new LinkedHashMap<>();
        Element section = doc.getElementById(sectionId);
        if (section == null) return out;
        Element table = section.selectFirst("table.data-table");
        if (table == null) return out;
        // Column headers: first <th> is the row label slot; rest carry data-date-key.
        Elements headerCells = table.select("thead th");
        java.util.List<String> periods = new java.util.ArrayList<>();
        for (int i = 1; i < headerCells.size(); i++) {
            String key = headerCells.get(i).attr("data-date-key");
            if (key.isBlank()) key = headerCells.get(i).text().trim();
            periods.add(key);
            out.put(key, new LinkedHashMap<>());
        }
        // Rows: first <td class="text"> has the label (often inside a <button>), rest are values.
        Elements rows = table.select("tbody tr");
        for (Element row : rows) {
            Elements cells = row.select("td");
            if (cells.size() < 2) continue;
            String label = cells.get(0).text().replace("+", "").trim();
            for (int i = 1; i < cells.size() && (i - 1) < periods.size(); i++) {
                Double v = parseScreenerNumber(cells.get(i).text().trim());
                if (v != null) out.get(periods.get(i - 1)).put(label, v);
            }
        }
        return out;
    }

    /** Top-ratios block: {@code <li class="flex flex-space-between">…</li>} pairs. */
    private static Map<String, Double> parseTopRatios(Document doc) {
        Map<String, Double> out = new LinkedHashMap<>();
        Element ul = doc.getElementById("top-ratios");
        if (ul == null) return out;
        Elements items = ul.select("li");
        for (Element li : items) {
            Element label = li.selectFirst(".name");
            Element value = li.selectFirst(".value");
            if (label == null || value == null) continue;
            String key = label.text().trim();
            // Value may contain ₹, "Cr.", "%" — strip and parse.
            String raw = value.text().replace("₹", "").replace("Cr.", "").replace(",", "").replace("%", "").trim();
            // Multiple sub-values (high/low) — take the first numeric token.
            String[] toks = raw.split("\\s+/\\s+|\\s+");
            for (String t : toks) {
                try { out.put(key, Double.parseDouble(t)); break; } catch (NumberFormatException ignored) {}
            }
        }
        return out;
    }

    /** Parse an Indian-format number like "1,234.56" or "(1,234)" (negative) → double. */
    private static Double parseScreenerNumber(String s) {
        if (s == null) return null;
        s = s.trim();
        if (s.isEmpty() || "-".equals(s)) return null;
        boolean neg = s.startsWith("(") && s.endsWith(")");
        if (neg) s = s.substring(1, s.length() - 1);
        s = s.replace(",", "").replace("%", "").trim();
        try { double v = Double.parseDouble(s); return neg ? -v : v; }
        catch (NumberFormatException e) { return null; }
    }

    /** Parse "2025-03-31" (data-date-key) → epoch-seconds. Falls back to 0 on unknown formats. */
    private static long parseDateEpoch(String periodKey) {
        if (periodKey == null || periodKey.isBlank()) return 0L;
        try {
            if (periodKey.matches("\\d{4}-\\d{2}-\\d{2}")) {
                return java.time.LocalDate.parse(periodKey).atStartOfDay(java.time.ZoneOffset.UTC).toEpochSecond();
            }
        } catch (Exception ignored) {}
        return 0L;
    }

    // ─── tiny put/convert helpers ────────────────────────────────────────────
    private static void putRaw(ObjectNode parent, String key, Double v) {
        if (v == null || v.isNaN()) return;
        parent.putObject(key).put("raw", v);
    }
    private static void putRawCr(ObjectNode parent, String key, Double crores) {
        if (crores == null || crores.isNaN()) return;
        parent.putObject(key).put("raw", crores * CRORE);
    }
    private static Double crToRupees(Double crores) {
        return crores == null ? null : crores * CRORE;
    }
    private static Double pctToFraction(Double pct) {
        return pct == null ? null : pct / 100.0;
    }
    private static Double sum(Double a, Double b) {
        if (a == null && b == null) return null;
        return (a == null ? 0 : a) + (b == null ? 0 : b);
    }
    private static Double diff(Double a, Double b) {
        if (a == null) return null;
        return a - (b == null ? 0 : b);
    }

    /* ─────────────────────────────────────────────────────────────────────── */
    /*  Historical OHLCV via Screener's undocumented chart API.                */
    /*                                                                         */
    /*  This is the 4th-tier fallback in HistoricalPriceTool's chain (Yahoo →  */
    /*  FMP → Stooq → here) for Indian tickers where Yahoo's v8 chart 429s     */
    /*  our datacenter IP and FMP's free quota is exhausted. Stooq locked      */
    /*  down free CSV access in 2026 (now requires API-key signup), so this    */
    /*  is effectively the ONLY working free historical source for .NS / .BO   */
    /*  from a server-class IP.                                                */
    /*                                                                         */
    /*  Source: https://www.screener.in/api/company/<ID>/chart/?q=Price-Volume */
    /*  Caveat: close-only (no O/H/L). Stub O=H=L=C so existing chart + TA     */
    /*  code paths still work — close-only is sufficient for SMA/EMA/RSI/MACD. */
    /* ─────────────────────────────────────────────────────────────────────── */

    /** Cache of ticker → Screener company_id. Looked up once per ticker, then reused. */
    private final Map<String, String> companyIdCache = new ConcurrentHashMap<>();
    private static final Pattern COMPANY_ID_RE = Pattern.compile("data-company-id=\"(\\d+)\"");

    /**
     * Fetch close-price + volume history from Screener.in. Returns ascending-by-date
     * {@link HistoricalPrice} list (O=H=L=C, volume from Screener's Volume dataset).
     * Returns {@code null} on any failure so the caller can propagate the original
     * Yahoo error to the user.
     *
     * @param ticker     Yahoo-style ticker (e.g. {@code FEDERALBNK.NS})
     * @param days       Lookback window in days. Screener caps practical depth at ~10y;
     *                   safe values: 30, 90, 180, 365, 1825, 3650.
     */
    public List<HistoricalPrice> historicalBars(String ticker, int days) {
        if (!isEnabled() || !isIndianTicker(ticker)) return null;
        String companyId = resolveCompanyId(ticker);
        if (companyId == null) return null;
        String url = "https://www.screener.in/api/company/" + companyId
                + "/chart/?q=Price-DMA50-DMA200-Volume&days=" + Math.max(7, Math.min(3650, days))
                + "&consolidated=true";
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                    .timeout(TIMEOUT)
                    .header("User-Agent", UA)
                    .header("Accept", "application/json")
                    .header("Referer", BASE + toScreenerSymbol(ticker) + "/consolidated/")
                    .GET().build();
            HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
            if (resp.statusCode() / 100 != 2) {
                LOG.debug("Screener chart HTTP {} for {}", resp.statusCode(), ticker);
                return null;
            }
            return parseChartJson(resp.body());
        } catch (Exception e) {
            LOG.debug("Screener historicalBars failed for {}: {}", ticker, e.toString());
            return null;
        }
    }

    /**
     * Look up Screener's internal numeric company_id for a ticker. Cached per-instance —
     * company_ids are stable, only re-fetched if the cached lookup later 404s on the
     * chart endpoint (currently never invalidated).
     */
    private String resolveCompanyId(String ticker) {
        String symbol = toScreenerSymbol(ticker);
        String cached = companyIdCache.get(symbol);
        if (cached != null) return cached;
        // Fetch the consolidated company page; company_id is embedded as
        // data-company-id="<N>" on the main container. Try /consolidated/ first
        // (most companies have it), fall back to standalone /.
        for (String suffix : new String[]{"/consolidated/", "/"}) {
            try {
                HttpRequest req = HttpRequest.newBuilder(URI.create(BASE + symbol + suffix))
                        .timeout(TIMEOUT)
                        .header("User-Agent", UA)
                        .header("Accept", "text/html,application/xhtml+xml")
                        .GET().build();
                HttpResponse<String> resp = http.send(req, BoundedHttp.ofString());
                if (resp.statusCode() / 100 != 2) continue;
                Matcher m = COMPANY_ID_RE.matcher(resp.body());
                if (m.find()) {
                    String id = m.group(1);
                    companyIdCache.put(symbol, id);
                    LOG.info("Screener company_id resolved: {} → {}", symbol, id);
                    return id;
                }
            } catch (Exception e) {
                LOG.debug("company_id lookup failed for {} @ {}: {}", symbol, suffix, e.toString());
            }
        }
        return null;
    }

    /**
     * Parse Screener's chart response into HistoricalPrice list. Shape:
     * <pre>{"datasets":[
     *   {"metric":"Price",  "values":[[date,closeStr], ...]},
     *   {"metric":"Volume", "values":[[date,volStr],   ...]},
     *   ... (DMA50, DMA200 unused — we compute these ourselves downstream)
     * ]}</pre>
     */
    private static List<HistoricalPrice> parseChartJson(String json) throws Exception {
        JsonNode root = MAPPER.readTree(json);
        JsonNode datasets = root.path("datasets");
        if (!datasets.isArray() || datasets.isEmpty()) return null;
        // Index by metric for the two we care about; ignore DMA series.
        Map<LocalDate, Double> priceByDate = new LinkedHashMap<>();
        Map<LocalDate, Long> volByDate = new LinkedHashMap<>();
        for (JsonNode ds : datasets) {
            String metric = ds.path("metric").asText("");
            JsonNode vals = ds.path("values");
            if (!vals.isArray()) continue;
            if ("Price".equalsIgnoreCase(metric)) {
                for (JsonNode pt : vals) {
                    if (!pt.isArray() || pt.size() < 2) continue;
                    LocalDate d = parseDateSafe(pt.get(0).asText(""));
                    double p = parseDoubleSafe(pt.get(1).asText(""));
                    if (d != null && !Double.isNaN(p)) priceByDate.put(d, p);
                }
            } else if ("Volume".equalsIgnoreCase(metric)) {
                for (JsonNode pt : vals) {
                    if (!pt.isArray() || pt.size() < 2) continue;
                    LocalDate d = parseDateSafe(pt.get(0).asText(""));
                    double v = parseDoubleSafe(pt.get(1).asText(""));
                    if (d != null && !Double.isNaN(v)) volByDate.put(d, (long) v);
                }
            }
        }
        if (priceByDate.isEmpty()) return null;
        List<HistoricalPrice> bars = new ArrayList<>(priceByDate.size());
        for (Map.Entry<LocalDate, Double> e : priceByDate.entrySet()) {
            double c = e.getValue();
            long v = volByDate.getOrDefault(e.getKey(), 0L);
            // Stub O=H=L=C — Screener's chart endpoint is close-only. Downstream
            // technical indicators (RSI/SMA/EMA/MACD) operate on close so this is fine;
            // candlestick rendering will collapse to line which is acceptable for a fallback.
            bars.add(new HistoricalPrice(e.getKey(), c, c, c, c, c, v));
        }
        bars.sort(Comparator.comparing(HistoricalPrice::date));
        return bars;
    }

    private static LocalDate parseDateSafe(String s) {
        if (s == null || s.isBlank()) return null;
        try { return LocalDate.parse(s.trim()); } catch (Exception e) { return null; }
    }

    private static double parseDoubleSafe(String s) {
        if (s == null) return Double.NaN;
        try { return Double.parseDouble(s.replace(",", "").trim()); } catch (Exception e) { return Double.NaN; }
    }
}
