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
 * Insider trading activity for a public stock.
 *
 * <p>Three modules combined:
 * <ul>
 *   <li>{@code insiderTransactions} — last ~100 individual filings (Form 4 entries on
 *       Yahoo)</li>
 *   <li>{@code netSharePurchaseActivity} — aggregate net buy/sell over the last 6 months
 *       (the "are insiders net buyers or net sellers" question)</li>
 *   <li>{@code insiderHolders} — current top 10 insider holders by position size</li>
 * </ul>
 *
 * <p>Output is consumed by the {@code chart-insider} fenced block on the front-end which
 * renders a timeline of buys (green) vs sells (red) sized by USD value, plus a net-activity
 * pill (Net Buying / Net Selling).
 */
public final class InsiderActivityTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(InsiderActivityTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String[] MODULES = {
            "insiderTransactions",
            "netSharePurchaseActivity",
            "insiderHolders"
    };

    private final YahooQuoteSummary yahoo;
    private final FinnhubClient finnhub;
    private final NseIndiaClient nseIndia;

    public InsiderActivityTool(YahooQuoteSummary yahoo) {
        this(yahoo, new FinnhubClient(), new NseIndiaClient());
    }

    public InsiderActivityTool(YahooQuoteSummary yahoo, FinnhubClient finnhub, NseIndiaClient nseIndia) {
        this.yahoo = yahoo;
        this.finnhub = finnhub;
        this.nseIndia = nseIndia;
    }

    @Override public String name() { return "stock_insider_activity"; }

    @Override
    public String description() {
        return "Insider trading activity for a public stock — last ~100 transactions (buys/sells "
                + "by execs and directors), 6-month net-activity summary, and top current insider "
                + "holders. Output goes verbatim into a `chart-insider` fenced block "
                + "(timeline + net-activity pill + holders table).";
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

            // Net 6-month activity
            JsonNode net = result.path("netSharePurchaseActivity");
            ObjectNode netOut = out.putObject("netActivity");
            netOut.put("period",            net.path("period").asText("6m"));
            long buyCount  = YahooQuoteSummary.rawLong(net.path("buyInfoCount"));
            long sellCount = YahooQuoteSummary.rawLong(net.path("sellInfoCount"));
            long buyShares  = YahooQuoteSummary.rawLong(net.path("buyInfoShares"));
            long sellShares = YahooQuoteSummary.rawLong(net.path("sellInfoShares"));
            long netShares  = YahooQuoteSummary.rawLong(net.path("netInfoShares"));
            netOut.put("buyTransactions",  buyCount);
            netOut.put("sellTransactions", sellCount);
            netOut.put("buyShares",        buyShares);
            netOut.put("sellShares",       sellShares);
            netOut.put("netShares",        netShares);
            netOut.put("netPercentInsiderShares",
                    YahooQuoteSummary.rawDouble(net.path("netPercentInsiderShares")));
            netOut.put("totalInsiderShares",
                    YahooQuoteSummary.rawLong(net.path("totalInsiderShares")));

            String trend;
            if (buyCount + sellCount == 0) {
                trend = "no activity";
            } else if (netShares > 0) {
                trend = "net buying";
            } else if (netShares < 0) {
                trend = "net selling";
            } else {
                trend = "balanced";
            }
            netOut.put("trend", trend);

            // Individual transactions (most recent first; cap at 50 for payload size)
            JsonNode tx = result.path("insiderTransactions").path("transactions");
            ArrayNode txArr = out.putArray("transactions");
            boolean wroteFromYahoo = false;
            if (tx.isArray() && tx.size() > 0) {
                int n = Math.min(50, tx.size());
                for (int i = 0; i < n; i++) {
                    JsonNode r = tx.get(i);
                    ObjectNode row = txArr.addObject();
                    long epoch = YahooQuoteSummary.rawLong(r.path("startDate"));
                    row.put("date", epoch > 0
                            ? LocalDate.ofInstant(Instant.ofEpochSecond(epoch), ZoneOffset.UTC).toString()
                            : "");
                    row.put("name",      r.path("filerName").asText(""));
                    row.put("title",     r.path("filerRelation").asText(""));
                    String txText = r.path("transactionText").asText("");
                    row.put("description", txText);
                    row.put("action",    classifyAction(txText, r.path("moneyText").asText("")));
                    row.put("shares",    YahooQuoteSummary.rawLong(r.path("shares")));
                    row.put("value",     YahooQuoteSummary.rawLong(r.path("value")));
                    row.put("ownership", r.path("ownership").asText(""));   // "D" direct, "I" indirect
                }
                wroteFromYahoo = true;
            }
            // Tier-2.5 fallback (Indian equities only): NSE India publishes SEBI Form-D/F
            // filings for free at /api/corporates-pit. This is the AUTHORITATIVE source
            // for insider trading on NSE-listed companies — Yahoo / Finnhub / FMP all
            // have spotty Indian coverage. NSE prefers fresh transactions for the last
            // ~6 months which matches the chart-insider widget's display window.
            if (!wroteFromYahoo && NseIndiaClient.isIndianTicker(ticker) && nseIndia.isEnabled()) {
                JsonNode nse = nseIndia.insiderTrading(ticker);
                JsonNode data = nse == null ? null : nse.path("data");
                if (data != null && data.isArray() && data.size() > 0) {
                    int n = Math.min(50, data.size());
                    long bShares = 0, sShares = 0;
                    long bCount = 0, sCount = 0;
                    for (int i = 0; i < n; i++) {
                        JsonNode r = data.get(i);
                        ObjectNode row = txArr.addObject();
                        // NSE date format: "10-Dec-2024 20:01" → take just the date.
                        String rawDate = r.path("date").asText("");
                        String date = rawDate;
                        int sp = date.indexOf(' ');
                        if (sp > 0) date = date.substring(0, sp);
                        row.put("date", date);
                        row.put("name", r.path("acqName").asText(""));
                        row.put("title", r.path("personCategory").asText(""));
                        String txType = r.path("tdpTransactionType").asText("");      // "Buy" / "Sell"
                        String mode = r.path("acqMode").asText("");                   // "Market Purchase" / "Allotment"
                        String action = txType.isEmpty()
                                ? (mode.toLowerCase().contains("sale") ? "Sale" : "Acquisition")
                                : (txType.equalsIgnoreCase("buy") ? "Buy" : "Sale");
                        row.put("description", action + (mode.isEmpty() ? "" : " (" + mode + ")"));
                        row.put("action", action);
                        // NSE shares come as a comma-separated string.
                        long shares = parseIntlLong(r.path("secAcq").asText(""));
                        long value = parseIntlLong(r.path("secVal").asText(""));
                        row.put("shares", shares);
                        row.put("value", value);
                        row.put("ownership", "");
                        if ("Buy".equalsIgnoreCase(action) || "Acquisition".equalsIgnoreCase(action)) {
                            bShares += shares; bCount++;
                        } else {
                            sShares += shares; sCount++;
                        }
                    }
                    // Back-fill netActivity if Yahoo didn't populate one.
                    JsonNode netExisting = out.path("netActivity");
                    if (netExisting.path("buyTransactions").asLong(0) == 0
                            && netExisting.path("sellTransactions").asLong(0) == 0) {
                        ObjectNode netOutNse = (ObjectNode) netExisting;
                        netOutNse.put("buyTransactions", bCount);
                        netOutNse.put("sellTransactions", sCount);
                        netOutNse.put("buyShares", bShares);
                        netOutNse.put("sellShares", sShares);
                        netOutNse.put("netShares", bShares - sShares);
                        String trendNse = (bShares > sShares) ? "net buying"
                                        : (sShares > bShares) ? "net selling" : "balanced";
                        netOutNse.put("trend", trendNse);
                    }
                    out.put("source", "yahoo_quoteSummary_v10+nse_india_insider");
                    wroteFromYahoo = true;   // mark as covered so Finnhub branch doesn't run
                }
            }

            // Tier-3 fallback: Yahoo + FMP returned no transactions → try Finnhub.
            // Finnhub's /stock/insider-transactions returns {data: [{name, share, change,
            // filingDate, transactionDate, transactionCode, transactionPrice}, ...]}.
            // 'change' is signed (negative = sell, positive = buy). 'transactionCode' uses
            // SEC Form 4 letter codes: P=buy, S=sell, A=grant, M=option exercise, G=gift, ...
            if (!wroteFromYahoo && finnhub.isEnabled()) {
                JsonNode fhResp = finnhub.insiderTransactions(ticker);
                JsonNode fhData = fhResp == null ? null : fhResp.path("data");
                if (fhData != null && fhData.isArray() && fhData.size() > 0) {
                    int n = Math.min(50, fhData.size());
                    long bShares = 0, sShares = 0;
                    long bCount = 0, sCount = 0;
                    for (int i = 0; i < n; i++) {
                        JsonNode r = fhData.get(i);
                        ObjectNode row = txArr.addObject();
                        row.put("date",      r.path("transactionDate").asText(""));
                        row.put("name",      r.path("name").asText(""));
                        row.put("title",     "");
                        String code = r.path("transactionCode").asText("");
                        long change = r.path("change").asLong(0);
                        double price = r.path("transactionPrice").asDouble(0);
                        String action = switch (code) {
                            case "P" -> "Buy"; case "S" -> "Sale"; case "A" -> "Grant";
                            case "M" -> "Option Exercise"; case "G" -> "Gift"; case "F" -> "Tax Withholding";
                            default -> change >= 0 ? "Acquisition" : "Disposition";
                        };
                        row.put("description", action + " (Form 4 code " + code + ")");
                        row.put("action",    action);
                        row.put("shares",    Math.abs(change));
                        row.put("value",     Math.round(Math.abs(change) * price));
                        row.put("ownership", "");
                        if (change > 0) { bShares += change; bCount++; }
                        else if (change < 0) { sShares += -change; sCount++; }
                    }
                    // Back-fill netActivity if Yahoo didn't provide one.
                    JsonNode netExisting = out.path("netActivity");
                    if (netExisting.path("buyTransactions").asLong(0) == 0
                            && netExisting.path("sellTransactions").asLong(0) == 0) {
                        ObjectNode netOutFh = (ObjectNode) netExisting;
                        netOutFh.put("buyTransactions", bCount);
                        netOutFh.put("sellTransactions", sCount);
                        netOutFh.put("buyShares", bShares);
                        netOutFh.put("sellShares", sShares);
                        netOutFh.put("netShares", bShares - sShares);
                        String fhTrend = (bShares > sShares) ? "net buying"
                                       : (sShares > bShares) ? "net selling" : "balanced";
                        netOutFh.put("trend", fhTrend);
                    }
                    out.put("source", "yahoo_quoteSummary_v10+finnhub_insider");
                }
            }

            // Top current insider holders
            JsonNode holders = result.path("insiderHolders").path("holders");
            ArrayNode holdArr = out.putArray("topHolders");
            if (holders.isArray()) {
                int n = Math.min(10, holders.size());
                for (int i = 0; i < n; i++) {
                    JsonNode r = holders.get(i);
                    ObjectNode row = holdArr.addObject();
                    row.put("name",      r.path("name").asText(""));
                    row.put("relation",  r.path("relation").asText(""));
                    row.put("position",  YahooQuoteSummary.rawLong(r.path("positionDirect")));
                    row.put("transactionDescription", r.path("latestTransDate").path("fmt").asText(""));
                }
            }

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("stock_insider_activity failed: {}", e.toString());
            return ToolResult.error("stock_insider_activity failed: " + e.getMessage());
        }
    }

    /** Parse a comma-grouped integer ("1,23,456" or "1,234,567") → long. Returns 0 on garbage. */
    private static long parseIntlLong(String s) {
        if (s == null || s.isBlank()) return 0L;
        try {
            return Long.parseLong(s.replace(",", "").trim());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    /** Yahoo encodes BUY/SELL in a free-text field — heuristic classification. */
    private static String classifyAction(String txText, String moneyText) {
        if (txText == null) txText = "";
        String lower = txText.toLowerCase();
        if (lower.contains("sale") || lower.contains("sell") || lower.contains("disposition")) {
            return "Sale";
        }
        if (lower.contains("purchase") || lower.contains("buy") || lower.contains("acquisition")) {
            return "Buy";
        }
        if (lower.contains("option") || lower.contains("award") || lower.contains("grant")) {
            return "Grant";
        }
        if (lower.contains("conversion")) {
            return "Conversion";
        }
        // Last-ditch: if there's a value on the row, infer Sale (insiders rarely report
        // pure-Buy as 'unknown' to Yahoo). Otherwise Other.
        return moneyText != null && !moneyText.isBlank() ? "Sale" : "Other";
    }
}
