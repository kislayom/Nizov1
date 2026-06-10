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
 * Structured financial statements + key statistics for a public stock.
 *
 * <p>Pulls from Yahoo Finance's {@code quoteSummary} v10 endpoint. Returns the last 4 fiscal
 * years of income statement, balance sheet, and cash flow statement plus a snapshot of
 * valuation/profitability/leverage ratios. The LLM should prefer this tool over scraping
 * macrotrends/SEC HTML — it's an order of magnitude faster and never 404s on hallucinated paths.
 *
 * <p>The output JSON has a stable shape and can be embedded verbatim inside a
 * {@code chart-financials} fenced block — the front-end renders 3 collapsible tables (IS / BS /
 * CF) with QoQ growth highlighting + a key-stats card.
 */
public final class StockFundamentalsTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(StockFundamentalsTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String[] MODULES = {
            "incomeStatementHistory",
            "balanceSheetHistory",
            "cashflowStatementHistory",
            "defaultKeyStatistics",
            "summaryDetail",
            "financialData",
            "assetProfile",
            "price"
    };

    private final YahooQuoteSummary yahoo;
    private final ScreenerInClient screener;

    public StockFundamentalsTool(YahooQuoteSummary yahoo) {
        this(yahoo, new ScreenerInClient());
    }

    public StockFundamentalsTool(YahooQuoteSummary yahoo, ScreenerInClient screener) {
        this.yahoo = yahoo;
        this.screener = screener;
    }

    @Override public String name() { return "stock_fundamentals"; }

    @Override
    public String description() {
        return "Structured financial statements + key statistics for a public stock — last 4 "
                + "fiscal years of income statement, balance sheet, cash flow plus current "
                + "valuation/profitability/leverage ratios. Use this INSTEAD of scraping "
                + "macrotrends/yahoo HTML — it's faster and never 404s. Output goes verbatim "
                + "into a `chart-financials` fenced block on the front-end.";
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

            // Tier-2 fallback for Indian-listed equities: Yahoo + FMP coverage drops
            // sharply below Nifty-100. Two failure modes we both want to handle:
            //   (a) incomeStatementHistory is empty entirely
            //   (b) rows exist but operatingIncome / grossProfit / ebit are all null
            //       (Yahoo HTML scrape often delivers only revenue + netIncome for
            //       mid/small-cap Indian names, leaving the rest as "—")
            JsonNode incRows = result.path("incomeStatementHistory").path("incomeStatementHistory");
            boolean yahooSparse;
            if (!incRows.isArray() || incRows.size() == 0) {
                yahooSparse = true;
            } else {
                // Inspect the latest row — if operatingIncome AND grossProfit AND ebit are
                // all null/missing, the row is just a stub and we should still try Screener.
                JsonNode latest = incRows.get(0);
                boolean opIncMissing = latest.path("operatingIncome").path("raw").asLong(0) == 0;
                boolean grossMissing = latest.path("grossProfit").path("raw").asLong(0) == 0;
                boolean ebitMissing  = latest.path("ebit").path("raw").asLong(0) == 0;
                yahooSparse = opIncMissing && grossMissing && ebitMissing;
            }
            if (yahooSparse && ScreenerInClient.isIndianTicker(ticker) && screener.isEnabled()) {
                JsonNode scr = screener.fetch(ticker);
                if (scr != null && scr.path("incomeStatementHistory")
                                       .path("incomeStatementHistory").size() > 0) {
                    LOG.info("stock_fundamentals: Yahoo+FMP sparse for {}, using Screener.in fallback", ticker);
                    result = scr;
                }
            }

            ObjectNode out = MAPPER.createObjectNode();
            out.put("ticker", ticker);
            out.put("source", "yahoo_quoteSummary_v10");
            out.put("asOf", LocalDate.now(ZoneOffset.UTC).toString());

            JsonNode price = result.path("price");
            out.put("currency", price.path("currency").asText(""));
            out.put("longName", price.path("longName").asText(""));
            out.put("exchange", price.path("exchangeName").asText(""));

            JsonNode profile = result.path("assetProfile");
            ObjectNode profOut = out.putObject("profile");
            profOut.put("sector", profile.path("sector").asText(""));
            profOut.put("industry", profile.path("industry").asText(""));
            profOut.put("country", profile.path("country").asText(""));
            profOut.put("fullTimeEmployees", profile.path("fullTimeEmployees").asLong(0));
            profOut.put("website", profile.path("website").asText(""));
            String summary = profile.path("longBusinessSummary").asText("");
            if (summary.length() > 600) summary = summary.substring(0, 597) + "...";
            profOut.put("summary", summary);

            ObjectNode annual = out.putObject("annual");
            annual.set("incomeStatement", parseIncomeStatement(result.path("incomeStatementHistory").path("incomeStatementHistory")));
            annual.set("balanceSheet",    parseBalanceSheet(result.path("balanceSheetHistory").path("balanceSheetStatements")));
            annual.set("cashFlow",        parseCashFlow(result.path("cashflowStatementHistory").path("cashflowStatements")));

            out.set("keyStats", parseKeyStats(
                    result.path("defaultKeyStatistics"),
                    result.path("summaryDetail"),
                    result.path("financialData"),
                    result.path("price")));

            out.set("ratios", parseRatios(
                    result.path("financialData"),
                    result.path("defaultKeyStatistics"),
                    result.path("balanceSheetHistory").path("balanceSheetStatements")));

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("stock_fundamentals failed: {}", e.toString());
            return ToolResult.error("stock_fundamentals failed: " + e.getMessage());
        }
    }

    private static ArrayNode parseIncomeStatement(JsonNode arr) {
        ArrayNode out = MAPPER.createArrayNode();
        if (!arr.isArray()) return out;
        // Yahoo orders newest-first; we emit oldest-first so the chart reads left→right.
        for (int i = arr.size() - 1; i >= 0; i--) {
            JsonNode r = arr.get(i);
            ObjectNode row = out.addObject();
            row.put("endDate", endDate(r));
            row.put("revenue",          YahooQuoteSummary.rawLong(r.path("totalRevenue")));
            row.put("costOfRevenue",    YahooQuoteSummary.rawLong(r.path("costOfRevenue")));
            row.put("grossProfit",      YahooQuoteSummary.rawLong(r.path("grossProfit")));
            row.put("operatingExpenses", YahooQuoteSummary.rawLong(r.path("totalOperatingExpenses")));
            row.put("operatingIncome",  YahooQuoteSummary.rawLong(r.path("operatingIncome")));
            row.put("ebit",             YahooQuoteSummary.rawLong(r.path("ebit")));
            row.put("interestExpense",  YahooQuoteSummary.rawLong(r.path("interestExpense")));
            row.put("incomeTax",        YahooQuoteSummary.rawLong(r.path("incomeTaxExpense")));
            row.put("netIncome",        YahooQuoteSummary.rawLong(r.path("netIncome")));
            row.put("eps",              YahooQuoteSummary.rawDouble(r.path("dilutedEPS")));
        }
        return out;
    }

    private static ArrayNode parseBalanceSheet(JsonNode arr) {
        ArrayNode out = MAPPER.createArrayNode();
        if (!arr.isArray()) return out;
        for (int i = arr.size() - 1; i >= 0; i--) {
            JsonNode r = arr.get(i);
            ObjectNode row = out.addObject();
            row.put("endDate", endDate(r));
            row.put("totalAssets",       YahooQuoteSummary.rawLong(r.path("totalAssets")));
            row.put("currentAssets",     YahooQuoteSummary.rawLong(r.path("totalCurrentAssets")));
            row.put("cash",              YahooQuoteSummary.rawLong(r.path("cash")));
            row.put("shortTermInvestments", YahooQuoteSummary.rawLong(r.path("shortTermInvestments")));
            row.put("totalLiabilities",  YahooQuoteSummary.rawLong(r.path("totalLiab")));
            row.put("currentLiabilities", YahooQuoteSummary.rawLong(r.path("totalCurrentLiabilities")));
            row.put("longTermDebt",      YahooQuoteSummary.rawLong(r.path("longTermDebt")));
            row.put("totalEquity",       YahooQuoteSummary.rawLong(r.path("totalStockholderEquity")));
            row.put("retainedEarnings",  YahooQuoteSummary.rawLong(r.path("retainedEarnings")));
        }
        return out;
    }

    private static ArrayNode parseCashFlow(JsonNode arr) {
        ArrayNode out = MAPPER.createArrayNode();
        if (!arr.isArray()) return out;
        for (int i = arr.size() - 1; i >= 0; i--) {
            JsonNode r = arr.get(i);
            ObjectNode row = out.addObject();
            row.put("endDate", endDate(r));
            long ocf  = YahooQuoteSummary.rawLong(r.path("totalCashFromOperatingActivities"));
            long capex = YahooQuoteSummary.rawLong(r.path("capitalExpenditures"));  // negative in Yahoo
            row.put("operatingCashFlow", ocf);
            row.put("capitalExpenditures", capex);
            row.put("freeCashFlow",        ocf + capex);  // capex is signed -ve
            row.put("dividendsPaid",       YahooQuoteSummary.rawLong(r.path("dividendsPaid")));
            row.put("repurchaseStock",     YahooQuoteSummary.rawLong(r.path("repurchaseOfStock")));
            row.put("issuanceStock",       YahooQuoteSummary.rawLong(r.path("issuanceOfStock")));
            row.put("netBorrowings",       YahooQuoteSummary.rawLong(r.path("netBorrowings")));
            row.put("netCashFromInvesting", YahooQuoteSummary.rawLong(r.path("totalCashflowsFromInvestingActivities")));
            row.put("netCashFromFinancing", YahooQuoteSummary.rawLong(r.path("totalCashFromFinancingActivities")));
        }
        return out;
    }

    private static ObjectNode parseKeyStats(JsonNode dks, JsonNode summary, JsonNode fd, JsonNode price) {
        ObjectNode k = MAPPER.createObjectNode();
        k.put("marketCap",       YahooQuoteSummary.rawLong(price.path("marketCap")));
        k.put("enterpriseValue", YahooQuoteSummary.rawLong(dks.path("enterpriseValue")));
        k.put("trailingPE",      YahooQuoteSummary.rawDouble(summary.path("trailingPE")));
        k.put("forwardPE",       YahooQuoteSummary.rawDouble(summary.path("forwardPE")));
        k.put("pegRatio",        YahooQuoteSummary.rawDouble(dks.path("pegRatio")));
        k.put("priceToBook",     YahooQuoteSummary.rawDouble(dks.path("priceToBook")));
        k.put("priceToSales",    YahooQuoteSummary.rawDouble(summary.path("priceToSalesTrailing12Months")));
        k.put("evToEbitda",      YahooQuoteSummary.rawDouble(dks.path("enterpriseToEbitda")));
        k.put("evToRevenue",     YahooQuoteSummary.rawDouble(dks.path("enterpriseToRevenue")));
        k.put("dividendYield",   YahooQuoteSummary.rawDouble(summary.path("dividendYield")));
        k.put("payoutRatio",     YahooQuoteSummary.rawDouble(summary.path("payoutRatio")));
        k.put("beta",            YahooQuoteSummary.rawDouble(summary.path("beta")));
        k.put("sharesOutstanding", YahooQuoteSummary.rawLong(dks.path("sharesOutstanding")));
        k.put("floatShares",     YahooQuoteSummary.rawLong(dks.path("floatShares")));
        k.put("shortRatio",      YahooQuoteSummary.rawDouble(dks.path("shortRatio")));
        k.put("shortPercentOfFloat", YahooQuoteSummary.rawDouble(dks.path("shortPercentOfFloat")));
        k.put("week52High",      YahooQuoteSummary.rawDouble(summary.path("fiftyTwoWeekHigh")));
        k.put("week52Low",       YahooQuoteSummary.rawDouble(summary.path("fiftyTwoWeekLow")));
        k.put("volume",          YahooQuoteSummary.rawLong(summary.path("volume")));
        k.put("avgVolume",       YahooQuoteSummary.rawLong(summary.path("averageVolume")));
        // FCF per Yahoo's calc is in financialData
        k.put("freeCashFlowTTM", YahooQuoteSummary.rawLong(fd.path("freeCashflow")));
        k.put("operatingCashFlowTTM", YahooQuoteSummary.rawLong(fd.path("operatingCashflow")));
        k.put("totalCash",        YahooQuoteSummary.rawLong(fd.path("totalCash")));
        k.put("totalDebt",        YahooQuoteSummary.rawLong(fd.path("totalDebt")));
        return k;
    }

    private static ObjectNode parseRatios(JsonNode fd, JsonNode dks, JsonNode bsArr) {
        ObjectNode r = MAPPER.createObjectNode();
        r.put("currentRatio",   YahooQuoteSummary.rawDouble(fd.path("currentRatio")));
        r.put("quickRatio",     YahooQuoteSummary.rawDouble(fd.path("quickRatio")));
        r.put("debtToEquity",   YahooQuoteSummary.rawDouble(fd.path("debtToEquity")));
        r.put("returnOnEquity", YahooQuoteSummary.rawDouble(fd.path("returnOnEquity")));
        r.put("returnOnAssets", YahooQuoteSummary.rawDouble(fd.path("returnOnAssets")));
        // ROCE — canonical Indian-equity profitability metric, sourced from Screener.in
        // for .NS/.BO tickers. Yahoo doesn't publish it; null for US/EU names.
        r.put("returnOnCapitalEmployed", YahooQuoteSummary.rawDouble(fd.path("returnOnCapitalEmployed")));
        r.put("grossMargin",    YahooQuoteSummary.rawDouble(fd.path("grossMargins")));
        r.put("operatingMargin", YahooQuoteSummary.rawDouble(fd.path("operatingMargins")));
        r.put("profitMargin",   YahooQuoteSummary.rawDouble(fd.path("profitMargins")));
        r.put("ebitdaMargin",   YahooQuoteSummary.rawDouble(fd.path("ebitdaMargins")));
        r.put("revenueGrowth",  YahooQuoteSummary.rawDouble(fd.path("revenueGrowth")));
        r.put("earningsGrowth", YahooQuoteSummary.rawDouble(fd.path("earningsGrowth")));
        r.put("revenueQuarterlyGrowth", YahooQuoteSummary.rawDouble(fd.path("revenueQuarterlyGrowth")));
        r.put("earningsQuarterlyGrowth", YahooQuoteSummary.rawDouble(fd.path("earningsQuarterlyGrowth")));
        return r;
    }

    /** Yahoo's nested {raw, fmt} for a fiscal-year-end date — we emit ISO yyyy-MM-dd. */
    private static String endDate(JsonNode statement) {
        JsonNode d = statement.path("endDate");
        long raw = YahooQuoteSummary.rawLong(d);
        if (raw <= 0) return "";
        return LocalDate.ofInstant(Instant.ofEpochSecond(raw), ZoneOffset.UTC).toString();
    }
}
