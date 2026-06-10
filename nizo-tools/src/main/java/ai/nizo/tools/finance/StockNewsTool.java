package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Company news via the Finnhub API — replaces the scrape-the-web path that the
 * news analyst used to rely on (StockTitan 403s, WSJ DataDome, Bing/DDG empty
 * results were burning 5+ minutes per stock run, and when everything failed the
 * LLM wrote "news" from training data).
 *
 * <p>Output is a compact markdown list (date · headline · source · one-line
 * summary · url), newest first, deduped by headline. Bounded to fit inside the
 * sub-agent tool-result budget (~4 KB after truncation).
 *
 * <p>Coverage note: Finnhub's free tier is strong on US listings, spottier on
 * NSE/BSE. When the response is empty or the client is disabled, the result
 * says so explicitly and tells the model to fall back to {@code web_search} —
 * the skill prompt mirrors this contract.
 */
public final class StockNewsTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(StockNewsTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final DateTimeFormatter DAY = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private static final int DEFAULT_MONTHS = 3;
    private static final int MAX_MONTHS = 12;
    private static final int DEFAULT_LIMIT = 15;
    private static final int MAX_LIMIT = 30;
    private static final int SUMMARY_CHARS = 160;

    private final FinnhubClient finnhub;

    public StockNewsTool() { this(new FinnhubClient()); }

    public StockNewsTool(FinnhubClient finnhub) { this.finnhub = finnhub; }

    @Override public String name() { return "stock_news"; }

    @Override
    public String description() {
        return "Recent company news headlines from the Finnhub API (no scraping, no bot-blocks). "
                + "Returns date · headline · source · summary · url, newest first. Use this FIRST "
                + "for any 'what happened with X' / news-analyst work; only fall back to web_search "
                + "if this returns no items (e.g. some NSE/BSE tickers).";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "ticker": { "type": "string", "description": "Ticker like AAPL, MSFT, RELIANCE.NS" },
                "months": { "type": "integer", "description": "Look-back window in months (default 3, max 12)" },
                "limit":  { "type": "integer", "description": "Max headlines to return (default 15, max 30)" }
              },
              "required": ["ticker"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            String ticker = args.path("ticker").asText("").trim().toUpperCase(Locale.ROOT);
            if (ticker.isEmpty()) return ToolResult.error("ticker is required");
            int months = clamp(args.path("months").asInt(DEFAULT_MONTHS), 1, MAX_MONTHS);
            int limit  = clamp(args.path("limit").asInt(DEFAULT_LIMIT), 1, MAX_LIMIT);

            if (!finnhub.isEnabled()) {
                return ToolResult.ok("stock_news unavailable (FINNHUB_TOKEN not configured). "
                        + "Fall back to web_search for news on " + ticker + ".");
            }

            LocalDate to = LocalDate.now(ZoneOffset.UTC);
            LocalDate from = to.minusMonths(months);
            JsonNode items = finnhub.companyNews(ticker, DAY.format(from), DAY.format(to));

            if (items == null || !items.isArray() || items.isEmpty()) {
                return ToolResult.ok("No API news found for " + ticker + " in the last " + months
                        + " months (coverage gap — common for some NSE/BSE listings). "
                        + "Fall back to web_search.");
            }

            StringBuilder sb = new StringBuilder(4096);
            sb.append("News for ").append(ticker)
              .append(" — last ").append(months).append(" months, ")
              .append("source: Finnhub API (fetched ").append(DAY.format(to)).append(")\n\n");

            Set<String> seen = new HashSet<>();
            int written = 0;
            for (JsonNode it : items) {
                if (written >= limit) break;
                String headline = it.path("headline").asText("").trim();
                if (headline.isEmpty()) continue;
                // Dedupe near-identical syndicated copies by normalized headline.
                String key = headline.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", " ").trim();
                if (!seen.add(key)) continue;

                long epochS = it.path("datetime").asLong(0);
                String day = epochS > 0
                        ? DAY.format(Instant.ofEpochSecond(epochS).atOffset(ZoneOffset.UTC).toLocalDate())
                        : "????-??-??";
                String source  = it.path("source").asText("?");
                String summary = it.path("summary").asText("").trim().replaceAll("\\s+", " ");
                if (summary.length() > SUMMARY_CHARS) summary = summary.substring(0, SUMMARY_CHARS) + "…";
                String url = it.path("url").asText("");

                sb.append("- [").append(day).append("] **").append(headline).append("** (").append(source).append(")");
                if (!summary.isEmpty() && !summary.equalsIgnoreCase(headline)) {
                    sb.append("\n  ").append(summary);
                }
                if (!url.isEmpty()) sb.append("\n  ").append(url);
                sb.append('\n');
                written++;
            }
            sb.append("\n(").append(written).append(" of ").append(items.size())
              .append(" items shown; deduped, newest first)");

            LOG.info("stock_news {}: {} items from Finnhub ({} after dedupe/limit)",
                    ticker, items.size(), written);
            return ToolResult.ok(sb.toString());
        } catch (Exception e) {
            LOG.warn("stock_news failed: {}", e.toString());
            return ToolResult.error("stock_news failed: " + e.getMessage() + ". Fall back to web_search.");
        }
    }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
}
