package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.Instant;

/**
 * Live equity quote via Yahoo Finance's public {@code v8/finance/chart} endpoint. No API key,
 * no scraping — Yahoo serves this JSON to anyone with a sane User-Agent.
 *
 * <p>Accepts standard tickers ({@code AAPL}, {@code MSFT}) plus exchange-suffixed forms used
 * outside the US:
 * <ul>
 *   <li>{@code .NS} — NSE India  ({@code HDFCBANK.NS})</li>
 *   <li>{@code .BO} — BSE India  ({@code INFY.BO})</li>
 *   <li>{@code .L}  — London     ({@code BARC.L})</li>
 *   <li>{@code .T}  — Tokyo      ({@code 7203.T})</li>
 *   <li>{@code .HK} — Hong Kong  ({@code 0700.HK})</li>
 *   <li>{@code .DE} — Frankfurt  ({@code SAP.DE})</li>
 *   <li>{@code .PA} — Paris      ({@code MC.PA})</li>
 * </ul>
 *
 * <p>Returns a clean structured JSON record (price, change, day range, 52-week range, currency,
 * exchange) — easy for the LLM to reason about and quote in a report.
 */
public final class StockQuoteTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .version(HttpClient.Version.HTTP_2)
            .build();

    @Override public String name() { return "stock_quote"; }

    @Override
    public String description() {
        return "Live stock quote — current price, day range, 52-week range, volume, currency. "
                + "Accepts US tickers (AAPL, MSFT) or exchange-suffixed (HDFCBANK.NS, BARC.L, "
                + "7203.T, 0700.HK, SAP.DE). Backed by Yahoo Finance's free public JSON.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "ticker": { "type": "string", "description": "Ticker symbol like AAPL or HDFCBANK.NS" }
              },
              "required": ["ticker"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String ticker = args.path("ticker").asText("").trim().toUpperCase();
        if (ticker.isBlank()) return ToolResult.error("ticker is required");
        // Light sanitization — symbols are letters / digits / dot / dash. Also allow an
        // optional leading caret for Yahoo-style index tickers (^NSEI, ^GSPC, ^AXJO).
        if (!ticker.matches("\\^?[A-Z0-9._-]{1,16}")) {
            return ToolResult.error("invalid ticker: " + ticker);
        }

        // Yahoo serves the same data on query1/query2; round-robin on 429 dramatically reduces
        // the rate-limit hit-rate. (Saw 429s in practice with a single host.)
        String[] hosts = { "query1.finance.yahoo.com", "query2.finance.yahoo.com" };
        String enc = java.net.URLEncoder.encode(ticker, java.nio.charset.StandardCharsets.UTF_8);

        HttpResponse<String> resp = null;
        String lastErr = "";
        for (int attempt = 0; attempt < hosts.length; attempt++) {
            String url = "https://" + hosts[attempt] + "/v8/finance/chart/" + enc
                    + "?interval=1d&range=5d";

            HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                    .timeout(Duration.ofSeconds(15))
                    .header("Accept", "application/json")
                    .header("Accept-Language", "en-US,en;q=0.9")
                    // The simple "Nizo/1.0" UA is less fingerprinted than a fake browser UA and
                    // doesn't trip Yahoo's "ban Mozilla-bots" rule.
                    .header("User-Agent", "Nizo/1.0 (+local agent)")
                    .GET().build();
            try {
                resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
            } catch (java.net.http.HttpTimeoutException te) {
                lastErr = "Yahoo timed out (15s)";
                continue;
            } catch (Exception ex) {
                lastErr = "network error: " + ex.getMessage();
                continue;
            }

            int sc = resp.statusCode();
            if (sc == 200) break;
            if (sc == 404) return ToolResult.error("ticker not found: " + ticker);
            if (sc == 429 || sc >= 500) {
                // Rate-limited or upstream hiccup — try the other host
                lastErr = "HTTP " + sc;
                resp = null;
                // small backoff between hosts so we don't hammer
                try { Thread.sleep(150); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
                continue;
            }
            // Other 4xx — body has the explanation
            return ToolResult.error("Yahoo HTTP " + sc + ": " + abbreviate(resp.body(), 240));
        }

        if (resp == null) {
            return ToolResult.error("Yahoo unavailable across all hosts: " + lastErr
                    + ". You can retry, or use http_json directly.");
        }

        JsonNode root = MAPPER.readTree(resp.body());
        JsonNode chartErr = root.path("chart").path("error");
        if (chartErr.isObject() && !chartErr.isEmpty()) {
            return ToolResult.error("Yahoo: " + chartErr.path("description").asText("unknown"));
        }
        JsonNode result = root.path("chart").path("result").path(0);
        if (result.isMissingNode() || result.isNull()) {
            return ToolResult.error("no data returned for " + ticker);
        }

        JsonNode meta = result.path("meta");

        double price       = meta.path("regularMarketPrice").asDouble();
        double prevClose   = meta.path("chartPreviousClose").asDouble(meta.path("previousClose").asDouble());
        double change      = price - prevClose;
        double changePct   = prevClose != 0 ? (change / prevClose) * 100.0 : 0.0;

        ObjectNode out = MAPPER.createObjectNode();
        out.put("symbol",        meta.path("symbol").asText(ticker));
        out.put("longName",      meta.path("longName").asText(meta.path("shortName").asText("")));
        out.put("exchange",      meta.path("fullExchangeName").asText(meta.path("exchangeName").asText("")));
        out.put("instrumentType", meta.path("instrumentType").asText("EQUITY"));
        out.put("currency",      meta.path("currency").asText(""));
        out.put("marketState",   meta.path("marketState").asText(""));     // PRE / REGULAR / POST / CLOSED
        out.put("price",         round2(price));
        out.put("previousClose", round2(prevClose));
        out.put("change",        round2(change));
        out.put("changePercent", round2(changePct));
        out.put("dayHigh",       round2(meta.path("regularMarketDayHigh").asDouble(0)));
        out.put("dayLow",        round2(meta.path("regularMarketDayLow").asDouble(0)));
        out.put("dayVolume",     meta.path("regularMarketVolume").asLong(0));
        out.put("week52High",    round2(meta.path("fiftyTwoWeekHigh").asDouble(0)));
        out.put("week52Low",     round2(meta.path("fiftyTwoWeekLow").asDouble(0)));
        long ts = meta.path("regularMarketTime").asLong(Instant.now().getEpochSecond());
        out.put("dataAt",        Instant.ofEpochSecond(ts).toString());

        return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
    private static String abbreviate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }
}
