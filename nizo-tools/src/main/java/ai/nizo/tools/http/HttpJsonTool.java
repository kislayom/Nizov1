package ai.nizo.tools.http;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;

/**
 * Generic JSON HTTP tool — usable by ANY API consumer (weather, crypto, GitHub, internal services,
 * stock data, …). Distinct from {@code web_fetch} which returns HTML/text and is meant for
 * human-readable pages.
 *
 * <p>Reusability is the design goal: it's intentionally not coupled to any one provider. Pass
 * a URL, optional headers (auth tokens go here), optional JSON body for POST/PUT, get back a
 * pretty-printed JSON response or a clean error.
 *
 * <p>Hard caps for safety:
 * <ul>
 *   <li>Response truncated above {@code max_response_kb} (default 256 KB) so the model isn't fed a
 *       megabyte of paginated nonsense.</li>
 *   <li>Total request timeout 30s — fast-fail rather than tying up an iteration.</li>
 *   <li>Only http:// and https://; the agent never gets to file:// or jar:// schemes.</li>
 * </ul>
 */
public final class HttpJsonTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** One client shared across all tools. Connection pooling is built-in. */
    private static final HttpClient HTTP = SharedHttpClient.INSTANCE;

    @Override public String name() { return "http_json"; }

    @Override
    public String description() {
        return "HTTP request to a JSON API. Returns the response body pretty-printed as JSON. "
                + "Use for ANY REST or public JSON endpoint — weather, crypto, stocks, GitHub, "
                + "internal APIs. For HTML pages use web_fetch instead. "
                + "Pass auth tokens via the 'headers' parameter (e.g. {\"Authorization\": \"Bearer ...\"}).";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "url":     { "type": "string",  "description": "Full http(s) URL" },
                "method":  { "type": "string",  "enum": ["GET","POST","PUT","DELETE","PATCH"], "default": "GET" },
                "headers": { "type": "object",  "description": "Optional request headers (e.g. Authorization)" },
                "body":    { "type": "object",  "description": "Optional JSON body for POST/PUT/PATCH" },
                "max_response_kb": { "type": "integer", "default": 256, "description": "Truncate larger responses" }
              },
              "required": ["url"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String url = args.path("url").asText("").trim();
        if (url.isBlank()) return ToolResult.error("url is required");
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            return ToolResult.error("url must start with http:// or https://");
        }
        String method = args.path("method").asText("GET").toUpperCase();
        int maxBytes = Math.max(1, args.path("max_response_kb").asInt(256)) * 1024;

        HttpRequest.Builder rb = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(30))
                .header("Accept", "application/json")
                .header("User-Agent", "Nizo/1.0 (+local agent)");

        // Custom headers (auth tokens go here)
        if (args.path("headers").isObject()) {
            Iterator<Map.Entry<String, JsonNode>> it = args.path("headers").fields();
            while (it.hasNext()) {
                Map.Entry<String, JsonNode> e = it.next();
                if (e.getValue().isTextual()) {
                    rb.header(e.getKey(), e.getValue().asText());
                }
            }
        }

        // Method + body
        switch (method) {
            case "GET"    -> rb.GET();
            case "DELETE" -> rb.DELETE();
            case "POST", "PUT", "PATCH" -> {
                String body = args.path("body").isMissingNode() || args.path("body").isNull()
                        ? "" : MAPPER.writeValueAsString(args.path("body"));
                rb.method(method, HttpRequest.BodyPublishers.ofString(body))
                  .header("Content-Type", "application/json");
            }
            default -> { return ToolResult.error("unsupported method: " + method); }
        }

        HttpResponse<String> resp;
        try {
            resp = HTTP.send(rb.build(), HttpResponse.BodyHandlers.ofString());
        } catch (java.net.http.HttpTimeoutException te) {
            return ToolResult.error("request timed out after 30s: " + url);
        } catch (Exception ex) {
            return ToolResult.error("request failed: " + ex.getMessage());
        }

        String body = resp.body() == null ? "" : resp.body();
        boolean truncated = false;
        if (body.length() > maxBytes) {
            body = body.substring(0, maxBytes);
            truncated = true;
        }

        // Pretty-print if it's JSON; otherwise return raw (some APIs return text/plain on errors)
        String formatted;
        try {
            JsonNode parsed = MAPPER.readTree(body);
            formatted = MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(parsed);
        } catch (Exception e) {
            formatted = body;
        }
        if (truncated) formatted += "\n\n[response truncated to " + (maxBytes / 1024) + " KB]";

        String header = "HTTP " + resp.statusCode() + " " + method + " " + url + "\n---\n";
        if (resp.statusCode() / 100 != 2) {
            return ToolResult.error(header + formatted);
        }
        return ToolResult.ok(header + formatted);
    }
}
