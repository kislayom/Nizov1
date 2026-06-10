package ai.nizo.tools.web;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Wikipedia REST API summary lookup. Free, no key, well-curated.
 * Use for proper-noun / encyclopedic factual queries.
 */
public final class WikipediaTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String UA = "Mozilla/5.0 (compatible; NizoAgent/1.0; +https://kislay.dev)";
    private final HttpClient http = SharedHttpClient.INSTANCE;

    @Override public String name() { return "wikipedia"; }

    @Override
    public String description() {
        return "Look up an encyclopedic summary on Wikipedia. Use for: definitions, history, "
                + "biographies, places, scientific concepts. Returns the article extract and URL. "
                + "Prefer this over web_search for factual lookups about well-known topics.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "title": { "type": "string", "description": "Article title or topic. Spaces ok." },
                "lang":  { "type": "string", "description": "Wikipedia language code (default 'en')." }
              },
              "required": ["title"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String title = args.path("title").asText("").trim();
        if (title.isEmpty()) return ToolResult.error("title is required");
        String lang = args.path("lang").asText("en").trim();
        if (lang.isEmpty()) lang = "en";

        String url = "https://" + lang + ".wikipedia.org/api/rest_v1/page/summary/"
                + URLEncoder.encode(title.replace(' ', '_'), StandardCharsets.UTF_8);
        URI uri;
        try {
            uri = URI.create(url);
        } catch (IllegalArgumentException iae) {
            return ToolResult.error("invalid wikipedia URL: " + iae.getMessage());
        }
        try {
            SsrfGuard.assertSafe(uri);
        } catch (SecurityException se) {
            return ToolResult.error(se.getMessage());
        }
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(10))
                        .header("User-Agent", UA)
                        .header("Accept", "application/json")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() == 404) {
            return ToolResult.ok("No Wikipedia article titled '" + title + "'. Try a different phrasing or use web_search.");
        }
        if (resp.statusCode() / 100 != 2) {
            return ToolResult.error("wikipedia HTTP " + resp.statusCode());
        }

        JsonNode root = MAPPER.readTree(resp.body());
        String pageTitle = root.path("title").asText(title);
        String desc = root.path("description").asText("");
        String extract = root.path("extract").asText("");
        String pageUrl = root.path("content_urls").path("desktop").path("page").asText(
                "https://" + lang + ".wikipedia.org/wiki/" + URLEncoder.encode(title.replace(' ', '_'), StandardCharsets.UTF_8));

        StringBuilder sb = new StringBuilder();
        sb.append("# ").append(pageTitle).append("\n");
        if (!desc.isBlank()) sb.append("_").append(desc).append("_\n");
        sb.append("\n").append(extract).append("\n\n").append(pageUrl);
        return ToolResult.ok(sb.toString());
    }
}
