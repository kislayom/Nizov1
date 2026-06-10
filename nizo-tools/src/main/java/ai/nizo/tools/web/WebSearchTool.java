package ai.nizo.tools.web;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.net.BoundedHttp;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Web search.
 *
 * <p>Provider chain (each one tried in order, fallback on error/empty):
 * <ol>
 *   <li><b>SearXNG</b> if {@code SEARXNG_BASE_URL} is set (recommended; self-hosted, no key)</li>
 *   <li><b>Brave Search API</b> if {@code BRAVE_API_KEY} is set</li>
 *   <li><b>Bing HTML</b> as no-key default (always available, stable selectors)</li>
 *   <li><b>DuckDuckGo HTML</b> as last-resort fallback (often bot-blocked → 202 anomaly page)</li>
 * </ol>
 *
 * <p><b>Why Bing first instead of DDG:</b> DuckDuckGo's html.duckduckgo.com and
 * lite.duckduckgo.com endpoints aggressively bot-detect server-side calls and serve a
 * 202 "anomaly modal" instead of results, regardless of User-Agent. Bing returns real
 * results with a Linux Firefox UA, no key needed. Verified May 2026 from our server box.
 */
public final class WebSearchTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(WebSearchTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** Real-browser UA — server-side scrapers using "compatible; bot/1.0" UAs trip
     *  bot defenses immediately. Linux Firefox blends in. */
    private static final String UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0";

    private final HttpClient http = SharedHttpClient.INSTANCE;

    private final SmartProxyClient smartProxy = new SmartProxyClient();

    @Override public String name() { return "web_search"; }

    @Override
    public String description() {
        return "Search the web for recent or factual information. Use for: current events, "
                + "real-world facts you're not sure about, prices, schedules, recent papers/news. "
                + "Returns a list of titles, URLs, and snippets — call web_fetch for full page content.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "query":  { "type": "string", "description": "Search query in natural language." },
                "limit":  { "type": "integer", "description": "Max results (default 5, max 10).",
                            "minimum": 1, "maximum": 10 }
              },
              "required": ["query"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String rawQuery = args.path("query").asText("").trim();
        String query = sanitizeQuery(rawQuery);
        if (query.isEmpty()) return ToolResult.error("query is required");
        int limit = clamp(args.path("limit").asInt(5), 1, 10);

        List<Result> results;
        try {
            results = searchPreferredProvider(query, limit);
        } catch (Exception e) {
            LOG.warn("web search failed: {}", e.toString());
            return ToolResult.error("search failed: " + e.getMessage());
        }
        if (results.isEmpty()) return ToolResult.ok("(no results)");

        StringBuilder sb = new StringBuilder();
        sb.append("Search results for: ").append(query).append("\n\n");
        for (int i = 0; i < results.size(); i++) {
            Result r = results.get(i);
            sb.append(i + 1).append(". **").append(r.title).append("**\n")
              .append("   ").append(r.url).append("\n");
            if (!r.snippet.isBlank()) {
                sb.append("   ").append(r.snippet.replaceAll("\\s+", " ").trim()).append("\n");
            }
            sb.append("\n");
        }
        return ToolResult.ok(sb.toString().trim());
    }

    private List<Result> searchPreferredProvider(String query, int limit) throws Exception {
        String searx = System.getenv("SEARXNG_BASE_URL");
        if (searx != null && !searx.isBlank()) {
            try {
                List<Result> r = searchSearxng(searx, query, limit);
                if (!r.isEmpty()) return r;
                LOG.info("SearXNG returned empty, falling back");
            } catch (Exception e) {
                LOG.warn("SearXNG failed: {}", e.toString());
            }
        }
        String braveKey = System.getenv("BRAVE_API_KEY");
        if (braveKey != null && !braveKey.isBlank()) {
            try {
                List<Result> r = searchBrave(braveKey, query, limit);
                if (!r.isEmpty()) return r;
                LOG.info("Brave returned empty, falling back");
            } catch (Exception e) {
                LOG.warn("Brave failed: {}", e.toString());
            }
        }
        // Bing — primary no-key fallback. Stable HTML, doesn't bot-block server scrapes.
        try {
            List<Result> r = searchBing(query, limit);
            if (!r.isEmpty()) return r;
            LOG.info("Bing returned empty, trying DDG");
        } catch (Exception e) {
            LOG.warn("Bing failed: {}", e.toString());
        }
        // DDG — flaky no-key fallback (often 202-blocked).
        try {
            List<Result> r = searchDuckDuckGoHtml(query, limit);
            if (!r.isEmpty()) return r;
            LOG.info("DDG returned empty, falling back to SmartProxy if enabled");
        } catch (Exception e) {
            LOG.warn("DDG failed: {}", e.toString());
        }
        // SmartProxy → scrape Bing through their fingerprint-rotated infra. Final layer.
        if (smartProxy.isEnabled()) {
            try {
                String bingUrl = "https://www.bing.com/search?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8)
                        + "&count=" + Math.max(limit, 10) + "&setLang=en&cc=US&setMkt=en-US";
                Optional<String> html = smartProxy.fetchHtml(bingUrl);
                if (html.isPresent()) return parseBingHtml(html.get(), bingUrl, limit);
            } catch (Exception e) {
                LOG.warn("SmartProxy fetch failed: {}", e.toString());
            }
        }
        return java.util.Collections.emptyList();
    }

    // ---- providers ----

    private List<Result> searchSearxng(String baseUrl, String query, int limit) throws Exception {
        String trimmed = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        String url = trimmed + "/search?format=json&safesearch=0&q=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept", "application/json")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("searxng HTTP " + resp.statusCode());
        }
        JsonNode root = MAPPER.readTree(resp.body());
        List<Result> out = new ArrayList<>();
        for (JsonNode r : root.path("results")) {
            if (out.size() >= limit) break;
            String title = r.path("title").asText("").trim();
            String u     = r.path("url").asText("").trim();
            String s     = r.path("content").asText("").trim();
            if (!u.isEmpty()) out.add(new Result(title, u, s));
        }
        return out;
    }

    private List<Result> searchBrave(String apiKey, String query, int limit) throws Exception {
        String url = "https://api.search.brave.com/res/v1/web/search?count=" + limit
                + "&q=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept", "application/json")
                        .header("X-Subscription-Token", apiKey)
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("brave HTTP " + resp.statusCode());
        }
        JsonNode root = MAPPER.readTree(resp.body());
        List<Result> out = new ArrayList<>();
        for (JsonNode r : root.path("web").path("results")) {
            if (out.size() >= limit) break;
            String title = r.path("title").asText("").trim();
            String u     = r.path("url").asText("").trim();
            String s     = r.path("description").asText("").trim();
            if (!u.isEmpty()) out.add(new Result(title, u, s));
        }
        return out;
    }

    /** Bing HTML scrape. No API key. Uses the public bing.com/search results page.
     *  Selectors target {@code li.b_algo > h2 > a} (title + URL) and
     *  {@code div.b_caption p} (snippet). Stable as of 2026-05.
     *  {@code &setLang=en&cc=US} forces English/US locale — without it, Bing localizes
     *  by source IP and serves German MOTOR-TALK or Indian results to non-US callers. */
    private List<Result> searchBing(String query, int limit) throws Exception {
        String url = "https://www.bing.com/search?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8)
                + "&count=" + Math.max(limit, 10)
                + "&setLang=en&cc=US&setMkt=en-US";
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept", "text/html,application/xhtml+xml")
                        .header("Accept-Language", "en-US,en;q=0.9")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("bing HTTP " + resp.statusCode());
        }
        return parseBingHtml(resp.body(), url, limit);
    }

    /** Shared Bing HTML parser — used by direct scrape AND SmartProxy fallback. */
    private List<Result> parseBingHtml(String html, String baseUrl, int limit) {
        Document doc = Jsoup.parse(html, baseUrl);
        Elements rows = doc.select("li.b_algo");
        List<Result> out = new ArrayList<>();
        for (Element row : rows) {
            if (out.size() >= limit) break;
            Element titleA = row.selectFirst("h2 > a");
            if (titleA == null) continue;
            String title = titleA.text().trim();
            String href  = titleA.absUrl("href");
            if (href.isEmpty()) href = titleA.attr("href");
            Element snipP = row.selectFirst("div.b_caption p");
            if (snipP == null) snipP = row.selectFirst("p");
            String snippet = snipP == null ? "" : snipP.text().trim();
            if (!href.isEmpty() && !title.isEmpty()) out.add(new Result(title, href, snippet));
        }
        if (out.isEmpty()) {
            LOG.warn("bing parse returned 0 results (htmlBytes={}, rows={})", html.length(), rows.size());
        }
        return out;
    }

    private List<Result> searchDuckDuckGoHtml(String query, int limit) throws Exception {
        String url = "https://html.duckduckgo.com/html/?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept-Language", "en-US,en;q=0.9")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("ddg HTTP " + resp.statusCode());
        }
        Document doc = Jsoup.parse(resp.body(), url);
        Elements rows = doc.select("div.result, div.web-result");
        List<Result> out = new ArrayList<>();
        for (Element row : rows) {
            if (out.size() >= limit) break;
            Element a = row.selectFirst("a.result__a, h2 a");
            if (a == null) continue;
            String title = a.text().trim();
            String href  = a.absUrl("href");
            // DDG redirects through /l/?uddg=...
            String real  = unwrapDdgRedirect(href);
            Element snippet = row.selectFirst(".result__snippet, .result__body");
            String s = snippet == null ? "" : snippet.text().trim();
            if (!real.isEmpty()) out.add(new Result(title, real, s));
        }
        return out;
    }

    private static String unwrapDdgRedirect(String href) {
        if (href == null) return "";
        int idx = href.indexOf("uddg=");
        if (idx < 0) return href;
        try {
            String enc = href.substring(idx + 5);
            int amp = enc.indexOf('&');
            if (amp > 0) enc = enc.substring(0, amp);
            return java.net.URLDecoder.decode(enc, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return href;
        }
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /**
     * Strip markdown that the model sometimes emits inside queries. Verified in May 2026 logs:
     * queries like {@code site:[reddit.com](http://reddit.com) AAPL} reach the search engine
     * verbatim with the brackets/parens, killing relevance.
     *
     * <p>Package-private for unit testing.
     */
    static String sanitizeQuery(String raw) {
        if (raw == null) return "";
        return raw
                .replaceAll("\\[([^\\]]+)\\]\\([^)]+\\)", "$1")  // [text](url) → text
                .replaceAll("\\s{2,}", " ")
                .trim();
    }

    private record Result(String title, String url, String snippet) {}
}
