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
 *   <li><b>Brave Search API</b> if {@code BRAVE_API_KEY} is set (best quality; free tier 2k/mo)</li>
 *   <li><b>Mojeek HTML</b> — no-key default. Independent UK crawler that does NOT
 *       bot-block home-server IPs (the only engine of 7 tested that still served
 *       organic results to our box, June 2026). Smaller index than Bing but real.</li>
 *   <li><b>Bing HTML</b> — kept in the chain in case the IP rehabilitates, but as of
 *       June 2026 Bing serves a challenge page (b_no, 0 results) to this box.</li>
 *   <li><b>DuckDuckGo HTML</b> — same story, serves the 202 "anomaly" page.</li>
 *   <li><b>SmartProxy → Bing</b> if creds configured (currently 403 — lapsed?).</li>
 * </ol>
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
        // Mojeek — primary no-key provider. Independent crawler, tolerant of
        // server-side callers (sole survivor of the June 2026 engine probe).
        try {
            List<Result> r = searchMojeek(query, limit);
            if (!r.isEmpty()) return r;
            LOG.info("Mojeek returned empty, trying Bing");
        } catch (Exception e) {
            LOG.warn("Mojeek failed: {}", e.toString());
        }
        // Bing — kept for the day this IP rehabilitates; currently serves a
        // challenge page (0 rows) to us.
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
        // ── Reliable keyless floor ──────────────────────────────────────────────────────────
        // Every engine above can be (and currently is) bot-blocked on this server's IP. These two
        // are different: official JSON APIs that do NOT bot-block server callers and need no key, so
        // they keep `web_search` from ever returning "(no results)" on a factual/encyclopedic query.
        // Lower coverage than full web search (instant answers + encyclopedia), so they run LAST.
        try {
            List<Result> r = searchDuckDuckGoInstant(query, limit);
            if (!r.isEmpty()) { LOG.info("served {} result(s) from DuckDuckGo Instant Answer floor", r.size()); return r; }
        } catch (Exception e) {
            LOG.warn("DDG instant-answer failed: {}", e.toString());
        }
        try {
            List<Result> r = searchWikipedia(query, limit);
            if (!r.isEmpty()) { LOG.info("served {} result(s) from Wikipedia floor", r.size()); return r; }
        } catch (Exception e) {
            LOG.warn("Wikipedia failed: {}", e.toString());
        }
        return java.util.Collections.emptyList();
    }

    /**
     * DuckDuckGo Instant Answer API — official keyless JSON endpoint (api.duckduckgo.com). Unlike the
     * HTML scrape it does not serve the "anomaly" wall to server IPs. Returns a topic abstract plus
     * related topics; good for definitional / entity queries, thin for long-tail. IP-agnostic.
     */
    List<Result> searchDuckDuckGoInstant(String query, int limit) throws Exception {
        String url = "https://api.duckduckgo.com/?format=json&no_html=1&no_redirect=1&skip_disambig=1&t=nizo&q="
                + URLEncoder.encode(query, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept", "application/json")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) throw new RuntimeException("ddg-ia HTTP " + resp.statusCode());
        return parseDuckDuckGoInstant(resp.body(), limit);
    }

    /** Parse the DDG Instant Answer JSON into results. Package-private for unit testing. */
    static List<Result> parseDuckDuckGoInstant(String json, int limit) throws Exception {
        JsonNode root = MAPPER.readTree(json);
        List<Result> out = new ArrayList<>();
        String abstractText = root.path("AbstractText").asText("").trim();
        String abstractUrl  = root.path("AbstractURL").asText("").trim();
        String heading      = root.path("Heading").asText("").trim();
        if (!abstractText.isEmpty() && !abstractUrl.isEmpty()) {
            out.add(new Result(heading.isEmpty() ? abstractText : heading, abstractUrl, abstractText));
        }
        // RelatedTopics: a mix of {Text, FirstURL} leaves and {Topics:[...]} groups. Flatten one level.
        collectDdgTopics(root.path("RelatedTopics"), out, limit);
        return out.size() > limit ? out.subList(0, limit) : out;
    }

    private static void collectDdgTopics(JsonNode topics, List<Result> out, int limit) {
        if (!topics.isArray()) return;
        for (JsonNode t : topics) {
            if (out.size() >= limit) return;
            if (t.has("Topics")) { collectDdgTopics(t.path("Topics"), out, limit); continue; }
            String text = t.path("Text").asText("").trim();
            String u    = t.path("FirstURL").asText("").trim();
            if (text.isEmpty() || u.isEmpty()) continue;
            // The leading clause before " - " is the entity name; keep the whole thing as snippet.
            int dash = text.indexOf(" - ");
            String title = dash > 0 ? text.substring(0, dash).trim() : text;
            out.add(new Result(title, u, text));
        }
    }

    /**
     * Wikipedia search via the MediaWiki API — keyless, never bot-blocks server callers, always up.
     * Excellent for the factual / encyclopedic slice of queries; the snippet HTML is stripped. The
     * absolute last resort so a factual question always gets a grounded, citable answer. IP-agnostic.
     */
    List<Result> searchWikipedia(String query, int limit) throws Exception {
        String url = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit="
                + Math.max(1, Math.min(limit, 10)) + "&srsearch=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        URI uri = URI.create(url);
        SsrfGuard.assertSafe(uri);
        HttpResponse<String> resp = http.send(
                HttpRequest.newBuilder(uri)
                        .timeout(Duration.ofSeconds(12))
                        .header("User-Agent", UA)
                        .header("Accept", "application/json")
                        .GET().build(),
                BoundedHttp.ofString());
        if (resp.statusCode() / 100 != 2) throw new RuntimeException("wikipedia HTTP " + resp.statusCode());
        return parseWikipedia(resp.body(), limit);
    }

    /** Parse MediaWiki search JSON into results. Package-private for unit testing. */
    static List<Result> parseWikipedia(String json, int limit) throws Exception {
        JsonNode root = MAPPER.readTree(json);
        List<Result> out = new ArrayList<>();
        for (JsonNode hit : root.path("query").path("search")) {
            if (out.size() >= limit) break;
            String title = hit.path("title").asText("").trim();
            if (title.isEmpty()) continue;
            String pageUrl = "https://en.wikipedia.org/wiki/"
                    + URLEncoder.encode(title.replace(' ', '_'), StandardCharsets.UTF_8).replace("%2F", "/");
            // The "snippet" field is HTML with <span class="searchmatch"> highlights — strip tags + entities.
            String snippet = hit.path("snippet").asText("")
                    .replaceAll("<[^>]+>", "").replace("&quot;", "\"").replace("&amp;", "&")
                    .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'").trim();
            out.add(new Result(title, pageUrl, snippet));
        }
        return out;
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

    /** Mojeek HTML scrape. No API key, no bot wall (verified June 2026). Selectors:
     *  {@code ul.results-standard li} → {@code h2 a.title} (title + direct href, no
     *  redirect unwrapping needed) + {@code p.s} snippet. */
    private List<Result> searchMojeek(String query, int limit) throws Exception {
        String url = "https://www.mojeek.com/search?q=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
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
            throw new RuntimeException("mojeek HTTP " + resp.statusCode());
        }
        Document doc = Jsoup.parse(resp.body(), url);
        Elements rows = doc.select("ul.results-standard li");
        List<Result> out = new ArrayList<>();
        for (Element row : rows) {
            if (out.size() >= limit) break;
            Element a = row.selectFirst("h2 a.title, h2 a");
            if (a == null) continue;
            String title = a.text().trim();
            String href  = a.absUrl("href");
            if (href.isEmpty()) href = a.attr("href");
            Element snipP = row.selectFirst("p.s");
            String snippet = snipP == null ? "" : snipP.text().trim();
            if (!href.isEmpty() && !title.isEmpty()) out.add(new Result(title, href, snippet));
        }
        if (out.isEmpty()) {
            LOG.warn("mojeek parse returned 0 results (htmlBytes={}, rows={})", resp.body().length(), rows.size());
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
