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
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Fetch a URL, strip nav/script/styles, return the readable main content as plain text/markdown.
 * Capped at ~12 KB so we don't blow the model's context.
 *
 * <p><b>Bot-bypass fallback:</b> when a direct fetch returns 403/429/5xx or a too-small
 * "anomaly modal" body (DDG, Cloudflare gates), we retry through {@link SmartProxyClient}
 * if it's enabled (env vars set). That gives us SmartProxy's fingerprint-rotated infra
 * for sites we can't reach directly. Free providers tried first to keep paid traffic low.
 *
 * <p><b>SSRF guard:</b> all URLs pass through {@link SsrfGuard} before we touch the network —
 * the LLM cannot trick us into fetching {@code 169.254.169.254} or internal RFC1918 hosts.
 */
public final class WebFetchTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(WebFetchTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** Real Firefox UA — the previous "compatible; NizoAgent/1.0" tripped Cloudflare. */
    private static final String UA = "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0";
    private static final int MAX_CHARS = 12_000;
    /** A response under this size is suspicious — usually a "JS required" or anomaly page. */
    private static final int SUSPICIOUS_BODY_THRESHOLD = 6_000;

    private final HttpClient http = SharedHttpClient.INSTANCE;

    private final SmartProxyClient smartProxy = new SmartProxyClient();

    @Override public String name() { return "web_fetch"; }

    @Override
    public String description() {
        return "Fetch a single URL and return its main readable content as text. Use after web_search "
                + "or when you have a specific URL the user mentioned. The result is truncated to ~12K chars.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "url":   { "type": "string", "description": "Absolute http(s) URL to fetch." },
                "max_chars": { "type": "integer", "description": "Optional override for output cap (default 12000)." }
              },
              "required": ["url"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String url = sanitizeUrl(args.path("url").asText(""));
        if (url.isEmpty()) return ToolResult.error("url is required");
        if (!(url.startsWith("http://") || url.startsWith("https://"))) {
            return ToolResult.error("url must be http(s)");
        }
        int max = args.path("max_chars").asInt(MAX_CHARS);
        if (max <= 0) max = MAX_CHARS;

        URI uri;
        try {
            uri = URI.create(url);
        } catch (IllegalArgumentException iae) {
            return ToolResult.error("invalid URL: " + iae.getMessage());
        }
        try {
            SsrfGuard.assertSafe(uri);
        } catch (SecurityException se) {
            return ToolResult.error(se.getMessage()
                    + ". This URL targets an internal/private address and won't be fetched.");
        }

        HttpResponse<String> resp = null;
        Exception directErr = null;
        try {
            resp = http.send(
                    HttpRequest.newBuilder(uri)
                            .timeout(Duration.ofSeconds(15))
                            .header("User-Agent", UA)
                            .header("Accept", "text/html,application/xhtml+xml")
                            .header("Accept-Language", "en-US,en;q=0.9")
                            .GET().build(),
                    BoundedHttp.ofString());
        } catch (Exception e) {
            directErr = e;
            LOG.warn("direct fetch failed {}: {}", url, e.toString());
        }

        // FIRST: detect a hard 404 (page genuinely doesn't exist) and short-circuit. The
        // LLM frequently invents URL patterns like "/apple-inc/aapl/financials/annual" that
        // aren't on the target site, then we waste 15s/req on a SmartProxy retry that ALSO
        // gets a 404. Catch this fast and tell the model to web_search instead of guessing.
        if (resp != null && resp.statusCode() == 404 && looksLikeRealNotFound(resp.body())) {
            LOG.info("hard 404 (real page-not-found, skipping SmartProxy retry): {}", url);
            return ToolResult.error(
                "404 — URL does not exist: " + url + ". " +
                "DO NOT retry this URL. DO NOT invent variant paths from training data. " +
                "Run web_search to find an actual URL on this site, then web_fetch that one. " +
                "Stop guessing URL templates."
            );
        }

        // Decide if direct fetch needs a SmartProxy retry: connection error, non-2xx,
        // OR suspiciously small body (anomaly/JS-required page).
        boolean directOk = resp != null
                && resp.statusCode() / 100 == 2
                && resp.body() != null
                && resp.body().length() >= SUSPICIOUS_BODY_THRESHOLD;

        if (!directOk && smartProxy.isEnabled()) {
            LOG.info("direct fetch underperformed (status={}, bodyLen={}, err={}) — retrying via SmartProxy",
                    resp == null ? -1 : resp.statusCode(),
                    resp == null || resp.body() == null ? -1 : resp.body().length(),
                    directErr == null ? "" : directErr.getMessage());
            var spHtml = smartProxy.fetchHtml(url);
            if (spHtml.isPresent()) {
                return renderHtml(url, "text/html (via smartproxy)", spHtml.get(), max);
            }
        }

        if (resp == null) {
            return ToolResult.error("fetch failed: " + (directErr == null ? "unknown" : directErr.getMessage())
                + ". Try a different URL — verify the page exists via web_search first.");
        }
        if (resp.statusCode() / 100 != 2) {
            // 404 reaches here when SmartProxy was tried and also failed (or wasn't enabled).
            // Either way, tell the LLM to find a real URL instead of retrying this one.
            String hint = (resp.statusCode() == 404)
                ? " — URL likely doesn't exist. Run web_search to find a real URL on this site; " +
                  "do not invent variant paths."
                : " — try a different source via web_search.";
            return ToolResult.error("HTTP " + resp.statusCode() + " from " + url + hint);
        }

        String contentType = resp.headers().firstValue("content-type").orElse("").toLowerCase();
        String body = resp.body();

        // For non-HTML, return raw (truncated).
        if (!contentType.contains("html")) {
            return ToolResult.ok(formatHeader(url, contentType) + "\n\n" + truncate(body, max));
        }
        return renderHtml(url, contentType, body, max);
    }

    /** Shared HTML cleanup → readable text. Used for both direct fetch and SmartProxy retries.
     *  Detects bot-protection block pages (Cloudflare, Akamai, captchas, etc.) and returns
     *  them as ERRORS — otherwise the model treats the block page as legitimate content and
     *  hallucinates an answer from "Please enable cookies. Sorry, you have been blocked." */
    private ToolResult renderHtml(String url, String contentType, String body, int max) {
        // Bot-block detection — these phrases are dead giveaways of CF/Akamai/captcha pages.
        // We check on the BODY (not parsed text) so we catch them even if Jsoup's pickMain
        // returns the visible part.
        String lc = body.toLowerCase();
        String blockReason = null;
        if      (lc.contains("attention required! | cloudflare"))           blockReason = "Cloudflare bot challenge";
        else if (lc.contains("checking your browser before accessing"))     blockReason = "Cloudflare interstitial";
        else if (lc.contains("just a moment...") && lc.contains("cloudflare")) blockReason = "Cloudflare wait page";
        else if (lc.contains("press &amp; hold to confirm you are a human")) blockReason = "PerimeterX captcha";
        else if (lc.contains("access denied") && lc.contains("akamai"))     blockReason = "Akamai block";
        else if (lc.contains("captcha-delivery.com"))                       blockReason = "DataDome captcha";
        else if (lc.contains("forbidden") && body.length() < 2000 && contentType.contains("html")) blockReason = "Forbidden (small HTML)";
        if (blockReason != null) {
            LOG.info("bot-block detected on {}: {}", url, blockReason);
            return ToolResult.error(blockReason + " on " + url
                    + ". Try a different source: Google cached version (cache:" + url
                    + ") · web.archive.org/web/" + url + " · or a different site entirely.");
        }

        Document doc = Jsoup.parse(body, url);
        doc.select("script, style, noscript, iframe, svg, nav, footer, header, form, aside, " +
                "[role=navigation], [aria-hidden=true], .cookie, .consent, .advertisement, .ads")
                .remove();
        Element main = pickMain(doc);
        String text = (main == null ? doc.body() : main).text().replaceAll("\\s+\n", "\n").trim();
        String title = doc.title().trim();
        StringBuilder out = new StringBuilder();
        if (!title.isEmpty()) out.append("# ").append(title).append("\n\n");
        out.append(formatHeader(url, contentType)).append("\n\n");
        out.append(truncate(text, max));
        return ToolResult.ok(out.toString());
    }

    private static Element pickMain(Document doc) {
        for (String sel : new String[]{
                "main", "article", "[role=main]",
                "#content", "#main", ".content", ".article",
                "div.markdown-body", "div.post"}) {
            Elements e = doc.select(sel);
            if (!e.isEmpty()) return e.first();
        }
        return null;
    }

    private static String formatHeader(String url, String contentType) {
        return "_source: " + url + " · " + contentType + "_";
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "\n\n…[truncated to " + max + " chars]";
    }

    /**
     * Clean up LLM-emitted URLs before handing to URI.create. Qwen3.6 sometimes
     * inserts a literal space where a slash should go (e.g. "https://www.gurufocus.com stock/AMZN"
     * — verified in May 2026 stock-analysis logs). Without this, every such URL
     * fails with IllegalArgumentException → SmartProxy 400 → wasted retry budget.
     *
     * <p>Heuristic: trim outer whitespace, strip wrapping quotes/brackets, then
     * if there's a space inside http(s)://host... replace it with a slash. We
     * prefer slash because the model is almost always missing a path separator,
     * not actually sending a query string with a space.
     */
    /**
     * Heuristic: is this 404 body a real "this page doesn't exist" page (vs. a 404 used
     * as a bot-block masquerade)? Real 404 pages from any normal site contain phrases like
     * "page not found", "not found", "doesn't exist", etc. Bot-block 404s tend to be empty
     * or contain "blocked" / "forbidden" / "captcha" markers (caught later in renderHtml).
     *
     * <p>Tightened May 2026: the previous {@code lc.contains("404")} matched any page that
     * mentioned "404" in passing (e.g. nav links, error code references). We now require a
     * 404 marker to appear in context — {@code "error 404"}, {@code "404 not found"},
     * {@code "http 404"}, or {@code "404 page"}.
     *
     * <p>Package-private for unit testing.
     */
    static boolean looksLikeRealNotFound(String body) {
        if (body == null) return false;
        if (body.length() > 10_000) return false; // anomaly pages are usually small
        String lc = body.toLowerCase();
        if (lc.contains("captcha") || lc.contains("blocked") || lc.contains("cloudflare")
                || lc.contains("akamai") || lc.contains("press &amp; hold")) return false;
        return lc.contains("page not found")
                || lc.contains("error 404")
                || lc.contains("404 not found")
                || lc.contains("http 404")
                || lc.contains("404 page")
                || lc.contains("err=404")
                || lc.contains("doesn't exist")
                || lc.contains("does not exist")
                || lc.contains("page you are looking for")
                || lc.contains("page you requested");
    }

    private static String sanitizeUrl(String raw) {
        if (raw == null) return "";
        String url = raw.trim();
        // Strip common wrappers the model accidentally includes.
        if (url.startsWith("<") && url.endsWith(">")) url = url.substring(1, url.length() - 1).trim();
        if ((url.startsWith("\"") && url.endsWith("\"")) ||
            (url.startsWith("'")  && url.endsWith("'"))) {
            url = url.substring(1, url.length() - 1).trim();
        }
        // Replace any internal whitespace with `/` — preserves path-like structure.
        // (This is only applied to the post-scheme portion to keep query strings safe;
        // we don't expect tools to legitimately need spaces in query values, but a
        // raw URL with a space is invalid per RFC 3986 anyway.)
        if (url.contains(" ") || url.contains("\t") || url.contains("\n")) {
            int schemeEnd = url.indexOf("://");
            if (schemeEnd > 0) {
                String scheme = url.substring(0, schemeEnd + 3);
                String rest = url.substring(schemeEnd + 3).replaceAll("\\s+", "/");
                // Collapse accidental double slashes from above (except after scheme).
                rest = rest.replaceAll("/+", "/");
                url = scheme + rest;
            } else {
                url = url.replaceAll("\\s+", "");
            }
        }
        return url;
    }
}
