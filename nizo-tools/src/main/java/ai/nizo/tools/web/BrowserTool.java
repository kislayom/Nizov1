package ai.nizo.tools.web;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.tools.net.SharedHttpClient;
import ai.nizo.tools.net.SsrfGuard;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;

/**
 * Drive a real headless browser — the capability {@code web_fetch} lacks. It renders
 * JavaScript-heavy pages, clicks, fills forms, and reads dynamic content across a multi-step
 * session, by talking to the Playwright {@code browser_sidecar.py} over loopback.
 *
 * <p>This is what unlocks app-like sites (Coles-style cart assembly, JS dashboards, multi-step
 * flows). It deliberately does NOT cross two lines, which stay with the human:
 * <ul>
 *   <li><b>secrets</b> — the sidecar refuses to type into password / payment fields;</li>
 *   <li><b>committing money</b> — the sidecar refuses to click "place order / pay now / confirm
 *       purchase". The agent assembles the cart and navigates to checkout; the human pays.</li>
 * </ul>
 *
 * <p>Outbound navigation is SSRF-guarded (loopback / RFC1918 / cloud-metadata blocked), same as the
 * other web tools. The browser session is held in this tool (single-user) and reused across calls;
 * {@code goto} with no session opens a fresh one.
 *
 * <p>Endpoint: {@code NIZO_BROWSER_URL} (default {@code http://127.0.0.1:7781}). If the sidecar is
 * not running the tool returns an actionable error rather than throwing.
 */
public final class BrowserTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(BrowserTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP = SharedHttpClient.INSTANCE;

    private final String sidecar;
    /** Single-user v1: one live browser session, reused across tool calls. */
    private volatile String sessionId;

    public BrowserTool() {
        this(envOr("NIZO_BROWSER_URL", "http://127.0.0.1:7781"));
    }

    BrowserTool(String sidecarUrl) {
        this.sidecar = sidecarUrl.replaceAll("/+$", "");
    }

    @Override public String name() { return "browser"; }

    @Override
    public String description() {
        return "Drive a real headless web browser for JS-heavy sites and multi-step web tasks that "
                + "web_fetch cannot do — it renders JavaScript, clicks, fills forms, and reads dynamic "
                + "content, keeping a session across calls. On goto it privacy-first dismisses cookie/"
                + "consent banners and returns the page's INTERACTIVE CONTROLS (buttons/inputs with "
                + "selectors) so you can target a search box or button precisely. Use it for app-like "
                + "sites (e.g. building a grocery cart), pages needing rendering, or login-gated flows. "
                + "It will NOT type passwords or payment details, and will NOT place an order / pay — it "
                + "surfaces those for the human. Actions: goto {url}; read; click {selector|text}; "
                + "type {selector,text,submit?}; wait {selector?}; dismiss; back; close.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "action":   { "type": "string", "enum": ["goto","read","click","type","wait","dismiss","back","close"],
                              "description": "What to do in the browser. 'wait' waits for {selector} (or ~1.5s); 'dismiss' re-attempts cookie/consent dismissal." },
                "url":      { "type": "string", "description": "For goto: the URL to open." },
                "selector": { "type": "string", "description": "CSS selector for click/type (preferred)." },
                "text":     { "type": "string", "description": "For click: visible link/button text. For type: the text to enter." },
                "submit":   { "type": "boolean", "description": "For type: press Enter after filling (default false)." }
              },
              "required": ["action"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        JsonNode a;
        try { a = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson); }
        catch (Exception e) { return ToolResult.error("bad arguments JSON: " + e.getMessage()); }

        String action = a.path("action").asText("").toLowerCase().trim();
        if (action.isEmpty()) return ToolResult.error("action is required (goto|read|click|type|back|close)");

        // SSRF-guard outbound navigation — never let the browser be pointed at internal hosts.
        if (action.equals("goto")) {
            String url = a.path("url").asText("");
            if (url.isBlank()) return ToolResult.error("goto requires a url");
            try { SsrfGuard.assertSafe(URI.create(url)); }
            catch (IllegalArgumentException e) { return ToolResult.error("blocked URL: " + e.getMessage()); }
            catch (Exception e) { return ToolResult.error("invalid url: " + e.getMessage()); }
        }

        // Build the sidecar request payload.
        var payload = new java.util.HashMap<String, Object>();
        payload.put("action", action);
        if (sessionId != null) payload.put("sessionId", sessionId);
        for (String f : new String[]{"url", "selector", "text"}) {
            if (a.hasNonNull(f)) payload.put(f, a.get(f).asText());
        }
        if (a.path("submit").asBoolean(false)) payload.put("submit", true);

        JsonNode resp;
        try {
            String body = MAPPER.writeValueAsString(payload);
            HttpRequest req = HttpRequest.newBuilder(URI.create(sidecar + "/act"))
                    .timeout(Duration.ofSeconds(90))
                    .header("Content-Type", "application/json")
                    // Force HTTP/1.1: the shared client defaults to HTTP/2, and a cleartext h2c
                    // upgrade against the HTTP/1.1 uvicorn sidecar silently DROPS the POST body
                    // (server saw body=b'' -> 422). The local sidecar is HTTP/1.1 only.
                    .version(HttpClient.Version.HTTP_1_1)
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> r = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
            if (r.statusCode() / 100 != 2)
                return ToolResult.error("browser sidecar HTTP " + r.statusCode() + ": " + brief(r.body()));
            resp = MAPPER.readTree(r.body());
        } catch (java.net.ConnectException e) {
            return ToolResult.error("browser sidecar not reachable at " + sidecar
                    + " — is the nizo-browser service running? (" + e.getMessage() + ")");
        } catch (Exception e) {
            return ToolResult.error("browser request failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }

        if (resp.hasNonNull("sessionId")) sessionId = resp.get("sessionId").asText();

        if (!resp.path("ok").asBoolean(false)) {
            String err = resp.path("error").asText("unknown browser error");
            if (resp.path("needs_human").asBoolean(false)) {
                return ToolResult.ok("[browser] HUMAN STEP REQUIRED — " + err
                        + "\nAssemble everything up to this point and hand the final step to the user.");
            }
            return ToolResult.error("[browser] " + err);
        }

        return new ToolResult(true, render(action, resp));
    }

    private static String render(String action, JsonNode r) {
        StringBuilder sb = new StringBuilder();
        sb.append("[browser ").append(action).append("] ").append(r.path("title").asText("")).append('\n');
        sb.append("URL: ").append(r.path("url").asText("")).append('\n');
        if (r.hasNonNull("note")) sb.append("note: ").append(r.path("note").asText("")).append('\n');
        String text = r.path("text").asText("");
        if (!text.isBlank()) sb.append("---\n").append(text).append('\n');
        JsonNode controls = r.path("controls");
        if (controls.isArray() && controls.size() > 0) {
            sb.append("--- interactive controls (target sel= with click/type):\n");
            int n = 0;
            for (JsonNode c : controls) {
                if (n++ >= 25) break;
                String tag = c.path("tag").asText(""), type = c.path("type").asText("");
                String label = c.path("label").asText(""), sel = c.path("sel").asText("");
                sb.append("  • ").append(tag);
                if (!type.isBlank()) sb.append('/').append(type);
                if (!label.isBlank()) sb.append("  \"").append(label).append('"');
                if (!sel.isBlank()) sb.append("  sel=").append(sel);
                sb.append('\n');
            }
        }
        JsonNode links = r.path("links");
        if (links.isArray() && links.size() > 0) {
            sb.append("--- links:\n");
            int n = 0;
            for (JsonNode l : links) {
                if (n++ >= 15) break;
                sb.append("  • ").append(l.path("t").asText("")).append("  →  ").append(l.path("h").asText("")).append('\n');
            }
        }
        return sb.toString();
    }

    private static String brief(String s) {
        if (s == null) return "";
        s = s.strip();
        return s.length() > 200 ? s.substring(0, 200) + "…" : s;
    }

    private static String envOr(String k, String d) {
        String v = System.getenv(k);
        return (v == null || v.isBlank()) ? d : v.trim();
    }
}
