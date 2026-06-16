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
        return "Drive a real headless web browser to complete multi-step tasks on any site (search, "
                + "pick a result, fill forms, build a cart) — what web_fetch can't. WORKFLOW: call "
                + "`observe` to get a numbered list of interactive elements ([index] role \"name\" state) "
                + "plus a snapshotVersion, then act BY INDEX: click {index}, type {index,text,submit?}. "
                + "Always pass the snapshotVersion you observed; if the page changed you'll get a "
                + "'stale, re-observe' error — call observe again. Each action reports changed/change_kind "
                + "so you know whether it did something. For lazy lists use `scroll`; to wait out a spinner "
                + "use wait {selector,state:hidden}; if the DOM is unreadable use `screenshot_marks` then "
                + "image_analyze the PNG (the box numbers match the indices). goto privacy-dismisses "
                + "cookie banners. It will NOT type passwords/payment or click place-order/pay — those "
                + "stay with the human. Actions: observe; goto {url}; click {index|selector|text}; "
                + "type {index|selector,text,submit?,sequential?}; scroll {selector?}; wait {selector,state?}; "
                + "screenshot; screenshot_marks; dismiss; read; back; close.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "action":   { "type": "string", "enum": ["observe","goto","read","click","type","scroll","wait","dismiss","screenshot","screenshot_marks","back","close"],
                              "description": "observe = numbered interactive elements + snapshotVersion (call before acting); then click/type BY INDEX. 'wait' {selector,state}; 'scroll' loads lazy lists; 'screenshot_marks' = numbered overlay for image_analyze." },
                "url":      { "type": "string", "description": "For goto: the URL to open." },
                "index":    { "type": "integer", "description": "Target element index from the latest observe (preferred over selector)." },
                "snapshotVersion": { "type": "integer", "description": "The snapshotVersion observe returned — pass it with click/type so a re-render fails safe." },
                "selector": { "type": "string", "description": "CSS selector (fallback for click/type; for wait/scroll the element to wait-for/scroll-to)." },
                "text":     { "type": "string", "description": "For click(by text): visible label. For type: the text to enter." },
                "submit":   { "type": "boolean", "description": "For type: press Enter after filling (default false)." },
                "sequential": { "type": "boolean", "description": "For type: per-key typing (use for autocomplete/typeahead fields)." },
                "state":    { "type": "string", "enum": ["visible","hidden","attached","detached"], "description": "For wait: element state to wait for (default visible)." }
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
        for (String f : new String[]{"url", "selector", "text", "state"}) {
            if (a.hasNonNull(f)) payload.put(f, a.get(f).asText());
        }
        if (a.path("index").isNumber()) payload.put("index", a.path("index").asInt());
        if (a.path("snapshotVersion").isNumber()) payload.put("snapshotVersion", a.path("snapshotVersion").asInt());
        if (a.path("submit").asBoolean(false)) payload.put("submit", true);
        if (a.path("sequential").asBoolean(false)) payload.put("sequential", true);

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
        if (r.hasNonNull("screenshot")) {
            String shot = r.path("screenshot").asText();
            sb.append("screenshot saved: ").append(shot)
              .append("  (view it with image_analyze path=\"").append(shot).append("\")\n");
        }
        if (r.has("changed")) {
            sb.append("changed: ").append(r.path("changed").asBoolean())
              .append(" (").append(r.path("change_kind").asText("?")).append(")\n");
        }
        JsonNode elements = r.path("elements");
        if (elements.isArray() && elements.size() > 0) {
            sb.append("snapshotVersion=").append(r.path("snapshotVersion").asInt())
              .append("  — act by [index] and pass this snapshotVersion:\n");
            int n = 0;
            for (JsonNode e : elements) {
                if (n++ >= 80) break;
                sb.append("  [").append(e.path("index").asInt()).append("] ").append(e.path("role").asText(""));
                String nm = e.path("name").asText("");  if (!nm.isBlank()) sb.append(" \"").append(nm).append('"');
                String stt = e.path("state").asText(""); if (!stt.isBlank()) sb.append("  ").append(stt);
                sb.append('\n');
            }
        }
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
