package ai.nizo.tools.web;

import ai.nizo.api.tool.ToolResult;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests for {@link BrowserTool} against a stub HTTP "sidecar". The headline guarantees:
 * outbound navigation is SSRF-guarded BEFORE any request leaves; the rendered page state comes
 * back to the model; the session id is captured and reused across calls; a {@code needs_human}
 * refusal (secret field / commit-purchase) is surfaced as a human step, not an error; and a
 * missing sidecar fails with an actionable message.
 */
class BrowserToolTest {

    private HttpServer server;
    private String baseUrl;
    private final AtomicInteger hits = new AtomicInteger();
    private final AtomicReference<String> lastBody = new AtomicReference<>("");
    private volatile String response = "{\"ok\":true}";

    @BeforeEach
    void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/act", ex -> {
            hits.incrementAndGet();
            lastBody.set(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] out = response.getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, out.length);
            ex.getResponseBody().write(out);
            ex.close();
        });
        server.start();
        baseUrl = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterEach
    void stop() { if (server != null) server.stop(0); }

    @Test
    void gotoRendersAndCapturesSession() {
        response = "{\"ok\":true,\"sessionId\":\"sess-1\",\"url\":\"https://example.com\","
                + "\"title\":\"Example Domain\",\"text\":\"Hello rendered world\","
                + "\"links\":[{\"t\":\"More\",\"h\":\"https://example.com/more\"}]}";
        BrowserTool t = new BrowserTool(baseUrl);

        ToolResult r = t.execute("{\"action\":\"goto\",\"url\":\"https://example.com\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("Hello rendered world"), r.content());
        assertTrue(r.content().contains("Example Domain"), r.content());
        assertTrue(r.content().contains("example.com/more"), "links should be rendered: " + r.content());

        // Second call must reuse the captured session id.
        response = "{\"ok\":true,\"sessionId\":\"sess-1\",\"url\":\"https://example.com\",\"title\":\"x\",\"text\":\"y\"}";
        t.execute("{\"action\":\"read\"}");
        assertTrue(lastBody.get().contains("\"sessionId\":\"sess-1\""),
                "second call should send the captured session: " + lastBody.get());
    }

    @Test
    void blocksInternalGotoBeforeAnyRequest() {
        BrowserTool t = new BrowserTool(baseUrl);
        ToolResult r = t.execute("{\"action\":\"goto\",\"url\":\"http://169.254.169.254/latest/meta-data\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("block"), r.content());
        assertEquals(0, hits.get(), "SSRF-blocked navigation must never reach the sidecar");
    }

    @Test
    void blocksLoopbackGoto() {
        BrowserTool t = new BrowserTool(baseUrl);
        ToolResult r = t.execute("{\"action\":\"goto\",\"url\":\"http://127.0.0.1:8080/admin\"}");
        assertFalse(r.ok());
        assertEquals(0, hits.get());
    }

    @Test
    void needsHumanIsSurfacedNotErrored() {
        response = "{\"ok\":false,\"needs_human\":true,\"error\":\"refusing to click a commit-purchase control\"}";
        BrowserTool t = new BrowserTool(baseUrl);
        // session-less click still reaches the sidecar (no SSRF gate on click)
        ToolResult r = t.execute("{\"action\":\"click\",\"text\":\"Place order\"}");
        assertTrue(r.ok(), "a human-step refusal should be a normal result, not a failure: " + r.content());
        assertTrue(r.content().contains("HUMAN STEP REQUIRED"), r.content());
    }

    @Test
    void observeRendersIndexedElements() {
        response = "{\"ok\":true,\"sessionId\":\"s1\",\"snapshotVersion\":2,\"url\":\"https://x.com\",\"title\":\"X\","
                + "\"elements\":[{\"index\":0,\"role\":\"textbox\",\"name\":\"Search\",\"state\":\"required\"},"
                + "{\"index\":1,\"role\":\"button\",\"name\":\"Go\",\"state\":\"\"}]}";
        BrowserTool t = new BrowserTool(baseUrl);
        ToolResult r = t.execute("{\"action\":\"observe\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("snapshotVersion=2"), r.content());
        assertTrue(r.content().contains("[0] textbox \"Search\""), r.content());
        assertTrue(r.content().contains("[1] button \"Go\""), r.content());
    }

    @Test
    void clickByIndexForwardsIndexVersionAndShowsChange() {
        response = "{\"ok\":true,\"sessionId\":\"s1\",\"changed\":true,\"change_kind\":\"navigated\","
                + "\"url\":\"https://x.com/2\",\"title\":\"2\",\"elements\":[]}";
        BrowserTool t = new BrowserTool(baseUrl);
        ToolResult r = t.execute("{\"action\":\"click\",\"index\":5,\"snapshotVersion\":3}");
        assertTrue(r.ok(), r.content());
        assertTrue(lastBody.get().contains("\"index\":5"), lastBody.get());
        assertTrue(lastBody.get().contains("\"snapshotVersion\":3"), lastBody.get());
        assertTrue(r.content().contains("changed: true (navigated)"), r.content());
    }

    @Test
    void missingActionRejected() {
        BrowserTool t = new BrowserTool(baseUrl);
        ToolResult r = t.execute("{}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("action is required"), r.content());
    }

    @Test
    void sidecarDownIsActionable() {
        // Point at a closed port (stop the stub first).
        int port = server.getAddress().getPort();
        server.stop(0);
        server = null;
        BrowserTool t = new BrowserTool("http://127.0.0.1:" + port);
        ToolResult r = t.execute("{\"action\":\"goto\",\"url\":\"https://example.com\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("not reachable") || r.content().toLowerCase().contains("failed"),
                r.content());
    }
}
