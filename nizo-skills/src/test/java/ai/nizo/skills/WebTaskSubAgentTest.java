package ai.nizo.skills;

import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.llm.ToolDef;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests for {@link WebTaskSubAgent}: the loop, the closed action space, and the stuck guard. */
class WebTaskSubAgentTest {

    private static final class CountingTool implements Tool {
        final String name; final String result; final AtomicInteger runs = new AtomicInteger();
        CountingTool(String name, String result) { this.name = name; this.result = result; }
        @Override public String name() { return name; }
        @Override public String description() { return name; }
        @Override public String parametersJsonSchema() { return "{\"type\":\"object\",\"properties\":{}}"; }
        @Override public ToolResult execute(String a) { runs.incrementAndGet(); return ToolResult.ok(result); }
    }

    private static final class FakeRegistry implements ToolRegistry {
        final List<Tool> tools;
        FakeRegistry(List<Tool> tools) { this.tools = tools; }
        @Override public List<Tool> all() { return tools; }
        @Override public Optional<Tool> byName(String n) { return tools.stream().filter(t -> t.name().equals(n)).findFirst(); }
        @Override public List<ToolDef> toolDefs() {
            List<ToolDef> d = new ArrayList<>();
            for (Tool t : tools) d.add(new ToolDef(t.name(), t.description(), t.parametersJsonSchema()));
            return d;
        }
    }

    /** Queue-scripted LLM (happy path). */
    private static final class ScriptedLlm implements LlmClient {
        final Deque<ChatResponse> script = new ArrayDeque<>();
        final List<ChatRequest> seen = new ArrayList<>();
        ScriptedLlm(ChatResponse... rs) { for (ChatResponse r : rs) script.add(r); }
        @Override public ChatResponse chat(ChatRequest req) {
            seen.add(req);
            return script.isEmpty() ? text("(no more)") : script.poll();
        }
    }

    /** Always asks for the same tool while tools are offered; returns text once tools are withdrawn. */
    private static final class StuckLlm implements LlmClient {
        @Override public ChatResponse chat(ChatRequest req) {
            if (req.tools() == null || req.tools().isEmpty()) return text("FINAL: could not complete, stopping");
            return toolCall("c", "browser");   // identical call+args every time -> identical fingerprint
        }
    }

    private static ChatResponse toolCall(String id, String tool) {
        return new ChatResponse(null, List.of(new ToolCall(id, tool, "{}")), "tool_calls", ChatResponse.Usage.EMPTY);
    }
    private static ChatResponse text(String c) {
        return new ChatResponse(c, List.of(), "stop", ChatResponse.Usage.EMPTY);
    }

    private static List<Tool> catalogue(CountingTool browser) {
        return List.of(browser, new CountingTool("image_analyze", "an image"),
                new CountingTool("code_exec", "42"), new CountingTool("web_search", "results"),
                new CountingTool("india_screener", "STOCKS"), new CountingTool("skill_stock_analysis", "REPORT"),
                new CountingTool("research", "RESEARCH"), new CountingTool("deep_work", "JOB"));
    }

    @Test
    void runsObserveActLoopAndReports() {
        CountingTool browser = new CountingTool("browser", "[0] button \"Go\"\nchanged: true (dom-updated)");
        ScriptedLlm llm = new ScriptedLlm(toolCall("c1", "browser"), toolCall("c2", "browser"), text("Done: added milk to cart."));
        WebTaskSubAgent a = new WebTaskSubAgent(llm, () -> new FakeRegistry(catalogue(browser)), "m", 24);
        ToolResult r = a.execute("{\"task\":\"add milk to the cart\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("Done: added milk"), r.content());
        assertEquals(2, browser.runs.get());
    }

    @Test
    void offersOnlyTheClosedActionSpace() {
        CountingTool browser = new CountingTool("browser", "ok");
        ScriptedLlm llm = new ScriptedLlm(text("done"));
        new WebTaskSubAgent(llm, () -> new FakeRegistry(catalogue(browser)), "m", 24).execute("{\"task\":\"x\"}");
        List<String> offered = llm.seen.get(0).tools().stream().map(ToolDef::name).toList();
        assertTrue(offered.contains("browser") && offered.contains("image_analyze") && offered.contains("code_exec"), offered.toString());
        assertFalse(offered.contains("india_screener"), offered.toString());
        assertFalse(offered.contains("skill_stock_analysis"), offered.toString());
        assertFalse(offered.contains("research"), offered.toString());
        assertFalse(offered.contains("deep_work"), offered.toString());
    }

    @Test
    void stuckLoopBreaksToSynthesisBounded() {
        CountingTool browser = new CountingTool("browser", "changed: false (none)");   // same result every time
        WebTaskSubAgent a = new WebTaskSubAgent(new StuckLlm(), () -> new FakeRegistry(catalogue(browser)), "m", 30);
        ToolResult r = a.execute("{\"task\":\"do the thing\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("FINAL"), "should early-stop with a synthesis report: " + r.content());
        assertTrue(browser.runs.get() <= 10, "stuck guard must break long before the round cap: " + browser.runs.get());
    }

    @Test
    void emptyTaskRejected() {
        ScriptedLlm llm = new ScriptedLlm(text("x"));
        WebTaskSubAgent a = new WebTaskSubAgent(llm, () -> new FakeRegistry(catalogue(new CountingTool("browser", "ok"))), "m", 24);
        ToolResult r = a.execute("{\"task\":\"\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"), r.content());
    }
}
