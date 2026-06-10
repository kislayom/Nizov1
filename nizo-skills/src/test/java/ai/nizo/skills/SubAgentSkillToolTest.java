package ai.nizo.skills;

import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.agent.AgentEventContext;
import ai.nizo.api.agent.AgentEventSink;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.Role;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.llm.ToolDef;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link SubAgentSkillTool} — the multi-agent skill executor that wraps each
 * sub-skill (analyst, researcher, trader) in its own LLM loop. Verifies:
 *
 * <ul>
 *   <li>Date + fiscal-year anchor injected into the sub-agent's user prompt
 *       (so Qwen3.6 stops querying stale years like "2024" when it's 2026)</li>
 *   <li>Anti-markdown query directive injected (so the sub-agent stops emitting
 *       {@code [reddit.com](http://reddit.com)} link syntax to web_search)</li>
 *   <li>Empty content + no tool calls → first nudge → second nudge → placeholder</li>
 *   <li>Tool catalogue excludes other {@code skill_*} tools (recursion guard)</li>
 *   <li>Inner tool calls emit {@link AgentEvent.ToolCallStart}/{@code Result} events
 *       through {@link AgentEventContext} so the UI sees them</li>
 *   <li>Iteration cap returns a placeholder, not an error</li>
 * </ul>
 */
class SubAgentSkillToolTest {

    @AfterEach
    void clearContext() {
        AgentEventContext.clear();
    }

    // -------- Anchor: date + fiscal year --------

    @Test
    void execute_injectsCurrentDateInUserMessage() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly("Final section"));
        SubAgentSkillTool tool = newTool(llm, registry());

        tool.execute("{\"input\":\"AAPL\"}");

        String today = LocalDate.now().toString();
        String userMsg = llm.lastRequest().messages().stream()
                .filter(m -> m.role() == Role.USER).findFirst().orElseThrow().content();
        assertTrue(userMsg.contains(today),
                "User message should include today's ISO date. Got: " + userMsg);
    }

    @Test
    void execute_injectsCompletedFiscalYearAnchor() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly("done"));
        SubAgentSkillTool tool = newTool(llm, registry());

        tool.execute("{\"input\":\"AAPL\"}");

        int latestFy = LocalDate.now().getYear() - 1;
        String userMsg = llm.lastRequest().messages().stream()
                .filter(m -> m.role() == Role.USER).findFirst().orElseThrow().content();
        assertTrue(userMsg.contains("FY" + latestFy),
                "Should anchor to FY" + latestFy + " (most recent completed). Got: " + userMsg);
        assertTrue(userMsg.toLowerCase().contains("most recently completed"),
                "Should explicitly call out 'most recently completed'");
    }

    @Test
    void execute_includesAntiMarkdownDirective() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly("done"));
        SubAgentSkillTool tool = newTool(llm, registry());

        tool.execute("{\"input\":\"AAPL\"}");

        String userMsg = llm.lastRequest().messages().stream()
                .filter(m -> m.role() == Role.USER).findFirst().orElseThrow().content();
        assertTrue(userMsg.toLowerCase().contains("no markdown")
                        || userMsg.contains("[text](url)"),
                "Should include anti-markdown rule. Got: " + userMsg);
    }

    @Test
    void execute_includesSkillBodyAsSystemPrompt() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly("done"));
        SubAgentSkillTool tool = new SubAgentSkillTool(
                manifest("fundamentals_analyst",
                        "BODY MARKER: pull real numbers from authoritative sources"),
                llm, () -> registry(), "test-model", 12);

        tool.execute("{\"input\":\"AAPL\"}");

        String sysMsg = llm.lastRequest().messages().stream()
                .filter(m -> m.role() == Role.SYSTEM).findFirst().orElseThrow().content();
        assertTrue(sysMsg.contains("BODY MARKER"),
                "System prompt should include the SKILL.md body");
    }

    // -------- Tool catalogue: skill_* recursion guard --------

    @Test
    void execute_filtersSkillStarToolsFromChildCatalogue() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly("done"));

        ToolRegistry parent = registry(
                fakeTool("web_search"),
                fakeTool("web_fetch"),
                fakeTool("skill_stock_news_analyst"),  // should be filtered
                fakeTool("skill_stock_trader"));        // should be filtered
        SubAgentSkillTool tool = newTool(llm, parent);

        tool.execute("{\"input\":\"AAPL\"}");

        List<ToolDef> childTools = llm.lastRequest().tools();
        List<String> names = childTools.stream().map(ToolDef::name).toList();
        assertTrue(names.contains("web_search"));
        assertTrue(names.contains("web_fetch"));
        assertFalse(names.contains("skill_stock_news_analyst"),
                "Sub-agent must not see other skill_* tools (recursion guard)");
        assertFalse(names.contains("skill_stock_trader"));
    }

    // -------- Empty-content nudge + placeholder --------

    @Test
    void emptyContent_nudgedOnce_thenSucceeds() {
        TestLlm llm = new TestLlm();
        llm.queueResponse(textOnly(""));         // first response: empty (no tools)
        llm.queueResponse(textOnly("Real section after nudge"));  // after nudge: writes
        SubAgentSkillTool tool = newTool(llm, registry());

        ToolResult r = tool.execute("{\"input\":\"AAPL\"}");

        assertTrue(r.ok());
        assertTrue(r.content().contains("Real section after nudge"));
        assertEquals(2, llm.requestCount(), "Should have made one nudge round-trip");
        // The nudge user message should have been appended on the 2nd request
        ChatRequest second = llm.requestAt(1);
        long userCount = second.messages().stream().filter(m -> m.role() == Role.USER).count();
        assertEquals(2, userCount, "Second request should have original user msg + nudge");
    }

    @Test
    void emptyContentPersistent_returnsPlaceholderNotError() {
        TestLlm llm = new TestLlm();
        // The nudge ladder is now THREE attempts before giving up (was two — bumped in May
        // 2026 after Qwen3.6 was observed needing a third "stop thinking, write now" push).
        // So we need 4 empty responses: initial + nudge 1 + nudge 2 + nudge 3.
        llm.queueResponse(textOnly(""));  // initial empty
        llm.queueResponse(textOnly(""));  // empty after nudge 1
        llm.queueResponse(textOnly(""));  // empty after nudge 2
        llm.queueResponse(textOnly(""));  // empty after nudge 3
        SubAgentSkillTool tool = newTool(llm, registry());

        ToolResult r = tool.execute("{\"input\":\"AAPL\"}");

        assertTrue(r.ok(),
                "Persistent empty content should return placeholder (ok), not error — "
                + "so the orchestrator can keep building the report");
        // Current placeholder wording (May 2026): "_Coverage note for **<analyst>** —
        // the analyst ran N source lookups but didn't converge on a written summary..."
        // Accept any of the synonym variants so the test survives minor copy edits.
        String c = r.content().toLowerCase();
        assertTrue(c.contains("did not produce a written section")
                        || c.contains("didn't converge")
                        || c.contains("coverage note")
                        || c.contains("placeholder")
                        || c.contains("did not"),
                "Placeholder should explain the section is unavailable. Got: " + r.content());
    }

    // -------- Iteration cap returns placeholder, not error --------

    @Test
    void iterationCap_returnsPlaceholderNotError() {
        TestLlm llm = new TestLlm();
        // Make the LLM keep returning tool calls so the loop never terminates on its own.
        for (int i = 0; i < 50; i++) {
            llm.queueResponse(toolCallOnly("call-" + i, "web_search", "{\"query\":\"x\"}"));
        }
        ToolRegistry parent = registry(fakeTool("web_search"));
        SubAgentSkillTool tool = new SubAgentSkillTool(
                manifest("a", "body"), llm, () -> parent, "m", /* cap */ 3);

        ToolResult r = tool.execute("{\"input\":\"AAPL\"}");

        assertTrue(r.ok(),
                "Hitting the iter cap should return ok+placeholder, not error — "
                + "otherwise the orchestrator can't recover and finish the report");
        assertTrue(r.content().contains("maximum")
                        || r.content().toLowerCase().contains("iteration")
                        || r.content().toLowerCase().contains("converge"),
                "Placeholder should explain the cap was hit. Got: " + r.content());
    }

    // -------- AgentEventContext propagation --------

    @Test
    void innerToolCalls_emitEventsThroughAgentEventContext() {
        TestLlm llm = new TestLlm();
        // First LLM response: call web_search. Second: write section (no tools).
        llm.queueResponse(toolCallOnly("c1", "web_search", "{\"query\":\"AAPL\"}"));
        llm.queueResponse(textOnly("Section text"));

        ToolRegistry parent = registry(fakeTool("web_search", "RESULT"));
        SubAgentSkillTool tool = newTool(llm, parent);

        // Bind a recording sink to thread-local AgentEventContext
        RecordingSink sink = new RecordingSink();
        AgentEventContext.set(sink);

        tool.execute("{\"input\":\"AAPL\"}");

        boolean sawStart = sink.events.stream().anyMatch(e ->
                e instanceof AgentEvent.ToolCallStart s && "web_search".equals(s.toolName()));
        boolean sawResult = sink.events.stream().anyMatch(e ->
                e instanceof AgentEvent.ToolCallResult r && "web_search".equals(r.toolName()));
        assertTrue(sawStart, "Sub-agent's inner web_search should emit ToolCallStart");
        assertTrue(sawResult, "Sub-agent's inner web_search should emit ToolCallResult");
    }

    @Test
    void noEventSinkBound_doesNotCrash() {
        // AgentEventContext defaults to NOOP when nothing is bound — sub-agent should still work.
        TestLlm llm = new TestLlm();
        llm.queueResponse(toolCallOnly("c1", "web_search", "{}"));
        llm.queueResponse(textOnly("done"));
        ToolRegistry parent = registry(fakeTool("web_search", "result"));
        SubAgentSkillTool tool = newTool(llm, parent);

        // No AgentEventContext.set() before this call
        ToolResult r = tool.execute("{}");

        assertTrue(r.ok());
        assertTrue(r.content().contains("done"));
    }

    // -------- Recursion guard at execution time (defense in depth) --------

    @Test
    void execute_refusesNestedSkillCallEvenIfLlmAttempts() {
        TestLlm llm = new TestLlm();
        // LLM (somehow) tries to call skill_stock_trader from inside a sub-agent.
        llm.queueResponse(toolCallOnly("c1", "skill_stock_trader", "{}"));
        llm.queueResponse(textOnly("done"));
        ToolRegistry parent = registry(fakeTool("skill_stock_trader", "should-not-be-called"));
        SubAgentSkillTool tool = newTool(llm, parent);

        tool.execute("{}");

        // The second LLM call should have received an error tool result, not the trader's output
        ChatRequest second = llm.requestAt(1);
        ChatMessage toolMsg = second.messages().stream()
                .filter(m -> m.role() == Role.TOOL).findFirst().orElseThrow();
        assertTrue(toolMsg.content().toLowerCase().contains("may not invoke"),
                "Recursive skill_* call should be refused with an error result. Got: "
                + toolMsg.content());
    }

    // ===== test infrastructure =====

    private static SubAgentSkillTool newTool(LlmClient llm, ToolRegistry parent) {
        return new SubAgentSkillTool(
                manifest("fundamentals_analyst", "Be a fundamentals analyst."),
                llm, () -> parent, "test-model", 12);
    }

    private static SkillManifest manifest(String name, String body) {
        return new SkillManifest(name, "desc", "when", List.of(), true, body, Path.of("test.md"));
    }

    private static ChatResponse textOnly(String content) {
        return new ChatResponse(content, List.of(), "stop", null);
    }

    private static ChatResponse toolCallOnly(String id, String name, String args) {
        return new ChatResponse("", List.of(new ToolCall(id, name, args)), "tool_calls", null);
    }

    private static ToolRegistry registry(Tool... tools) {
        List<Tool> list = new ArrayList<>(List.of(tools));
        return new ToolRegistry() {
            @Override public List<Tool> all() { return list; }
            @Override public Optional<Tool> byName(String n) {
                return list.stream().filter(t -> t.name().equals(n)).findFirst();
            }
            @Override public List<ToolDef> toolDefs() {
                return list.stream().map(t -> new ToolDef(t.name(), t.description(), t.parametersJsonSchema())).toList();
            }
        };
    }

    private static Tool fakeTool(String name) { return fakeTool(name, "ok"); }

    private static Tool fakeTool(String name, String result) {
        return new Tool() {
            @Override public String name() { return name; }
            @Override public String description() { return name + " (test)"; }
            @Override public String parametersJsonSchema() { return "{\"type\":\"object\"}"; }
            @Override public ToolResult execute(String args) { return ToolResult.ok(result); }
        };
    }

    /** A record-and-replay LlmClient. */
    static class TestLlm implements LlmClient {
        private final List<ChatRequest> requests = new ArrayList<>();
        private final Deque<ChatResponse> queued = new ArrayDeque<>();

        void queueResponse(ChatResponse r) { queued.add(r); }
        int requestCount() { return requests.size(); }
        ChatRequest lastRequest() { return requests.get(requests.size() - 1); }
        ChatRequest requestAt(int i) { return requests.get(i); }

        @Override public ChatResponse chat(ChatRequest req) {
            requests.add(req);
            if (queued.isEmpty()) return new ChatResponse("(out of queued responses)", List.of(), "stop", null);
            return queued.pollFirst();
        }
    }

    static class RecordingSink implements AgentEventSink {
        final List<AgentEvent> events = new ArrayList<>();
        @Override public void emit(AgentEvent event) { events.add(event); }
    }
}
