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

/**
 * Offline tests for {@link DeepAgentTool} using a scripted {@link LlmClient} — no live model.
 * They lock the three load-bearing properties: the worker actually runs its tool loop and returns
 * the final summary; recursion tools never reach the worker's catalogue; and an empty task is
 * rejected.
 */
class DeepAgentToolTest {

    /** A tool that records how many times it ran and returns a fixed result. */
    private static final class CountingTool implements Tool {
        final String name;
        final String result;
        final AtomicInteger runs = new AtomicInteger();
        CountingTool(String name, String result) { this.name = name; this.result = result; }
        @Override public String name() { return name; }
        @Override public String description() { return name + " tool"; }
        @Override public String parametersJsonSchema() { return "{\"type\":\"object\",\"properties\":{}}"; }
        @Override public ToolResult execute(String args) { runs.incrementAndGet(); return ToolResult.ok(result); }
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

    /** Returns scripted responses in order; records every request it saw (to inspect offered tools). */
    private static final class ScriptedLlm implements LlmClient {
        final Deque<ChatResponse> script = new ArrayDeque<>();
        final List<ChatRequest> seen = new ArrayList<>();
        ScriptedLlm(ChatResponse... responses) { for (ChatResponse r : responses) script.add(r); }
        @Override public ChatResponse chat(ChatRequest req) {
            seen.add(req);
            return script.isEmpty()
                    ? new ChatResponse("(no more script)", List.of(), "stop", ChatResponse.Usage.EMPTY)
                    : script.poll();
        }
    }

    private static ChatResponse toolCall(String id, String tool) {
        return new ChatResponse(null, List.of(new ToolCall(id, tool, "{}")), "tool_calls", ChatResponse.Usage.EMPTY);
    }
    private static ChatResponse text(String content) {
        return new ChatResponse(content, List.of(), "stop", ChatResponse.Usage.EMPTY);
    }

    private static List<Tool> catalogueWithRecursionBait(CountingTool calc) {
        // Includes the tools the worker must NOT be offered: research (itself), deep_work, skill_*.
        return List.of(
                calc,
                new CountingTool("web_search", "results"),
                new CountingTool("research", "RECURSION"),
                new CountingTool("deep_work", "LONGJOB"),
                new CountingTool("skill_stock_analysis", "STOCK"));
    }

    @Test
    void delegatesThroughToolLoopAndReturnsSummary() {
        CountingTool calc = new CountingTool("code_exec", "exit=0\n42");
        ScriptedLlm llm = new ScriptedLlm(
                toolCall("c1", "code_exec"),          // round 0: worker calls a tool
                text("The answer is 42 (code_exec)")); // round 1: worker writes its summary
        FakeRegistry reg = new FakeRegistry(catalogueWithRecursionBait(calc));

        DeepAgentTool agent = new DeepAgentTool(llm, () -> reg, "test-model", 8);
        ToolResult r = agent.execute("{\"task\":\"what is 6 times 7\"}");

        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("42"), "summary should carry the worker's answer: " + r.content());
        assertEquals(1, calc.runs.get(), "the worker should have actually executed the tool once");
    }

    @Test
    void excludesRecursionAndSkillToolsFromWorkerCatalogue() {
        CountingTool calc = new CountingTool("code_exec", "exit=0\n42");
        ScriptedLlm llm = new ScriptedLlm(text("done"));
        FakeRegistry reg = new FakeRegistry(catalogueWithRecursionBait(calc));

        DeepAgentTool agent = new DeepAgentTool(llm, () -> reg, "test-model", 8);
        agent.execute("{\"task\":\"anything\"}");

        // Inspect the tools the worker was offered on its first call.
        assertFalse(llm.seen.isEmpty(), "the worker must have called the LLM at least once");
        List<String> offered = llm.seen.get(0).tools().stream().map(ToolDef::name).toList();
        assertTrue(offered.contains("code_exec"), "primitive tools must be offered: " + offered);
        assertTrue(offered.contains("web_search"), offered.toString());
        assertFalse(offered.contains("research"), "must not offer itself (recursion): " + offered);
        assertFalse(offered.contains("deep_work"), "must not offer deep_work: " + offered);
        assertFalse(offered.contains("skill_stock_analysis"), "must not offer skill_* tools: " + offered);
    }

    @Test
    void refusesRecursionToolEvenIfTheModelTriesIt() {
        CountingTool research = new CountingTool("research", "RECURSION");
        ScriptedLlm llm = new ScriptedLlm(
                toolCall("c1", "research"),  // model tries to recurse despite it not being offered
                text("recovered"));
        FakeRegistry reg = new FakeRegistry(List.of(new CountingTool("code_exec", "ok"), research));

        DeepAgentTool agent = new DeepAgentTool(llm, () -> reg, "test-model", 8);
        ToolResult r = agent.execute("{\"task\":\"x\"}");

        assertTrue(r.ok(), r.content());
        assertEquals(0, research.runs.get(), "a recursion attempt must never actually execute");
    }

    @Test
    void emptyTaskRejected() {
        ScriptedLlm llm = new ScriptedLlm(text("unused"));
        FakeRegistry reg = new FakeRegistry(List.of(new CountingTool("code_exec", "ok")));
        DeepAgentTool agent = new DeepAgentTool(llm, () -> reg, "test-model", 8);

        ToolResult r = agent.execute("{\"task\":\"\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"), r.content());
    }
}
