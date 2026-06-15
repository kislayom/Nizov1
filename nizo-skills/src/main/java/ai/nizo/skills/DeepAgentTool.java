package ai.nizo.skills;

import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.agent.AgentEventContext;
import ai.nizo.api.agent.AgentEventSink;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.llm.ToolDef;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

/**
 * The general deep-agent: delegate a focused sub-investigation to an isolated worker that runs its
 * own bounded tool loop and returns ONE grounded summary.
 *
 * <p>{@link SubAgentSkillTool} already proves this pattern works — but it is skill-bound (its system
 * prompt is a SKILL.md) and carries stock-report machinery (chart fences, section skeletons). This
 * is the skill-free generalisation the rest of the industry converged on in 2026 ("sub-agents as
 * tools" / OpenClaw's {@code prepareSubagentSpawn}): any task — and any Deep Work step — can spin up
 * a worker with a FRESH context, let it fan out across several tool calls, and get back a dense
 * answer <em>without</em> polluting the caller's context with the intermediate steps.
 *
 * <p>Why this matters for accuracy and long jobs: the caller stays focused (only summaries return,
 * not 50 KB of raw tool output), big tasks decompose into independent sub-questions, and the worker
 * inherits the same grounding discipline as the Deep Work executor — compute derived numbers with
 * {@code code_exec}, fetch real pages for figures, say "could not verify X" instead of guessing.
 *
 * <h2>Hardening (mirrors {@link SubAgentSkillTool})</h2>
 * <ul>
 *   <li>Per-{@code llm.chat()} timeout on a daemon executor (never the common ForkJoinPool), so a
 *       wedged llama-server fails fast instead of pinning the caller's thread. Configurable via
 *       {@code NIZO_SUBAGENT_LLM_TIMEOUT_MS}.</li>
 *   <li>Tool results truncated before re-feeding ({@code NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS}).</li>
 *   <li>Recursion blocked: the worker's catalogue excludes {@code skill_*}, {@code research}, and
 *       {@code deep_work} — no nested deep-agents or long-jobs from inside a worker.</li>
 *   <li>Round + tool budget; on exhaustion the worker is forced to answer with tools removed. The
 *       empty-content / Qwen-stuck-in-thinking failure mode is handled with the same no-think nudge.</li>
 * </ul>
 */
public final class DeepAgentTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(DeepAgentTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static final long DEFAULT_PER_CALL_TIMEOUT_MS = 180_000L;
    public static final int DEFAULT_TOOL_RESULT_MAX_CHARS = 4_000;
    private static final int SUMMARY_MAX_CHARS = 12_000;

    /** Tools the worker may NEVER call — recursion / nested-long-job guard. {@code skill_*} is matched by prefix. */
    static final Set<String> EXCLUDED = Set.of("research", "deep_work");

    private static final Executor TIMEOUT_EXECUTOR = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "nizo-research-llm");
        t.setDaemon(true);
        return t;
    });

    private final LlmClient llm;
    private final Supplier<ToolRegistry> parentToolsRef;
    private final String model;
    private final int maxRounds;
    private final int toolBudget;
    private final long perCallTimeoutMs;
    private final int toolResultMaxChars;

    public DeepAgentTool(LlmClient llm, Supplier<ToolRegistry> parentToolsRef, String model, int maxRounds) {
        this.llm = llm;
        this.parentToolsRef = parentToolsRef;
        this.model = model;
        this.maxRounds = Math.max(2, maxRounds);
        this.toolBudget = Math.max(2, maxRounds);
        this.perCallTimeoutMs = resolveLong("NIZO_SUBAGENT_LLM_TIMEOUT_MS", DEFAULT_PER_CALL_TIMEOUT_MS);
        this.toolResultMaxChars = (int) resolveLong("NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS", DEFAULT_TOOL_RESULT_MAX_CHARS);
    }

    @Override public String name() { return "research"; }

    @Override
    public String description() {
        return "Delegate a focused sub-investigation to an isolated worker agent. It runs its own "
             + "multi-step tool loop (web_search, web_fetch, code_exec, finance, files, …) in a FRESH "
             + "context and returns ONE dense, grounded summary — without filling your context with the "
             + "intermediate steps. Use it to (a) split a big task into independent sub-questions you run "
             + "separately, or (b) go deep on a single thread that needs several lookups. Give a specific, "
             + "self-contained task and state exactly what to return.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "task":   { "type": "string", "description": "A self-contained question or task for the worker to investigate and answer." },
                "output": { "type": "string", "description": "Optional: the exact form/structure of the answer you want back." }
              },
              "required": ["task"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        String task, output;
        try {
            JsonNode n = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            task = n.path("task").asText("");
            output = n.path("output").asText("");
        } catch (Exception e) {
            task = argumentsJson == null ? "" : argumentsJson;
            output = "";
        }
        if (task.isBlank()) return ToolResult.error("task is required");

        ToolRegistry tools = parentToolsRef.get();
        if (tools == null) return ToolResult.error("research sub-agent cannot run: tool registry not initialized");

        List<ToolDef> childToolDefs = new ArrayList<>();
        for (Tool t : tools.all()) {
            String tn = t.name();
            if (tn.startsWith("skill_") || EXCLUDED.contains(tn)) continue;
            childToolDefs.add(new ToolDef(tn, t.description(), t.parametersJsonSchema()));
        }

        String userMsg = datePreamble() + "TASK: " + task
                + (output.isBlank() ? "" : "\n\nReturn it in this form: " + output);

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(RESEARCHER_SYSTEM));
        messages.add(ChatMessage.user(userMsg));

        AgentEventSink sink = AgentEventContext.current();
        long t0 = System.nanoTime();
        int totalCalls = 0, nudges = 0;
        LOG.info("research sub-agent starting (task.len={})", task.length());

        for (int round = 0; round < maxRounds; round++) {
            boolean budgetHit = totalCalls >= toolBudget;
            List<ToolDef> effective = budgetHit ? List.of() : childToolDefs;
            Map<String, Object> extras = (nudges >= 1)
                    ? Map.of("chat_template_kwargs", Map.of("enable_thinking", false))
                    : Map.of();

            ChatResponse resp;
            try {
                resp = chatWithTimeout(ChatRequest.of(model, messages).withTools(effective).withExtraBody(extras));
            } catch (TimeoutException te) {
                return ToolResult.error("research sub-agent LLM call timed out after " + (perCallTimeoutMs / 1000)
                        + "s (set NIZO_SUBAGENT_LLM_TIMEOUT_MS to override).");
            } catch (Exception e) {
                return ToolResult.error("research sub-agent LLM call failed: "
                        + e.getClass().getSimpleName() + ": " + e.getMessage());
            }

            if (!resp.hasToolCalls()) {
                String content = resp.content() == null ? "" : resp.content().strip();
                if (content.isBlank() && nudges < 2) {
                    // Qwen3.6 sometimes returns no tool calls AND empty content (spent the budget in
                    // reasoning_content). Push it to write; the 2nd nudge rides with enable_thinking=false.
                    messages.add(ChatMessage.assistant("(gathering done)"));
                    messages.add(ChatMessage.user(nudges == 0
                            ? "Now write your final answer to the task as the assistant message — grounded, "
                            + "concise, sources inline. No tool calls, no reasoning. Begin now."
                            : "Write the answer now. Use what you have; say 'could not verify X' for gaps. "
                            + "Plain prose, no reasoning. Begin immediately."));
                    nudges++;
                    continue;
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                LOG.info("research sub-agent done in {}ms, {} rounds, {} calls, {} chars",
                        ms, round + 1, totalCalls, content.length());
                if (content.isBlank()) {
                    return ToolResult.ok("_(The research sub-agent could not converge on an answer after "
                            + totalCalls + " tool calls. Try a narrower task.)_");
                }
                return ToolResult.ok(cap(content));
            }

            messages.add(ChatMessage.assistantToolCalls(resp.toolCalls()));
            for (ToolCall call : resp.toolCalls()) {
                totalCalls++;
                String tag = "research::" + call.id();
                if (sink != null) sink.emit(new AgentEvent.ToolCallStart(round, tag, call.name(), call.argumentsJson()));
                long ts = System.nanoTime();
                ToolResult r = executeChild(tools, call);
                long ems = (System.nanoTime() - ts) / 1_000_000;
                if (sink != null) sink.emit(new AgentEvent.ToolCallResult(round, tag, call.name(), r.ok(), r.content(), ems));
                messages.add(ChatMessage.toolResult(call.id(), truncate(r.content())));
            }
        }

        // Round budget exhausted while still calling tools — force a final no-tools synthesis.
        try {
            messages.add(ChatMessage.user("Round budget reached. Write your best grounded answer to the task "
                    + "NOW from what you've gathered. No more tools. Mark anything unverified."));
            ChatResponse fin = chatWithTimeout(ChatRequest.of(model, messages).withTools(List.of())
                    .withExtraBody(Map.of("chat_template_kwargs", Map.of("enable_thinking", false))));
            String c = fin.content() == null ? "" : fin.content().strip();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            LOG.warn("research sub-agent forced synthesis after round cap ({}ms, {} calls)", ms, totalCalls);
            return ToolResult.ok(c.isBlank()
                    ? "_(Research ran out of rounds after " + totalCalls + " tool calls without a final answer.)_"
                    : cap(c));
        } catch (Exception e) {
            return ToolResult.error("research sub-agent ran out of rounds and the final synthesis failed: "
                    + e.getClass().getSimpleName());
        }
    }

    private ToolResult executeChild(ToolRegistry tools, ToolCall call) {
        String n = call.name();
        if (n != null && (n.startsWith("skill_") || EXCLUDED.contains(n)))
            return ToolResult.error("research sub-agent may not call '" + n + "'");
        Tool t = tools.byName(n).orElse(null);
        if (t == null) return ToolResult.error("unknown tool '" + n + "'");
        try {
            return t.execute(call.argumentsJson() == null ? "{}" : call.argumentsJson());
        } catch (Exception e) {
            return ToolResult.error(t.name() + " threw: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private ChatResponse chatWithTimeout(ChatRequest req)
            throws TimeoutException, ExecutionException, InterruptedException {
        CompletableFuture<ChatResponse> fut = CompletableFuture.supplyAsync(() -> llm.chat(req), TIMEOUT_EXECUTOR);
        try {
            return fut.get(perCallTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException te) {
            fut.cancel(true);
            throw te;
        }
    }

    private String truncate(String s) {
        if (s == null) return "";
        if (s.length() <= toolResultMaxChars) return s;
        return s.substring(0, toolResultMaxChars) + "\n…[truncated to " + toolResultMaxChars + " chars]";
    }

    private static String cap(String s) {
        if (s.length() <= SUMMARY_MAX_CHARS) return s;
        return s.substring(0, SUMMARY_MAX_CHARS) + "\n…[summary truncated to " + SUMMARY_MAX_CHARS + " chars]";
    }

    private static String datePreamble() {
        LocalDate today = LocalDate.now(ZoneId.systemDefault());
        return "**Today is " + today.format(DateTimeFormatter.ISO_LOCAL_DATE) + ".** Treat your own memory of "
                + "current prices, versions, dates and limits as stale — fetch them.\n\n";
    }

    private static long resolveLong(String env, long dflt) {
        String v = System.getenv(env);
        if (v == null || v.isBlank()) return dflt;
        try { return Long.parseLong(v.trim()); } catch (NumberFormatException e) { return dflt; }
    }

    private static final String RESEARCHER_SYSTEM = """
        You are a focused research sub-agent. You are given ONE self-contained task. Investigate it
        with the available tools and return a single, dense, grounded answer — nothing else.

        Rules:
        - Gather real evidence with tools; never fabricate numbers, dates, prices, or quotes.
        - COMPUTE every derived number (rates, ratios, sums, statistics, date math) with code_exec —
          never do multi-step arithmetic in your head.
        - For any figure you cite, prefer web_fetch of the actual page over a search snippet.
        - Cite each key fact's source inline (URL or tool name).
        - Be efficient: batch related lookups; conclude as soon as the task is covered. Your round
          budget is small — when it is gone you MUST answer with no further tools.
        - Return ONLY the answer: dense findings, each grounded. No preamble, no "I will…".
        - If you cannot ground something, write "could not verify X" rather than guessing.
        """;
}
