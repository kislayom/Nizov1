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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
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
 * Completes a high-level WEB TASK by driving the browser through a deterministic perceive→act→verify
 * loop in Java — the SOTA web-agent pattern (browser-use / WebVoyager / Manus), and the same lesson
 * as the stock pipeline: don't trust the model to bound or unstick itself, do it in code.
 *
 * <p>The model gets a small CLOSED action space (browser + image_analyze + a little web/compute) and
 * is told to: observe() → act BY INDEX with the snapshotVersion → read changed/change_kind → diverge
 * on failure → fall back to screenshot_marks + image_analyze when the DOM is unreadable → stop with a
 * report (and surface any human-only step, since the sidecar refuses passwords/payment/checkout).
 *
 * <p>Deterministic guards (outside the LLM): bounded rounds, a per-call LLM timeout, and a
 * STUCK-LOOP detector — if the same (tool + args + result) fingerprint repeats, it injects a
 * "diverge" nudge once and then breaks to an early-stop synthesis, so it never burns the budget
 * repeating a dead action (WebVoyager: navigation-stuck is ~44% of failures).
 */
public final class WebTaskSubAgent implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(WebTaskSubAgent.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** The closed action space — web interaction + vision + light compute, nothing else. */
    static final Set<String> ALLOWED_TOOLS = Set.of(
            "browser", "image_analyze", "web_search", "web_fetch", "code_exec", "current_time");

    private static final long DEFAULT_PER_CALL_TIMEOUT_MS = 180_000L;
    private static final int DEFAULT_TOOL_RESULT_MAX_CHARS = 6_000;   // observe snapshots are bigger
    private static final int SUMMARY_MAX_CHARS = 8_000;

    private static final Executor TIMEOUT_EXECUTOR = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "nizo-webtask-llm");
        t.setDaemon(true);
        return t;
    });

    private final LlmClient llm;
    private final Supplier<ToolRegistry> parentToolsRef;
    private final String model;
    private final int maxRounds;
    private final long perCallTimeoutMs;
    private final int toolResultMaxChars;

    public WebTaskSubAgent(LlmClient llm, Supplier<ToolRegistry> parentToolsRef, String model, int maxRounds) {
        this.llm = llm;
        this.parentToolsRef = parentToolsRef;
        this.model = model;
        this.maxRounds = Math.max(4, maxRounds);
        this.perCallTimeoutMs = resolveLong("NIZO_SUBAGENT_LLM_TIMEOUT_MS", DEFAULT_PER_CALL_TIMEOUT_MS);
        this.toolResultMaxChars = (int) resolveLong("NIZO_WEBTASK_TOOL_RESULT_MAX_CHARS", DEFAULT_TOOL_RESULT_MAX_CHARS);
    }

    @Override public String name() { return "web_task"; }

    @Override
    public String description() {
        return "Complete a multi-step task on a website by driving a real browser end-to-end — "
                + "search, pick a result, fill forms, set a location, build a cart, work a booking "
                + "flow. Give it a self-contained task and the exact result you want back. It runs an "
                + "isolated observe→act→verify loop (acting by element index), recovers from failures, "
                + "and falls back to screenshot+vision when a page is hard to parse. It will NOT enter "
                + "passwords/payment or place an order/pay — it stops and tells you the human step. Use "
                + "this instead of calling the browser tool yourself for anything beyond a single "
                + "lookup.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "task":   { "type": "string", "description": "A self-contained web task, ideally with the starting site/URL." },
                "output": { "type": "string", "description": "Optional: the exact result to report back." }
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
        if (tools == null) return ToolResult.error("web_task cannot run: tool registry not initialized");

        List<ToolDef> childToolDefs = new ArrayList<>();
        for (Tool t : tools.all()) {
            if (ALLOWED_TOOLS.contains(t.name())) {
                childToolDefs.add(new ToolDef(t.name(), t.description(), t.parametersJsonSchema()));
            }
        }

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(WEB_TASK_SYSTEM));
        messages.add(ChatMessage.user("TASK: " + task + (output.isBlank() ? "" : "\n\nReport back: " + output)));

        AgentEventSink sink = AgentEventContext.current();
        Deque<String> recentFingerprints = new ArrayDeque<>();
        long t0 = System.nanoTime();
        int totalCalls = 0, nudges = 0, divergeNudges = 0;
        LOG.info("web_task starting (task.len={})", task.length());

        for (int round = 0; round < maxRounds; round++) {
            boolean budgetHit = totalCalls >= maxRounds;   // ~one tool call per round
            List<ToolDef> effective = budgetHit ? List.of() : childToolDefs;
            Map<String, Object> extras = (nudges >= 1)
                    ? Map.of("chat_template_kwargs", Map.of("enable_thinking", false))
                    : Map.of();

            ChatResponse resp;
            try {
                resp = chatWithTimeout(ChatRequest.of(model, messages).withTools(effective).withExtraBody(extras));
            } catch (TimeoutException te) {
                return ToolResult.error("web_task LLM call timed out after " + (perCallTimeoutMs / 1000) + "s");
            } catch (Exception e) {
                return ToolResult.error("web_task LLM call failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }

            if (!resp.hasToolCalls()) {
                String content = resp.content() == null ? "" : resp.content().strip();
                if (content.isBlank() && nudges < 2) {
                    messages.add(ChatMessage.assistant("(working)"));
                    messages.add(ChatMessage.user(nudges == 0
                            ? "Continue the task, or if it's done report what you accomplished, the result, and any "
                              + "human step left. No empty replies."
                            : "Give your final report now — what you did, the result, and any remaining human step. "
                              + "Plain prose, no reasoning."));
                    nudges++;
                    continue;
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                LOG.info("web_task done in {}ms, {} rounds, {} calls, {} chars", ms, round + 1, totalCalls, content.length());
                return content.isBlank()
                        ? ToolResult.ok("_(web_task produced no final report after " + totalCalls + " actions.)_")
                        : ToolResult.ok(cap(content));
            }

            messages.add(ChatMessage.assistantToolCalls(resp.toolCalls()));
            for (ToolCall call : resp.toolCalls()) {
                totalCalls++;
                String tag = "web_task::" + call.id();
                if (sink != null) sink.emit(new AgentEvent.ToolCallStart(round, tag, call.name(), call.argumentsJson()));
                ToolResult r = executeChild(tools, call);
                if (sink != null) sink.emit(new AgentEvent.ToolCallResult(round, tag, call.name(), r.ok(), r.content(), 0));
                messages.add(ChatMessage.toolResult(call.id(), truncate(r.content())));

                // Stuck-loop guard: same (tool + args + result-preview) repeating = going nowhere.
                String fp = call.name() + "|" + brief(call.argumentsJson(), 80) + "|" + brief(r.content(), 80);
                recentFingerprints.addLast(fp);
                if (recentFingerprints.size() > 3) recentFingerprints.removeFirst();
            }

            if (isStuck(recentFingerprints)) {
                recentFingerprints.clear();
                if (divergeNudges == 0) {
                    LOG.warn("web_task stuck (repeating action) — injecting diverge nudge");
                    messages.add(ChatMessage.user(
                            "You are repeating the SAME action with the same result — it is not working. STOP and try a "
                          + "FUNDAMENTALLY different approach: re-observe; scroll to reveal the target; dismiss an overlay; "
                          + "use screenshot_marks + image_analyze to see the page; or pick a different element. Do not "
                          + "repeat that action."));
                    divergeNudges++;
                } else {
                    LOG.warn("web_task stuck twice — breaking to early-stop synthesis");
                    break;
                }
            }
        }

        // Bounded out (round/stuck) — one final tool-less synthesis so we return the best partial result.
        try {
            messages.add(ChatMessage.user("You've hit the action budget or got stuck. Give your best final report NOW "
                    + "from what you achieved: what's done, the result so far, and exactly what remains (incl. any human "
                    + "step). No more tools."));
            ChatResponse fin = chatWithTimeout(ChatRequest.of(model, messages).withTools(List.of())
                    .withExtraBody(Map.of("chat_template_kwargs", Map.of("enable_thinking", false))));
            String c = fin.content() == null ? "" : fin.content().strip();
            LOG.warn("web_task early-stop synthesis after {} calls", totalCalls);
            return ToolResult.ok(c.isBlank()
                    ? "_(web_task ran out of steps after " + totalCalls + " actions without finishing.)_"
                    : cap(c));
        } catch (Exception e) {
            return ToolResult.error("web_task ran out of steps and final synthesis failed: " + e.getClass().getSimpleName());
        }
    }

    /** Stuck if the last 3 tool fingerprints are all identical. */
    private static boolean isStuck(Deque<String> fps) {
        if (fps.size() < 3) return false;
        String first = fps.peekFirst();
        for (String f : fps) if (!f.equals(first)) return false;
        return true;
    }

    private ToolResult executeChild(ToolRegistry tools, ToolCall call) {
        String n = call.name();
        if (n == null || !ALLOWED_TOOLS.contains(n))
            return ToolResult.error("web_task may only use: " + String.join(", ", ALLOWED_TOOLS) + " (got '" + n + "')");
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
        return s.length() <= toolResultMaxChars ? s
                : s.substring(0, toolResultMaxChars) + "\n…[truncated to " + toolResultMaxChars + " chars]";
    }

    private static String cap(String s) {
        return s.length() <= SUMMARY_MAX_CHARS ? s : s.substring(0, SUMMARY_MAX_CHARS) + "\n…[truncated]";
    }

    private static String brief(String s, int n) {
        if (s == null) return "";
        String x = s.replaceAll("\\s+", " ").trim();
        return x.length() <= n ? x : x.substring(0, n);
    }

    private static long resolveLong(String env, long dflt) {
        String v = System.getenv(env);
        if (v == null || v.isBlank()) return dflt;
        try { return Long.parseLong(v.trim()); } catch (NumberFormatException e) { return dflt; }
    }

    private static final String WEB_TASK_SYSTEM = """
        You are a web task agent. Complete the user's task by driving a real browser through a strict
        PERCEIVE -> ACT -> VERIFY loop. One action per step.

        LOOP:
        1. PERCEIVE: browser{action:observe} -> a numbered list of interactive elements
           ([index] role "name" state) and a snapshotVersion. (Use goto{url} first to open a page.)
        2. ACT BY INDEX: browser{action:click,index:N,snapshotVersion:V} or
           {action:type,index:N,text:"...",snapshotVersion:V,submit:true}. ALWAYS pass the
           snapshotVersion you just observed. scroll to load more results; wait{selector,state:"hidden"}
           to wait out a spinner; type sequential:true for autocomplete/typeahead fields.
        3. VERIFY: each action returns changed + change_kind. If changed=false your action did nothing —
           do NOT repeat it; choose a different element/approach. If you get "stale, re-observe", the
           page changed: call observe again before acting.

        RULES:
        - NEVER repeat an identical failed action. On failure read the SPECIFIC error and DIVERGE:
          re-observe, scroll to the target, dismiss an overlay, or for a custom dropdown click it then
          click the option.
        - If the DOM is unreadable or you can't find a control: browser{action:screenshot_marks}, then
          image_analyze that PNG (the box numbers match the element indices) and act on what you see.
        - You CANNOT enter passwords/payment or place an order/pay — the browser refuses these and
          returns 'HUMAN STEP REQUIRED'. When you hit that, STOP and report exactly what's left for the human.
        - Compute any totals/numbers with code_exec; never eyeball arithmetic.
        - When the task is verifiably done, reply with NO tool calls: a concise report of what you
          accomplished, the key result, and any remaining human step.
        """;
}
