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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Wraps a filesystem skill as a callable sub-agent.
 *
 * <p>Unlike {@link FilesystemSkillTool} (which returns the SKILL.md body as instructions for the
 * outer LLM to follow), this tool runs its own inner LLM loop:
 *
 * <ol>
 *   <li>System prompt = the SKILL.md body (the analyst/researcher's playbook).</li>
 *   <li>User prompt = the {@code input} argument from the caller.</li>
 *   <li>Tool catalogue = parent registry MINUS all {@code skill_*} tools (no recursion).</li>
 *   <li>Iterate: {@code chat → tool calls → results → chat …} until the sub-agent produces
 *       text (no more tool calls) or the iteration cap is hit.</li>
 *   <li>Return the sub-agent's final text as the tool result, which becomes the orchestrator's
 *       view of "what the analyst found".</li>
 * </ol>
 *
 * <p>This is the proper multi-agent pattern (Hermes, TradingAgents): each sub-skill genuinely
 * does its work in isolation and returns a real result, instead of relying on the outer LLM to
 * voluntarily follow instructions in the SKILL.md body.
 *
 * <h2>Hardening</h2>
 * <ul>
 *   <li>Each {@code llm.chat()} call is wrapped in a per-call timeout. If llama-server hangs,
 *       the sub-agent fails fast with an actionable error rather than blocking its caller (and
 *       wedging the orchestrator's virtual thread). Configurable via constructor or
 *       {@code NIZO_SUBAGENT_LLM_TIMEOUT_MS} env var.</li>
 *   <li>Tool results are truncated before being appended to the message history. Web fetches in
 *       particular can return ~50KB; with cap=20 the context blows past the model's window.
 *       Configurable via {@code NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS}.</li>
 *   <li>Timeouts run on a dedicated daemon executor — never the common ForkJoinPool — so a
 *       wedged LLM can't starve other CPU-bound work in the JVM.</li>
 * </ul>
 */
public final class SubAgentSkillTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(SubAgentSkillTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Default per-{@code llm.chat()} timeout — 3 min. Long enough for Q8 + 256K decode of long prompts. */
    public static final long DEFAULT_PER_CALL_TIMEOUT_MS = 180_000L;
    /** Default per-tool-result truncation cap. */
    public static final int DEFAULT_TOOL_RESULT_MAX_CHARS = 4_000;

    /**
     * Daemon executor used to enforce per-call LLM timeouts. Keep a dedicated pool: blocking on
     * the common ForkJoinPool would starve other CPU-bound work in the JVM. Daemon threads so
     * the JVM can shut down cleanly even if a wedged llama-server connection is still pending.
     */
    private static final Executor TIMEOUT_EXECUTOR = Executors.newCachedThreadPool(new ThreadFactory() {
        private final AtomicInteger seq = new AtomicInteger(0);
        @Override public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "nizo-subagent-llm-" + seq.incrementAndGet());
            t.setDaemon(true);
            return t;
        }
    });

    /**
     * Tool-name → fenced-block-language map for the chart-fence injection feature.
     *
     * <p>When a sub-agent calls one of these tools during its LLM loop, we keep the FULL JSON
     * result and after the LLM finishes its section we replace any {@code [CHART:type]}
     * placeholder it emitted with a real fenced block. Concretely the LLM is instructed (via
     * SKILL.md) to write things like:
     *
     * <pre>{@code
     * ### Income statement
     * [CHART:chart-financials]
     *
     * Apple grew revenue 7.2% y/y in FY2025 driven by Services...
     * }</pre>
     *
     * <p>And the post-process expands {@code [CHART:chart-financials]} into
     * {@code ```chart-financials\n{...50KB JSON...}\n```}. The LLM never has to retype JSON it
     * already received as a tool result — saves ~5x on token output and avoids the per-call
     * timeout we kept hitting on data-heavy fundamentals/analyst sections.
     */
    static final Map<String, String> TOOL_TO_CHART;
    static {
        Map<String, String> m = new HashMap<>();
        m.put("stock_fundamentals",       "chart-financials");
        m.put("stock_analyst_ratings",    "chart-analyst");
        m.put("stock_insider_activity",   "chart-insider");
        m.put("stock_earnings_history",   "chart-earnings");
        m.put("stock_buffett_score",      "chart-buffett");
        m.put("historical_price",         "chart-interactive");
        m.put("technical_indicators",     "chart-tech");
        TOOL_TO_CHART = Map.copyOf(m);
    }

    /** Matches a placeholder line like {@code [CHART:chart-financials]} (whole-line, optional surrounding whitespace). */
    private static final Pattern CHART_PLACEHOLDER = Pattern.compile(
            "(?m)^[ \\t]*\\[CHART:(chart-[a-z0-9-]+)\\][ \\t]*$");

    /**
     * LLMs sometimes emit the placeholder INSIDE an empty triple-backtick fence wrapper
     * because SKILL.md shows the placeholder example wrapped in {@code ```} for readability.
     * After pass 1 expands the inner placeholder, the surrounding empty fence markers stay
     * and render as stray code blocks. This regex catches the wrapped form so we can strip
     * the wrapping in a pre-pass before pass 1 runs. Flow:
     * <pre>
     *     ```                     ← stray opener
     *     [CHART:chart-X]
     *     ```                     ← stray closer
     * </pre>
     * is rewritten to just {@code [CHART:chart-X]} on its own line. The capture group
     * accepts ONE OR MORE placeholder lines so a fence wrapping multiple placeholders also
     * gets stripped (the LLM occasionally batches them together). Blank lines between the
     * fence opener / closer and the placeholder are tolerated.
     */
    private static final Pattern WRAPPED_CHART_PLACEHOLDER = Pattern.compile(
            "(?ms)^[ \\t]*```[ \\t]*\\r?\\n(?:[ \\t]*\\r?\\n)*"
          + "((?:[ \\t]*\\[CHART:chart-[a-z0-9-]+\\][ \\t]*\\r?\\n(?:[ \\t]*\\r?\\n)*)+)"
          + "[ \\t]*```[ \\t]*$");

    /**
     * LLMs sometimes wrap a canonical {@code ```chart-X ... ```} fence inside ANOTHER empty
     * pair of triple-backtick markers (same root cause: SKILL.md shows the example wrapped
     * for readability). The outer empty fence renders as a stray code block in markdown.
     * This is matched in a separate pre-pass and the outer wrapper is stripped, leaving
     * only the canonical inner fence. Blank lines between the wrappers and the inner fence
     * are tolerated (LLM frequently inserts a blank line for readability).
     */
    private static final Pattern WRAPPED_CHART_FENCE = Pattern.compile(
            "(?ms)^[ \\t]*```[ \\t]*\\r?\\n(?:[ \\t]*\\r?\\n)*"
          + "([ \\t]*```chart-[a-z0-9-]+[ \\t]*\\r?\\n.*?\\r?\\n[ \\t]*```[ \\t]*\\r?\\n)"
          + "(?:[ \\t]*\\r?\\n)*[ \\t]*```[ \\t]*$");

    /**
     * Pattern that finds a candidate "malformed chart reference" — chart-X immediately
     * followed by an opening brace, with optional leading backticks. Used as a CANDIDATE
     * detector; the rescue routine then does balanced-brace scanning in plain Java because
     * regex can't handle nested braces in JSON.
     *
     * <p>Examples this captures (the JSON body extends past the regex match):
     * <pre>
     *   `chart-financials {"ticker":"AAPL", ...}`            ← single-backtick wrap (May 2026 bug)
     *   chart-financials {"ticker":...}                       ← no backticks at all
     *   `\`\`chart-financials {...}\`\`\``                   ← triple-backtick all on one line
     * </pre>
     */
    private static final Pattern MALFORMED_CHART_OPEN = Pattern.compile(
            "(?ms)`{0,3}\\s*(chart-[a-z0-9-]+)\\s*\\{",
            Pattern.CASE_INSENSITIVE);

    /**
     * Virtual-thread executor for fanning out concurrent child tool calls within one sub-agent.
     * Most child tools (web_search, web_fetch, http_json, FMP HTTP, stock_*) are I/O-bound so
     * virtual threads are essentially free. Distinct from {@link #TIMEOUT_EXECUTOR} which is
     * used to enforce per-call LLM timeouts.
     */
    private static final java.util.concurrent.ExecutorService CHILD_DISPATCH_EXECUTOR =
            java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor();

    /** One child tool call's result + timing. {@code iter} preserved so events tag the correct iteration. */
    private record TimedChildResult(int iter, ToolCall call, ToolResult result, long elapsedMs) {}

    private final SkillManifest manifest;
    private final LlmClient llm;
    private final Supplier<ToolRegistry> parentToolsRef;  // lazy: registry is built around us
    private final String model;
    private final int maxIterations;
    private final long perCallTimeoutMs;
    private final int toolResultMaxChars;

    public SubAgentSkillTool(SkillManifest manifest,
                             LlmClient llm,
                             Supplier<ToolRegistry> parentToolsRef,
                             String model,
                             int maxIterations) {
        this(manifest, llm, parentToolsRef, model, maxIterations,
                resolveTimeoutMs(),
                resolveToolResultMaxChars());
    }

    public SubAgentSkillTool(SkillManifest manifest,
                             LlmClient llm,
                             Supplier<ToolRegistry> parentToolsRef,
                             String model,
                             int maxIterations,
                             long perCallTimeoutMs,
                             int toolResultMaxChars) {
        this.manifest = manifest;
        this.llm = llm;
        this.parentToolsRef = parentToolsRef;
        this.model = model;
        this.maxIterations = Math.max(1, maxIterations);
        this.perCallTimeoutMs = perCallTimeoutMs <= 0 ? DEFAULT_PER_CALL_TIMEOUT_MS : perCallTimeoutMs;
        this.toolResultMaxChars = toolResultMaxChars <= 0 ? DEFAULT_TOOL_RESULT_MAX_CHARS : toolResultMaxChars;
    }

    private static long resolveTimeoutMs() {
        String v = System.getenv("NIZO_SUBAGENT_LLM_TIMEOUT_MS");
        if (v == null || v.isBlank()) return DEFAULT_PER_CALL_TIMEOUT_MS;
        try { return Long.parseLong(v.trim()); }
        catch (NumberFormatException e) {
            LOG.warn("invalid NIZO_SUBAGENT_LLM_TIMEOUT_MS='{}', falling back to default {}ms", v, DEFAULT_PER_CALL_TIMEOUT_MS);
            return DEFAULT_PER_CALL_TIMEOUT_MS;
        }
    }

    private static int resolveToolResultMaxChars() {
        String v = System.getenv("NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS");
        if (v == null || v.isBlank()) return DEFAULT_TOOL_RESULT_MAX_CHARS;
        try { return Integer.parseInt(v.trim()); }
        catch (NumberFormatException e) {
            LOG.warn("invalid NIZO_SUBAGENT_TOOL_RESULT_MAX_CHARS='{}', falling back to default {}", v, DEFAULT_TOOL_RESULT_MAX_CHARS);
            return DEFAULT_TOOL_RESULT_MAX_CHARS;
        }
    }

    @Override public String name() { return "skill_" + manifest.name(); }

    @Override
    public String description() {
        String d = manifest.modelDescription();
        return (d == null || d.isBlank())
                ? "Run the '" + manifest.name() + "' sub-agent."
                : d;
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "input": { "type": "string", "description": "Context to hand to the sub-agent (e.g. ticker, question, focus area)." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        String input = parseInput(argumentsJson);
        LOG.info("sub-agent {} starting (input.length={})", name(), input.length());

        ToolRegistry parentTools = parentToolsRef.get();
        if (parentTools == null) {
            return ToolResult.error("sub-agent " + manifest.name()
                    + " cannot run: parent tool registry not yet initialized");
        }

        // Build the sub-agent's tool catalogue: everything except other skill_* tools.
        // This prevents recursion and keeps the sub-agent focused on doing primitive work
        // (web_search, web_fetch, http_json, stock_quote, current_time, …).
        List<ToolDef> childToolDefs = new ArrayList<>();
        for (Tool t : parentTools.all()) {
            if (t.name().startsWith("skill_")) continue;
            childToolDefs.add(new ToolDef(t.name(), t.description(), t.parametersJsonSchema()));
        }

        // Anchor the current date AND fiscal-year context. Qwen3.6's internal "today" is
        // ~1 year stale, so it queries "FY2024" when FY2025 has already completed. Without
        // this prefix the analyst returns out-of-date numbers.
        LocalDate today = LocalDate.now(ZoneId.systemDefault());
        int year = today.getYear();
        // Most US companies' fiscal year aligns with calendar year; Apple's ends in September.
        // The most recently *completed* fiscal year is the prior calendar year.
        int latestFy = year - 1;
        String datedInput = "**Today's date is " + today.format(DateTimeFormatter.ISO_LOCAL_DATE) + "**.\n\n"
                + "Time anchoring (do not deviate):\n"
                + "- Current calendar year: " + year + "\n"
                + "- Most recently COMPLETED fiscal year for most US companies: FY" + latestFy + "\n"
                + "- When searching financials, query for FY" + latestFy + " FIRST. Use FY" + (latestFy - 1)
                + " or FY" + (latestFy - 2) + " only as historical comparisons.\n"
                + "- 'Recent' = last 90 days. 'This year' = " + year + ".\n"
                + "- DO NOT query stale years like " + (year - 3) + "/" + (year - 4)
                + " as if they were current.\n\n"
                + "Query formatting (do not deviate):\n"
                + "- Pass plain text only to web_search. NEVER markdown.\n"
                + "- Bare domains: `site:macrotrends.net` — NOT `site:[macrotrends.net](http://macrotrends.net)`.\n"
                + "- No `[text](url)` patterns ANYWHERE in your queries.\n\n"
                + "Input: " + input;

        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(buildSystemPrompt()));
        messages.add(ChatMessage.user(datedInput));

        long t0 = System.nanoTime();
        int totalCalls = 0;
        String lastContent = "";
        int nudgeCount = 0;

        // Chart-fence injection cache: when the sub-agent calls one of the structured-data tools
        // below, we keep the FULL JSON result (NOT the truncated copy fed to the LLM) so we can
        // splice it back into the LLM's final markdown after it finishes. The LLM only emits a
        // tiny placeholder like "[CHART:chart-financials]" — it never has to retype 50KB of JSON
        // into a fenced block. This is the dominant fix for the 180s LLM-timeout we kept seeing
        // on data-heavy sections (fundamentals, analyst-estimates).
        Map<String, String> chartCache = new HashMap<>();

        // Per-sub-skill hard tool budget. Once exceeded, the LLM is FORCED to produce text
        // (no tools list passed) on the next call. Without this enforcement, the LLM's
        // SKILL.md "STOP-AND-WRITE" rules were unreliable — sub-skills like news_analyst
        // would keep firing web_search / web_fetch indefinitely and never emit a section.
        // Result was a "Coverage gap" placeholder in the master report (May 2026 incident:
        // 4 of 8 sub-skills returned 0 chars on a single run; user saw 4 visible gaps).
        int hardToolBudget = budgetFor(manifest.name());

        // Track the tool calls + brief summaries the sub-agent made. Used both to dump
        // a "Data gathered" appendix when the LLM stays silent (so the user sees concrete
        // evidence rather than just a placeholder) and to feed back into the forced-write
        // nudge as ammunition.
        java.util.List<String> toolCallLog = new java.util.ArrayList<>();

        for (int iter = 0; iter < maxIterations; iter++) {
            // Code-level STOP-AND-WRITE: budget exceeded → force text-only completion.
            // Pass an empty tools list so the LLM physically cannot call another tool;
            // append a user message that re-orients it to writing. This bypasses the
            // SKILL.md text rules which Qwen sometimes ignores.
            boolean budgetExceeded = totalCalls >= hardToolBudget;
            List<ToolDef> effectiveTools = budgetExceeded ? java.util.List.of() : childToolDefs;
            if (budgetExceeded && nudgeCount == 0) {
                LOG.warn("sub-agent {} hit hard tool budget ({} calls); forcing text-only completion",
                        name(), totalCalls);
                // Structured nudge with a section skeleton — gives the LLM a concrete
                // shape to fill in rather than a vague "write your section" instruction.
                // Qwen3.6 was sometimes emitting empty content because the open-ended
                // instruction left it unsure what to write (May 2026 — news_analyst still
                // silent even with tools stripped).
                String skeleton = sectionSkeletonFor(manifest.name());
                messages.add(ChatMessage.user(
                        "TOOL BUDGET EXHAUSTED. You made " + totalCalls + " tool calls — "
                      + "no more are allowed. The assistant MUST emit a written section now. "
                      + "Empty / silent replies will be rejected and replaced by a coverage-gap "
                      + "placeholder in the master report, which frustrates the user.\n\n"
                      + "Fill in this template using the evidence you gathered above. Replace "
                      + "the bracketed prompts with concrete text drawn from the tool results "
                      + "in this conversation. Keep it short if data is sparse — one sentence "
                      + "per slot is fine. DO NOT request more tools. Begin your reply now:\n\n"
                      + skeleton));
                nudgeCount++;
            }
            ChatResponse resp;
            try {
                // After the SECOND nudge (i.e., we're about to attempt a third round and the
                // model has stayed silent twice), force-disable Qwen's thinking mode for this
                // call. Symptom we're fixing: Qwen3.6 sometimes spends its whole completion
                // budget on reasoning_content and emits zero `content` tokens, so the loop
                // returns blank. Disabling thinking via the OpenAI extra_body field flushes
                // straight to content tokens. Only applies to dispatchers that understand the
                // chat_template_kwargs convention (Qwen via llama-server does).
                java.util.Map<String, Object> extras = (nudgeCount >= 2)
                        ? java.util.Map.of("chat_template_kwargs",
                                            java.util.Map.of("enable_thinking", false))
                        : java.util.Map.of();
                resp = chatWithTimeout(new ChatRequest(model, messages, effectiveTools, null, null, false, extras));
            } catch (TimeoutException te) {
                LOG.warn("sub-agent {} chat TIMED OUT at iter {} after {}ms (cachedCharts={})",
                        name(), iter, perCallTimeoutMs, chartCache.size());
                // If we already collected at least one chart-bound tool result, return those as
                // a partial success rather than a hard error — the analyst commentary is nice
                // but the data widgets are the main user-facing deliverable. The orchestrator
                // sees this as "done" instead of "failed" and the master report still includes
                // the chart blocks.
                if (!chartCache.isEmpty()) {
                    String note = "_(The " + manifest.name() + " sub-agent timed out at iter "
                            + iter + " after " + (perCallTimeoutMs / 1000) + "s of LLM synthesis, "
                            + "but successfully collected " + chartCache.size()
                            + " chart-bound tool result(s) below.)_";
                    return ToolResult.ok(injectChartFences(note, chartCache, /*appendUnused=*/true));
                }
                return ToolResult.error("Sub-agent " + manifest.name() + "'s LLM call timed out after "
                        + (perCallTimeoutMs / 1000) + "s. Either the prompt is too long or "
                        + "llama-server is unresponsive. (Set NIZO_SUBAGENT_LLM_TIMEOUT_MS to override.)");
            } catch (Exception e) {
                LOG.warn("sub-agent {} chat failed at iter {}: {}", name(), iter, e.toString());
                return ToolResult.error("sub-agent " + manifest.name() + " LLM call failed: "
                        + e.getClass().getSimpleName() + ": " + e.getMessage());
            }

            if (!resp.hasToolCalls()) {
                lastContent = resp.content() == null ? "" : resp.content();
                if (lastContent.isBlank() && nudgeCount < 3) {
                    // Qwen3.6 sometimes returns no tool calls AND empty content — model gave up
                    // mid-task. Push back THREE times with progressively harder instructions:
                    //   1) Plain "write now"
                    //   2) "Stop, output the section, do not think"
                    //   3) Same message, but the NEXT iteration's chatRequest gets
                    //      enable_thinking=false in extra_body (see ChatRequest construction
                    //      below) to bypass the reasoning_content-only failure mode entirely.
                    LOG.warn("sub-agent {} returned empty content at iter {}; nudge {} (thinkingMode={})",
                            name(), iter, nudgeCount + 1,
                            (nudgeCount >= 1) ? "OFF on next call" : "default");
                    messages.add(ChatMessage.assistant("(researching done)"));
                    String nudge;
                    if (nudgeCount == 0) {
                        nudge = "Now WRITE your final section as the assistant message. " +
                                "No tool calls, no thinking — just the structured markdown the playbook " +
                                "describes. Begin immediately with the first heading.";
                    } else if (nudgeCount == 1) {
                        nudge = "STOP. Output your section now. Even if data is partial, write what you have. " +
                                "Use the structure: ### Snapshot, then key numbers, then verdict. " +
                                "Do not search, do not fetch, do not think — write the markdown text right now.";
                    } else {
                        // Final attempt — paired with enable_thinking=false in the next request.
                        nudge = "Write the section. One paragraph minimum. Use what you have. " +
                                "Do not output any reasoning. Begin with the section heading and write " +
                                "plain prose underneath. Output it now.";
                    }
                    messages.add(ChatMessage.user(nudge));
                    nudgeCount++;
                    continue;
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                LOG.info("sub-agent {} done in {}ms, {} iters, {} calls, {} chars",
                        name(), ms, iter + 1, totalCalls, lastContent.length());
                if (lastContent.isBlank()) {
                    // Persistent silence — return a structured placeholder so the orchestrator
                    // can keep going and produce a partial report rather than dying on this section.
                    // Even a silent sub-agent gets its cached charts auto-attached at the end so
                    // the master report still has the data widgets to render. We ALSO surface
                    // the tool-call log we accumulated so the user can see what was looked at —
                    // far more useful than a generic "coverage gap" placeholder.
                    LOG.warn("sub-agent {} silent after {} nudges, returning placeholder ({} tool-call summaries available)",
                            name(), nudgeCount, toolCallLog.size());
                    String fallback = humanFallbackWithEvidence(manifest.name(), totalCalls, toolCallLog);
                    return ToolResult.ok(injectChartFences(fallback, chartCache, /*appendUnused=*/true));
                }
                return ToolResult.ok(injectChartFences(lastContent, chartCache, /*appendUnused=*/true));
            }

            messages.add(ChatMessage.assistantToolCalls(resp.toolCalls()));
            AgentEventSink sink = AgentEventContext.current();

            // Parallel tool dispatch — fan tool calls out onto virtual threads when the LLM
            // emits more than one in a single round. Most sub-agent tools are I/O-bound
            // (web_search, web_fetch, http_json, FMP) so the CPU is rarely the bottleneck.
            // Order is preserved: we collect futures in declaration order, then join in
            // declaration order so messages.add(toolResult) lines up with the call list.
            //
            // chartCache is a HashMap (not thread-safe), so we collect cache entries into a
            // separate concurrent map and merge them after the join — keeps the hot loop
            // synchronization-free.
            List<ToolCall> calls = resp.toolCalls();
            List<java.util.concurrent.CompletableFuture<TimedChildResult>> futures = new ArrayList<>(calls.size());
            // Capture the sink so each VT can re-bind AgentEventContext (thread-locals don't
            // inherit across VTs — without this any nested tool that emits via
            // AgentEventContext.current() goes to NOOP and its events vanish from the UI).
            final AgentEventSink capturedSink = sink;
            // Prefix child tool call IDs with this sub-agent's name so the UI can attribute
            // events correctly during parallel execution. Without the prefix, when multiple
            // sub-agents (e.g. fundamentals + analyst + news) run concurrently and each makes
            // inner tool calls, the frontend can't tell which parent each inner call belongs
            // to — they all collapse onto whichever activeAgent variable happened to be set
            // most recently. Result: tiles flip to "done" but their detail panel says "no tool
            // calls yet" because all inner calls were attributed to a single tile. Tag format
            // is "<parent-skill>::<original-callId>"; the frontend extracts the prefix.
            final String agentTag = name();
            for (ToolCall call : calls) {
                totalCalls++;
                String taggedId = agentTag + "::" + call.id();
                sink.emit(new AgentEvent.ToolCallStart(iter, taggedId, call.name(), call.argumentsJson()));
                final int iterFinal = iter;
                futures.add(java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                    AgentEventContext.set(capturedSink);
                    try {
                        long ts = System.nanoTime();
                        ToolResult r = executeChildCall(call);
                        long ms = (System.nanoTime() - ts) / 1_000_000;
                        return new TimedChildResult(iterFinal, call, r, ms);
                    } finally {
                        AgentEventContext.clear();
                    }
                }, CHILD_DISPATCH_EXECUTOR));
            }
            for (java.util.concurrent.CompletableFuture<TimedChildResult> f : futures) {
                TimedChildResult tr;
                try {
                    tr = f.get();
                } catch (Exception e) {
                    Throwable cause = (e instanceof java.util.concurrent.ExecutionException && e.getCause() != null)
                            ? e.getCause() : e;
                    LOG.warn("sub-agent {} parallel tool dispatch failed: {}", name(), cause.toString());
                    continue;
                }
                ToolResult result = tr.result;
                String taggedResultId = agentTag + "::" + tr.call.id();
                sink.emit(new AgentEvent.ToolCallResult(tr.iter, taggedResultId, tr.call.name(),
                        result.ok(), result.content(), tr.elapsedMs));

                // Log this call for the silent-fallback. We keep a one-line summary per call
                // — tool name + the most informative arg + first ~60 chars of result. When
                // the LLM goes silent, we render this list as a "Data gathered" appendix so
                // the user can see what was looked at even when no synthesis was written.
                if (result.ok()) {
                    String summary = summarizeOneCall(tr.call, result.content());
                    if (summary != null) toolCallLog.add(summary);
                }

                // Cache the FULL (un-truncated) JSON result of any chart-bound tool. The LLM
                // sees a truncated copy in its message history; the cache holds the original
                // for post-process injection.
                if (result.ok()) {
                    String chartType = TOOL_TO_CHART.get(tr.call.name());
                    if (chartType != null && result.content() != null && !result.content().isBlank()) {
                        chartCache.put(chartType, result.content());
                    }
                }

                // Truncate before appending to the running message list. A 50KB web_fetch x 20
                // iterations would otherwise blow past the model's context window.
                String trimmed = truncateToolResult(result.content());
                messages.add(ChatMessage.toolResult(tr.call.id(), trimmed));
            }
        }

        long ms = (System.nanoTime() - t0) / 1_000_000;
        LOG.warn("sub-agent {} hit iteration cap after {}ms, {} calls", name(), ms, totalCalls);
        // Don't return error — that aborts this sub-skill in a way the orchestrator may not
        // recover from. Return a placeholder noting the cap was hit so the orchestrator can
        // keep building the report with the other analysts' output. Even on cap-out, attach
        // any chart fences we cached so at least the data widgets render.
        String capFallback = humanFallback(manifest.name(), totalCalls);
        return ToolResult.ok(injectChartFences(capFallback, chartCache, /*appendUnused=*/true));
    }

    /**
     * Friendly fallback section for the user when a sub-skill failed to produce a written
     * section. The previous text ("Treat this section as unavailable in the master report")
     * leaked implementation details into the user-facing report. May 2026: user pointed out
     * these placeholders look like errors, not informative gaps.
     */
    private static String humanFallback(String skillName, int toolCalls) {
        String topic = humanSectionTitle(skillName);
        return "_Coverage gap for **" + topic + "** — the analyst gathered evidence ("
             + toolCalls + " source lookups) but couldn't converge on a confident summary "
             + "in this run. Often a sign of sparse public data on the ticker, or a "
             + "model-context window that ran out before the section was written. The other "
             + "sections of this report are still authoritative; rerun the pipeline if you "
             + "need a fresh attempt at this dimension._";
    }

    /**
     * Hard tool-call budget per sub-skill. After {@code totalCalls} reaches this number,
     * the loop drops the tools list from the next LLM call, forcing the model to produce
     * text. Matches the budgets advertised in each SKILL.md's STOP-AND-WRITE section.
     * Bear was lowered from 14 to 10 in May 2026 after observing it silently bailing
     * at ~10 calls without ever hitting the original budget — the forced-write didn't
     * fire so the LLM never got the "you must write now" nudge.
     */
    private static int budgetFor(String skillName) {
        if (skillName == null) return 10;
        return switch (skillName) {
            case "stock_news_analyst"          -> 8;
            case "stock_sentiment_analyst"     -> 10;
            case "stock_fundamentals_analyst"  -> 8;
            case "stock_analyst_estimates"     -> 6;
            case "stock_technical_analyst"     -> 6;
            case "stock_bear_researcher"       -> 10;
            case "stock_bull_researcher"       -> 12;
            case "stock_trader"                -> 4;
            default                            -> 10;
        };
    }

    /**
     * Per-skill section skeleton handed to the LLM after the tool budget is exhausted.
     * Gives the model a concrete shape to fill in rather than the open-ended "write your
     * section". Concrete templates with bracketed prompts dramatically reduce empty-reply
     * rate on Qwen3.6 — it has a specific shape to populate from the evidence already
     * in its message history.
     */
    private static String sectionSkeletonFor(String skillName) {
        if (skillName == null) skillName = "";
        return switch (skillName) {
            case "stock_news_analyst" -> """
                ### A. Headline
                [The single most important event for this ticker / market in the last
                3-6 months — one bold sentence + date + source URL]

                ### B. Catalysts table

                | Date | Type | Event | Impact | Source |
                |---|---|---|---|---|
                | YYYY-MM-DD | [Earnings/Product/Regulatory/M&A/Macro] | [Brief description] | [tailwind/headwind/neutral] | [bare URL] |

                Aim for 3-6 rows from the evidence you gathered. If a row is missing data,
                fill it with `—`.

                ### C. Macro context (1-2 sentences)
                [What's the broader market doing that affects this name?]

                ### D. Earnings expectation (if applicable)
                [Next earnings date, consensus EPS, days away — OR "_(not applicable for
                this ticker)_"]

                ### E. Risk events on the calendar
                [1-3 upcoming dates: earnings, FDA approvals, court rulings, policy
                announcements — OR "_(none identified in this run)_"]
                """;
            case "stock_sentiment_analyst" -> """
                ### A. Sentiment grade
                [BULLISH / BEARISH / MIXED / NEUTRAL] — [one-sentence justification]

                ### B. Volume + trend
                [Mention volume: high / normal / spike. Direction: rising / flat / falling.]

                ### C. The case from each side
                **Bull camp says** — [1-2 quotes / paraphrases with attribution, or
                "_(sparse data)_"]
                **Bear camp says** — [1-2 quotes / paraphrases with attribution, or
                "_(sparse data)_"]

                ### D. Smart money vs retail divergence
                [One sentence — retail-driven? institution-driven? — OR "_(unclear from
                this run)_"]

                ### E. Red flag indicators
                [Any pump-and-dump signals, short-squeeze setups, capitulation? OR
                "_(none identified)_"]
                """;
            case "stock_fundamentals_analyst" -> """
                ### A. Snapshot
                [1-paragraph executive summary + 4-5 key numbers + one-line takeaway]

                ### B. Financial statements

                [CHART:chart-financials]

                [1 paragraph: revenue trend, margin trend, debt trajectory]

                ### C. Profitability
                - Gross / operating / net margin (latest year)
                - ROE (target ≥ 15%)
                - ROIC (target ≥ 10%)

                ### D. Cash flow
                - FCF trajectory (3y trend)
                - Buybacks + dividends as % of FCF

                ### E. Buffett-Munger scorecard

                [CHART:chart-buffett]

                [2-paragraph commentary: verdict + key drivers; address the gaps]

                **Verdict on financial quality:** [one sentence]
                """;
            case "stock_analyst_estimates" -> """
                ### A. Wall Street consensus

                [CHART:chart-analyst]

                [1 paragraph: rating + price target + recent action drift]

                ### B. Earnings track record

                [CHART:chart-earnings]

                [1 paragraph: beat-rate + streak + next earnings date]

                ### C. Insider activity

                [CHART:chart-insider]

                [1 paragraph: net 6m direction + outsized trades]

                ### D. Smart-money read
                [1 paragraph synthesizing all three above]
                """;
            case "stock_technical_analyst" -> """
                ### A. Trend
                [Direction call: bullish / neutral / bearish + one-sentence justification]

                ### B. Support / resistance
                - **Support:** ~$X
                - **Resistance:** ~$Y

                ### C. Indicators

                [CHART:chart-tech]

                ### D. Price chart

                [CHART:chart-interactive]

                ### E. Pattern recognition
                [Pattern forming? Cup-and-handle / head-and-shoulders / etc., OR
                "_(no clear pattern in this window)_"]

                ### F. Tactical read
                [One sentence: "Buy now / wait for pullback to $X / avoid until trend
                resolves"]
                """;
            case "stock_bear_researcher" -> """
                ### Bear thesis: [ONE-SENTENCE CORE THESIS]

                ### Top 3 risks
                1. **[Risk 1]** — [evidence + source]
                2. **[Risk 2]** — [evidence + source]
                3. **[Risk 3]** — [evidence + source]

                ### Counterpoint to consensus
                [1-paragraph: what's the consensus missing?]

                ### Stop-loss reference
                [Where would a bear admit they're wrong? Price level + rationale]
                """;
            case "stock_bull_researcher" -> """
                ### Bull thesis: [ONE-SENTENCE CORE THESIS]

                ### Top 3 catalysts
                1. **[Catalyst 1]** — [evidence + source]
                2. **[Catalyst 2]** — [evidence + source]
                3. **[Catalyst 3]** — [evidence + source]

                ### Counterpoint to bears
                [1-paragraph: what's the bear case missing / overweighting?]

                ### Conviction level
                [HIGH / MEDIUM / LOW + one-sentence why]
                """;
            case "stock_trader" -> """
                ### Rating: [STRONG BUY / BUY / HOLD / AVOID / SELL]

                ### Rationale
                [2-3 sentences synthesizing the bear + bull arguments — what tips the scale?]

                ### Position sizing
                [Suggested allocation as % of portfolio + one-sentence why]

                ### Key price levels
                - **Entry:** $X
                - **Take-profit:** $Y
                - **Stop-loss:** $Z

                ### Time horizon
                [Days / weeks / months / years + one-sentence why]
                """;
            default -> """
                ### Summary
                [1-paragraph synthesis of the evidence gathered]

                ### Key findings
                - [Finding 1]
                - [Finding 2]
                - [Finding 3]

                ### Conclusion
                [One sentence]
                """;
        };
    }

    /**
     * One-line summary of a tool call: tool name + most useful arg + first ~80 chars of
     * the result. Used to populate the "Data gathered" fallback when the LLM goes silent.
     * Skips noisy fields like full JSON dumps — we want what the user / analyst would
     * find informative, not the raw tool wire.
     */
    private static String summarizeOneCall(ToolCall call, String content) {
        try {
            String toolName = call.name();
            String args = call.argumentsJson();
            String snippet = "";
            if (content != null && !content.isBlank()) {
                String s = content.strip().replaceAll("\\s+", " ");
                snippet = s.length() > 80 ? s.substring(0, 80) + "…" : s;
            }
            String argSnippet = "";
            if (args != null && !args.isBlank()) {
                String a = args.strip().replaceAll("\\s+", " ");
                argSnippet = a.length() > 60 ? a.substring(0, 60) + "…" : a;
            }
            if (argSnippet.isEmpty()) return "- `" + toolName + "` → " + snippet;
            return "- `" + toolName + "` `" + argSnippet + "` → " + snippet;
        } catch (Exception e) { return null; }
    }

    /**
     * Friendly fallback section that includes the actual tool calls + their summaries —
     * far more useful than a generic "Coverage gap" placeholder. The user can see at a
     * glance what data was gathered, even when the LLM failed to synthesize a section.
     */
    private static String humanFallbackWithEvidence(String skillName, int toolCalls,
                                                    java.util.List<String> toolCallLog) {
        String topic = humanSectionTitle(skillName);
        StringBuilder sb = new StringBuilder();
        sb.append("_Coverage note for **").append(topic).append("** — the analyst ran ")
          .append(toolCalls).append(" source lookups but didn't converge on a written ")
          .append("summary. Data gathered, in case it's useful:_\n\n");
        if (toolCallLog == null || toolCallLog.isEmpty()) {
            sb.append("_(No usable tool output captured this run. Re-run the pipeline to ")
              .append("try again.)_\n");
        } else {
            // Cap to first 12 entries — too many drowns the user.
            int shown = Math.min(toolCallLog.size(), 12);
            for (int i = 0; i < shown; i++) sb.append(toolCallLog.get(i)).append('\n');
            if (toolCallLog.size() > shown) {
                sb.append("- _(...and ").append(toolCallLog.size() - shown)
                  .append(" more calls; full set in the agent panel)_\n");
            }
        }
        return sb.toString();
    }

    /** Map an internal skill name to a user-friendly topic. */
    private static String humanSectionTitle(String skillName) {
        if (skillName == null) return "this section";
        return switch (skillName) {
            case "stock_news_analyst"          -> "News + catalysts";
            case "stock_sentiment_analyst"     -> "Sentiment read";
            case "stock_fundamentals_analyst"  -> "Fundamentals";
            case "stock_analyst_estimates"     -> "Analyst estimates";
            case "stock_technical_analyst"     -> "Technical timing";
            case "stock_bear_researcher"       -> "Bear case";
            case "stock_bull_researcher"       -> "Bull case";
            case "stock_trader"                -> "Trader verdict";
            default                            -> skillName.replace('_', ' ');
        };
    }

    /**
     * Run {@code llm.chat()} on a daemon thread and wait at most {@link #perCallTimeoutMs} for
     * the response. If the LLM hangs (e.g. llama-server frozen, network partition), throws
     * {@link TimeoutException} so the caller can surface an actionable error rather than wedging
     * the calling virtual thread indefinitely.
     */
    private ChatResponse chatWithTimeout(ChatRequest req) throws TimeoutException, ExecutionException, InterruptedException {
        CompletableFuture<ChatResponse> fut = CompletableFuture.supplyAsync(() -> llm.chat(req), TIMEOUT_EXECUTOR);
        try {
            return fut.get(perCallTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException te) {
            // Best-effort cancel — the underlying HTTP call may not actually interrupt, but we
            // free the calling thread immediately and the daemon worker will be reaped on JVM exit.
            fut.cancel(true);
            throw te;
        }
    }

    /**
     * Replace {@code [CHART:type]} placeholders in the LLM's output with actual fenced blocks
     * built from cached tool JSON. Optionally appends any cached charts that the LLM didn't
     * reference at all (so the data widgets still show up in the master report).
     *
     * <p>Idempotent and safe: if {@code chartCache} is empty, returns {@code content} unchanged.
     * If a placeholder references a chart type that wasn't collected, the placeholder is left
     * in place (visible in the report so the bug is obvious — better than silent data loss).
     *
     * @param content       the LLM's final markdown
     * @param chartCache    chartType → full JSON result (e.g. "chart-financials" → "{...}")
     * @param appendUnused  if true, any chart in cache that the LLM didn't reference is
     *                      appended to the end of the section under a "Data" subheading
     * @return the markdown with placeholders expanded
     */
    public static String injectChartFences(String content, Map<String, String> chartCache, boolean appendUnused) {
        if (content == null) content = "";
        if (chartCache == null || chartCache.isEmpty()) return content;

        java.util.Set<String> usedTypes = new java.util.HashSet<>();

        // ── Pre-pass A: strip empty fence wrap around [CHART:type] placeholders.
        // LLM may emit ```\n[CHART:chart-X]\n``` (placeholder INSIDE an empty triple-backtick
        // fence) because SKILL.md formats the example with backticks for readability. Without
        // this pre-pass, pass 1 expands the inner placeholder and the outer empty fence stays
        // as a stray code block. Run twice in case nested wrappings exist (rare).
        for (int i = 0; i < 2; i++) {
            String stripped = WRAPPED_CHART_PLACEHOLDER.matcher(content).replaceAll("$1");
            if (stripped.equals(content)) break;
            content = stripped;
        }
        // ── Pre-pass B: strip empty fence wrap around an already-canonical chart fence.
        // Same root cause but the LLM (or a previous injectChartFences pass) already expanded
        // the placeholder so we have ```\n```chart-X\n{json}\n```\n```. Strip the outer pair.
        for (int i = 0; i < 2; i++) {
            String stripped = WRAPPED_CHART_FENCE.matcher(content).replaceAll("$1");
            if (stripped.equals(content)) break;
            content = stripped;
        }

        // ── Pass 1: expand canonical [CHART:type] placeholders ─────────────────
        Matcher m = CHART_PLACEHOLDER.matcher(content);
        StringBuilder pass1 = new StringBuilder(content.length() + 4096);
        while (m.find()) {
            String type = m.group(1);
            String json = chartCache.get(type);
            String replacement;
            if (json != null && !json.isBlank()) {
                replacement = "```" + type + "\n" + json + "\n```";
                usedTypes.add(type);
            } else {
                // Chart data unavailable (e.g. .BO ticker with no Yahoo/Stooq/Finnhub bars).
                // Replace with a tiny italic notice instead of leaving the raw "[CHART:type]"
                // placeholder text in the report — that looks like a templating bug to users.
                String friendly = type.startsWith("chart-") ? type.substring("chart-".length()) : type;
                replacement = "_(" + friendly + " chart unavailable — data source returned no rows)_";
            }
            // appendReplacement uses $-substitution by default; quote so $ in JSON doesn't blow up.
            m.appendReplacement(pass1, Matcher.quoteReplacement(replacement));
        }
        m.appendTail(pass1);

        // ── Pass 2: rescue malformed LLM-emitted chart references ──────────────
        // Qwen3.6 sometimes ignores SKILL.md instructions and tries to embed the chart by
        // writing `chart-X {JSON}` with single backticks (or no backticks, or only opening
        // triple-backticks). The result is markdown-rendered as text — JSON underscores
        // become italics, the data widget never shows up. This pass catches such patterns
        // and rewrites them to canonical fenced blocks using the AUTHORITATIVE cached JSON
        // (not whatever the LLM might have transcribed wrong).
        StringBuilder out = rescueMalformedChartRefs(pass1.toString(), chartCache, usedTypes);

        if (appendUnused) {
            // Auto-append any cached chart the LLM didn't reference. Visible in the order the
            // tools were called (HashMap iteration is undefined; that's fine — sections are
            // labelled by their fence type which the front-end renders distinctively).
            //
            // CRITICAL: also scan the already-rendered output for any pre-existing canonical
            // `\`\`\`chart-X` fences and mark those types used. Without this, when a sub-skill
            // emits its own canonical fence (its OWN injectChartFences pass already expanded
            // the placeholder before returning the section text), the orchestrator's outer
            // injectChartFences sees those as "not in usedTypes" and appends ANOTHER copy at
            // the bottom under "Data widgets". Symptom: the price chart renders twice in the
            // master report (May 2026 user report).
            String renderedSoFar = out.toString();
            java.util.regex.Matcher canonical =
                    java.util.regex.Pattern.compile("```(chart-[a-z0-9-]+)").matcher(renderedSoFar);
            while (canonical.find()) usedTypes.add(canonical.group(1));

            StringBuilder extras = new StringBuilder();
            for (Map.Entry<String, String> e : chartCache.entrySet()) {
                if (usedTypes.contains(e.getKey())) continue;
                if (e.getValue() == null || e.getValue().isBlank()) continue;
                extras.append("\n\n```").append(e.getKey()).append("\n")
                      .append(e.getValue()).append("\n```");
            }
            if (extras.length() > 0) {
                if (out.length() > 0 && out.charAt(out.length() - 1) != '\n') out.append('\n');
                out.append("\n_Data widgets:_").append(extras);
            }
        }

        // ── Final pass: dedupe canonical chart fences ──────────────────────────
        // After all other passes, the master report can still contain the same
        // chart-X fence multiple times when different sub-skills each chose to
        // surface it (e.g. fundamentals + analyst_estimates both emit chart-insider,
        // technical_analyst emits chart-tech in both its "F. Technical Indicators"
        // AND "G. Tactical Read" sub-sections). Keep the FIRST occurrence of each
        // chart type and collapse subsequent ones to a brief reference link so the
        // commentary still makes sense without re-rendering 50-200KB of duplicate
        // JSON. Only fires when appendUnused=true so per-sub-skill passes inside
        // SubAgentSkillTool aren't affected (those are intra-section and dedup-safe).
        if (appendUnused) {
            return dedupeChartFences(out.toString());
        }
        return out.toString();
    }

    /** Compiled once: matches a complete {@code ```chart-X\n{json}\n```} block. */
    private static final java.util.regex.Pattern CANONICAL_CHART_FENCE =
            java.util.regex.Pattern.compile("(?ms)```(chart-[a-z0-9-]+)\\r?\\n.*?\\r?\\n```");

    /**
     * Keep only the first occurrence of each canonical {@code ```chart-X``` ... ```}
     * fence in {@code content}. Replace 2nd+ occurrences with a short markdown note
     * pointing back to the chart (so any prose referencing "the chart above" still
     * reads naturally).
     */
    static String dedupeChartFences(String content) {
        if (content == null || content.isEmpty()) return content;
        java.util.Set<String> seen = new java.util.HashSet<>();
        StringBuilder out = new StringBuilder(content.length());
        java.util.regex.Matcher m = CANONICAL_CHART_FENCE.matcher(content);
        int cursor = 0;
        while (m.find()) {
            out.append(content, cursor, m.start());
            String type = m.group(1);
            if (seen.add(type)) {
                // First occurrence — keep verbatim.
                out.append(m.group());
            } else {
                // Duplicate — collapse to a short reference instead of repeating the JSON.
                out.append("_(see ").append(humanChartLabel(type)).append(" above)_");
            }
            cursor = m.end();
        }
        out.append(content, cursor, content.length());
        return out.toString();
    }

    /** User-friendly label for a chart fence type, used in the dedupe reference note. */
    private static String humanChartLabel(String type) {
        return switch (type) {
            case "chart-financials"  -> "Financial statements widget";
            case "chart-buffett"     -> "Buffett-Munger scorecard";
            case "chart-analyst"     -> "Analyst ratings widget";
            case "chart-earnings"    -> "Earnings history widget";
            case "chart-insider"     -> "Insider activity widget";
            case "chart-interactive" -> "interactive price chart";
            case "chart-tech"        -> "technical indicators dashboard";
            default                  -> type.replace("chart-", "") + " widget";
        };
    }

    /**
     * Walk {@code content} looking for malformed chart-fence-attempts (e.g. {@code `chart-X
     * {JSON}`} with single backticks instead of triple) and rewrite each one as a canonical
     * fenced block sourced from {@code chartCache}. Already-canonical fences (lines starting
     * with {@code ```chart-X}) are skipped intact.
     *
     * <p>Brace matching is done in plain Java with a depth counter (regex can't handle nested
     * braces). String-interior braces inside JSON values aren't a problem because Yahoo /
     * FMP responses don't contain literal {@code {} or }} inside any string field.
     */
    public static StringBuilder rescueMalformedChartRefs(String input,
                                                  Map<String, String> chartCache,
                                                  java.util.Set<String> usedTypes) {
        StringBuilder out = new StringBuilder(input.length() + 1024);
        Matcher mm = MALFORMED_CHART_OPEN.matcher(input);
        int cursor = 0;
        while (mm.find(cursor)) {
            int matchStart = mm.start();
            int braceStart = mm.end() - 1;          // position of the opening '{'
            String chartType = mm.group(1).toLowerCase();

            // Skip if not a recognized chart type or no cached JSON — leave LLM's text alone.
            String cached = chartCache.get(chartType);
            if (cached == null || cached.isBlank()) {
                out.append(input, cursor, matchStart + 1);
                cursor = matchStart + 1;
                continue;
            }

            // Skip if already a canonical fence: detect EITHER (a) 3 backticks consumed at
            // the start of the regex match (matchStart through to chart token), which means
            // the regex matched a ```chart-X { canonical fence — already handled by pass 1
            // expansion or already-correct from a sub-skill, OR (b) prior content on the same
            // line ends with ``` (the LLM tried to single-line the whole fence, less common).
            int btCount = 0;
            for (int p = matchStart; p < input.length() && p < matchStart + 4 && input.charAt(p) == '`'; p++) {
                btCount++;
            }
            boolean atStartOfLine = matchStart == 0 || input.charAt(matchStart - 1) == '\n';
            if (btCount >= 3 && atStartOfLine) {
                // Already a clean canonical fence — emit the opening backticks and continue
                // searching past them.
                out.append(input, cursor, matchStart + 3);
                cursor = matchStart + 3;
                continue;
            }
            // Fallback prior-line check (catches edge cases like inline ``` on prev line).
            int searchFrom = Math.max(0, matchStart - 1);
            int lineStart = input.lastIndexOf('\n', searchFrom) + 1;
            String linePrefix = (lineStart < matchStart)
                    ? input.substring(lineStart, matchStart).trim()
                    : "";
            if (linePrefix.equals("```") || linePrefix.endsWith("```")) {
                // Already a clean fence opening — pass through.
                out.append(input, cursor, matchStart + 1);
                cursor = matchStart + 1;
                continue;
            }

            // Balanced-brace scan to find the closing }.
            int depth = 1;
            int j = braceStart + 1;
            boolean inString = false;
            boolean escape = false;
            while (j < input.length() && depth > 0) {
                char c = input.charAt(j);
                if (escape) { escape = false; j++; continue; }
                if (inString) {
                    if (c == '\\') escape = true;
                    else if (c == '"') inString = false;
                } else {
                    if (c == '"') inString = true;
                    else if (c == '{') depth++;
                    else if (c == '}') depth--;
                }
                j++;
            }
            if (depth != 0) {
                // Unbalanced — bail without modifying. Better to show malformed text than
                // chew off a chunk of unrelated content.
                out.append(input, cursor, matchStart + 1);
                cursor = matchStart + 1;
                continue;
            }
            int matchEnd = j;
            // Consume any trailing backticks (rescue closing of `single-backtick` wrap).
            while (matchEnd < input.length() && input.charAt(matchEnd) == '`') matchEnd++;

            // Emit content before the match, then a canonical fence using cached JSON.
            out.append(input, cursor, matchStart);
            // Ensure the fence starts on its own line (markdown requires line-anchor).
            if (out.length() > 0 && out.charAt(out.length() - 1) != '\n') out.append('\n');
            if (out.length() < 2 || out.charAt(out.length() - 2) != '\n') out.append('\n');
            out.append("```").append(chartType).append('\n')
               .append(cached).append("\n```\n");
            usedTypes.add(chartType);
            cursor = matchEnd;
        }
        out.append(input, cursor, input.length());
        return out;
    }

    /**
     * Truncate a tool result to {@link #toolResultMaxChars} chars before appending to message
     * history. Adds a marker explaining what was cut so the LLM knows the slice is partial.
     * Results already under the limit are passed through unchanged (no marker).
     */
    String truncateToolResult(String content) {
        if (content == null) return "";
        if (content.length() <= toolResultMaxChars) return content;
        String head = content.substring(0, toolResultMaxChars);
        return head + "\n…[truncated to " + toolResultMaxChars + " chars; full content was "
                + content.length() + " chars]";
    }

    /** Build the sub-agent's system prompt from the SKILL.md body, plus a small framing preamble. */
    private String buildSystemPrompt() {
        StringBuilder sb = new StringBuilder();
        sb.append("You are the '").append(manifest.name()).append("' sub-agent. ");
        sb.append("Your job is described in the playbook below. Use the available tools to ");
        sb.append("gather real evidence — never fabricate data. When you have enough, write your ");
        sb.append("section as the FINAL assistant message (no tool calls). ");
        sb.append("Be thorough but bounded — finish in a small number of iterations.\n\n");
        sb.append("---\n\n");
        sb.append(manifest.body());
        return sb.toString();
    }

    /** Execute a tool call inside the sub-agent. Refuses skill_* (defense in depth against recursion). */
    private ToolResult executeChildCall(ToolCall call) {
        String n = call.name();
        if (n != null && n.startsWith("skill_")) {
            return ToolResult.error("sub-agent may not invoke other skills (got '" + n + "')");
        }
        ToolRegistry parentTools = parentToolsRef.get();
        if (parentTools == null) {
            return ToolResult.error("parent tool registry not yet initialized");
        }
        Tool t = parentTools.byName(n).orElse(null);
        if (t == null) {
            return ToolResult.error("unknown tool '" + n + "' in sub-agent");
        }
        try {
            return t.execute(call.argumentsJson() == null ? "{}" : call.argumentsJson());
        } catch (Exception e) {
            return ToolResult.error(t.name() + " threw: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private static String parseInput(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) return "";
        try {
            JsonNode node = MAPPER.readTree(argumentsJson);
            JsonNode in = node.get("input");
            if (in == null || in.isNull()) return "";
            return in.asText("");
        } catch (Exception e) {
            // If the orchestrator passed something malformed, hand the raw string through —
            // it's still useful context for the sub-agent.
            return argumentsJson;
        }
    }

    public SkillManifest manifest() { return manifest; }

    /** Visible for tests. */
    long perCallTimeoutMs() { return perCallTimeoutMs; }
    /** Visible for tests. */
    int toolResultMaxChars() { return toolResultMaxChars; }
}
