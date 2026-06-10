package ai.nizo.agent.loop;

import ai.nizo.agent.cache.StockReportStore;
import ai.nizo.agent.condense.CondenseConstants;
import ai.nizo.agent.condense.CondenseEngine;
import ai.nizo.agent.condense.TokenEstimator;
import ai.nizo.api.condense.CondenseRequest;
import ai.nizo.api.condense.CondenseResult;
import ai.nizo.api.memory.UserFactStore;
import ai.nizo.agent.session.SessionStore;
import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.agent.AgentEventSink;
import ai.nizo.api.chat.ChatHandler;
import ai.nizo.api.chat.IncomingMessage;
import ai.nizo.api.chat.OutgoingMessage;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.ChatStreamHandler;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Production agent loop.
 *
 * <p>Two execution modes:
 * <ul>
 *   <li>{@link #handle(IncomingMessage)} — blocking; returns the final {@link OutgoingMessage}</li>
 *   <li>{@link #runStreaming(IncomingMessage, AgentEventSink)} — emits {@link AgentEvent}s as
 *       tokens, tool calls, and the final reply land. Used by the SSE web channel.</li>
 * </ul>
 *
 * <p>State per turn:
 * <ol>
 *   <li>Recent messages from {@link SessionStore} are loaded as prefix history.</li>
 *   <li>The new user message is appended (and persisted).</li>
 *   <li>Iterative loop: {@code llm.chat(messages, tools) → execute tool calls → continue / stop}.</li>
 *   <li>Final assistant reply is persisted; tool-call intermediate turns are NOT persisted
 *       (they're expensive context that the agent re-derives next turn from current input + memory).</li>
 * </ol>
 *
 * <p>Inspired by Hermes Agent's {@code run_conversation()} loop.
 */
public final class AgentLoop implements ChatHandler {

    private static final Logger LOG = LoggerFactory.getLogger(AgentLoop.class);

    /**
     * Virtual-thread executor used to fan out concurrent tool calls when the LLM emits more
     * than one in a single assistant message. Java 21's per-task virtual threads are ideal for
     * this — most tool calls are I/O-bound (HTTP / SQL / sub-agent LLM round-trips), so the
     * usual platform-thread cost of "one OS thread per call" is unnecessary.
     *
     * <p>Pre-condition for parallelism actually saving wall time: {@code llama-server} must be
     * configured with {@code --parallel >= N}. With {@code --parallel 1}, the LLM serializes
     * concurrent requests at the GPU and the parallel dispatch only saves a small amount of
     * Java-side latency.
     */
    private static final java.util.concurrent.ExecutorService TOOL_DISPATCH_EXECUTOR =
            java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor();

    /** Result tuple from one parallel tool call — preserves the LLM's emit-order via {@code idx}. */
    private record TimedToolResult(int idx, ToolCall call, ToolResult result, long elapsedMs) {}

    /** Pre-compiled fence pattern reused across tool-result extraction. */
    private static final java.util.regex.Pattern FENCE_PATTERN =
            java.util.regex.Pattern.compile("(?ms)```(chart-[a-z0-9-]+)\\n(.*?)\\n```");

    /**
     * Hard cap on per-sub-skill LLM-visible output. Each sub-skill's text response is 5-10 KB
     * of paragraphs + scorecards + commentary; with 5+ sub-skills called in multiple rounds,
     * total context blows past llama's per-slot limit (87K tokens at parallel-3, 67K at
     * parallel-5). Truncating to 2500 chars keeps the orchestrator able to read each
     * analyst's gist + chart placeholders without burning context. The full sub-skill output
     * survives in its own per-call return value (which the orchestrator's master synthesis
     * doesn't need verbatim — it has the chart fences and a brief summary). Configurable via
     * {@code NIZO_AGENT_TOOL_RESULT_MAX_CHARS}.
     */
    private static final int SUB_SKILL_MAX_CHARS = resolveSubSkillMaxChars();
    private static int resolveSubSkillMaxChars() {
        String v = System.getenv("NIZO_AGENT_TOOL_RESULT_MAX_CHARS");
        if (v == null || v.isBlank()) return 2500;
        try { return Math.max(500, Integer.parseInt(v.trim())); }
        catch (NumberFormatException e) { return 2500; }
    }

    /**
     * Scan a tool-result string for {@code ```chart-X\n{...JSON...}\n```} blocks. For each
     * match, store the JSON in {@code chartCache} (keyed by chart-type) AND replace the entire
     * fenced block in the returned string with a tiny {@code [CHART:type]} placeholder line.
     *
     * <p>This is the token-saver: a sub-skill's tool-result with full chart fences is typically
     * 50-100 KB. After this transform the LLM-visible copy is &lt;5 KB while the canonical
     * fences live in {@code chartCache} ready for the final-reply rescue. Without this,
     * 5+ sub-skill calls accumulate &gt;100K tokens of context and llama-server returns HTTP 400
     * "exceeds context size" (verified May 2026 — 177K tokens vs 87K limit).
     *
     * @param content      the tool-result text from a sub-skill (may contain fenced charts)
     * @param chartCache   accumulated cache; canonical JSON is added per discovered fence
     * @return same content with each fence replaced by {@code [CHART:type]} (one line)
     */
    static String extractFencesAndReplaceWithPlaceholders(String content,
                                                           Map<String, String> chartCache) {
        if (content == null || content.isEmpty()) return content == null ? "" : content;
        java.util.regex.Matcher m = FENCE_PATTERN.matcher(content);
        StringBuilder out = new StringBuilder(content.length());
        while (m.find()) {
            String type = m.group(1);
            String json = m.group(2);
            // First sub-skill to emit a given chart type wins; per-ticker each type is unique
            // in practice (chart-financials only comes from stock_fundamentals_analyst).
            chartCache.putIfAbsent(type, json);
            // Replace fenced block with placeholder. Use Matcher.quoteReplacement defensively
            // since user content could include $ characters that appendReplacement treats specially.
            m.appendReplacement(out, java.util.regex.Matcher.quoteReplacement("[CHART:" + type + "]"));
        }
        m.appendTail(out);
        return out.toString();
    }

    private final LlmClient llm;
    private final ToolRegistry tools;
    private final SessionStore sessions;
    private final UserFactStore userFacts;   // nullable; null = no cross-session memory
    private final CondenseEngine condense;   // nullable; null = no auto/reactive condense
    private final StockReportStore stockReports;  // nullable; null = don't persist
    private final String model;
    private final String systemPrompt;
    private final int maxIterations;
    private final int historyMessages;

    public AgentLoop(LlmClient llm,
                     ToolRegistry tools,
                     SessionStore sessions,
                     UserFactStore userFacts,
                     CondenseEngine condense,
                     StockReportStore stockReports,
                     String model,
                     String systemPrompt,
                     int maxIterations,
                     int historyMessages) {
        this.llm = llm;
        this.tools = tools;
        this.sessions = sessions;
        this.userFacts = userFacts;
        this.condense = condense;
        this.stockReports = stockReports;
        this.model = model;
        this.systemPrompt = systemPrompt;
        this.maxIterations = Math.max(1, maxIterations);
        this.historyMessages = Math.max(2, historyMessages);
    }

    /** Backwards-compat constructor: memory but no condense. */
    public AgentLoop(LlmClient llm,
                     ToolRegistry tools,
                     SessionStore sessions,
                     UserFactStore userFacts,
                     String model,
                     String systemPrompt,
                     int maxIterations,
                     int historyMessages) {
        this(llm, tools, sessions, userFacts, null, null, model, systemPrompt, maxIterations, historyMessages);
    }

    /** Backwards-compat constructor for callers without a memory or condense layer. */
    public AgentLoop(LlmClient llm,
                     ToolRegistry tools,
                     SessionStore sessions,
                     String model,
                     String systemPrompt,
                     int maxIterations,
                     int historyMessages) {
        this(llm, tools, sessions, null, null, null, model, systemPrompt, maxIterations, historyMessages);
    }

    /** Backwards-compat constructor without stock-reports cache. */
    public AgentLoop(LlmClient llm,
                     ToolRegistry tools,
                     SessionStore sessions,
                     UserFactStore userFacts,
                     CondenseEngine condense,
                     String model,
                     String systemPrompt,
                     int maxIterations,
                     int historyMessages) {
        this(llm, tools, sessions, userFacts, condense, null, model, systemPrompt, maxIterations, historyMessages);
    }

    public CondenseEngine condenseEngine() { return condense; }
    public String systemPrompt() { return systemPrompt; }

    @Override
    public OutgoingMessage handle(IncomingMessage in) {
        AccumulatingSink sink = new AccumulatingSink();
        runStreaming(in, sink);
        return OutgoingMessage.of(sink.finalReply.get());
    }

    public void runStreaming(IncomingMessage in, AgentEventSink sink) {
        long t0 = System.nanoTime();
        // Tag the calling thread with the chatId so MeasuredTool (down-stack) records
        // each tool invocation against the right conversation. Cleared in finally below.
        ai.nizo.tools.registry.UsageTracker.setCurrentChatId(in.chatId());
        // Bind the SSE sink to the thread for the WHOLE turn, not just the in-loop tool-call
        // branch. Virtual threads are reused across requests; if a future change calls a tool
        // (or anything that emits an event) outside the inner block, leaving the binding in
        // place during e.g. cleanup ensures we don't surface to a stale sink from the previous
        // request that landed on this thread. try/finally guarantees we always clear.
        ai.nizo.api.agent.AgentEventContext.set(sink);
        try {
            runStreamingInner(in, sink, t0);
        } finally {
            ai.nizo.api.agent.AgentEventContext.clear();
            ai.nizo.tools.registry.UsageTracker.clearCurrentChatId();
        }
    }

    private void runStreamingInner(IncomingMessage in, AgentEventSink sink, long t0) {
        ChatMessage userMessage = buildUserMessage(in);

        // Auto-condense check: if estimated prompt tokens are within CONDENSE_BUFFER_TOKENS of the
        // effective context window, condense BEFORE we send. Skipped when the breaker is open or
        // there's no engine wired.
        checkTokenBudget(in, sink);

        List<ChatMessage> messages = new ArrayList<>();
        String effectiveSystem = buildSystemPromptWithMemory(in);
        if (effectiveSystem != null && !effectiveSystem.isBlank()) {
            messages.add(ChatMessage.system(effectiveSystem));
        }
        messages.addAll(sessions.recent(in.chatId(), historyMessages));
        messages.add(userMessage);

        // Per-turn cache of canonical chart fences extracted from sub-skill tool-results.
        // Each sub-skill emits markdown like ```chart-financials\n{50KB JSON}\n```. We extract
        // those fences ONCE, store the JSON here keyed by chart-type, and replace the LLM-visible
        // copy with a tiny [CHART:type] placeholder. At FinalReply time we expand any
        // placeholders the orchestrator's LLM emitted back into canonical fences. Net effect:
        // the LLM never has to see (or re-emit) tens of thousands of JSON tokens.
        Map<String, String> orchestratorChartCache = new HashMap<>();

        // When true, a tool returned content prefixed with VERBATIM_MARKER and we want to
        // skip the next LLM round entirely — the tool's content is the assistant's reply.
        boolean verbatimShortCircuit = false;

        sessions.append(in.chatId(), userMessage);

        int iteration = 0;
        int totalTools = 0;
        int promptTokens = 0;
        int completionTokens = 0;
        String finalContent = null;
        String stopReason = "max_iterations";
        boolean reactiveTried = false;

        // Cap completion size for stock-skill chats — Qwen3.6 thinking mode can
        // emit unbounded `<think>` blocks, hanging the run for 15+ minutes
        // before producing any user content. 12000 per iteration is
        // generous: leaves plenty of headroom for the trader's master report
        // (typically 3-5k content tokens) even after Qwen burns several
        // thousand thinking tokens. Lower caps (tested 6000) starved the
        // final synthesis iteration → stop=length mid-report.
        boolean isStockChat = in.chatId() != null && in.chatId().startsWith("stock-");
        for (; iteration < maxIterations; iteration++) {
            ChatRequest req = ChatRequest.of(model, messages).withTools(tools.toolDefs());
            if (isStockChat) req = req.withMaxTokens(12000);
            CompletableFuture<ChatResponse> future = new CompletableFuture<>();
            final int currentIter = iteration;

            llm.streamChat(req, new ChatStreamHandler() {
                @Override public void onToken(String token) {
                    sink.emit(new AgentEvent.TokenChunk(currentIter, token));
                }
                @Override public void onThinking(String token) {
                    sink.emit(new AgentEvent.ThinkingChunk(currentIter, token));
                }
                @Override public void onComplete(ChatResponse response) { future.complete(response); }
                @Override public void onError(Throwable t) { future.completeExceptionally(t); }
            });

            ChatResponse resp;
            try {
                resp = future.get();
            } catch (Exception e) {
                Throwable cause = rootCause(e);
                // User pressed Stop — propagate as a clean cancellation, not a model error.
                if (e instanceof InterruptedException || cause instanceof InterruptedException) {
                    Thread.currentThread().interrupt(); // restore so worker sees the interrupt
                    finalContent = "[stopped]";
                    stopReason = "stopped";
                    sink.emit(new AgentEvent.Warning(iteration, "[user stop]"));
                    break;
                }
                String causeMsg = cause.getMessage() == null ? "" : cause.getMessage();
                // Reactive condense: if this looks like a prompt-too-long error, condense once
                // and retry the same iteration. Guard with reactiveTried so we don't loop.
                if (!reactiveTried && condense != null && !condense.isOpen() && isPromptTooLong(causeMsg)) {
                    reactiveTried = true;
                    LOG.warn("prompt-too-long detected; firing reactive condense for chat={}", in.chatId());
                    sink.emit(new AgentEvent.Warning(iteration,
                            "context overflow — condensing conversation and retrying"));
                    CondenseResult r = condense.condense(
                            CondenseRequest.full(in.chatId(), in.userId(), CondenseRequest.Trigger.REACTIVE));
                    if (r.ok()) {
                        // Rebuild the messages list from the now-condensed history. Keep the new
                        // user message at the tail (we already appended it before this turn).
                        messages.clear();
                        if (effectiveSystem != null && !effectiveSystem.isBlank()) {
                            messages.add(ChatMessage.system(effectiveSystem));
                        }
                        messages.addAll(sessions.recent(in.chatId(), historyMessages));
                        // sessions.recent already includes the user message we appended pre-loop.
                        iteration--; // re-run this iteration with the slimmer prompt
                        continue;
                    } else {
                        sink.emit(new AgentEvent.Warning(iteration,
                                "reactive condense failed: " + r.error()));
                    }
                }
                LOG.warn("streamChat failed at iter {}: {}", iteration, e.toString());
                // Map common transport errors to friendlier messages so the UI is useful.
                String userFacing;
                String lower = causeMsg.toLowerCase();
                if (lower.contains("error in input stream") || lower.contains("connection reset")
                        || lower.contains("broken pipe")) {
                    userFacing = "The model dropped the connection mid-stream. This usually means the "
                            + "request hit the HTTP read timeout (try a shorter prompt or set "
                            + "NIZO_LLM_TIMEOUT_MIN higher), or llama-server crashed.";
                } else if (lower.contains("connect")) {
                    userFacing = "Could not reach the LLM endpoint — is llama-server running on :8080?";
                } else {
                    userFacing = "Model error: " + causeMsg;
                }
                finalContent = userFacing;
                stopReason = "llm_error";
                sink.emit(new AgentEvent.Warning(iteration, "[" + cause.getClass().getSimpleName() + "] " + causeMsg));
                break;
            }
            if (resp.usage() != null) {
                promptTokens += resp.usage().promptTokens();
                completionTokens += resp.usage().completionTokens();
            }

            if (!resp.hasToolCalls()) {
                finalContent = resp.content() == null ? "" : resp.content();
                stopReason = resp.finishReason() == null ? "stop" : resp.finishReason();
                break;
            }

            // Continue: append assistant tool_calls turn, then execute and append tool results.
            // AgentEventContext is bracketed at runStreaming() level — sub-agents inherit the
            // same sink for the entire turn so their inner ToolCallStart/Result events surface
            // to the UI even if they happen outside this block in the future.
            messages.add(ChatMessage.assistantToolCalls(resp.toolCalls()));

            // Parallel tool dispatch — fan out independent tool calls onto virtual threads so
            // the orchestrator can run e.g. all five skill_stock_*_analyst sub-agents
            // concurrently. Pre-condition for this win: llama-server must be configured with
            // --parallel >1 (otherwise the LLM serializes them at the GPU).
            //
            // Order preservation: we wait for ALL futures to complete, then append
            // tool-results to messages in the same order the LLM emitted them. This keeps the
            // chat-message history deterministic regardless of which sub-agent finishes first.
            //
            // Per-tool UserContext: ThreadLocals don't propagate across virtual-thread
            // boundaries, so we set/clear inside each task instead of relying on outer scope.
            final String userId = in.userId();
            // Capture the sink so it can be re-bound inside each virtual thread. ThreadLocals
            // (including AgentEventContext) are NOT inherited by VTs by default — without this,
            // tools that emit events via AgentEventContext.current() see NOOP and their inner
            // ToolCallStart/Result events never reach the SSE stream (UI tiles stay idle).
            final AgentEventSink capturedSink = sink;
            int callIdx = 0;
            List<ToolCall> calls = resp.toolCalls();
            List<java.util.concurrent.CompletableFuture<TimedToolResult>> futures = new ArrayList<>(calls.size());
            for (ToolCall call : calls) {
                final int idx = callIdx++;
                totalTools++;
                sink.emit(new AgentEvent.ToolCallStart(iteration, call.id(), call.name(), call.argumentsJson()));
                futures.add(java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                    // Re-bind both contexts on this virtual thread so nested tools (sub-agents,
                    // deterministic orchestrators) can read them.
                    ai.nizo.api.tool.UserContext.set(userId);
                    ai.nizo.api.agent.AgentEventContext.set(capturedSink);
                    long ts = System.nanoTime();
                    try {
                        ToolResult r = executeOneCall(call);
                        long ms = (System.nanoTime() - ts) / 1_000_000;
                        return new TimedToolResult(idx, call, r, ms);
                    } finally {
                        ai.nizo.api.tool.UserContext.clear();
                        ai.nizo.api.agent.AgentEventContext.clear();
                    }
                }, TOOL_DISPATCH_EXECUTOR));
            }

            // Join in declaration order so toolResult messages line up with the LLM's call list.
            for (java.util.concurrent.CompletableFuture<TimedToolResult> f : futures) {
                TimedToolResult tr;
                try {
                    tr = f.get();
                } catch (Exception e) {
                    Throwable cause = (e instanceof java.util.concurrent.ExecutionException && e.getCause() != null)
                            ? e.getCause() : e;
                    LOG.warn("parallel tool dispatch failed: {}", cause.toString());
                    continue;
                }
                sink.emit(new AgentEvent.ToolCallResult(iteration, tr.call.id(), tr.call.name(),
                        tr.result.ok(), tr.result.content(), tr.elapsedMs));

                // ── Verbatim short-circuit: a tool may return content prefixed with the
                //    DeterministicStockOrchestratorTool.VERBATIM_MARKER, signalling "use this
                //    text as the assistant's final reply; do NOT pass it through another LLM
                //    round (which would summarize it and lose the carefully-rendered report)".
                //    Verified May 2026 — the orchestrator returned 276K chars and Qwen wrote
                //    a 1090-char summary instead. This bypass guarantees the assembled report
                //    reaches the user verbatim.
                if (tr.result.ok() && tr.result.content() != null
                        && tr.result.content().startsWith(ai.nizo.skills.DeterministicStockOrchestratorTool.VERBATIM_MARKER)) {
                    finalContent = tr.result.content().substring(
                            ai.nizo.skills.DeterministicStockOrchestratorTool.VERBATIM_MARKER.length());
                    stopReason = "verbatim_render";
                    LOG.info("verbatim short-circuit: tool {} returned {} chars, using as final reply",
                            tr.call.name(), finalContent.length());
                    verbatimShortCircuit = true;
                    break;   // break inner future-join loop
                }

                // ── Token-saver pass 1: replace canonical chart fences with [CHART:type]
                //    placeholders. Each fence is 5-50 KB of JSON; the placeholder is ~30 bytes.
                //    The canonical JSON is cached in orchestratorChartCache for final-reply rescue.
                String llmVisibleContent = extractFencesAndReplaceWithPlaceholders(
                        tr.result.content(), orchestratorChartCache);

                // ── Token-saver pass 2: hard-cap remaining prose at SUB_SKILL_MAX_CHARS.
                //    Even after fence removal, a sub-skill's prose (paragraphs + commentary +
                //    bullet scorecards) is 5-10 KB. With 5 sub-skills × multiple rounds plus
                //    the system prompt, we burn 100K+ tokens (verified May 2026: 164K vs 87K
                //    per-slot context). Cap = 2500 chars per tool-result. Master report assembly
                //    only needs the gist + chart placeholders; the full prose is preserved in
                //    each sub-skill's own session message history if anything needs it later.
                if (tr.call.name() != null && tr.call.name().startsWith("skill_")
                        && llmVisibleContent.length() > SUB_SKILL_MAX_CHARS) {
                    // Extract any [CHART:type] placeholders so they survive the truncation —
                    // critical because the orchestrator's master synthesis NEEDS to see these
                    // markers to know to emit them (they're how the final-reply rescue knows
                    // where each chart goes). Without preserving them, the orchestrator never
                    // emits chart placeholders and the master report has zero widgets.
                    java.util.List<String> placeholders = new java.util.ArrayList<>();
                    java.util.regex.Matcher pm = java.util.regex.Pattern
                            .compile("(?m)^\\[CHART:chart-[a-z0-9-]+\\]\\s*$")
                            .matcher(llmVisibleContent);
                    while (pm.find()) placeholders.add(pm.group().trim());

                    String head = llmVisibleContent.substring(0, SUB_SKILL_MAX_CHARS);
                    StringBuilder rebuilt = new StringBuilder(SUB_SKILL_MAX_CHARS + 256);
                    rebuilt.append(head)
                           .append("\n\n…[").append(tr.call.name()).append(" output truncated at ")
                           .append(SUB_SKILL_MAX_CHARS).append(" chars; full was ")
                           .append(llmVisibleContent.length()).append(".]");
                    if (!placeholders.isEmpty()) {
                        rebuilt.append("\n\nCharts emitted by this analyst (orchestrator MUST include these placeholders verbatim in the master report so the data widgets render):\n");
                        for (String p : placeholders) rebuilt.append(p).append('\n');
                    }
                    llmVisibleContent = rebuilt.toString();
                }

                messages.add(ChatMessage.toolResult(tr.call.id(), llmVisibleContent));
            }
            // After the future-join loop, if a tool short-circuited (e.g. the deterministic
            // stock orchestrator returned a fully-assembled report), break the outer iteration
            // loop too — finalContent is already set, no LLM round needed.
            if (verbatimShortCircuit) break;
        }

        // Fallback for cases where the loop exits without a meaningful reply:
        //   - max iterations reached (model still calling tools when budget hit)
        //   - model returned an empty content string (we previously stored "" here, leaving
        //     the assistant message blank in the session, looking like the chat "lost" its
        //     answer — verified May 2026 with the AMZN bug).
        // Surface a clear note + the tool-call count so the user knows the run actually did
        // work, just didn't synthesize. Bumping NIZO_AGENT_MAX_ITERATIONS to 30 prevents
        // most of these in practice.
        // Fallback path: model exited without producing user-visible content.
        // Surface the issue but DON'T persist the placeholder text as a real
        // assistant reply — saving it pollutes session history and shows up
        // as "Previous run ended without a report" on every reload (May 2026
        // bug papa hit). Instead, emit a FinalReply with the stop reason but
        // skip sessions.append; the UI checks for this case and offers retry.

        // ── Chart-fence post-process at orchestrator level ─────────────────────
        // Sub-skill outputs (which arrive in our `messages` list as TOOL role)
        // contain canonical fences like ```chart-financials\n{JSON}\n``` thanks to
        // SubAgentSkillTool's injection. The orchestrator's LLM, when assembling
        // the master report, sometimes RE-TRANSCRIBES these fences and corrupts
        // them — observed pattern: single-backtick wrap, JSON underscores parsed
        // as <em> emphasis (May 2026 chrome-inspect bug, surfaced after fixing
        // the sub-agent-level version of the same bug).
        //
        // The chart-fence cache was populated incrementally during the tool-dispatch loop
        // (see extractFencesAndReplaceWithPlaceholders). We just rescue the final reply.
        if (finalContent != null && !finalContent.isBlank() && !orchestratorChartCache.isEmpty()) {
            String before = finalContent;
            finalContent = ai.nizo.skills.SubAgentSkillTool.injectChartFences(
                    finalContent, orchestratorChartCache, /*appendUnused=*/false);
            if (!before.equals(finalContent)) {
                LOG.info("orchestrator chart-fence rescue applied: {} chart-type(s) cached, content {} → {} chars",
                        orchestratorChartCache.size(), before.length(), finalContent.length());
            }
        }

        boolean noSynthesis = (finalContent == null || finalContent.isBlank());
        if (noSynthesis) {
            String note = "[Run ended without a final synthesis after " + (iteration + 1) +
                    " iteration(s) and " + totalTools + " tool call(s). Reason: " + stopReason +
                    ". This usually means max iterations was hit while the model was still " +
                    "researching. Try again, or run a tighter query.]";
            finalContent = note;
            LOG.warn("turn chat={} produced no final content (iters={}, tools={}, reason={})",
                    in.chatId(), iteration + 1, totalTools, stopReason);
        }

        if (!noSynthesis) {
            ChatMessage assistantReply = ChatMessage.assistant(finalContent);
            sessions.append(in.chatId(), assistantReply);
        }

        sink.emit(new AgentEvent.FinalReply(iteration, finalContent, promptTokens, completionTokens, stopReason));

        long ms = (System.nanoTime() - t0) / 1_000_000;
        LOG.info("turn chat={} channel={} iters={} tools={} ptok={} ctok={} ms={} stop={}",
                in.chatId(), in.channel(), iteration + 1, totalTools, promptTokens, completionTokens, ms, stopReason);

        // Persist completed stock-analysis runs so the Library survives page refreshes.
        // Save if (a) chat looks like a stock run, (b) we got real synthesis content
        // (any non-fallback finalContent counts — this includes runs that ended on
        // `stop=length` after hitting max_tokens but still produced a usable report).
        // Best-effort — a save failure does NOT fail the turn.
        if (stockReports != null
                && in.chatId() != null && in.chatId().startsWith("stock-")
                && finalContent != null && !finalContent.isBlank()
                && !finalContent.startsWith("[Run ended")
                && !"interrupted".equals(stopReason)
                && !"llm_error".equals(stopReason)) {
            try {
                String prompt = (in.text() == null) ? "" : in.text();
                stockReports.save(new StockReportStore.Report(
                        in.chatId(),
                        StockReportStore.DEFAULT_USER,
                        StockReportStore.tickerFromChatId(in.chatId()),
                        System.currentTimeMillis(),
                        finalContent,
                        prompt,
                        in.channel(),
                        iteration + 1,
                        totalTools,
                        ms,
                        stopReason
                ));
            } catch (Exception e) {
                LOG.warn("stock-report save failed for {}: {}", in.chatId(), e.toString());
            }
        }
    }

    private ToolResult executeOneCall(ToolCall call) {
        Tool tool = tools.byName(call.name()).orElse(null);
        if (tool == null) {
            String msg = "ERROR: unknown tool '" + call.name() + "'. Available: "
                    + String.join(", ", tools.all().stream().map(Tool::name).toList());
            LOG.warn("unknown tool: {}", call.name());
            return ToolResult.error(msg);
        }
        try {
            return tool.execute(call.argumentsJson() == null ? "{}" : call.argumentsJson());
        } catch (Exception e) {
            LOG.warn("tool {} failed: {}", tool.name(), e.toString());
            return ToolResult.error(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private ChatMessage buildUserMessage(IncomingMessage in) {
        if (in.hasImages()) {
            return ChatMessage.userWithImages(in.text() == null ? "" : in.text(), in.images());
        }
        return ChatMessage.user(in.text());
    }

    /**
     * Compose the system prompt with the user's known facts inlined. This is what makes the agent
     * remember across conversations: every turn, every chat, the agent sees what it already knows.
     */
    private String buildSystemPromptWithMemory(IncomingMessage in) {
        StringBuilder sb = new StringBuilder();
        if (systemPrompt != null && !systemPrompt.isBlank()) sb.append(systemPrompt).append("\n\n");

        // Voice mode hint — when the user spoke (rather than typed), we want a CRISP
        // reply that reads well aloud. No tables, no markdown, no headings, no code.
        // Keep it 2-4 sentences. The full long answer can stay in the chat history if
        // needed but the spoken layer should be conversational.
        if (in.isVoice()) {
            sb.append("## You are in voice mode\n")
              .append("The user spoke this question (it was transcribed from speech). ")
              .append("Respond in a way that reads well aloud:\n")
              .append("- 2-4 sentences max, conversational tone\n")
              .append("- Plain prose only — NO markdown, NO tables, NO bullet lists, NO code blocks, NO headings\n")
              .append("- Numbers and technical detail kept minimal — just the headline answer\n")
              .append("- If the user wants more depth, they will ask a follow-up\n")
              .append("- Do NOT say 'as you can see' or refer to visual elements — they're listening, not looking\n\n");
        }

        // Language hint — reply in the user's spoken/typed language. Especially important
        // for voice: if they spoke Hindi, reply in Hindi (Devanagari) so the TTS picks it up.
        String lang = in.language();
        if (lang != null && !lang.isBlank() && !"en".equalsIgnoreCase(lang)) {
            String langName = languageDisplayName(lang);
            sb.append("## Reply language\n")
              .append("Respond in ").append(langName).append(" (").append(lang).append("). ")
              .append("The user wrote/spoke in this language; keep your reply in the same language ")
              .append("unless they explicitly switch.\n\n");
        }

        if (userFacts == null) return sb.length() == 0 ? systemPrompt : sb.toString().trim();
        var facts = userFacts.recall(in.userId(), 200);
        if (facts.isEmpty()) return sb.length() == 0 ? systemPrompt : sb.toString().trim();

        sb.append("## What you remember about this user (id=").append(in.userId()).append(")\n");
        for (var f : facts) sb.append("- ").append(f.content()).append("\n");
        sb.append("\nUse these facts naturally. If the user contradicts or updates one, call ")
          .append("`memory_forget` then `memory_remember` with the new value.");
        return sb.toString();
    }

    /** Map ISO 639-1 codes → human-readable name for the model's system prompt. */
    private static String languageDisplayName(String code) {
        return switch (code.toLowerCase()) {
            case "hi"    -> "Hindi (हिन्दी, Devanagari script)";
            case "es"    -> "Spanish";
            case "fr"    -> "French";
            case "de"    -> "German";
            case "it"    -> "Italian";
            case "pt"    -> "Portuguese";
            case "pl"    -> "Polish";
            case "tr"    -> "Turkish";
            case "ru"    -> "Russian (Cyrillic)";
            case "nl"    -> "Dutch";
            case "cs"    -> "Czech";
            case "ar"    -> "Arabic (العربية)";
            case "ja"    -> "Japanese (日本語)";
            case "ko"    -> "Korean (한국어)";
            case "hu"    -> "Hungarian";
            case "zh", "zh-cn", "zh-tw" -> "Chinese (中文)";
            case "ta"    -> "Tamil (தமிழ்)";
            case "te"    -> "Telugu (తెలుగు)";
            case "bn"    -> "Bengali (বাংলা)";
            case "mr"    -> "Marathi (मराठी)";
            case "gu"    -> "Gujarati (ગુજરાતી)";
            case "ml"    -> "Malayalam (മലയാളം)";
            case "kn"    -> "Kannada (ಕನ್ನಡ)";
            case "pa"    -> "Punjabi (ਪੰਜਾਬੀ)";
            case "ur"    -> "Urdu (اردو)";
            default      -> code;
        };
    }

    /**
     * Auto-condense at the start of each turn. Estimates the prompt cost (system prompt + history
     * + tool schemas + new user message) and fires a full condense when the estimate is within
     * {@link CondenseConstants#CONDENSE_BUFFER_TOKENS} of the model's context window.
     *
     * <p>Skipped when there's no condense engine wired or the breaker is open.
     */
    private void checkTokenBudget(IncomingMessage in, AgentEventSink sink) {
        if (condense == null || condense.isOpen()) return;
        try {
            String sys = buildSystemPromptWithMemory(in);
            int sysTokens = TokenEstimator.estimate(sys);
            int historyTokens = TokenEstimator.estimateMessages(sessions.recent(in.chatId(), historyMessages));
            int userTokens = TokenEstimator.estimate(buildUserMessage(in));
            int toolTokens = TokenEstimator.estimateTools(tools.toolDefs());
            int totalTokens = sysTokens + historyTokens + userTokens + toolTokens;

            int threshold = CondenseConstants.autoCondenseThreshold();
            if (totalTokens >= threshold) {
                LOG.info("auto-condense threshold hit for chat={}: {} tokens >= threshold {}",
                        in.chatId(), totalTokens, threshold);
                sink.emit(new AgentEvent.Warning(0,
                        "auto-condensing: " + totalTokens + " tokens approaches context limit"));
                CondenseResult r = condense.condense(
                        CondenseRequest.full(in.chatId(), in.userId(), CondenseRequest.Trigger.AUTO));
                if (!r.ok()) {
                    sink.emit(new AgentEvent.Warning(0, "auto-condense failed: " + r.error()));
                }
            }
        } catch (Exception e) {
            LOG.warn("checkTokenBudget threw, ignored: {}", e.toString());
        }
    }

    /**
     * Heuristic match for "prompt too long" errors across providers. Anthropic emits the literal
     * string {@code prompt-too-long}; llama.cpp returns HTTP 400 with messages like
     * {@code "the request exceeds the available context size"} or just {@code "context"}; vLLM uses
     * {@code "maximum context length"}. We match on substrings so the loop is forgiving.
     */
    static boolean isPromptTooLong(String message) {
        if (message == null || message.isBlank()) return false;
        String m = message.toLowerCase();
        if (m.contains("prompt-too-long") || m.contains("prompt_too_long")) return true;
        if (m.contains("context")
                && (m.contains("exceed") || m.contains("too long") || m.contains("too large")
                    || m.contains("overflow") || m.contains("maximum"))) return true;
        if (m.contains("maximum context length")) return true;
        if (m.contains("token") && m.contains("limit")) return true;
        return false;
    }

    /**
     * Manual condense entry point. Returns the engine's result; throws if no engine wired.
     * Manual condenses bypass the circuit breaker but still increment the failure counter on error
     * (so back-to-back manual attempts can still trip protection).
     */
    public CondenseResult condenseNow(CondenseRequest request) {
        if (condense == null) throw new IllegalStateException("condense engine not wired");
        return condense.condense(request);
    }

    private static Throwable rootCause(Throwable t) {
        Throwable c = t;
        while (c.getCause() != null && c.getCause() != c) c = c.getCause();
        return c;
    }

    /** Sink that materialises just the final reply, for the blocking {@link #handle} path. */
    private static final class AccumulatingSink implements AgentEventSink {
        final AtomicReference<String> finalReply = new AtomicReference<>("");

        @Override
        public void emit(AgentEvent event) {
            if (event instanceof AgentEvent.FinalReply fr) {
                finalReply.set(fr.text());
            }
        }
    }
}
