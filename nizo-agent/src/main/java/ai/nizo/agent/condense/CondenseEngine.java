package ai.nizo.agent.condense;

import ai.nizo.agent.session.SessionStore;
import ai.nizo.api.condense.CondenseHook;
import ai.nizo.api.condense.CondenseMode;
import ai.nizo.api.condense.CondenseRequest;
import ai.nizo.api.condense.CondenseResult;
import ai.nizo.api.condense.FileCache;
import ai.nizo.api.condense.FileCacheEntry;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.skills.SkillManifest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

/**
 * Orchestrates a {@code condense} operation: pre-hooks → strip non-essential content →
 * build the condense prompt → run a single-turn forked worker → parse → wrap → re-inject context →
 * post-hooks → persist new history.
 *
 * <p><b>Forked-worker semantics</b>: this engine sends an identical-prefix request to the LLM
 * (same model, same system prompt, same tool schemas, same message prefix) so llama.cpp's
 * KV-cache reuse fires on every token before the new condense user message. The single appended
 * message asks for the summary. We use {@link LlmClient#chat} (blocking) — it's a one-shot turn,
 * no need for streaming, and we don't want tool execution.
 *
 * <p><b>Tool denial</b>: the prompt explicitly forbids tool calls (twice — preamble and trailer),
 * and we discard any tool_calls returned without executing them. Tool definitions are still in
 * the request for cache-prefix parity. (Anthropic's {@code canUseTool: deny} is API-enforced;
 * llama.cpp doesn't expose that switch, so we belt-and-suspenders the constraint.)
 *
 * <p><b>Circuit breaker</b>: {@link #isOpen()} flips after {@link CondenseConstants#CONDENSE_MAX_FAILURES}
 * consecutive failures. Callers should check it before invoking another condense (auto and reactive
 * paths must respect it; manual {@code /condense} bypasses).
 */
public final class CondenseEngine {

    private static final Logger LOG = LoggerFactory.getLogger(CondenseEngine.class);

    private final LlmClient llm;
    private final SessionStore sessions;
    private final ToolRegistry tools;
    private final FileCache fileCache;
    private final List<CondenseHook> hooks;
    private final Supplier<List<SkillManifest>> skills;
    private final String model;
    private final Supplier<String> systemPromptSupplier;

    private final AtomicInteger consecutiveFailures = new AtomicInteger(0);

    public CondenseEngine(LlmClient llm,
                          SessionStore sessions,
                          ToolRegistry tools,
                          FileCache fileCache,
                          List<CondenseHook> hooks,
                          Supplier<List<SkillManifest>> skills,
                          String model,
                          Supplier<String> systemPromptSupplier) {
        this.llm = llm;
        this.sessions = sessions;
        this.tools = tools;
        this.fileCache = fileCache;
        this.hooks = hooks == null ? List.of() : List.copyOf(hooks);
        this.skills = skills == null ? List::of : skills;
        this.model = model;
        this.systemPromptSupplier = systemPromptSupplier;
    }

    /** True when the breaker is open and condenses must not auto-fire. */
    public boolean isOpen() { return consecutiveFailures.get() >= CondenseConstants.CONDENSE_MAX_FAILURES; }

    /** Force-close the breaker (used after a manual successful condense). */
    public void resetBreaker() { consecutiveFailures.set(0); }

    public FileCache fileCache() { return fileCache; }

    /**
     * Run a condense. This:
     * <ol>
     *   <li>Loads the chat's full history from {@link SessionStore}.</li>
     *   <li>Determines the keep/summarize split based on {@link CondenseMode}.</li>
     *   <li>Runs pre-condense hooks; collects extra instructions.</li>
     *   <li>Strips images / non-essential parts from the messages-to-summarize.</li>
     *   <li>Builds the condense prompt and invokes the forked worker.</li>
     *   <li>Parses {@code <summary>} out of the response and wraps it.</li>
     *   <li>Re-attaches recently-read files and active skills as separate user messages.</li>
     *   <li>Replaces the chat history with: kept-prefix + summary + re-injected context + kept-suffix
     *       (depending on mode).</li>
     *   <li>Runs post-condense hooks.</li>
     * </ol>
     */
    public CondenseResult condense(CondenseRequest req) {
        long t0 = System.nanoTime();
        try {
            List<ChatMessage> all = sessions.recent(req.chatId(), 10_000);
            int total = all.size();
            if (total == 0) {
                return CondenseResult.failure(req, "no messages to condense", elapsedMs(t0));
            }

            int[] split = splitFor(req, total);
            int keepHead = split[0];   // [0, keepHead)
            int sumStart = split[1];   // [sumStart, sumEnd) is the summarized range
            int sumEnd   = split[2];
            int keepTailStart = split[3];

            List<ChatMessage> keptHead = sliceCopy(all, 0, keepHead);
            List<ChatMessage> toSummarize = sliceCopy(all, sumStart, sumEnd);
            List<ChatMessage> keptTail = sliceCopy(all, keepTailStart, total);

            if (toSummarize.isEmpty()) {
                return CondenseResult.failure(req, "split produced empty summarize range", elapsedMs(t0));
            }

            int tokensBefore = TokenEstimator.estimateMessages(all);

            // 1) Pre-condense hooks
            List<String> hookInstructions = new ArrayList<>();
            for (CondenseHook h : hooks) {
                try {
                    String s = h.preCondense(req);
                    if (s != null && !s.isBlank()) hookInstructions.add(s);
                } catch (Exception e) {
                    LOG.warn("preCondense hook {} threw, ignored: {}", h.getClass().getSimpleName(), e.toString());
                }
            }

            // 2) Strip images and other non-essential bytes from the to-summarize range. Keeps
            //    the prefix cache hit cleaner and the request body within sane limits.
            List<ChatMessage> sanitized = stripNonEssential(toSummarize);

            // 3) Build the condense prompt
            String condensePrompt = CondensePromptBuilder.build(req.mode(), hookInstructions);

            // 4) Forked worker — single turn, tool-denied
            String rawResponse = invokeForkedWorker(sanitized, condensePrompt);
            if (rawResponse == null || rawResponse.isBlank()) {
                noteFailure();
                return CondenseResult.failure(req, "empty response from forked worker", elapsedMs(t0));
            }

            // 5) Parse
            String rawSummary = CondenseSummaryParser.extractRawSummary(rawResponse);
            if (rawSummary == null || rawSummary.isBlank()) {
                noteFailure();
                return CondenseResult.failure(req, "could not extract <summary> from response", elapsedMs(t0));
            }
            String formatted = CondenseSummaryParser.formatForReinjection(rawSummary);

            // 6) Re-inject context (files + skills) as auxiliary user messages
            List<FileCacheEntry> reinjected = fileCache.topN(
                    CondenseConstants.REINJECT_FILE_BUDGET,
                    CondenseConstants.REINJECT_FILE_PER_FILE,
                    CondenseConstants.REINJECT_FILE_MAX_COUNT);
            List<SkillManifest> activeSkills = pickSkillsToReinject();

            ChatMessage summaryMsg = ChatMessage.user(formatted);
            List<ChatMessage> reinjectionMsgs = buildReinjectionMessages(reinjected, activeSkills);

            // 7) Stitch the new history
            List<ChatMessage> rebuilt = new ArrayList<>();
            rebuilt.addAll(keptHead);
            rebuilt.add(summaryMsg);
            rebuilt.addAll(reinjectionMsgs);
            rebuilt.addAll(keptTail);

            sessions.replaceHistory(req.chatId(), rebuilt);

            // FileCache lifecycle: per-spec, clear on condense and let subsequent reads repopulate it
            // (the re-injected entries we just wrote are now part of the chat history, so the cache
            // serves only the post-condense session).
            fileCache.clear();

            int tokensAfter = TokenEstimator.estimateMessages(rebuilt);
            long ms = elapsedMs(t0);
            LOG.info("condense ok chat={} mode={} trigger={} before={} -> after={} (msgs {} -> {}) ms={}",
                    req.chatId(), req.mode(), req.trigger(), tokensBefore, tokensAfter, total, rebuilt.size(), ms);

            CondenseResult ok = new CondenseResult(
                    true, req.mode(), req.trigger(),
                    total, rebuilt.size(),
                    tokensBefore, tokensAfter,
                    rawSummary, formatted,
                    reinjected.stream().map(FileCacheEntry::path).toList(),
                    activeSkills.stream().map(SkillManifest::name).toList(),
                    ms, Instant.now(), null
            );

            consecutiveFailures.set(0);
            for (CondenseHook h : hooks) {
                try { h.postCondense(ok); }
                catch (Exception e) {
                    LOG.warn("postCondense hook {} threw, ignored: {}", h.getClass().getSimpleName(), e.toString());
                }
            }
            return ok;
        } catch (Exception e) {
            noteFailure();
            LOG.warn("condense failed for chat={}: {}", req.chatId(), e.toString(), e);
            return CondenseResult.failure(req, e.getClass().getSimpleName() + ": " + e.getMessage(), elapsedMs(t0));
        }
    }

    // ===================== internals =====================

    /**
     * Compute four indices over the {@code total}-length history:
     * <pre>[0, keepHead) ; [sumStart, sumEnd) ; [keepTailStart, total)</pre>
     * The result of the condense is: keptHead + summary + reinjection + keptTail.
     */
    static int[] splitFor(CondenseRequest req, int total) {
        return switch (req.mode()) {
            case FULL          -> new int[] { 0, 0, total, total };
            case PARTIAL_FROM  -> {
                int p = clamp(req.pivotIndex(), 0, total);
                yield new int[] { p, p, total, total };
            }
            case PARTIAL_UP_TO -> {
                int p = clamp(req.pivotIndex(), 0, total);
                yield new int[] { 0, 0, p, p };
            }
        };
    }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    private static List<ChatMessage> sliceCopy(List<ChatMessage> src, int from, int to) {
        if (from >= to) return List.of();
        return new ArrayList<>(src.subList(from, to));
    }

    /**
     * Strip images and oversized tool-result payloads from the messages we'll send to the
     * forked worker. The model still gets text continuity (which is what summarization needs);
     * we save tokens and dodge the multimodal load on a job that doesn't need it.
     */
    private static List<ChatMessage> stripNonEssential(List<ChatMessage> ms) {
        List<ChatMessage> out = new ArrayList<>(ms.size());
        for (ChatMessage m : ms) {
            if (m.hasImages()) {
                String txt = m.content() == null ? "" : m.content();
                String marker = "[image attachment omitted: " + m.images().size() + " image(s)]";
                String body = txt.isBlank() ? marker : txt + "\n" + marker;
                out.add(ChatMessage.user(body));
            } else {
                out.add(m);
            }
        }
        return out;
    }

    /**
     * Invoke the LLM with the same cache-prefix params used during normal turns, plus a single
     * appended user message containing the condense prompt. We pass tool defs (for cache parity)
     * but ignore any returned tool_calls — text content is what we want.
     */
    private String invokeForkedWorker(List<ChatMessage> history, String condensePrompt) {
        List<ChatMessage> messages = new ArrayList<>();
        String sys = systemPromptSupplier == null ? null : systemPromptSupplier.get();
        if (sys != null && !sys.isBlank()) messages.add(ChatMessage.system(sys));
        messages.addAll(history);
        messages.add(ChatMessage.user(condensePrompt));

        ChatRequest req = ChatRequest.of(model, messages).withTools(tools.toolDefs());
        ChatResponse resp = llm.chat(req);
        if (resp.hasToolCalls()) {
            LOG.warn("forked worker emitted {} tool_call(s); discarding (no-tools constraint)",
                    resp.toolCalls().size());
        }
        return resp.content();
    }

    private List<ChatMessage> buildReinjectionMessages(List<FileCacheEntry> files, List<SkillManifest> skillList) {
        List<ChatMessage> out = new ArrayList<>();
        if (!files.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("# Recently-read files (re-attached after condense)\n\n");
            sb.append("These were in your working set before the summary. They are still relevant.\n\n");
            for (FileCacheEntry f : files) {
                sb.append("## ").append(f.path()).append('\n');
                sb.append("```\n").append(f.content()).append("\n```\n\n");
            }
            out.add(ChatMessage.user(sb.toString()));
        }
        if (!skillList.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("# Active skills (re-attached after condense)\n\n");
            for (SkillManifest s : skillList) {
                sb.append("## ").append(s.name()).append('\n');
                if (s.description() != null) sb.append(s.description()).append("\n\n");
                if (s.whenToUse() != null && !s.whenToUse().isBlank()) {
                    sb.append("When to use: ").append(s.whenToUse()).append("\n\n");
                }
            }
            out.add(ChatMessage.user(sb.toString()));
        }
        return out;
    }

    private List<SkillManifest> pickSkillsToReinject() {
        List<SkillManifest> all = skills.get();
        if (all == null || all.isEmpty()) return List.of();
        List<SkillManifest> out = new ArrayList<>();
        int spent = 0;
        for (SkillManifest s : all) {
            int cost = TokenEstimator.estimate(s.name())
                    + TokenEstimator.estimate(s.description())
                    + TokenEstimator.estimate(s.whenToUse());
            cost = Math.min(cost, CondenseConstants.REINJECT_SKILL_PER_SKILL);
            if (spent + cost > CondenseConstants.REINJECT_SKILL_BUDGET) break;
            out.add(s);
            spent += cost;
        }
        return out;
    }

    private void noteFailure() { consecutiveFailures.incrementAndGet(); }

    private static long elapsedMs(long t0) { return (System.nanoTime() - t0) / 1_000_000; }
}
