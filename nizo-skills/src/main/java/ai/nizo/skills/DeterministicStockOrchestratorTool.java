package ai.nizo.skills;

import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.agent.AgentEventContext;
import ai.nizo.api.agent.AgentEventSink;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Deterministic orchestrator for {@code stock_analysis}. Replaces the previous behaviour
 * where the orchestrator was a {@link FilesystemSkillTool} that returned SKILL.md text and
 * relied on the outer LLM to follow the playbook — observed in practice (May 2026) to bypass
 * sub-skill invocations and hallucinate a report from training data.
 *
 * <h2>Deterministic dispatch</h2>
 * <ol>
 *   <li><b>Sanity:</b> call {@code stock_quote} once. If "ticker not found", abort early.</li>
 *   <li><b>Analysts (parallel):</b> fan out 5 sub-skills via virtual threads:
 *       fundamentals, analyst-estimates, news, sentiment, technicals. They run concurrently
 *       (gated only by llama-server's {@code --parallel} setting).</li>
 *   <li><b>Debate (sequential):</b> bear → bull → trader. Each sees the prior outputs as
 *       context. The trader produces the final verdict.</li>
 *   <li><b>Assembly:</b> stitch a master report with section headers, embed sub-skill
 *       commentary verbatim, and run the {@code [CHART:type]} placeholder rescue with the
 *       canonical chart fences extracted from sub-skill outputs.</li>
 * </ol>
 *
 * <p>The outer LLM never decides what to call — it just calls {@code skill_stock_analysis}
 * once with a ticker and receives a fully-assembled master report. Hallucination eliminated;
 * sub-skill coverage guaranteed.
 *
 * <p>This is the architecture the user requested: "LLM feeds in data, runtime renders the
 * graph". The runtime now also does the orchestration.
 */
public final class DeterministicStockOrchestratorTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(DeterministicStockOrchestratorTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Virtual-thread executor — sub-skills are I/O + LLM bound, perfect for virtual threads. */
    private static final ExecutorService DISPATCH = Executors.newVirtualThreadPerTaskExecutor();

    /**
     * Bounds how many sub-skills hit the single llama-server concurrently. The server exposes only
     * {@code --parallel} slots (3 on Kimaya); fanning out all 5 analysts at once overflows them, so
     * each context-switch evicts a sub-agent's KV cache and the next of its many calls must re-prefill
     * the whole (tens-of-thousands-of-tokens) prompt. Measured June 2026: the news analyst took 315s
     * for ~800 tokens of output — i.e. ~290s of pure re-prefill/queueing, not generation. Capping
     * concurrency to fit inside the slots (with one to spare for the outer agent / prefetch) keeps each
     * sub-agent's slot — and KV cache — warm across all its calls, so only the incremental tokens
     * prefill. Default 2; tune via {@code NIZO_STOCK_ANALYST_CONCURRENCY}.
     */
    private static final java.util.concurrent.Semaphore ANALYST_GATE =
            new java.util.concurrent.Semaphore(analystConcurrency());

    private static int analystConcurrency() {
        String v = System.getenv("NIZO_STOCK_ANALYST_CONCURRENCY");
        if (v != null) {
            try { return Math.max(1, Integer.parseInt(v.trim())); } catch (NumberFormatException ignore) { /* fall through */ }
        }
        return 2;
    }

    /** Pre-compiled fence pattern for extracting chart-X JSON from sub-skill outputs. */
    private static final Pattern FENCE_PATTERN = Pattern.compile(
            "(?ms)```(chart-[a-z0-9-]+)\\n(.*?)\\n```");

    /** Where each sub-skill's output goes in the final master report. Order = display order. */
    private static final List<SectionSpec> SECTIONS = List.of(
            new SectionSpec("skill_stock_fundamentals_analyst", "Fundamentals + Buffett scorecard"),
            new SectionSpec("skill_stock_analyst_estimates",    "Analyst estimates + earnings + insider activity"),
            new SectionSpec("skill_stock_news_analyst",         "News + catalysts"),
            new SectionSpec("skill_stock_sentiment_analyst",    "Sentiment read"),
            new SectionSpec("skill_stock_technical_analyst",    "Technical / timing"),
            new SectionSpec("skill_stock_bear_researcher",      "Bear case"),
            new SectionSpec("skill_stock_bull_researcher",      "Bull case"),
            new SectionSpec("skill_stock_trader",               "Verdict + targets + sizing")
    );

    private final Supplier<ToolRegistry> parentToolsRef;

    public DeterministicStockOrchestratorTool(Supplier<ToolRegistry> parentToolsRef) {
        this.parentToolsRef = parentToolsRef;
    }

    @Override public String name() { return "skill_stock_analysis"; }

    @Override
    public String description() {
        return "Investment-banking grade research on a public company. "
                + "Runs a deterministic multi-agent pipeline (quote → 5 analysts in parallel → "
                + "bear/bull/trader debate) and returns an assembled master report with embedded "
                + "interactive chart widgets (financials, analyst targets, earnings beat-rate, "
                + "insider activity, technical chart, indicators dashboard, Buffett scorecard). "
                + "Use when the user names a ticker (AAPL, MSFT, HDFCBANK.NS) or asks 'is X a buy'. "
                + "Internally invokes 8 sub-skills + ~6 data tools; takes 10-25 minutes to complete. "
                + "Never call this in parallel with other stock_* tools — it owns its pipeline.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "input": { "type": "string", "description": "Ticker symbol (e.g. AAPL, MSFT, HDFCBANK.NS) or a one-line ask like 'should I buy AAPL'" }
              },
              "required": ["input"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        long t0 = System.nanoTime();
        String input = parseInput(argumentsJson);
        if (input.isEmpty()) return ToolResult.error("input is required (ticker symbol)");

        String ticker = extractTicker(input);
        boolean isIndex = isIndexTicker(ticker);
        LOG.info("deterministic stock-analysis pipeline starting for ticker={}{}",
                ticker, isIndex ? " (INDEX mode — stock-only sub-skills skipped)" : "");

        ToolRegistry tools = parentToolsRef.get();
        if (tools == null) return ToolResult.error("tool registry not initialized yet");

        AgentEventSink sink = AgentEventContext.current();

        // ── Stage 0: stock_quote sanity ─────────────────────────────────────
        Tool stockQuote = tools.byName("stock_quote").orElse(null);
        if (stockQuote == null) return ToolResult.error("stock_quote tool not registered — server misconfigured");
        ToolResult quoteResult = invokeTool(stockQuote, "{\"ticker\":\"" + ticker + "\"}", sink);
        if (!quoteResult.ok()) {
            return ToolResult.ok(buildAbortReport(ticker, "Quote lookup failed: " + quoteResult.content()));
        }
        String quoteSummary = summarizeQuote(quoteResult.content());

        // ── Stage 1: 5 analysts in parallel ─────────────────────────────────
        Map<String, ToolResult> outputs = new LinkedHashMap<>();
        Map<String, String> chartCache = new HashMap<>();

        // Deterministic pre-fetch of chart data the technical sub-skill is supposed to
        // surface but doesn't always call. May 2026: the LLM technical_analyst sub-skill
        // intermittently skips the technical_indicators tool call (~50% of runs), leaving
        // chart-tech un-rendered. Pre-fetch here guarantees both interactive + tech charts
        // are in the cache; the assembly stage's appendUnused=true then surfaces any
        // placeholder the analyst forgot to emit. Fundamentals + analyst-estimates are
        // already reliable so we leave those to their sub-skills.
        //
        // historical_price MUST start before technical_indicators so the 60s in-memory bars
        // cache warms first (technical_indicators reuses 1y/1d bars from historical_price).
        // To save wall-clock time we kick historical_price first, then technical_indicators
        // shortly after — the cache lookup inside fetchOne is fast enough that the second
        // call almost always reuses the bars without a fresh Yahoo hit. If the cache is
        // already populated (e.g. recent run on same ticker), both prefetches return in
        // milliseconds.
        final String _prefetchUserId = ai.nizo.api.tool.UserContext.current();
        java.util.concurrent.CompletableFuture<Void> hpFuture =
                java.util.concurrent.CompletableFuture.runAsync(
                        boundRunnable(sink, _prefetchUserId, () -> prefetchChart(
                                tools, "historical_price", "chart-interactive",
                                "{\"ticker\":\"" + ticker + "\",\"range\":\"all_timeframes\"}", sink, chartCache)),
                        DISPATCH);
        // Give historical_price a small head-start so it caches the 1y/1d bars before
        // technical_indicators tries to fetch them (cuts a Yahoo round-trip).
        try { Thread.sleep(800); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
        java.util.concurrent.CompletableFuture<Void> tiFuture =
                java.util.concurrent.CompletableFuture.runAsync(
                        boundRunnable(sink, _prefetchUserId, () -> prefetchChart(
                                tools, "technical_indicators", "chart-tech",
                                "{\"ticker\":\"" + ticker + "\"}", sink, chartCache)),
                        DISPATCH);
        // Wait for both to finish — both populate chartCache which downstream needs.
        java.util.concurrent.CompletableFuture.allOf(hpFuture, tiFuture).join();

        // For indices, skip sub-skills that only make sense for an individual company:
        //   • fundamentals_analyst — indices don't have income statements / balance sheets
        //   • analyst_estimates    — no Wall-Street ratings on an index ticker
        // Keep the news + sentiment + technical analysts, then run the bear/bull/trader
        // debate over what remains — for an index the debate becomes "macro outlook on
        // this market" rather than "buy/sell on this stock".
        List<String> parallelStage = isIndex
                ? List.of(
                        "skill_stock_news_analyst",
                        "skill_stock_sentiment_analyst",
                        "skill_stock_technical_analyst")
                : List.of(
                        "skill_stock_fundamentals_analyst",
                        "skill_stock_analyst_estimates",
                        "skill_stock_news_analyst",
                        "skill_stock_sentiment_analyst",
                        "skill_stock_technical_analyst");
        runParallel(parallelStage, ticker, tools, sink, outputs, chartCache);

        // ── Stage 2: bear + bull in PARALLEL ──────────────────────────────
        // Previously bull was sequential-after-bear with bear's argument as input. That
        // sequencing cost ~2 min on every run for negligible analytical value: bull and
        // bear independently research opposite cases on the same underlying data; the
        // back-and-forth happens at the trader stage anyway. Running them in parallel
        // each over the same analyst-context cuts total pipeline wall-clock by ~120-180s.
        // Accuracy is preserved because:
        //   (a) both see the full debateContext (all 5 analyst outputs),
        //   (b) the trader stage downstream reads both arguments and weighs them.
        String debateContext = buildDebateContext(ticker, outputs);
        final String _debateUserId = ai.nizo.api.tool.UserContext.current();
        java.util.concurrent.CompletableFuture<ToolResult> bearFuture =
                java.util.concurrent.CompletableFuture.supplyAsync(
                        boundSupplier(sink, _debateUserId,
                                () -> invokeSubSkill(tools, "skill_stock_bear_researcher", debateContext, sink)),
                        DISPATCH);
        java.util.concurrent.CompletableFuture<ToolResult> bullFuture =
                java.util.concurrent.CompletableFuture.supplyAsync(
                        boundSupplier(sink, _debateUserId,
                                () -> invokeSubSkill(tools, "skill_stock_bull_researcher", debateContext, sink)),
                        DISPATCH);
        ToolResult bear, bull;
        try {
            bear = bearFuture.get();
            bull = bullFuture.get();
        } catch (Exception e) {
            LOG.warn("debate stage failed: {}", e.toString());
            bear = ToolResult.error("bear researcher: " + e.getMessage());
            bull = ToolResult.error("bull researcher: " + e.getMessage());
        }
        outputs.put("skill_stock_bear_researcher", bear);
        outputs.put("skill_stock_bull_researcher", bull);
        absorbCharts(bear.content(), chartCache);
        absorbCharts(bull.content(), chartCache);

        // ── Stage 3: trader synthesis (sees both bear + bull arguments) ───
        ToolResult trader = invokeSubSkill(tools, "skill_stock_trader",
                debateContext + "\n\n[Bear's case]\n\n" + truncate(bear.content(), 2000)
                        + "\n\n[Bull's case]\n\n" + truncate(bull.content(), 2000), sink);
        outputs.put("skill_stock_trader", trader);
        absorbCharts(trader.content(), chartCache);

        // ── Stage 5: assemble master report ────────────────────────────────
        String masterReport = assembleMasterReport(ticker, quoteSummary, outputs, chartCache);

        long ms = (System.nanoTime() - t0) / 1_000_000;
        LOG.info("deterministic stock-analysis pipeline DONE for ticker={} in {}ms ({} chars)",
                ticker, ms, masterReport.length());
        // Prefix with a marker that {@code AgentLoop} recognizes and uses to short-circuit:
        // the assembled master report becomes the assistant's final reply directly, without
        // a downstream LLM round that would otherwise try to "summarize" 276K chars down to
        // a 1KB blurb (verified May 2026 — the outer Qwen took our perfectly-rendered report
        // and rewrote it as a paragraph). The marker is consumed in AgentLoop before saving.
        return ToolResult.ok(VERBATIM_MARKER + masterReport);
    }

    /** Magic prefix that tells AgentLoop "use my result as finalContent verbatim, no LLM round". */
    public static final String VERBATIM_MARKER = " NIZO_RENDER_VERBATIM \n";

    // ───────────────────────────────────────────────────────────────────────
    // Pipeline helpers
    // ───────────────────────────────────────────────────────────────────────

    /** Run a list of sub-skill names concurrently with the given ticker as input. */
    /**
     * Wrap a {@code Runnable} so it re-binds {@link AgentEventContext} + {@code UserContext}
     * inside the virtual thread before running. Without this, every sub-skill's INNER tool
     * events read {@link AgentEventSink#NOOP} from {@code AgentEventContext.current()} and
     * silently vanish — tile detail panels show "0 tool calls" for sub-skills that made many
     * (May 2026 bug, applied to all 5 DISPATCH sites in this orchestrator).
     */
    private static Runnable boundRunnable(AgentEventSink sink, String userId, Runnable body) {
        return () -> {
            ai.nizo.api.agent.AgentEventContext.set(sink);
            if (userId != null) ai.nizo.api.tool.UserContext.set(userId);
            try { body.run(); }
            finally {
                ai.nizo.api.agent.AgentEventContext.clear();
                ai.nizo.api.tool.UserContext.clear();
            }
        };
    }

    /** Same as {@link #boundRunnable} but for a {@code Supplier<T>}. */
    private static <T> java.util.function.Supplier<T> boundSupplier(
            AgentEventSink sink, String userId, java.util.function.Supplier<T> body) {
        return () -> {
            ai.nizo.api.agent.AgentEventContext.set(sink);
            if (userId != null) ai.nizo.api.tool.UserContext.set(userId);
            try { return body.get(); }
            finally {
                ai.nizo.api.agent.AgentEventContext.clear();
                ai.nizo.api.tool.UserContext.clear();
            }
        };
    }

    private void runParallel(List<String> skillNames, String ticker, ToolRegistry tools,
                             AgentEventSink sink, Map<String, ToolResult> outputs,
                             Map<String, String> chartCache) {
        List<CompletableFuture<Map.Entry<String, ToolResult>>> futures = new java.util.ArrayList<>();
        String input = "{\"input\":\"" + ticker + "\"}";
        // ThreadLocals (AgentEventContext, UserContext) do NOT inherit across virtual threads,
        // so every VT must re-bind the sink + userId or sub-skill INNER tool events go to NOOP
        // and the tile detail panels show "0 tool calls" for sub-skills that made many.
        final String parallelUserId = ai.nizo.api.tool.UserContext.current();
        for (String name : skillNames) {
            futures.add(CompletableFuture.supplyAsync(
                    boundSupplier(sink, parallelUserId, () -> {
                        Tool t = tools.byName(name).orElse(null);
                        if (t == null) {
                            return Map.entry(name, ToolResult.error("sub-skill " + name + " not registered"));
                        }
                        // Gate on the llama slot budget: at most ANALYST_CONCURRENCY sub-skills call the
                        // model at once, so each keeps a warm KV slot across its calls instead of being
                        // evicted and re-prefilled. acquireUninterruptibly: the supplier can't throw checked.
                        ANALYST_GATE.acquireUninterruptibly();
                        ToolResult r;
                        try {
                            r = invokeTool(t, input, sink);
                        } finally {
                            ANALYST_GATE.release();
                        }
                        return Map.entry(name, r);
                    }), DISPATCH));
        }
        // Join in declaration order so outputs map iterates in user-visible order.
        for (int i = 0; i < skillNames.size(); i++) {
            try {
                Map.Entry<String, ToolResult> e = futures.get(i).get();
                outputs.put(e.getKey(), e.getValue());
                if (e.getValue().ok()) absorbCharts(e.getValue().content(), chartCache);
            } catch (ExecutionException | InterruptedException ex) {
                String name = skillNames.get(i);
                LOG.warn("sub-skill {} threw during parallel join: {}", name, ex.toString());
                outputs.put(name, ToolResult.error("dispatch error: " + ex.getMessage()));
                if (ex instanceof InterruptedException) Thread.currentThread().interrupt();
            }
        }
    }

    /** Invoke a sub-skill with a structured input string. */
    private ToolResult invokeSubSkill(ToolRegistry tools, String name, String input, AgentEventSink sink) {
        Tool t = tools.byName(name).orElse(null);
        if (t == null) return ToolResult.error("sub-skill " + name + " not registered");
        String json;
        try {
            json = MAPPER.writeValueAsString(Map.of("input", input));
        } catch (Exception e) {
            return ToolResult.error("input serialization failed: " + e.getMessage());
        }
        return invokeTool(t, json, sink);
    }

    /** Run a tool, emitting ToolCallStart/Result events to the parent sink. */
    private ToolResult invokeTool(Tool t, String args, AgentEventSink sink) {
        String fakeId = "det-" + Long.toHexString(System.nanoTime());
        sink.emit(new AgentEvent.ToolCallStart(0, fakeId, t.name(), args));
        long ts = System.nanoTime();
        ToolResult r;
        try {
            r = t.execute(args == null ? "{}" : args);
        } catch (Exception e) {
            r = ToolResult.error(t.name() + " threw: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
        long ms = (System.nanoTime() - ts) / 1_000_000;
        sink.emit(new AgentEvent.ToolCallResult(0, fakeId, t.name(), r.ok(), r.content(), ms));
        return r;
    }

    /**
     * Run a single tool deterministically and seed the orchestrator's chart cache with its
     * raw JSON output. Used to guarantee chart presence regardless of whether the sub-skill
     * decides to call the tool itself. Failures are logged but don't abort the pipeline —
     * the chart simply won't render, which is better than a hard error.
     */
    private void prefetchChart(ToolRegistry tools, String toolName, String chartType,
                               String args, AgentEventSink sink, Map<String, String> chartCache) {
        Tool t = tools.byName(toolName).orElse(null);
        if (t == null) {
            LOG.warn("prefetch: tool {} not registered, skipping chart={}", toolName, chartType);
            return;
        }
        // Retry on Yahoo 429s — pipeline starts make several Yahoo calls within seconds of
        // each other; the v8 chart endpoint occasionally rate-limits even single hits.
        // Backoff: 0s → 4s → 12s → 30s. Bail if still failing.
        int[] backoffSec = {0, 4, 12, 30};
        ToolResult r = null;
        for (int attempt = 0; attempt < backoffSec.length; attempt++) {
            if (backoffSec[attempt] > 0) {
                try { Thread.sleep(backoffSec[attempt] * 1000L); }
                catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
            }
            r = invokeTool(t, args, sink);
            if (r.ok() && r.content() != null && !r.content().isBlank()) break;
            String reason = r.content() == null ? "null" : r.content();
            // Only retry rate-limit / transient HTTP errors. Hard failures (bad ticker etc.)
            // shouldn't be retried.
            boolean transientErr = reason.contains("429") || reason.contains("503")
                    || reason.contains("timeout") || reason.contains("Read timed out")
                    || reason.contains("connection reset");
            if (!transientErr) break;
            LOG.info("prefetch: {} attempt {} hit transient error, retrying in {}s",
                    toolName, attempt + 1, attempt + 1 < backoffSec.length ? backoffSec[attempt + 1] : "n/a");
        }
        if (r != null && r.ok() && r.content() != null && !r.content().isBlank()) {
            chartCache.put(chartType, r.content());
            LOG.info("prefetch: {} → chartCache[{}] = {} chars", toolName, chartType, r.content().length());
        } else {
            LOG.warn("prefetch: {} failed after retries, chart={} won't render: {}",
                    toolName, chartType,
                    r == null ? "null" : (r.content() == null ? "null" : r.content().substring(0, Math.min(120, r.content().length()))));
        }
    }

    /** Extract canonical {@code ```chart-X\n{...}\n```} fences from sub-skill output into the cache. */
    private static void absorbCharts(String content, Map<String, String> chartCache) {
        if (content == null || content.isEmpty()) return;
        Matcher m = FENCE_PATTERN.matcher(content);
        while (m.find()) {
            chartCache.putIfAbsent(m.group(1), m.group(2));
        }
    }

    /** Build a single concatenated context for the debate stage from all 5 analyst outputs. */
    private static String buildDebateContext(String ticker, Map<String, ToolResult> outputs) {
        StringBuilder sb = new StringBuilder();
        sb.append("Ticker: ").append(ticker).append("\n\n");
        sb.append("Analyst reports (from prior parallel stage):\n\n");
        for (SectionSpec sec : SECTIONS) {
            if (sec.skill.contains("researcher") || sec.skill.contains("trader")) continue;
            ToolResult r = outputs.get(sec.skill);
            if (r == null || !r.ok()) continue;
            sb.append("=== ").append(sec.heading).append(" ===\n");
            sb.append(stripFences(r.content())).append("\n\n");
        }
        return sb.toString();
    }

    /** Remove ```chart-X``` fences from text — keeps the prose for debate context only. */
    private static String stripFences(String content) {
        if (content == null) return "";
        return FENCE_PATTERN.matcher(content).replaceAll("[CHART:$1 — see master report]");
    }

    /** Assemble the final master report markdown from sub-skill outputs + cached chart fences. */
    private String assembleMasterReport(String ticker, String quoteSummary,
                                        Map<String, ToolResult> outputs, Map<String, String> chartCache) {
        boolean isIndex = isIndexTicker(ticker);
        StringBuilder out = new StringBuilder(32 * 1024);
        out.append("# ").append(ticker)
           .append(isIndex ? " — Market Index Research" : " — Investment Research")
           .append("\n\n");
        out.append(quoteSummary).append("\n\n");
        out.append("---\n\n");

        // Pull verdict line out of trader output for the top "Quick Verdict" section.
        ToolResult traderOut = outputs.get("skill_stock_trader");
        if (traderOut != null && traderOut.ok()) {
            String verdict = extractVerdict(traderOut.content());
            if (verdict != null) {
                out.append(isIndex ? "## Market Outlook\n\n" : "## Quick Verdict\n\n");
                out.append("> **").append(isIndex ? "Bias" : "Rating").append(": ").append(verdict).append("**\n\n");
                out.append(extractVerdictRationale(traderOut.content())).append("\n\n");
                out.append("---\n\n");
            }
        }

        // Numbered sections in display order. For indices, skip sub-skills that are
        // stock-only — they didn't run and their "_(sub-skill X did not run)_" placeholder
        // would just clutter the report.
        int sectionNum = 1;
        for (SectionSpec sec : SECTIONS) {
            if (isIndex && isStockOnlySkill(sec.skill)) continue;
            ToolResult r = outputs.get(sec.skill);
            out.append("## ").append(sectionNum++).append(". ").append(sec.heading).append("\n\n");
            if (r == null) {
                out.append("_(sub-skill ").append(sec.skill).append(" did not run)_\n\n");
            } else if (!r.ok()) {
                out.append("_(").append(sec.skill).append(" failed: ").append(r.content()).append(")_\n\n");
            } else {
                // Include the sub-skill's own markdown verbatim — it already contains [CHART:type]
                // placeholders OR canonical fences depending on whether SubAgentSkillTool already
                // expanded them. Either way, the rescue pass below normalizes everything.
                out.append(r.content()).append("\n\n");
            }
            out.append("---\n\n");
        }

        // Disclaimer
        out.append("> *This is research, not advice. AI-generated. Verify all numbers before acting.*\n");

        // ── Apply chart-fence rescue at the end so any [CHART:type] placeholder anywhere
        //    in the assembled report gets expanded to the canonical fence using chartCache.
        //    appendUnused=true so deterministically pre-fetched charts (chart-tech,
        //    chart-interactive) still surface even when the technical sub-skill forgot to
        //    emit their placeholders.
        String assembled = out.toString();
        return SubAgentSkillTool.injectChartFences(assembled, chartCache, /*appendUnused=*/true);
    }

    /** Build a concise quote-summary block from the stock_quote tool's JSON. */
    private static String summarizeQuote(String quoteJson) {
        try {
            JsonNode n = MAPPER.readTree(quoteJson);
            String name = n.path("longName").asText("");
            String exchange = n.path("exchange").asText("");
            String currency = n.path("currency").asText("USD");
            double price = n.path("price").asDouble(0);
            double change = n.path("change").asDouble(0);
            double pct = n.path("changePercent").asDouble(0);
            String sym = currency.equals("USD") ? "$" : currency + " ";
            StringBuilder sb = new StringBuilder();
            if (!name.isEmpty()) sb.append("**").append(name).append("**");
            if (!exchange.isEmpty()) sb.append(" · ").append(exchange);
            sb.append(" · ").append(java.time.LocalDate.now()).append("\n\n");
            sb.append(sym).append(String.format(Locale.ROOT, "%.2f", price));
            if (change != 0) {
                sb.append(" ").append(change >= 0 ? "+" : "").append(String.format(Locale.ROOT, "%.2f", change));
                sb.append(" (").append(pct >= 0 ? "+" : "").append(String.format(Locale.ROOT, "%.2f", pct)).append("%)");
            }
            return sb.toString();
        } catch (Exception e) {
            return "_(quote data unavailable)_";
        }
    }

    private static String buildAbortReport(String ticker, String reason) {
        return "# " + ticker + " — Analysis Aborted\n\n"
                + "_(stock_analysis aborted before running sub-skills)_\n\n"
                + "**Reason:** " + reason + "\n\n"
                + "Try again with a different ticker, or check that the data sources are reachable.\n";
    }

    /** Pull a verdict word from trader output ("Rating: STRONG BUY", "Verdict: HOLD", etc.). */
    private static String extractVerdict(String content) {
        if (content == null) return null;
        Matcher m = Pattern.compile("(?i)\\b(?:Rating|Verdict)\\s*[:\\-]\\s*\\*{0,2}(STRONG\\s*BUY|BUY|HOLD|AVOID|SELL)\\*{0,2}").matcher(content);
        if (m.find()) return m.group(1).toUpperCase().replaceAll("\\s+", " ");
        return null;
    }

    /**
     * First paragraph of trader output, with the verdict/rating line stripped — that line
     * is already shown in the quote block above so emitting it again is a duplicate.
     * Patterns stripped (whole-line, anywhere in the paragraph, case-insensitive):
     *   "Rating: HOLD", "**Rating:** STRONG BUY", "Verdict: BUY", "Verdict — SELL", etc.
     */
    private static String extractVerdictRationale(String content) {
        if (content == null) return "";
        // Take the first paragraph of trader output as the rationale.
        String firstPara = content.split("\n\n", 2)[0];
        // Strip ANY line that is JUST a Rating: / Verdict: line (with optional bold markup
        // and surrounding whitespace). Keeps lines where the rating word is part of prose.
        String stripped = firstPara.replaceAll(
                "(?im)^[ \\t]*\\*{0,2}(?:Rating|Verdict)\\s*[:\\-—]\\s*\\*{0,2}(STRONG\\s*BUY|BUY|HOLD|AVOID|SELL)\\*{0,2}[ \\t]*\\.?[ \\t]*$\\r?\\n?",
                "");
        // Tidy up leading / trailing whitespace left behind.
        stripped = stripped.replaceAll("^\\s+", "").replaceAll("\\s+$", "");
        if (stripped.isEmpty()) {
            // Fallback: no prose left after stripping → use the second paragraph instead.
            String[] paras = content.split("\n\n", 3);
            stripped = paras.length > 1 ? paras[1].trim() : "";
        }
        if (stripped.length() > 600) stripped = stripped.substring(0, 600) + "…";
        return stripped;
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "\n…[truncated]";
    }

    private static String parseInput(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) return "";
        try {
            JsonNode n = MAPPER.readTree(argumentsJson);
            JsonNode in = n.get("input");
            return in == null ? argumentsJson : in.asText("");
        } catch (Exception e) { return argumentsJson; }
    }

    /**
     * Index aliases — map a colloquial name to the Yahoo Finance symbol. Keys are
     * UPPER-CASE, spaces stripped. Covers the markets a typical user asks about by name.
     * Symbols starting with {@code ^} are Yahoo's convention for indices and signal the
     * orchestrator to skip stock-only sub-skills (fundamentals / analyst ratings /
     * insider activity / earnings — none of those apply to a market index).
     */
    private static final Map<String, String> INDEX_ALIASES = Map.ofEntries(
            // India
            Map.entry("NIFTY",          "^NSEI"),
            Map.entry("NIFTY50",        "^NSEI"),
            Map.entry("NIFTY 50",       "^NSEI"),
            Map.entry("NSE",            "^NSEI"),
            Map.entry("NIFTYBANK",      "^NSEBANK"),
            Map.entry("BANKNIFTY",      "^NSEBANK"),
            Map.entry("NIFTYIT",        "^CNXIT"),
            Map.entry("SENSEX",         "^BSESN"),
            Map.entry("BSE",            "^BSESN"),
            // US
            Map.entry("SP500",          "^GSPC"),
            Map.entry("S&P500",         "^GSPC"),
            Map.entry("S&P 500",        "^GSPC"),
            Map.entry("SPX",            "^GSPC"),
            Map.entry("DOW",            "^DJI"),
            Map.entry("DOWJONES",       "^DJI"),
            Map.entry("NASDAQ",         "^IXIC"),
            Map.entry("NASDAQ100",      "^NDX"),
            Map.entry("NDX",            "^NDX"),
            Map.entry("RUSSELL",        "^RUT"),
            Map.entry("RUSSELL2000",    "^RUT"),
            Map.entry("VIX",            "^VIX"),
            // Australia
            Map.entry("ASX",            "^AXJO"),
            Map.entry("ASX200",         "^AXJO"),
            Map.entry("ASX 200",        "^AXJO"),
            Map.entry("AORD",           "^AORD"),
            // Europe / Asia common requests
            Map.entry("FTSE",           "^FTSE"),
            Map.entry("FTSE100",        "^FTSE"),
            Map.entry("DAX",            "^GDAXI"),
            Map.entry("NIKKEI",         "^N225"),
            Map.entry("NIKKEI225",      "^N225"),
            Map.entry("HANGSENG",       "^HSI")
    );

    /** True if {@code ticker} represents a market index (Yahoo convention: leading caret). */
    public static boolean isIndexTicker(String ticker) {
        return ticker != null && ticker.startsWith("^");
    }

    /** Sub-skills that only make sense for an individual company, not for a market index. */
    private static boolean isStockOnlySkill(String skill) {
        return "skill_stock_fundamentals_analyst".equals(skill)
            || "skill_stock_analyst_estimates".equals(skill);
    }

    /** Extract a ticker from "AAPL", "should I buy AAPL", "TICKER: MSFT",
     *  "NIFTY 50", "^GSPC", etc. Maps colloquial index names to Yahoo symbols. */
    private static String extractTicker(String input) {
        if (input == null) return "";
        String trimmed = input.trim();
        // 1. Trivial case — input IS just a ticker, possibly an index with the ^ prefix.
        if (trimmed.matches("\\^?[A-Za-z][A-Za-z0-9.-]{0,15}")) {
            String t = trimmed.toUpperCase();
            // Trivial-case shortcut: still resolve via alias map in case user typed bare "NIFTY".
            String aliased = INDEX_ALIASES.get(t);
            return aliased != null ? aliased : t;
        }
        // 2. Index alias by full name / spaces (e.g. "S&P 500", "NIFTY 50", "ASX 200").
        String normalized = trimmed.toUpperCase().replaceAll("[^A-Z0-9& ]", "").trim();
        if (INDEX_ALIASES.containsKey(normalized)) return INDEX_ALIASES.get(normalized);
        String collapsed = normalized.replace(" ", "");
        if (INDEX_ALIASES.containsKey(collapsed)) return INDEX_ALIASES.get(collapsed);
        // 3. Yahoo-style symbol with caret (^NSEI inside a sentence).
        Matcher caret = Pattern.compile("\\^[A-Z]{2,10}").matcher(trimmed.toUpperCase());
        if (caret.find()) return caret.group();
        // 4. Stock-symbol-shaped token, ignoring common noise words.
        Matcher m = Pattern.compile("\\b([A-Z]{1,6}(?:\\.[A-Z]{1,3})?)\\b").matcher(trimmed.toUpperCase());
        java.util.Set<String> noise = java.util.Set.of("THE", "AND", "FOR", "BUY", "SELL", "HOLD",
                "STOCK", "ANALYSIS", "FROM", "WITH", "INTO", "ON", "OF", "TO", "USA", "USD", "AI", "OK", "IT", "IS", "AT",
                "INDEX", "INDICES");
        while (m.find()) {
            String t = m.group(1);
            if (!noise.contains(t)) {
                String aliased = INDEX_ALIASES.get(t);
                return aliased != null ? aliased : t;
            }
        }
        return trimmed.toUpperCase();
    }

    /** Section spec — sub-skill name + display heading in the master report. */
    private record SectionSpec(String skill, String heading) {}
}
