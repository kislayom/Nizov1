package ai.nizo.app.bootstrap;

import ai.nizo.agent.condense.CondenseEngine;
import ai.nizo.agent.condense.InMemoryFileCache;
import ai.nizo.agent.exec.ChatExecutor;
import ai.nizo.agent.loop.AgentLoop;
import ai.nizo.agent.memory.SqliteUserFactStore;
import ai.nizo.agent.cache.StockReportStore;
import ai.nizo.api.condense.CondenseHook;
import ai.nizo.api.condense.FileCache;
import ai.nizo.api.memory.UserFactStore;
import ai.nizo.agent.session.SessionStore;
import ai.nizo.agent.session.SqliteSessionStore;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.mcp.client.McpClientPool;
import ai.nizo.mcp.config.McpServersFile;
import ai.nizo.app.config.NizoHome;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.llm.FailoverLlmClient;
import ai.nizo.llm.LlmConfig;
import ai.nizo.llm.OpenAiCompatibleClient;
import ai.nizo.skills.FilesystemSkillTool;
import ai.nizo.skills.SaveSkillTool;
import ai.nizo.skills.SkillLoader;
import ai.nizo.skills.SkillManifest;
import ai.nizo.skills.SubAgentSkillTool;
import ai.nizo.tools.code.CodeExecTool;
import ai.nizo.tools.file.FileListTool;
import ai.nizo.tools.file.FileReadTool;
import ai.nizo.tools.file.FileWriteTool;
import ai.nizo.tools.finance.AnalystRatingsTool;
import ai.nizo.tools.finance.EarningsHistoryTool;
import ai.nizo.tools.finance.FmpClient;
import ai.nizo.tools.finance.HistoricalPriceTool;
import ai.nizo.tools.finance.InsiderActivityTool;
import ai.nizo.tools.finance.StockBuffettScoreTool;
import ai.nizo.tools.finance.StockFundamentalsTool;
import ai.nizo.tools.finance.StockQuoteTool;
import ai.nizo.tools.finance.TechnicalIndicatorsTool;
import ai.nizo.tools.finance.YahooHtmlScraper;
import ai.nizo.tools.finance.YahooQuoteSummary;
import ai.nizo.tools.http.HttpJsonTool;
import ai.nizo.tools.memory.MemoryForgetTool;
import ai.nizo.tools.memory.MemoryRecallTool;
import ai.nizo.tools.memory.MemoryRememberTool;
import ai.nizo.tools.registry.InMemoryToolRegistry;
import ai.nizo.tools.registry.MeasuredTool;
import ai.nizo.tools.registry.UsageTracker;
import ai.nizo.tools.shell.ShellTool;
import ai.nizo.tools.system.CurrentTimeTool;
import ai.nizo.tools.web.SmartProxyClient;
import ai.nizo.tools.web.WebFetchTool;
import ai.nizo.tools.web.WebSearchTool;
import ai.nizo.tools.web.WikipediaTool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Wire the whole agent stack from env + ~/.nizo. Single place that knows about all the modules.
 *
 * <p>Senior-CTO note: this is intentionally hand-wired (no DI framework). Six dependencies,
 * three lifecycle objects ({@link SqliteSessionStore} is {@link AutoCloseable}). A DI container
 * would be more ceremony than insight here.
 */
public final class Bootstrap implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(Bootstrap.class);

    public final NizoHome home;
    public final LlmConfig llmConfig;
    public final LlmClient llmClient;   // resilient wrapper (FailoverLlmClient) over the primary endpoint
    public final SqliteSessionStore sessions;
    public final UserFactStore userFacts;
    public final StockReportStore stockReports;
    public final ai.nizo.agent.deepwork.JobStore deepJobs;
    public final ai.nizo.agent.schedule.ScheduleStore schedules;
    public final ai.nizo.agent.deepwork.DeepWorkEngine deepWork;
    public final FileCache fileCache;
    public final ToolRegistry tools;
    public final List<SkillManifest> skills;
    public final List<CondenseHook> condenseHooks;
    public final CondenseEngine condense;
    public final McpServersFile mcpConfig;
    public final McpClientPool mcpPool;
    public final AgentLoop agent;
    public final ChatExecutor chatExecutor;
    public final UsageTracker usage;

    public Bootstrap(String systemPrompt, int maxIterations, int historyMessages) {
        this.home = NizoHome.resolve();
        this.llmConfig = LlmConfig.fromEnv();
        OpenAiCompatibleClient primaryLlm = new OpenAiCompatibleClient(llmConfig.baseUrl(), llmConfig.authToken());
        // Resilience layer: retry transport blips / the cold-start tail with short backoff and, if a
        // second provider is configured, fail over to it. Local-first by default — a single provider
        // makes this a pure retry wrapper that never leaves the box. A fallback engages ONLY if
        // NIZO_LLM_FALLBACK_URL is set (opt-in; that one does send prompts off-box).
        java.util.List<LlmClient> llmProviders = new java.util.ArrayList<>();
        llmProviders.add(primaryLlm);
        String fallbackUrl = System.getenv("NIZO_LLM_FALLBACK_URL");
        if (fallbackUrl != null && !fallbackUrl.isBlank()) {
            llmProviders.add(new OpenAiCompatibleClient(fallbackUrl.trim(), System.getenv("NIZO_LLM_FALLBACK_TOKEN")));
            LOG.info("LLM fallback provider configured ({})", fallbackUrl.trim());
        }
        this.llmClient = new FailoverLlmClient(llmProviders, 2, new long[]{1000, 3000, 6000});
        this.sessions = new SqliteSessionStore(home.sessionsDb());
        this.userFacts = new SqliteUserFactStore(home.memoryDb());
        this.stockReports = new StockReportStore(home.stockReportsDb());
        this.deepJobs = new ai.nizo.agent.deepwork.JobStore(home.root().resolve("deep_work.db"));
        this.schedules = new ai.nizo.agent.schedule.ScheduleStore(home.root().resolve("scheduler.db"));
        this.fileCache = new InMemoryFileCache();

        SkillLoader skillLoader = new SkillLoader();
        this.skills = skillLoader.discover(home.skillsDir());

        // Single shared usage tracker. Every tool gets wrapped with MeasuredTool so the
        // tracker sees per-name counts, last-used, error rate, and a deque of the most
        // recent N invocations (each tagged with the chatId). The tracker is exposed
        // via WebChannel's /api/usage for the UI.
        this.usage = new UsageTracker();
        java.util.function.Function<Tool, Tool> measure = t -> new MeasuredTool(t, usage);

        // Registered now, bound to the engine after the registry exists (see below).
        ai.nizo.agent.deepwork.DeepWorkTool deepWorkTool = new ai.nizo.agent.deepwork.DeepWorkTool();

        // Tier-0 PRIMARY data source: FMP. Constructed up-front so we can wire it into
        // HistoricalPriceTool as the rate-limit fallback.
        FmpClient fmp = new FmpClient();

        // Shared Indian-specific clients — must be constructed BEFORE HistoricalPriceTool
        // so we can wire ScreenerInClient as the 4th-tier historical-bars fallback for
        // .NS / .BO names. Stooq's free CSV path was paywalled in 2026; Screener's chart
        // API is now the only working free source for Indian historicals from a
        // datacenter IP.
        ai.nizo.tools.finance.ScreenerInClient screenerInClient = new ai.nizo.tools.finance.ScreenerInClient();
        ai.nizo.tools.finance.NseIndiaClient nseIndiaClient = new ai.nizo.tools.finance.NseIndiaClient();

        // Shared price-history tool — TechnicalIndicatorsTool reuses it to fetch OHLCV.
        // Fallback chain: Yahoo v8 chart → FMP /historical-price-eod → Stooq CSV →
        // Screener.in chart API (Indian tickers only). Each tier handles a different
        // failure mode (429, 402, paywall, geo-block).
        HistoricalPriceTool historicalPriceTool = new HistoricalPriceTool(fmp, screenerInClient);

        // SmartProxy client (paid residential-IP scraper). Acts as the LAST RESORT for any
        // outbound fetch that gets bot-blocked. WebFetchTool/WebSearchTool already use it; we
        // also pass it to YahooHtmlScraper so the structured-data tools have a fallback when
        // Yahoo's getcrumb endpoint rate-limits our datacenter IP (observed in production).
        SmartProxyClient smartProxy = new SmartProxyClient();
        YahooHtmlScraper yahooScraper = new YahooHtmlScraper(smartProxy);
        // Yahoo client — Tier 1 (direct API w/ crumb dance) and Tier 2 (HTML scraper) chained
        // behind FMP. If FMP serves all requested modules, Yahoo paths aren't even tried.
        // FmpClient was constructed earlier (above HistoricalPriceTool) so we share one instance.
        YahooQuoteSummary yahooQs = new YahooQuoteSummary(fmp, yahooScraper);

        ai.nizo.tools.finance.StockBuffettScoreTool buffettScoreTool =
                new ai.nizo.tools.finance.StockBuffettScoreTool(yahooQs);

        InMemoryToolRegistry.Builder builder = InMemoryToolRegistry.builder()
                .add(measure.apply(new CurrentTimeTool()))
                .add(measure.apply(new WebSearchTool()))
                .add(measure.apply(new WebFetchTool()))
                .add(measure.apply(new ai.nizo.tools.web.BrowserTool()))  // headless browser (Playwright sidecar) for JS/interactive sites
                .add(measure.apply(new WikipediaTool()))
                .add(measure.apply(new HttpJsonTool()))          // generic JSON HTTP — reusable for any API
                .add(measure.apply(new StockQuoteTool()))        // live quotes via Yahoo Finance public JSON
                .add(measure.apply(historicalPriceTool))         // multi-timeframe OHLCV via Yahoo v8 chart
                .add(measure.apply(new TechnicalIndicatorsTool(historicalPriceTool)))
                .add(measure.apply(new StockFundamentalsTool(yahooQs, screenerInClient)))  // 4y financial statements + key stats
                .add(measure.apply(new AnalystRatingsTool(yahooQs)))      // sell-side consensus + price targets
                .add(measure.apply(new ai.nizo.tools.finance.StockNewsTool()))  // company news via Finnhub API (no scraping)
                .add(measure.apply(deepWorkTool))                               // long-horizon background jobs (plan→execute→verify)
                .add(measure.apply(new ai.nizo.agent.deepwork.JobStatusTool(deepJobs)))
                .add(measure.apply(new ai.nizo.agent.schedule.ScheduleTool(schedules, java.time.ZoneId.systemDefault())))
                .add(measure.apply(new InsiderActivityTool(yahooQs)))     // insider buys/sells
                .add(measure.apply(new EarningsHistoryTool(yahooQs)))     // beat/miss + next reporting date
                .add(measure.apply(buffettScoreTool))                     // Buffett-Munger 0-100 scorecard (no LLM)
                .add(measure.apply(new ai.nizo.tools.finance.IndiaUniverseTool(nseIndiaClient))) // NIFTY 500 / sector constituents
                .add(measure.apply(new ai.nizo.tools.finance.IndiaScreenerTool(
                        yahooQs, screenerInClient, historicalPriceTool, buffettScoreTool)))  // multi-factor INR ranker
                .add(measure.apply(new ai.nizo.tools.finance.IndiaMacroDashboardTool()))  // India regime classifier (Phase 2)
                .add(measure.apply(new ai.nizo.tools.finance.IndiaSectorViewTool(nseIndiaClient)))  // Per-sector momentum (Phase 3)
                .add(measure.apply(new ai.nizo.tools.finance.IndiaEventCalendarTool()))  // RBI/budget/election calendar (Phase 4)
                .add(measure.apply(new ai.nizo.tools.finance.IndiaPicksBacktestTool(historicalPriceTool)))  // Backtest harness (Phase 5)
                .add(measure.apply(new FileReadTool(home.workspace(), fileCache)))
                .add(measure.apply(new FileWriteTool(home.workspace())))
                .add(measure.apply(new FileListTool(home.workspace())))
                .add(measure.apply(new ShellTool(home.workspace())))
                .add(measure.apply(new CodeExecTool(home.workspace())))  // compute exact numbers — don't guess
                .add(measure.apply(new ai.nizo.tools.vision.ImageAnalyzeTool(llmClient, llmConfig.model(), home.workspace())))  // the agent's eyes
                .add(measure.apply(new MemoryRememberTool(userFacts)))
                .add(measure.apply(new MemoryRecallTool(userFacts)))
                .add(measure.apply(new MemoryForgetTool(userFacts)))
                .add(measure.apply(new SaveSkillTool(home.skillsDir())));

        // Forward-reference holder: sub-agent skills need the eventual full registry to dispatch
        // their inner tool calls. Filled in after builder.build() below.
        AtomicReference<ToolRegistry> registryRef = new AtomicReference<>();
        // India Top Picks orchestrator — chains india_universe -> india_screener with
        // sector diversification. Deterministic, no LLM in the hot path (like
        // DeterministicStockOrchestratorTool). Needs the registry to dispatch sub-tools,
        // hence its registration is after the registryRef is allocated.
        builder.add(measure.apply(new ai.nizo.skills.IndiaTopPicksTool(registryRef::get)));
        // General deep-agent: `research` delegates a focused sub-investigation to an isolated
        // worker (fresh context, bounded tool loop, summary-return). Generalizes the stock
        // pipeline's fan-out to any task; needs the registry to dispatch its inner tools.
        builder.add(measure.apply(new ai.nizo.skills.DeepAgentTool(
                llmClient, registryRef::get, llmConfig.model(), 8)));
        // Web-task agent: drives the browser through a deterministic observe→act→verify loop to
        // complete general multi-step web tasks (search, forms, cart, bookings). Closed action space.
        builder.add(measure.apply(new ai.nizo.skills.WebTaskSubAgent(
                llmClient, registryRef::get, llmConfig.model(), 24)));
        // Sub-agents get their own iteration cap, separate from the orchestrator's. 20 gives
        // enough headroom for analysts to fight through bot-blocks (DataDome, 404s, retries via
        // SmartProxy) before being asked to write — 12 was sometimes hit mid-research.
        int subAgentMaxIterations = 20;
        for (SkillManifest m : skills) {
            try {
                Tool skillTool;
                if ("stock_analysis".equals(m.name())) {
                    // Special case: the stock_analysis orchestrator is now deterministic Java
                    // code, NOT an LLM-driven skill. The previous behaviour (FilesystemSkillTool
                    // returning SKILL.md text + outer LLM following the playbook) was unreliable
                    // — observed May 2026 the LLM bypassed sub-skill calls and hallucinated a
                    // report from training data. This new tool runs the 8 sub-skills in a fixed
                    // schedule (5 in parallel + 3 sequential debate) and assembles the master
                    // report itself. The LLM only reads the result.
                    skillTool = new ai.nizo.skills.DeterministicStockOrchestratorTool(registryRef::get);
                } else {
                    skillTool = m.agent()
                            ? new SubAgentSkillTool(m, llmClient, registryRef::get,
                                                    llmConfig.model(), subAgentMaxIterations)
                            : new FilesystemSkillTool(m);
                }
                builder.add(measure.apply(skillTool));
            } catch (Exception e) {
                LOG.warn("skipping skill {} (duplicate name?): {}", m.name(), e.toString());
            }
        }

        // External MCP servers — load config, start subprocesses, register their tools alongside ours.
        // Failures per-server are logged and skipped; we never abort bootstrap because one MCP died.
        this.mcpConfig = McpServersFile.loadOrEmpty(home.mcpConfigFile());
        this.mcpPool = new McpClientPool();
        List<Tool> mcpTools = mcpPool.startAll(mcpConfig);
        int mcpAdded = 0;
        for (Tool t : mcpTools) {
            try { builder.add(measure.apply(t)); mcpAdded++; }
            catch (Exception e) { LOG.warn("skipping MCP tool {}: {}", t.name(), e.toString()); }
        }

        this.tools = builder.build();
        registryRef.set(this.tools);  // close the forward-reference loop for sub-agent skills

        // India Picks daily refresh — fires the picks pipeline once at 09:00 IST so the
        // library is always pre-populated. Disable via NIZO_INDIA_PICKS_AUTO_REFRESH=0.
        try {
            new ai.nizo.skills.IndiaPicksDailyScheduler(registryRef::get).start();
        } catch (Exception e) {
            LOG.warn("india-picks daily scheduler failed to arm: {}", e.toString());
        }
        this.condenseHooks = List.of();

        this.condense = new CondenseEngine(
                llmClient,
                sessions,
                tools,
                fileCache,
                condenseHooks,
                () -> skills,
                llmConfig.model(),
                () -> systemPrompt);

        this.agent = new AgentLoop(
                llmClient,
                tools,
                sessions,
                userFacts,
                condense,
                stockReports,
                llmConfig.model(),
                systemPrompt,
                maxIterations,
                historyMessages);

        // Reflective Phase — post-task self-learning (skills + user facts). Optional by
        // design: NIZO_REFLECT=0 disables, and the engine no-ops on trivial turns.
        agent.setReflection(new ai.nizo.agent.reflect.ReflectionEngine(
                llmClient, llmConfig.model(), home.skillsDir(), userFacts));

        // Skill curation — the other half of the self-learning loop. Reflection WRITES skills;
        // this grades them and reversibly retires the vacuous/overfit/duplicative ones on a
        // schedule (Hermes "writes, grades, prunes"). Daemon-scheduled; NIZO_SKILL_CURATOR=off disables.
        if (ai.nizo.agent.reflect.SkillCurator.enabled()) {
            new ai.nizo.agent.reflect.SkillCurator(llmClient, llmConfig.model(), home.skillsDir()).start();
        }

        // Deep Work — long-horizon background jobs. The engine needs the FINISHED tool
        // registry (its steps call tools), while the deep_work tool had to be registered
        // while the registry was still being built — hence the late bind() + boot resume.
        this.deepWork = new ai.nizo.agent.deepwork.DeepWorkEngine(
                llmClient, llmConfig.model(), tools, deepJobs, sessions);
        deepWorkTool.bind(deepWork);
        deepWork.resumeAll();

        // Server-owned per-chat workers — this is what makes "leave the page, come back, see
        // results" work. The chat keeps running in a virtual thread regardless of which (if any)
        // SSE subscribers are currently attached.
        this.chatExecutor = new ChatExecutor(agent);

        // Scheduler — fire reminders / recurring jobs, delivering each by running the stored prompt
        // through the agent into its chat. Daemon-ticked; schedules survive restart via the store.
        ai.nizo.agent.schedule.ScheduleRunner schedRunner = task -> {
            ai.nizo.api.tool.UserContext.set(task.userId());
            ai.nizo.api.tool.UserContext.setChat(task.chatId());
            try {
                ai.nizo.api.chat.OutgoingMessage out = agent.handle(new ai.nizo.api.chat.IncomingMessage(
                        task.userId(), task.chatId(), task.prompt(), java.util.List.of(), "schedule"));
                // Proactive push so a reminder reaches the user even off the web UI (no-op without a bot token).
                if (out != null && out.text() != null && !out.text().isBlank()) {
                    ai.nizo.channels.telegram.TelegramNotifier.push(task.chatId(), "⏰ " + out.text());
                }
            } finally {
                ai.nizo.api.tool.UserContext.clear();
            }
        };
        new ai.nizo.agent.schedule.SchedulerEngine(schedules, schedRunner,
                java.time.ZoneId.systemDefault(),
                java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()).start();

        LOG.info("bootstrap ready: home={} model={} tools={} (mcp={}) skills={} condense=on",
                home.root(), llmConfig.model(), tools.all().size(), mcpAdded, skills.size());
    }

    public List<String> toolNames() {
        return tools.all().stream().map(Tool::name).toList();
    }

    @Override
    public void close() {
        // Order matters: stop chat workers first (so they don't try to call into a torn-down
        // MCP pool), then MCP subprocesses, then anything else.
        try { if (chatExecutor != null) chatExecutor.close(); }
        catch (Exception e) { LOG.warn("chat executor close failed: {}", e.toString()); }
        try { if (mcpPool != null) mcpPool.close(); }
        catch (Exception e) { LOG.warn("mcp pool close failed: {}", e.toString()); }
    }
}
