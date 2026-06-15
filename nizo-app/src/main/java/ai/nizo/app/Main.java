package ai.nizo.app;

import ai.nizo.api.chat.IncomingMessage;
import ai.nizo.app.bootstrap.Bootstrap;
import ai.nizo.channels.telegram.TelegramChannel;
import ai.nizo.channels.telegram.TelegramConfig;
import ai.nizo.channels.web.WebChannel;
import ai.nizo.channels.web.WebConfig;

import java.awt.Desktop;
import java.net.URI;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Entry point.
 *
 * <p>Modes (argv[0]):
 * <ul>
 *   <li>{@code chat <prompt…>} — one-shot CLI chat (default)</li>
 *   <li>{@code ui} / {@code web} — local web UI with streaming SSE on :7777 (default)</li>
 *   <li>{@code telegram} — long-poll Telegram bot</li>
 *   <li>{@code tools} — print the registered tool catalogue and exit</li>
 * </ul>
 *
 * <p>Configuration: env vars + {@code ~/.nizo/} home directory.
 */
public final class Main {

    private static final String DEFAULT_SYSTEM_PROMPT = """
            You are Nizo, a personal assistant running on a local GPU (Qwen3.6-27B via llama.cpp).

            Style: concise, direct, no fluff. No hedging. No "I'm an AI…" preambles.
            Plain text by default; markdown only when it helps (lists, code).

            You have a real toolset, persistent per-chat session history, AND persistent
            cross-conversation memory about the user. Anything you remember about the user
            via `memory_remember` is shown to you as "## What you remember about this user"
            at the top of the prompt — read it on every turn.

            Use tools when:
            - The user shares a durable fact about themselves (name, role, location, project,
              preference, team) — call `memory_remember` with a third-person sentence.
            - The user asks "what do you know about me" / "who am I" — `memory_recall`.
            - The user says "forget about X" or corrects a stored fact — `memory_forget`.
            - The user asks about now / today / current state — `current_time` or `web_search`.
            - You need facts beyond your training cutoff — `web_search` → `web_fetch`.
            - The user gives a URL or file — `web_fetch` / `file_read`.
            - The user asks to run, inspect, or edit something on disk — `shell` / `file_*`.
            - You worked out a procedure worth keeping — `save_skill`.

            Tool calling is cheap. Prefer one well-scoped call over guessing. After a tool
            returns, integrate the result into your answer rather than just repeating it.
            Be proactive about memory — remember names, roles, and preferences without being
            asked. But only DURABLE facts; don't memorise per-message context.

            ## Stock / ticker inputs — MUST route through skill_stock_analysis

            If the user's message is a ticker symbol (AAPL, NVDA, RELIANCE.NS, BHP.AX,
            7203.T, ^NSEI, ^GSPC, NIFTY, NIFTY 50, S&P 500, SENSEX, ASX 200) — or any
            short phrase asking for stock / market analysis on one specific symbol — your
            FIRST and ONLY tool call MUST be `skill_stock_analysis` with that symbol as
            input. Do NOT call individual sub-skills (`skill_stock_news_analyst`,
            `skill_stock_fundamentals_analyst`, etc.) directly — those are internal to the
            pipeline. Do NOT call `web_search` / `web_fetch` first to "gather context"
            before delegating; the orchestrator handles that. Do NOT answer from training
            data; the orchestrator produces a multi-agent research report with live data
            + interactive charts + Buffett scorecard that you cannot replicate by hand.

            After `skill_stock_analysis` returns, the runtime short-circuits the response
            to the user — your job is just to call it. One tool call, no preamble.

            ## Web fetches that get blocked

            If `web_fetch` returns "Cloudflare bot challenge" / "Akamai block" / similar,
            it means the site detected scraping. DON'T pretend the data is there. Try:
            - Different source on a different domain (most facts have multiple homes)
            - Wikipedia for canonical facts
            - The site's RSS / sitemap / open data feed if known
            - As a last resort: tell the user the source blocked us and suggest they paste
              the content manually.
            Do NOT report partial / wrong data because the actual page was a captcha.

            ## Multilingual rules (CRITICAL — STRICT)

            When you reply in a non-English language, you must feel NATIVE — not translated.

            **HINDI** — this is enforced strictly:
            - Output MUST be in Devanagari script (देवनागरी): नमस्ते, क्या हाल है, धन्यवाद.
            - Romanized / Latin Hindi is FORBIDDEN.
              ❌ WRONG: "Haan bilkul. Main Hindi mein baat kar sakta hoon."
              ❌ WRONG: "Namaste! Aap kaise hain?"
              ✅ RIGHT: "हाँ बिल्कुल। मैं हिंदी में बात कर सकता हूँ।"
              ✅ RIGHT: "नमस्ते! आप कैसे हैं?"
            - Even if the user writes "in Hinglish" or types in Latin, you reply in Devanagari.
            - Use natural Hindi register, not literal-translated English.
            - Mix in common English nouns when natural (AI, GPU, app) — that's how Indians
              actually speak Hindi. But the surrounding sentence stays in Devanagari.
            - **Tamil/Telugu/Bengali/Marathi/Gujarati/Punjabi/Malayalam/Kannada**: native
              script always (தமிழ், తెలుగు, বাংলা, etc.).
            - **Urdu**: Nastaliq/Arabic script.
            - **Chinese**: Simplified or Traditional per user's preference; default Simplified.
            - **Arabic**: Right-to-left, Modern Standard Arabic unless user uses dialect.
            - **Japanese**: Mix of kanji + hiragana/katakana as a native would, not all hiragana.
            - **Korean**: Hangul.

            Match the user's register: casual if they're casual, formal if they're formal.
            If the user mixes English and another language (code-switching), match their pattern.
            """;

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0].toLowerCase() : "ui";
        switch (mode) {
            case "telegram"      -> runTelegram();
            case "ui", "web"     -> runWebUi();
            case "tools"         -> runToolsList();
            case "mcp"           -> runMcp(restArgs(args));
            case "chat", "--chat" -> runOneShotChat(restArgs(args));
            default              -> runOneShotChat(args);
        }
    }

    // --------- modes ---------

    private static void runOneShotChat(String[] argv) {
        try (Bootstrap b = new Bootstrap(systemPrompt(),
                envInt("NIZO_AGENT_MAX_ITERATIONS", 8),
                envInt("NIZO_HISTORY_MESSAGES", 12))) {
            announce(b);
            String prompt = argv.length > 0 ? String.join(" ", argv)
                    : "Hello! Introduce yourself in one sentence.";
            var out = b.agent.handle(new IncomingMessage(
                    "cli-user", "cli-default", prompt, List.of(), "cli"));
            System.out.println(out.text());
        }
    }

    private static void runWebUi() throws Exception {
        WebConfig webCfg = WebConfig.fromEnv();
        try (Bootstrap b = new Bootstrap(systemPrompt(),
                envInt("NIZO_AGENT_MAX_ITERATIONS", 50),
                envInt("NIZO_HISTORY_MESSAGES", 24))) {
            announce(b);
            ai.nizo.channels.web.WebUiContext ctx = new ai.nizo.channels.web.WebUiContext(
                    b.llmConfig.model(),
                    b.llmConfig.baseUrl(),
                    b.home.root(),
                    b.home.workspace(),
                    b.sessions,
                    b.tools,
                    () -> new ai.nizo.skills.SkillLoader().discover(b.home.skillsDir()),
                    b.condense,
                    b.mcpPool,
                    b.mcpConfig,
                    b.chatExecutor,
                    b.usage,
                    b.stockReports
            );
            try (WebChannel web = new WebChannel(webCfg, b.agent, b.agent, ctx)) {
                web.start();
                System.err.println("[nizo] open " + web.url());
                tryOpenBrowser(web.url());
                // Also serve Telegram alongside web if a bot token is configured (same agent/stores).
                TelegramChannel tg = maybeStartTelegram(b);
                try {
                    blockUntilShutdown();
                } finally {
                    if (tg != null) try { tg.close(); } catch (Exception ignored) {}
                }
            }
        }
    }

    /** Start Telegram alongside the web server if TELEGRAM_BOT_TOKEN is set; null (web-only) otherwise. */
    private static TelegramChannel maybeStartTelegram(Bootstrap b) {
        String token = System.getenv("TELEGRAM_BOT_TOKEN");
        if (token == null || token.isBlank()) return null;
        try {
            TelegramConfig cfg = TelegramConfig.fromEnv();
            TelegramChannel tg = new TelegramChannel(cfg, b.agent);
            tg.start();
            System.err.println("[nizo] Telegram bot @" + cfg.botUsername() + " started alongside web");
            return tg;
        } catch (Exception e) {
            System.err.println("[nizo] Telegram start failed (continuing web-only): " + e.getMessage());
            return null;
        }
    }

    private static void runTelegram() throws Exception {
        TelegramConfig tgCfg = TelegramConfig.fromEnv();
        try (Bootstrap b = new Bootstrap(systemPrompt(),
                envInt("NIZO_AGENT_MAX_ITERATIONS", 50),
                envInt("NIZO_HISTORY_MESSAGES", 24))) {
            announce(b);
            System.err.println("[nizo] starting Telegram bot @" + tgCfg.botUsername());
            try (TelegramChannel channel = new TelegramChannel(tgCfg, b.agent)) {
                channel.start();
                blockUntilShutdown();
            }
        }
    }

    private static void runToolsList() {
        try (Bootstrap b = new Bootstrap(systemPrompt(), 1, 1)) {
            System.out.println("Tools registered (" + b.tools.all().size() + "):");
            for (var t : b.tools.all()) {
                String origin = (t instanceof ai.nizo.mcp.client.McpClientTool m)
                        ? " (mcp:" + m.serverName() + ")" : "";
                System.out.println("  " + t.name() + origin);
                System.out.println("      " + t.description().replace("\n", " "));
            }
            System.out.println("\nSkills loaded (" + b.skills.size() + "):");
            for (var s : b.skills) {
                System.out.println("  " + s.name() + " — " + s.description());
            }
        }
    }

    /**
     * MCP subcommands.
     * <pre>
     *   nizo mcp list             — show configured servers, tool counts, failures
     *   nizo mcp add &lt;name&gt; &lt;cmd&gt; [args...]
     *                             — add a stdio server entry to ~/.nizo/mcp.json
     *   nizo mcp remove &lt;name&gt;     — drop an entry
     * </pre>
     */
    private static void runMcp(String[] argv) {
        String sub = argv.length > 0 ? argv[0].toLowerCase() : "list";
        switch (sub) {
            case "list" -> mcpList();
            case "add"  -> mcpAdd(argv);
            case "remove", "rm" -> mcpRemove(argv);
            default -> {
                System.err.println("Usage: nizo mcp [list|add <name> <cmd> [args...]|remove <name>]");
                System.exit(1);
            }
        }
    }

    private static void mcpList() {
        // Boot fully so we know which servers actually started and what tools they offered.
        try (Bootstrap b = new Bootstrap(systemPrompt(), 1, 1)) {
            System.out.println("MCP config: " + b.home.mcpConfigFile());
            if (b.mcpConfig.isEmpty()) {
                System.out.println("(no servers configured — add one with `nizo mcp add ...`)");
                return;
            }
            System.out.println();
            System.out.println("Servers (" + b.mcpConfig.list().size() + "):");
            var counts = b.mcpPool.toolCounts();
            var failures = b.mcpPool.failures();
            for (var cfg : b.mcpConfig.list()) {
                String tail;
                if (cfg.disabled())             tail = "[disabled]";
                else if (counts.containsKey(cfg.name())) tail = "ok · " + counts.get(cfg.name()) + " tool(s)";
                else if (failures.containsKey(cfg.name())) tail = "FAILED · " + failures.get(cfg.name());
                else tail = "(unknown)";
                System.out.println("  " + cfg.name() + " [" + cfg.transport() + "] — " + tail);
                if (cfg.transport() == ai.nizo.mcp.config.McpServerConfig.Transport.STDIO) {
                    System.out.println("      cmd: " + String.join(" ", cfg.commandLine()));
                } else {
                    System.out.println("      url: " + cfg.url());
                }
            }
            if (!b.tools.all().isEmpty()) {
                System.out.println();
                System.out.println("MCP tools registered:");
                int n = 0;
                for (var t : b.tools.all()) {
                    if (t instanceof ai.nizo.mcp.client.McpClientTool m) {
                        System.out.println("  " + t.name() + "  →  " + m.serverName() + ":" + m.remoteName());
                        n++;
                    }
                }
                if (n == 0) System.out.println("  (none — all servers failed or are disabled)");
            }
        }
    }

    private static void mcpAdd(String[] argv) {
        if (argv.length < 3) {
            System.err.println("Usage: nizo mcp add <name> <command> [args...]");
            System.exit(1);
        }
        String name = argv[1];
        String command = argv[2];
        java.util.List<String> args = java.util.List.of();
        if (argv.length > 3) {
            args = new java.util.ArrayList<>();
            for (int i = 3; i < argv.length; i++) args.add(argv[i]);
        }
        var home = ai.nizo.app.config.NizoHome.resolve();
        var file = ai.nizo.mcp.config.McpServersFile.loadOrEmpty(home.mcpConfigFile());
        file.put(ai.nizo.mcp.config.McpServerConfig.stdio(name, command, args, java.util.Map.of()));
        try {
            file.save();
            System.out.println("Added MCP server " + name + " → " + command + " " + String.join(" ", args));
            System.out.println("Config: " + home.mcpConfigFile());
        } catch (Exception e) {
            System.err.println("Save failed: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void mcpRemove(String[] argv) {
        if (argv.length < 2) {
            System.err.println("Usage: nizo mcp remove <name>");
            System.exit(1);
        }
        var home = ai.nizo.app.config.NizoHome.resolve();
        var file = ai.nizo.mcp.config.McpServersFile.loadOrEmpty(home.mcpConfigFile());
        var removed = file.remove(argv[1]);
        if (removed == null) {
            System.err.println("No such MCP server: " + argv[1]);
            System.exit(1);
        }
        try {
            file.save();
            System.out.println("Removed MCP server: " + argv[1]);
        } catch (Exception e) {
            System.err.println("Save failed: " + e.getMessage());
            System.exit(1);
        }
    }

    // --------- helpers ---------

    private static void announce(Bootstrap b) {
        System.err.println("[nizo] LLM endpoint=" + b.llmConfig.baseUrl() + " model=" + b.llmConfig.model());
        System.err.println("[nizo] home=" + b.home.root() + " tools=" + b.tools.all().size()
                + " skills=" + b.skills.size());
    }

    private static void tryOpenBrowser(String url) {
        try {
            if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(URI.create(url));
            }
        } catch (Throwable ignore) { /* headless server: nothing to do */ }
    }

    private static void blockUntilShutdown() throws InterruptedException {
        CountDownLatch shutdown = new CountDownLatch(1);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.err.println("[nizo] shutdown signal");
            shutdown.countDown();
        }, "nizo-shutdown"));
        System.err.println("[nizo] running. Ctrl-C to stop.");
        shutdown.await();
    }

    private static String systemPrompt() {
        String override = System.getenv("NIZO_SYSTEM_PROMPT");
        return (override != null && !override.isBlank()) ? override : DEFAULT_SYSTEM_PROMPT;
    }

    private static int envInt(String name, int dflt) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return dflt;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { return dflt; }
    }

    private static String[] restArgs(String[] args) {
        if (args.length <= 1) return new String[0];
        String[] rest = new String[args.length - 1];
        System.arraycopy(args, 1, rest, 0, rest.length);
        return rest;
    }

    private Main() {}
}
