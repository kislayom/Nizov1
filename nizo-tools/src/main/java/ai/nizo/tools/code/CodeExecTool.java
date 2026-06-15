package ai.nizo.tools.code;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Execute a short Python 3 program and capture its output — the agent's "compute, don't guess"
 * primitive.
 *
 * <p><b>Why this exists.</b> An LLM that does multi-step arithmetic, financial math (CAGR,
 * returns, ratios), statistics, or date math <em>in its head</em> hallucinates numbers. The
 * Deep Work verifier keeps catching exactly this — a CAGR cited with zero tool calls behind it.
 * The fix is to let the model run real code: it writes a tiny Python program, we run it, and the
 * model cites the printed result. That is the difference between "sounds right" and machine-level
 * accuracy. {@code shell} could technically do this, but multi-line Python through {@code bash -c}
 * is a quoting minefield the model avoids — so it falls back to guessing. A first-class tool that
 * takes raw source and returns structured stdout/stderr removes that friction.
 *
 * <p><b>Security posture (matches {@link ai.nizo.tools.shell.ShellTool}, NOT a hardened sandbox).</b>
 * <ul>
 *   <li>Working directory is a fresh per-call subdirectory under {@code NIZO_HOME/workspace}
 *       (auto-created, deleted on exit). Concurrent calls never collide.</li>
 *   <li>Default timeout 30 s, cap 120 s; the process is force-killed on timeout.</li>
 *   <li>stdout and stderr are captured <em>separately</em> (stdout = the answer, stderr =
 *       tracebacks) and each truncated to 16 KB before returning to the model.</li>
 *   <li><b>Environment is scrubbed</b> to the same innocuous allowlist as the shell tool —
 *       secrets ({@code HF_TOKEN}, {@code BRAVE_API_KEY}, {@code TELEGRAM_BOT_TOKEN},
 *       {@code NIZO_LLM_TOKEN}, OAuth tokens, …) are NOT inherited by the child. This is the
 *       primary defence against a prompt-injected exfiltration script.</li>
 *   <li>Optional network isolation: set {@code NIZO_CODEEXEC_NO_NET=1} and, if {@code unshare}
 *       is present and usable, the interpreter runs in a fresh network namespace
 *       ({@code unshare -n}) so even a malicious script cannot phone home. Best-effort: if the
 *       unshare wrapper cannot start (no user-namespace privilege), we log and fall back to a
 *       plain run rather than failing the call.</li>
 * </ul>
 *
 * <p>This is deliberately Python-only for now: Python's stdlib covers the overwhelming majority of
 * "analyse this like a human would" work, and if the host has numpy/pandas installed they import
 * normally (system site-packages are found via the interpreter prefix, not {@code PYTHONPATH},
 * which we strip). The interpreter is {@code python3} by default; override with
 * {@code NIZO_CODEEXEC_PYTHON}.
 */
public final class CodeExecTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(CodeExecTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final int DEFAULT_TIMEOUT_S = 30;
    private static final int MAX_TIMEOUT_S = 120;
    private static final int MAX_OUTPUT_CHARS = 16_000;
    private static final int MAX_SOURCE_CHARS = 200_000;

    /** Innocuous environment variables passed through to the interpreter. Mirrors ShellTool. */
    static final Set<String> DEFAULT_ENV_ALLOWLIST = Set.of(
            "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
            "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR");

    static final String EXTRA_ALLOWLIST_ENV = "NIZO_CODEEXEC_ENV_EXTRA";
    static final String PYTHON_ENV = "NIZO_CODEEXEC_PYTHON";
    static final String NO_NET_ENV = "NIZO_CODEEXEC_NO_NET";

    private final Path workspace;
    private final Map<String, String> filteredEnv;
    private final String python;
    private final boolean noNet;

    public CodeExecTool(Path workspace) {
        this(workspace, System.getenv());
    }

    /** Test-friendly: pass an explicit parent environment to filter and to read knobs from. */
    CodeExecTool(Path workspace, Map<String, String> parentEnv) {
        this.workspace = workspace;
        try { Files.createDirectories(workspace); }
        catch (IOException e) { throw new RuntimeException("could not create code-exec workspace " + workspace, e); }
        this.filteredEnv = Collections.unmodifiableMap(buildFilteredEnv(parentEnv));
        String py = parentEnv.get(PYTHON_ENV);
        this.python = (py == null || py.isBlank()) ? "python3" : py.trim();
        String nn = parentEnv.get(NO_NET_ENV);
        this.noNet = nn != null && (nn.equals("1") || nn.equalsIgnoreCase("true"));
    }

    private static Map<String, String> buildFilteredEnv(Map<String, String> parent) {
        Set<String> allow = new LinkedHashSet<>(DEFAULT_ENV_ALLOWLIST);
        String extra = parent.get(EXTRA_ALLOWLIST_ENV);
        if (extra != null && !extra.isBlank()) {
            Arrays.stream(extra.split(","))
                    .map(String::trim).filter(s -> !s.isEmpty()).forEach(allow::add);
        }
        Map<String, String> out = new java.util.HashMap<>();
        for (String k : allow) {
            String v = parent.get(k);
            if (v != null) out.put(k, v);
        }
        return out;
    }

    @Override public String name() { return "code_exec"; }

    @Override
    public String description() {
        return "Run a Python 3 program and capture its output. THIS IS HOW YOU COMPUTE EXACT "
                + "VALUES — never do multi-step arithmetic, financial math (CAGR, returns, ratios, "
                + "compounding), statistics, date/duration math, unit conversion, or large-table "
                + "aggregation in your head; write the calculation here and cite the printed result. "
                + "Also use it to parse and reduce JSON/CSV you fetched from other tools. The program "
                + "runs in a private scratch directory with the Python standard library (numpy/pandas "
                + "if the host has them). Print answers to stdout. stdout and stderr come back "
                + "separately, with the exit code; a traceback on stderr means fix the code and retry. "
                + "Default timeout 30 s (max 120). No persistent state between calls.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "code":      { "type": "string", "description": "Python 3 source. Print results you want to read to stdout." },
                "stdin":     { "type": "string", "description": "Optional text piped to the program's standard input." },
                "timeout_s": { "type": "integer", "description": "Hard timeout in seconds (default 30, max 120).",
                               "minimum": 1, "maximum": 120 }
              },
              "required": ["code"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String code = args.path("code").asText("");
        if (code.isBlank()) return ToolResult.error("code is required");
        if (code.length() > MAX_SOURCE_CHARS)
            return ToolResult.error("code too large (" + code.length() + " chars; max " + MAX_SOURCE_CHARS + ")");
        String stdin = args.path("stdin").asText("");
        int timeoutS = clamp(args.path("timeout_s").asInt(DEFAULT_TIMEOUT_S), 1, MAX_TIMEOUT_S);

        Path dir = Files.createTempDirectory(workspace, "code-");
        try {
            Path script = dir.resolve("main.py");
            Files.writeString(script, code, StandardCharsets.UTF_8);
            return run(dir, script, stdin, timeoutS, this.noNet);
        } finally {
            deleteRecursive(dir);
        }
    }

    private ToolResult run(Path dir, Path script, String stdin, int timeoutS, boolean isolateNet) throws Exception {
        List<String> cmd = new ArrayList<>();
        boolean wrappedNet = false;
        if (isolateNet && hasUnshare()) {
            // Fresh network namespace: the program sees only loopback, cannot reach the network.
            cmd.addAll(List.of("unshare", "-n", "--"));
            wrappedNet = true;
        }
        cmd.add(python);
        cmd.add(script.toString());

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.directory(dir.toFile());
        Map<String, String> env = pb.environment();
        env.clear();
        env.putAll(filteredEnv);
        // Unbuffer so partial output survives a timeout kill; keep tracebacks readable.
        env.put("PYTHONUNBUFFERED", "1");

        long t0 = System.nanoTime();
        Process p;
        try {
            p = pb.start();
        } catch (IOException e) {
            if (wrappedNet) {
                // unshare not usable (no user-ns privilege) — retry once without isolation.
                LOG.warn("code_exec: unshare -n unavailable ({}), running without network isolation", e.getMessage());
                return run(dir, script, stdin, timeoutS, false);
            }
            return ToolResult.error("could not start " + python + ": " + e.getMessage()
                    + " (is Python 3 installed and on PATH?)");
        }

        // Feed stdin (if any) on a side thread so a program that ignores stdin can't deadlock us.
        Thread stdinThread = null;
        if (!stdin.isEmpty()) {
            stdinThread = new Thread(() -> {
                try (OutputStream os = p.getOutputStream()) {
                    os.write(stdin.getBytes(StandardCharsets.UTF_8));
                } catch (IOException ignored) { /* process may have exited */ }
            }, "code-exec-stdin");
            stdinThread.setDaemon(true);
            stdinThread.start();
        } else {
            try { p.getOutputStream().close(); } catch (IOException ignored) {}
        }

        // Drain BOTH streams on daemon threads so the main thread is never blocked on a read —
        // waitFor() is the sole authority on the timeout. (Reading stdout inline would block until
        // the child exits, letting a silent sleep outlast its deadline.) Sinks are bounded; when we
        // destroyForcibly() on timeout the pipes close and the drains hit EOF and finish.
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        Thread outThread = new Thread(() -> drain(p.getInputStream(), out), "code-exec-stdout");
        Thread errThread = new Thread(() -> drain(p.getErrorStream(), err), "code-exec-stderr");
        outThread.setDaemon(true); errThread.setDaemon(true);
        outThread.start(); errThread.start();

        boolean exited;
        try {
            exited = p.waitFor(timeoutS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            p.destroyForcibly();
            return ToolResult.error("interrupted");
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;

        if (!exited) {
            p.destroyForcibly();
            joinQuietly(outThread, 500); joinQuietly(errThread, 500);
            return ToolResult.error("timeout after " + timeoutS + "s — the program did not finish. "
                    + "partial stdout:\n" + truncate(out.toString(), MAX_OUTPUT_CHARS)
                    + "\npartial stderr:\n" + truncate(err.toString(), MAX_OUTPUT_CHARS));
        }
        joinQuietly(outThread, 1000); joinQuietly(errThread, 1000);

        int rc = p.exitValue();
        String header = "exit=" + rc + " time=" + ms + "ms" + (wrappedNet ? " net=isolated" : "");
        String body = header
                + "\n--- stdout ---\n" + (out.length() == 0 ? "(empty)" : truncate(out.toString(), MAX_OUTPUT_CHARS))
                + "\n--- stderr ---\n" + (err.length() == 0 ? "(empty)" : truncate(err.toString(), MAX_OUTPUT_CHARS));
        if (rc != 0) LOG.debug("code_exec rc={} ({}ms)", rc, ms);
        return new ToolResult(rc == 0, body);
    }

    /** Read a stream into a bounded buffer; stop accumulating past the cap (keeps memory bounded). */
    private static void drain(java.io.InputStream is, StringBuilder sink) {
        byte[] buf = new byte[4096];
        int n;
        try {
            while ((n = is.read(buf)) != -1) {
                if (sink.length() < MAX_OUTPUT_CHARS + 1024) {
                    sink.append(new String(buf, 0, n, StandardCharsets.UTF_8));
                }
                // keep reading to drain the pipe even after the sink is full, so the process
                // isn't blocked indefinitely on a full pipe before the timeout fires
            }
        } catch (IOException ignored) {
            // process exited / stream closed
        }
    }

    private static void joinQuietly(Thread t, long ms) {
        try { t.join(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    private boolean hasUnshare() {
        for (String d : pathDirs()) {
            if (Files.isExecutable(Path.of(d, "unshare"))) return true;
        }
        return false;
    }

    private List<String> pathDirs() {
        String path = filteredEnv.getOrDefault("PATH", "/usr/bin:/bin:/usr/local/bin");
        return Arrays.asList(path.split(":"));
    }

    private static void deleteRecursive(Path dir) {
        if (dir == null) return;
        try (var walk = Files.walk(dir)) {
            walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try { Files.deleteIfExists(p); } catch (IOException ignored) {}
            });
        } catch (IOException ignored) {}
    }

    /** Visible for tests. */
    Map<String, String> filteredEnvForTest() { return filteredEnv; }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "\n…[truncated to " + max + " chars]";
    }
}
