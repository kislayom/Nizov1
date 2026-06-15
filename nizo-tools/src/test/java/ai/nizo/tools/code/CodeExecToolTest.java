package ai.nizo.tools.code;

import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link CodeExecTool}. The headline guarantees: it actually computes (so the model can
 * stop guessing numbers), it scrubs secrets from the child env, and it surfaces failures
 * (tracebacks, timeouts) honestly rather than swallowing them.
 *
 * <p>These spawn a real {@code python3}; they're skipped automatically if no interpreter is on PATH
 * so the suite still passes on a machine without Python.
 */
class CodeExecToolTest {

    private static boolean pythonAvailable() {
        for (String d : System.getenv().getOrDefault("PATH", "/usr/bin:/bin").split(":")) {
            if (Files.isExecutable(Path.of(d, "python3"))) return true;
        }
        return false;
    }

    private static Map<String, String> hostileEnv() {
        Map<String, String> env = new HashMap<>();
        env.put("PATH", System.getenv("PATH")); // need real PATH so python3 resolves
        env.put("HOME", System.getProperty("user.home"));
        env.put("HF_TOKEN", "hf_super_secret_should_NOT_leak");
        env.put("BRAVE_API_KEY", "brv_super_secret");
        env.put("TELEGRAM_BOT_TOKEN", "tg_secret_token");
        env.put("NIZO_LLM_TOKEN", "llm_secret");
        return env;
    }

    private static CodeExecTool tool(Path tmp) throws Exception {
        Path ws = tmp.resolve("ws");
        Files.createDirectories(ws);
        return new CodeExecTool(ws, hostileEnv());
    }

    @Test
    void computesExactArithmetic(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute("{\"code\":\"print(2+2)\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("exit=0"), r.content());
        assertTrue(r.content().contains("\n4"), "expected 4 on stdout:\n" + r.content());
    }

    @Test
    void computesCagrToMachinePrecision(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        // CAGR of 100 -> 200 over 10y = (2)^(1/10) - 1 = 7.1773% — the kind of number an LLM
        // routinely fumbles in its head. The tool must return the precise value.
        ToolResult r = t.execute("{\"code\":\"print(f'{((200/100)**(1/10)-1)*100:.4f}')\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("7.1773"), "expected exact CAGR:\n" + r.content());
    }

    @Test
    void scrubsSecretsFromChildEnv(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute(
                "{\"code\":\"import os; print('HF_TOKEN' in os.environ, 'PATH' in os.environ)\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("False True"),
                "HF_TOKEN must be absent and PATH present:\n" + r.content());
        assertFalse(r.content().contains("hf_super_secret_should_NOT_leak"));
    }

    @Test
    void surfacesTracebackAndNonzeroExit(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute("{\"code\":\"raise ValueError('boom')\"}");
        assertFalse(r.ok(), "a raised exception must be a failed result: " + r.content());
        assertTrue(r.content().contains("ValueError"), r.content());
        assertTrue(r.content().contains("boom"), r.content());
        assertFalse(r.content().contains("exit=0"), r.content());
    }

    @Test
    void timesOut(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute("{\"code\":\"import time; time.sleep(10)\",\"timeout_s\":1}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("timeout"), r.content());
    }

    @Test
    void pipesStdin(@TempDir Path tmp) throws Exception {
        if (!pythonAvailable()) return;
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute(
                "{\"code\":\"import sys; print(sys.stdin.read().strip().upper())\",\"stdin\":\"hello\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("HELLO"), r.content());
    }

    @Test
    void emptyCodeRejected(@TempDir Path tmp) throws Exception {
        CodeExecTool t = tool(tmp);
        ToolResult r = t.execute("{\"code\":\"\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"), r.content());
    }

    @Test
    void filteredEnvHasNoSecrets(@TempDir Path tmp) throws Exception {
        CodeExecTool t = tool(tmp);
        Map<String, String> env = t.filteredEnvForTest();
        for (String k : env.keySet()) {
            assertTrue(CodeExecTool.DEFAULT_ENV_ALLOWLIST.contains(k),
                    "child env contains non-allowlisted key: " + k);
        }
        assertFalse(env.containsKey("HF_TOKEN"));
        assertFalse(env.containsKey("BRAVE_API_KEY"));
        assertFalse(env.containsKey("TELEGRAM_BOT_TOKEN"));
        assertFalse(env.containsKey("NIZO_LLM_TOKEN"));
    }
}
