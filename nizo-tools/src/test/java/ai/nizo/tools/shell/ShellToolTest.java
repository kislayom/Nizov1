package ai.nizo.tools.shell;

import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ShellTool} — the headline being env scrubbing. We never set real secrets in
 * {@code System.getenv()} (it's effectively read-only and we don't want CI dependencies), so we
 * use the package-private constructor that takes a fake parent env map.
 */
class ShellToolTest {

    /** Build a fake "parent" env that mimics the agent's runtime env including secrets. */
    private static Map<String, String> hostileEnv() {
        Map<String, String> env = new HashMap<>();
        env.put("PATH", System.getenv("PATH")); // need real PATH so /usr/bin/env etc. work
        env.put("HOME", System.getProperty("user.home"));
        env.put("HF_TOKEN", "hf_super_secret_should_NOT_leak");
        env.put("BRAVE_API_KEY", "brv_super_secret");
        env.put("TELEGRAM_BOT_TOKEN", "tg_secret_token");
        env.put("NIZO_LLM_TOKEN", "llm_secret");
        env.put("SMARTPROXY_PASSWORD", "proxy_pw_should_not_leak");
        env.put("AWS_SECRET_ACCESS_KEY", "aws_secret");
        env.put("RANDOM_PROJECT_VAR", "random_value");
        return env;
    }

    @Test
    void echoStillWorks(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        ShellTool t = new ShellTool(workspace, hostileEnv());

        ToolResult r = t.execute("{\"command\":\"echo hello-world\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("hello-world"),
                "expected echo output, got: " + r.content());
    }

    @Test
    void envContainsAllowlistedVarsOnly(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        ShellTool t = new ShellTool(workspace, hostileEnv());

        ToolResult r = t.execute("{\"command\":\"env\"}");
        assertTrue(r.ok(), r.content());
        String body = r.content();

        // Secrets must not be in the child env.
        assertFalse(body.contains("HF_TOKEN="),
                "HF_TOKEN must not be inherited:\n" + body);
        assertFalse(body.contains("hf_super_secret_should_NOT_leak"));
        assertFalse(body.contains("BRAVE_API_KEY="));
        assertFalse(body.contains("brv_super_secret"));
        assertFalse(body.contains("TELEGRAM_BOT_TOKEN="));
        assertFalse(body.contains("NIZO_LLM_TOKEN="));
        assertFalse(body.contains("SMARTPROXY_PASSWORD="));
        assertFalse(body.contains("AWS_SECRET_ACCESS_KEY="));
        assertFalse(body.contains("RANDOM_PROJECT_VAR="));

        // Allowlisted vars that we set in the fake parent env are passed through.
        assertTrue(body.contains("PATH="),
                "PATH must be in allowlisted env: " + body);
    }

    @Test
    void filteredEnvMapMatchesAllowlist() {
        ShellTool t = new ShellTool(Path.of(System.getProperty("java.io.tmpdir"), "nizo-test-ws-1"),
                hostileEnv());
        Map<String, String> env = t.filteredEnvForTest();
        // Only allowlisted keys (those we provided in hostileEnv that are in allowlist).
        for (String k : env.keySet()) {
            assertTrue(ShellTool.DEFAULT_ENV_ALLOWLIST.contains(k),
                    "child env contains non-allowlisted key: " + k);
        }
        // Secret keys must not appear no matter what.
        assertFalse(env.containsKey("HF_TOKEN"));
        assertFalse(env.containsKey("BRAVE_API_KEY"));
        assertFalse(env.containsKey("TELEGRAM_BOT_TOKEN"));
        assertFalse(env.containsKey("NIZO_LLM_TOKEN"));
        assertFalse(env.containsKey("SMARTPROXY_PASSWORD"));
    }

    @Test
    void extraAllowlistAddsCommaSeparatedNames(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Map<String, String> parent = hostileEnv();
        parent.put("MY_DEBUG", "1");
        parent.put("MY_FLAG", "on");
        parent.put(ShellTool.EXTRA_ALLOWLIST_ENV, "MY_DEBUG, MY_FLAG"); // with whitespace

        ShellTool t = new ShellTool(workspace, parent);
        Map<String, String> env = t.filteredEnvForTest();
        assertEquals("1", env.get("MY_DEBUG"), "MY_DEBUG should be allowlisted via extras");
        assertEquals("on", env.get("MY_FLAG"));
        // Secret values still excluded — extras don't override the implicit deny.
        assertFalse(env.containsKey("HF_TOKEN"));

        // And confirm via running env that the new entries are visible to the child.
        ToolResult r = t.execute("{\"command\":\"env\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("MY_DEBUG=1"));
        assertTrue(r.content().contains("MY_FLAG=on"));
    }

    @Test
    void emptyExtraAllowlistIsHarmless(@TempDir Path tmp) {
        Path workspace = tmp.resolve("ws");
        Map<String, String> parent = hostileEnv();
        parent.put(ShellTool.EXTRA_ALLOWLIST_ENV, "");
        ShellTool t = new ShellTool(workspace, parent);
        Map<String, String> env = t.filteredEnvForTest();
        for (String k : env.keySet()) {
            assertTrue(ShellTool.DEFAULT_ENV_ALLOWLIST.contains(k));
        }
    }

    @Test
    void exitCodePropagated(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        ShellTool t = new ShellTool(workspace, hostileEnv());
        ToolResult r = t.execute("{\"command\":\"exit 7\"}");
        assertFalse(r.ok());
        assertTrue(r.content().contains("exit=7"), r.content());
    }

    @Test
    void emptyCommandRejected(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        ShellTool t = new ShellTool(workspace, hostileEnv());
        ToolResult r = t.execute("{\"command\":\"\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"));
    }
}
