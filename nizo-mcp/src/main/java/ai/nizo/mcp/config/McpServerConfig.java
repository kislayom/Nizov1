package ai.nizo.mcp.config;

import java.util.List;
import java.util.Map;

/**
 * One MCP server entry in {@code ~/.nizo/mcp.json}.
 *
 * <p>Supports two transports — only stdio is implemented in v1. URL servers are loaded but
 * skipped at start time with a warning until {@code HttpSseMcpClient} lands.
 *
 * <p>Format mirrors Claude Desktop's {@code claude_desktop_config.json} so user configs are
 * portable in either direction.
 *
 * <pre>
 * {
 *   "mcpServers": {
 *     "filesystem": {
 *       "command": "npx",
 *       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/kislay/work"]
 *     },
 *     "github": {
 *       "command": "npx",
 *       "args": ["-y", "@modelcontextprotocol/server-github"],
 *       "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..." }
 *     }
 *   }
 * }
 * </pre>
 */
public record McpServerConfig(
        String name,
        Transport transport,
        String command,
        List<String> args,
        Map<String, String> env,
        String url,
        String authToken,
        boolean disabled
) {
    public McpServerConfig {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("name required");
        args = args == null ? List.of() : List.copyOf(args);
        env = env == null ? Map.of() : Map.copyOf(env);
    }

    public enum Transport { STDIO, HTTP }

    public static McpServerConfig stdio(String name, String command, List<String> args, Map<String, String> env) {
        return new McpServerConfig(name, Transport.STDIO, command, args, env, null, null, false);
    }

    public static McpServerConfig http(String name, String url, String authToken) {
        return new McpServerConfig(name, Transport.HTTP, null, List.of(), Map.of(), url, authToken, false);
    }

    /** Full argv (command + args), the form {@link ProcessBuilder} expects. */
    public List<String> commandLine() {
        if (transport != Transport.STDIO) {
            throw new IllegalStateException("commandLine() called on non-stdio MCP config: " + name);
        }
        java.util.List<String> out = new java.util.ArrayList<>(1 + args.size());
        out.add(command);
        out.addAll(args);
        return out;
    }
}
