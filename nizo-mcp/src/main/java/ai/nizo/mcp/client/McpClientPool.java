package ai.nizo.mcp.client;

import ai.nizo.api.tool.Tool;
import ai.nizo.mcp.McpClient;
import ai.nizo.mcp.config.McpServerConfig;
import ai.nizo.mcp.config.McpServersFile;
import ai.nizo.mcp.protocol.McpToolDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages a set of MCP clients started from a {@link McpServersFile}. One bad server (crashed
 * subprocess, missing token, unsupported transport) never blocks the others — failures are
 * logged and the rest start normally.
 *
 * <p>{@link #close()} on the pool tears every client down in reverse order — call from
 * Bootstrap's {@code close()} to ensure subprocesses die when the agent exits.
 */
public final class McpClientPool implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(McpClientPool.class);

    /** Servers we successfully started (may be fewer than the config requested). */
    private final Map<String, McpClient> clients = new LinkedHashMap<>();
    /** Per-server tool counts after start, useful for status output. */
    private final Map<String, Integer> toolCounts = new LinkedHashMap<>();
    /** Servers that failed to start, with their failure reason. */
    private final Map<String, String> failures = new LinkedHashMap<>();

    /**
     * Spin up every server in {@code config} and return all the resulting tools as a single list.
     * Bootstrap drops these into its {@code ToolRegistry} alongside native tools.
     */
    public synchronized List<Tool> startAll(McpServersFile config) {
        List<Tool> out = new ArrayList<>();
        if (config == null || config.isEmpty()) {
            LOG.info("MCP: no servers configured");
            return out;
        }
        for (McpServerConfig cfg : config.list()) {
            if (cfg.disabled()) {
                LOG.info("MCP[{}] disabled in config; skipping", cfg.name());
                continue;
            }
            try {
                McpClient client = newClient(cfg);
                client.start();
                List<McpToolDescriptor> tools = client.listTools();
                for (McpToolDescriptor td : tools) {
                    out.add(new McpClientTool(client, td));
                }
                clients.put(cfg.name(), client);
                toolCounts.put(cfg.name(), tools.size());
                LOG.info("MCP[{}] ready: {} tool(s)", cfg.name(), tools.size());
            } catch (Exception e) {
                String reason = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                failures.put(cfg.name(), reason);
                LOG.warn("MCP[{}] failed to start: {}", cfg.name(), reason);
            }
        }
        return out;
    }

    private McpClient newClient(McpServerConfig cfg) {
        return switch (cfg.transport()) {
            case STDIO -> new StdioMcpClient(cfg.name(), cfg.commandLine(), cfg.env());
            case HTTP  -> throw new UnsupportedOperationException(
                    "HTTP MCP transport not implemented yet (server: " + cfg.name() + ")");
        };
    }

    public synchronized List<String> serverNames() { return List.copyOf(clients.keySet()); }
    public synchronized Map<String, Integer> toolCounts() { return Map.copyOf(toolCounts); }
    public synchronized Map<String, String> failures() { return Map.copyOf(failures); }

    /**
     * Hot-start a single new server and return the tools it exposed. Throws on failure
     * (the workbench wants to surface the error to the user, not silently swallow it).
     *
     * <p>Idempotent on the failed-state map: clears any prior failure entry so a successful
     * start "heals" the pool.
     */
    public synchronized List<Tool> startOne(McpServerConfig cfg) {
        if (cfg == null) throw new IllegalArgumentException("cfg required");
        if (clients.containsKey(cfg.name())) {
            throw new IllegalStateException("MCP server already running: " + cfg.name());
        }
        if (cfg.disabled()) {
            throw new IllegalStateException("MCP server is disabled in config: " + cfg.name());
        }
        McpClient client = newClient(cfg);
        try {
            client.start();
            List<McpToolDescriptor> tools = client.listTools();
            List<Tool> wrapped = new ArrayList<>();
            for (McpToolDescriptor td : tools) wrapped.add(new McpClientTool(client, td));
            clients.put(cfg.name(), client);
            toolCounts.put(cfg.name(), tools.size());
            failures.remove(cfg.name());
            LOG.info("MCP[{}] hot-started: {} tool(s)", cfg.name(), tools.size());
            return wrapped;
        } catch (Exception e) {
            // Make sure we don't leak a half-started subprocess on failure.
            try { client.close(); } catch (Exception ignore) {}
            String reason = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            failures.put(cfg.name(), reason);
            throw (e instanceof RuntimeException re) ? re : new RuntimeException(e);
        }
    }

    /**
     * Stop a single server. Returns the tool-name prefix that was used so callers can clean their
     * registry ({@code "<server>__"}). Returns null if the server wasn't running.
     */
    public synchronized String stopOne(String name) {
        McpClient client = clients.remove(name);
        toolCounts.remove(name);
        if (client == null) return null;
        try { client.close(); }
        catch (Exception e) { LOG.warn("MCP[{}] close failed: {}", name, e.toString()); }
        return name + McpClientTool.NAME_SEP;
    }

    /** True if a server with this name is currently running. */
    public synchronized boolean isRunning(String name) {
        return clients.containsKey(name);
    }

    @Override
    public synchronized void close() {
        // Reverse order so dependents shut down before their dependencies (if any).
        List<String> ordered = new ArrayList<>(clients.keySet());
        java.util.Collections.reverse(ordered);
        for (String name : ordered) {
            try { clients.get(name).close(); }
            catch (Exception e) { LOG.warn("MCP[{}] close failed: {}", name, e.toString()); }
        }
        clients.clear();
        toolCounts.clear();
        failures.clear();
    }
}
