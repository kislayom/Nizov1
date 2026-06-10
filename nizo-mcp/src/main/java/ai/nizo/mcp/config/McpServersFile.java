package ai.nizo.mcp.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads / saves the {@code mcp.json} config file.
 *
 * <p>Supports two top-level shapes:
 * <ul>
 *   <li>Claude-Desktop / OpenClaw style: {@code { "mcpServers": { "name": {...}, ... } }}</li>
 *   <li>Bare dict: {@code { "filesystem": {...}, "github": {...} }}</li>
 * </ul>
 * Either form parses; we always write the {@code mcpServers}-keyed form for portability.
 *
 * <p>Environment-variable expansion: any string value of the form {@code "${VAR}"} is replaced
 * by the value of {@code System.getenv("VAR")} at load time. Lets users keep tokens out of
 * the file ({@code "GITHUB_TOKEN": "${GITHUB_TOKEN}"}).
 */
public final class McpServersFile {

    private static final Logger LOG = LoggerFactory.getLogger(McpServersFile.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Path path;
    private final Map<String, McpServerConfig> servers;

    public McpServersFile(Path path, Map<String, McpServerConfig> servers) {
        this.path = path;
        this.servers = new LinkedHashMap<>(servers == null ? Map.of() : servers);
    }

    public Path path() { return path; }
    public Map<String, McpServerConfig> servers() { return Map.copyOf(servers); }
    public List<McpServerConfig> list() { return List.copyOf(servers.values()); }
    public boolean isEmpty() { return servers.isEmpty(); }

    /** Load from disk, or return an empty file pointer if the path doesn't exist. */
    public static McpServersFile loadOrEmpty(Path path) {
        if (path == null || !Files.exists(path)) return new McpServersFile(path, Map.of());
        try {
            byte[] bytes = Files.readAllBytes(path);
            if (bytes.length == 0) return new McpServersFile(path, Map.of());
            JsonNode root = MAPPER.readTree(bytes);
            JsonNode body = root.has("mcpServers") ? root.get("mcpServers") : root;
            Map<String, McpServerConfig> out = new LinkedHashMap<>();
            for (Iterator<Map.Entry<String, JsonNode>> it = body.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> e = it.next();
                try {
                    out.put(e.getKey(), parseEntry(e.getKey(), e.getValue()));
                } catch (Exception ex) {
                    LOG.warn("MCP config: skipping {} ({})", e.getKey(), ex.getMessage());
                }
            }
            LOG.info("MCP config loaded from {} ({} server{})", path, out.size(), out.size() == 1 ? "" : "s");
            return new McpServersFile(path, out);
        } catch (IOException e) {
            LOG.warn("MCP config: could not read {}: {}", path, e.toString());
            return new McpServersFile(path, Map.of());
        }
    }

    private static McpServerConfig parseEntry(String name, JsonNode n) {
        boolean disabled = n.path("disabled").asBoolean(false);
        // HTTP/SSE transport: presence of "url" wins.
        if (n.has("url")) {
            String url = expand(n.path("url").asText(""));
            String tok = expand(n.path("authToken").asText(null));
            return new McpServerConfig(
                    name, McpServerConfig.Transport.HTTP,
                    null, List.of(), Map.of(), url, tok, disabled);
        }
        // stdio
        String command = n.path("command").asText("");
        if (command.isBlank()) throw new IllegalArgumentException("missing command/url");
        List<String> args = new ArrayList<>();
        for (JsonNode a : n.path("args")) args.add(expand(a.asText("")));
        Map<String, String> env = new LinkedHashMap<>();
        if (n.has("env") && n.get("env").isObject()) {
            for (Iterator<Map.Entry<String, JsonNode>> it = n.get("env").fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> e = it.next();
                env.put(e.getKey(), expand(e.getValue().asText("")));
            }
        }
        return new McpServerConfig(name, McpServerConfig.Transport.STDIO,
                command, args, env, null, null, disabled);
    }

    /** Replace {@code ${VAR}} occurrences with environment values. Unset vars become empty. */
    static String expand(String s) {
        if (s == null) return null;
        if (!s.contains("${")) return s;
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < s.length()) {
            int open = s.indexOf("${", i);
            if (open < 0) { out.append(s, i, s.length()); break; }
            out.append(s, i, open);
            int close = s.indexOf('}', open);
            if (close < 0) { out.append(s, open, s.length()); break; }
            String var = s.substring(open + 2, close);
            String val = System.getenv(var);
            out.append(val == null ? "" : val);
            i = close + 1;
        }
        return out.toString();
    }

    /** Persist the current config back to {@link #path}. Creates parent dirs as needed. */
    public synchronized void save() throws IOException {
        if (path == null) throw new IllegalStateException("no path set");
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        ObjectNode root = MAPPER.createObjectNode();
        ObjectNode list = root.putObject("mcpServers");
        for (var e : servers.entrySet()) {
            ObjectNode n = list.putObject(e.getKey());
            McpServerConfig c = e.getValue();
            if (c.disabled()) n.put("disabled", true);
            if (c.transport() == McpServerConfig.Transport.HTTP) {
                if (c.url() != null) n.put("url", c.url());
                if (c.authToken() != null) n.put("authToken", c.authToken());
            } else {
                n.put("command", c.command());
                if (!c.args().isEmpty()) {
                    var arr = n.putArray("args");
                    for (String a : c.args()) arr.add(a);
                }
                if (!c.env().isEmpty()) {
                    ObjectNode env = n.putObject("env");
                    c.env().forEach(env::put);
                }
            }
        }
        Files.writeString(path, MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(root),
                StandardCharsets.UTF_8);
    }

    /** Add or replace a server entry. */
    public synchronized void put(McpServerConfig cfg) {
        servers.put(cfg.name(), cfg);
    }

    /** Remove a server entry. Returns the removed config, or null if not present. */
    public synchronized McpServerConfig remove(String name) {
        return servers.remove(name);
    }
}
