package ai.nizo.mcp.client;

import ai.nizo.mcp.McpClient;
import ai.nizo.mcp.McpException;
import ai.nizo.mcp.protocol.McpToolDescriptor;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * MCP client over stdio: spawns a subprocess and pipes JSON-RPC line-by-line over stdin/stdout.
 *
 * <p>Wire format: each JSON-RPC message is a single line of UTF-8 JSON terminated by {@code \n}.
 * (MCP also defines a {@code Content-Length}-framed variant for HTTP; stdio servers in the wild
 * universally use line-delimited JSON, which is what Claude Desktop and OpenClaw both speak.)
 *
 * <p>Concurrency model: a dedicated reader thread parses inbound lines, looks up the request id
 * in the {@link #pending} map, and completes the matching {@link CompletableFuture}. Callers
 * block on the future with a timeout. Writes synchronize on {@link #stdin} so concurrent
 * {@code callTool} invocations interleave cleanly.
 *
 * <p>Stderr is consumed by a separate thread and logged at WARN — left unread, the subprocess's
 * stderr buffer fills up and the process eventually deadlocks.
 */
public final class StdioMcpClient implements McpClient {

    private static final Logger LOG = LoggerFactory.getLogger(StdioMcpClient.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Protocol version we advertise during {@code initialize}. */
    private static final String PROTOCOL_VERSION = "2024-11-05";

    private static final long CALL_TIMEOUT_SEC = 60;
    private static final long INIT_TIMEOUT_SEC = 30;
    private static final long CLOSE_GRACE_SEC = 3;

    private final String serverName;
    private final List<String> command;
    private final Map<String, String> envOverrides;

    private final AtomicLong idSeq = new AtomicLong(1);
    private final ConcurrentHashMap<Long, CompletableFuture<JsonNode>> pending = new ConcurrentHashMap<>();
    private final AtomicBoolean ready = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private Process process;
    private BufferedWriter stdin;
    private BufferedReader stdout;
    private ExecutorService readerPool;

    private List<McpToolDescriptor> toolCache;

    public StdioMcpClient(String serverName, List<String> command, Map<String, String> envOverrides) {
        if (serverName == null || serverName.isBlank()) throw new IllegalArgumentException("serverName required");
        if (command == null || command.isEmpty()) throw new IllegalArgumentException("command required");
        this.serverName = serverName;
        this.command = List.copyOf(command);
        this.envOverrides = envOverrides == null ? Map.of() : Map.copyOf(envOverrides);
    }

    @Override public String serverName() { return serverName; }
    @Override public boolean isReady() { return ready.get() && !closed.get(); }

    @Override
    public synchronized void start() throws McpException {
        if (ready.get() || closed.get()) return;
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            // Inherit parent env; layer overrides on top (so we can pass tokens, etc.).
            pb.environment().putAll(envOverrides);
            // Don't merge stderr into stdout — our wire format is line-delimited JSON on stdout
            // and a stray stderr line would break parsing.
            pb.redirectErrorStream(false);
            process = pb.start();

            stdin = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            stdout = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));

            readerPool = Executors.newFixedThreadPool(2, r -> {
                Thread t = new Thread(r, "mcp-" + serverName + "-reader");
                t.setDaemon(true);
                return t;
            });
            readerPool.submit(this::readStdoutLoop);
            readerPool.submit(this::readStderrLoop);

            initializeHandshake();
            ready.set(true);
            LOG.info("MCP[{}] started: {}", serverName, command);
        } catch (IOException e) {
            cleanup();
            throw new McpException("could not spawn MCP subprocess: " + e.getMessage(), e);
        } catch (McpException e) {
            cleanup();
            throw e;
        }
    }

    private void initializeHandshake() throws McpException {
        ObjectNode params = MAPPER.createObjectNode();
        params.put("protocolVersion", PROTOCOL_VERSION);
        ObjectNode caps = params.putObject("capabilities");
        // We're a tools-only client for now. Resources / prompts can be added later by extending capabilities.
        caps.putObject("tools");
        ObjectNode info = params.putObject("clientInfo");
        info.put("name", "nizo");
        info.put("version", "1.0.0");

        JsonNode result = sendRequest("initialize", params, INIT_TIMEOUT_SEC);
        String serverProto = result.path("protocolVersion").asText("");
        if (!serverProto.isEmpty() && !serverProto.equals(PROTOCOL_VERSION)) {
            LOG.info("MCP[{}] server speaks protocol {}, we requested {}", serverName, serverProto, PROTOCOL_VERSION);
        }

        // Spec requires a notification (no id, no response) after a successful initialize.
        sendNotification("notifications/initialized", MAPPER.createObjectNode());
    }

    @Override
    public synchronized List<McpToolDescriptor> listTools() throws McpException {
        if (!ready.get()) throw new McpException("MCP[" + serverName + "] not started");
        if (toolCache != null) return toolCache;

        JsonNode result = sendRequest("tools/list", null, CALL_TIMEOUT_SEC);
        List<McpToolDescriptor> out = new ArrayList<>();
        for (JsonNode t : result.path("tools")) {
            String n = t.path("name").asText("");
            if (n.isEmpty()) continue;
            String desc = t.path("description").asText("");
            JsonNode schema = t.path("inputSchema");
            String schemaJson = (schema.isMissingNode() || schema.isNull())
                    ? "{\"type\":\"object\",\"properties\":{}}"
                    : schema.toString();
            out.add(new McpToolDescriptor(n, desc, schemaJson));
        }
        toolCache = List.copyOf(out);
        return toolCache;
    }

    @Override
    public JsonNode callTool(String toolName, JsonNode argumentsJson) throws McpException {
        if (!ready.get()) throw new McpException("MCP[" + serverName + "] not ready");
        ObjectNode params = MAPPER.createObjectNode();
        params.put("name", toolName);
        params.set("arguments", argumentsJson == null ? MAPPER.createObjectNode() : argumentsJson);
        return sendRequest("tools/call", params, CALL_TIMEOUT_SEC);
    }

    // ------------------- transport -------------------

    private JsonNode sendRequest(String method, JsonNode params, long timeoutSec) throws McpException {
        long id = idSeq.getAndIncrement();
        ObjectNode req = MAPPER.createObjectNode();
        req.put("jsonrpc", "2.0");
        req.put("id", id);
        req.put("method", method);
        if (params != null) req.set("params", params);

        CompletableFuture<JsonNode> future = new CompletableFuture<>();
        pending.put(id, future);
        try {
            writeLine(MAPPER.writeValueAsString(req));
            JsonNode response = future.get(timeoutSec, TimeUnit.SECONDS);
            JsonNode error = response.get("error");
            if (error != null && !error.isNull()) {
                int code = error.path("code").asInt(0);
                String msg = error.path("message").asText("(no message)");
                throw new McpException(code, "MCP[" + serverName + "] " + method + " error " + code + ": " + msg);
            }
            return response.path("result");
        } catch (TimeoutException te) {
            pending.remove(id);
            throw new McpException("MCP[" + serverName + "] " + method + " timed out after " + timeoutSec + "s", te);
        } catch (McpException e) {
            throw e;
        } catch (Exception e) {
            pending.remove(id);
            throw new McpException("MCP[" + serverName + "] " + method + " failed: " + e.getMessage(), e);
        }
    }

    private void sendNotification(String method, JsonNode params) throws McpException {
        ObjectNode req = MAPPER.createObjectNode();
        req.put("jsonrpc", "2.0");
        req.put("method", method);
        if (params != null) req.set("params", params);
        try {
            writeLine(MAPPER.writeValueAsString(req));
        } catch (IOException e) {
            throw new McpException("MCP[" + serverName + "] notification " + method + " failed", e);
        }
    }

    private void writeLine(String json) throws IOException {
        synchronized (stdin) {
            stdin.write(json);
            stdin.write('\n');
            stdin.flush();
        }
    }

    private void readStdoutLoop() {
        try {
            String line;
            while ((line = stdout.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                JsonNode msg;
                try { msg = MAPPER.readTree(line); }
                catch (Exception e) {
                    LOG.warn("MCP[{}] non-JSON stdout line, ignoring: {}", serverName, abbrev(line));
                    continue;
                }
                JsonNode idNode = msg.get("id");
                if (idNode != null && idNode.isNumber()) {
                    CompletableFuture<JsonNode> f = pending.remove(idNode.asLong());
                    if (f != null) f.complete(msg);
                    else LOG.debug("MCP[{}] orphan response id={}", serverName, idNode.asLong());
                } else {
                    // Notification from server (e.g. log messages). Ignore for v1.
                    LOG.debug("MCP[{}] notification: {}", serverName, msg.path("method").asText());
                }
            }
        } catch (IOException e) {
            if (!closed.get()) LOG.warn("MCP[{}] stdout reader died: {}", serverName, e.toString());
        } finally {
            // Drain any pending callers so they fail fast instead of hanging on the timeout.
            McpException dead = new McpException("MCP[" + serverName + "] subprocess closed stdout");
            pending.values().forEach(f -> f.completeExceptionally(dead));
            pending.clear();
        }
    }

    private void readStderrLoop() {
        try (BufferedReader err = new BufferedReader(
                new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = err.readLine()) != null) {
                if (!line.isBlank()) LOG.warn("MCP[{}] stderr: {}", serverName, abbrev(line));
            }
        } catch (IOException e) {
            if (!closed.get()) LOG.debug("MCP[{}] stderr reader done: {}", serverName, e.toString());
        }
    }

    private static String abbrev(String s) {
        return s.length() <= 240 ? s : s.substring(0, 240) + "…";
    }

    @Override
    public synchronized void close() {
        if (!closed.compareAndSet(false, true)) return;
        ready.set(false);
        cleanup();
    }

    private void cleanup() {
        try { if (stdin != null) stdin.close(); } catch (Exception ignore) {}
        try { if (stdout != null) stdout.close(); } catch (Exception ignore) {}
        if (process != null) {
            process.destroy();
            try {
                if (!process.waitFor(CLOSE_GRACE_SEC, TimeUnit.SECONDS)) process.destroyForcibly();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
        }
        if (readerPool != null) readerPool.shutdownNow();
        LOG.info("MCP[{}] stopped", serverName);
    }
}
