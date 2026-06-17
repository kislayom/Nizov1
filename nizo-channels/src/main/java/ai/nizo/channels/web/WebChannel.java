package ai.nizo.channels.web;

import ai.nizo.agent.exec.ChatEvent;
import ai.nizo.agent.exec.ChatExecution;
import ai.nizo.agent.exec.ChatExecutor;
import ai.nizo.agent.exec.EventSubscriber;
import ai.nizo.agent.loop.AgentLoop;
import ai.nizo.agent.session.ChatSummary;
import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.agent.AgentEventSink;
import ai.nizo.api.chat.ChatHandler;
import ai.nizo.api.chat.IncomingMessage;
import ai.nizo.api.chat.OutgoingMessage;
import ai.nizo.agent.condense.CondenseConstants;
import ai.nizo.agent.condense.TokenEstimator;
import ai.nizo.api.condense.CondenseMode;
import ai.nizo.api.condense.CondenseRequest;
import ai.nizo.api.condense.CondenseResult;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.tool.Tool;
import ai.nizo.mcp.client.McpClientPool;
import ai.nizo.mcp.client.McpClientTool;
import ai.nizo.mcp.config.McpServerConfig;
import ai.nizo.mcp.config.McpServersFile;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.time.Duration;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

/**
 * Local web workbench for Nizo.
 *
 * <p>Routes:
 * <ul>
 *   <li>{@code GET  /}                                — single-page workbench UI</li>
 *   <li>{@code GET  /api/health}                      — liveness</li>
 *   <li>{@code GET  /api/status}                      — model, endpoint, counts</li>
 *   <li>{@code GET  /api/tools}                       — registered tools (name, description, schema)</li>
 *   <li>{@code GET  /api/skills}                      — discovered filesystem skills</li>
 *   <li>{@code GET  /api/sessions}                    — all conversations, newest first</li>
 *   <li>{@code GET  /api/sessions/{id}/messages}      — full conversation history</li>
 *   <li>{@code DELETE /api/sessions/{id}}             — wipe a conversation</li>
 *   <li>{@code GET  /api/workspace?path=…}            — list a directory under ~/.nizo/workspace</li>
 *   <li>{@code POST /api/chat}                        — JSON in / JSON out (blocking)</li>
 *   <li>{@code POST /api/chat/stream}                 — SSE stream of {@link AgentEvent}s</li>
 * </ul>
 */
public final class WebChannel implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(WebChannel.class);
    private static final String CHANNEL_NAME = "web";

    private final WebConfig config;
    private final ChatHandler handler;        // blocking
    private final AgentLoop streamingLoop;    // optional; if non-null, /api/chat/stream is enabled
    private final WebUiContext ctx;           // optional; if non-null, inspector endpoints are enabled
    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicLong sessionSeq = new AtomicLong();
    private final long maxBodyBytes = BoundedBodyReader.configuredMaxBytes();
    private WebAuth auth;                     // initialized in start()
    private HttpServer server;

    /** Blocking-only mode (no SSE, no inspector endpoints). */
    public WebChannel(WebConfig config, ChatHandler handler) {
        this(config, handler, null, null);
    }

    /** Streaming + inspector. Production. */
    public WebChannel(WebConfig config, ChatHandler handler, AgentLoop streamingLoop, WebUiContext ctx) {
        this.config = config;
        this.handler = handler;
        this.streamingLoop = streamingLoop;
        this.ctx = ctx;
    }

    /** Test-only: inject a fixed token instead of reading {@code ~/.nizo/web-token}. */
    public synchronized void setAuthForTest(WebAuth auth) { this.auth = auth; }

    /** Expose so tests can poke at the auth helper. Returns {@code null} before {@link #start()}. */
    public WebAuth auth() { return auth; }

    public synchronized void start() throws IOException {
        if (server != null) return;
        if (auth == null) auth = WebAuth.loadOrCreate();
        server = HttpServer.create(new InetSocketAddress(config.host(), config.port()), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.createContext("/api/health", this::handleHealth);
        server.createContext("/api/status", this::handleStatus);
        server.createContext("/api/tools", this::handleTools);
        server.createContext("/api/skills", this::handleSkills);
        server.createContext("/api/sessions", this::handleSessions);  // matches /api/sessions and /api/sessions/...
        server.createContext("/api/workspace/file", this::handleWorkspaceFile);
        server.createContext("/api/workspace", this::handleWorkspace);
        server.createContext("/api/chat/stream", this::handleChatStream);
        server.createContext("/api/chat/condense", this::handleChatCondense);
        server.createContext("/api/chat/stats", this::handleChatStats);
        // ─── Server-owned chat execution (post = enqueue, get = subscribe via EventSource) ───
        server.createContext("/api/chat/messages", this::handleChatEnqueue);
        server.createContext("/api/chat/events",   this::handleChatEvents);   // EventSource subscribe
        server.createContext("/api/chat/stop",     this::handleChatStop);
        server.createContext("/api/chat/state",    this::handleChatState);
        server.createContext("/api/usage",         this::handleUsage);          // tool/skill usage telemetry
        server.createContext("/api/voice/transcribe", this::handleVoiceTranscribe); // STT proxy → sidecar
        server.createContext("/api/voice/speak",      this::handleVoiceSpeak);      // TTS proxy → sidecar
        server.createContext("/api/voice/voices",     this::handleVoiceVoices);     // sidecar /voices (speakers + langs)
        server.createContext("/api/voice/health",     this::handleVoiceHealth);     // sidecar /health
        server.createContext("/api/music/compose",    this::handleMusicCompose);    // text-to-music proxy → sidecar
        server.createContext("/api/music/compose-async", this::handleMusicComposeAsync); // queues job, returns jobId
        server.createContext("/api/music/jobs",       this::handleMusicJobs);       // list jobs OR get/delete by id
        server.createContext("/api/stock/history",    this::handleStockHistory);    // GET → list of past completed reports (single-user for now)
        server.createContext("/api/stock/report",     this::handleStockReport);     // GET /api/stock/report/{chatId} → full record; DELETE removes
        server.createContext("/api/stock/rerun-stage", this::handleStockRerunStage); // POST → re-run a single sub-skill (fundamentals, news, …) and patch the master report
        server.createContext("/api/ticker/search",     this::handleTickerSearch);    // GET /api/ticker/search?q=APP → typeahead matches (NSE/BSE/US/ASX)
        server.createContext("/api/notify/vpn-event",  this::handleNotifyVpnEvent);  // POST → log + forward to Telegram if TELEGRAM_BOT_TOKEN + TELEGRAM_NOTIFY_CHAT_ID set
        server.createContext("/api/chat", this::handleChat);
        server.createContext("/api/mcp", this::handleMcp);
        server.createContext("/", this::handleIndex);
        server.start();
        LOG.info("Web UI listening on http://{}:{}{}",
                config.host(), config.port(),
                streamingLoop != null ? " (streaming enabled)" : "");
    }

    @Override
    public synchronized void close() {
        if (server != null) {
            server.stop(0);
            server = null;
            LOG.info("Web UI stopped");
        }
    }

    public String url() {
        return "http://" + config.host() + ":" + config.port() + "/";
    }

    // ============================ handlers =================================

    private void handleHealth(HttpExchange ex) throws IOException {
        respondText(ex, 200, "ok");
    }

    private void handleIndex(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        // Index is public — no token required to *load* the page. But if the caller is on
        // loopback we plant the cookie that lets every subsequent request sail through. The
        // result: a user on the box just hits / and the UI works without any setup. Remote
        // callers (NIZO_WEB_HOST=0.0.0.0) get the page but no cookie — they'll need to paste
        // the token into the Settings dialog.
        if (auth != null && WebAuth.isLoopback(ex)) {
            ex.getResponseHeaders().add("Set-Cookie", WebAuth.setCookieValue(auth.token()));
        }
        // Dev override: if NIZO_WEB_DIR points at a directory with index.html, serve from disk so
        // UI changes hot-reload on a browser refresh (no jar rebuild). Falls back to the bundled
        // classpath resource for production. Path-guarded to the configured dir.
        String webDir = System.getenv("NIZO_WEB_DIR");
        if (webDir != null && !webDir.isBlank()) {
            Path idx = Path.of(webDir).resolve("index.html").toAbsolutePath().normalize();
            if (idx.startsWith(Path.of(webDir).toAbsolutePath().normalize()) && Files.isRegularFile(idx)) {
                byte[] body = Files.readAllBytes(idx);
                ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
                ex.getResponseHeaders().add("Cache-Control", "no-store");
                ex.sendResponseHeaders(200, body.length);
                try (OutputStream os = ex.getResponseBody()) { os.write(body); }
                return;
            }
        }
        try (InputStream is = WebChannel.class.getResourceAsStream("/web/index.html")) {
            if (is == null) { respondText(ex, 500, "index.html not bundled"); return; }
            byte[] body = is.readAllBytes();
            ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
            ex.getResponseHeaders().add("Cache-Control", "no-store");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        }
    }

    private void handleStatus(HttpExchange ex) throws IOException {
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", ctx.modelName());
        body.put("llmEndpoint", ctx.llmEndpoint());
        body.put("home", ctx.home().toString());
        body.put("workspace", ctx.workspaceDir().toString());
        body.put("toolCount", ctx.tools().all().size());
        body.put("skillCount", ctx.skills().get().size());
        body.put("streamingEnabled", streamingLoop != null);
        body.put("now", java.time.Instant.now().toString());
        respondJson(ex, 200, mapper.writeValueAsString(body));
    }

    private void handleTools(HttpExchange ex) throws IOException {
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Tool t : ctx.tools().all()) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("name", t.name());
            r.put("description", t.description());
            // parametersJsonSchema is a string; embed as parsed JSON for clean rendering.
            try { r.put("parameters", mapper.readTree(t.parametersJsonSchema())); }
            catch (Exception e) { r.put("parameters", t.parametersJsonSchema()); }
            rows.add(r);
        }
        respondJson(ex, 200, mapper.writeValueAsString(Map.of("tools", rows)));
    }

    private void handleSkills(HttpExchange ex) throws IOException {
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (var s : ctx.skills().get()) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("name", s.name());
            r.put("description", s.description());
            r.put("whenToUse", s.whenToUse());
            r.put("tags", s.tags());
            r.put("source", s.source() == null ? null : s.source().toString());
            rows.add(r);
        }
        respondJson(ex, 200, mapper.writeValueAsString(Map.of("skills", rows)));
    }

    private void handleSessions(HttpExchange ex) throws IOException {
        // GET /api/sessions returns chat metadata (low-sensitivity but per-user).
        // DELETE removes a session — definitely needs auth.
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        String path = ex.getRequestURI().getPath();
        String method = ex.getRequestMethod();

        if (path.equals("/api/sessions") || path.equals("/api/sessions/")) {
            if (!"GET".equalsIgnoreCase(method)) { respondText(ex, 405, "method not allowed"); return; }
            // ?userId=papa → only that user's chats. Absent → unfiltered (legacy
            // behavior; the web UI always passes its picked identity).
            String userFilter = null;
            String rawQ = ex.getRequestURI().getRawQuery();
            if (rawQ != null) {
                for (String pair : rawQ.split("&")) {
                    int eq = pair.indexOf('=');
                    if (eq > 0 && "userId".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                        String v = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                        if (!v.isBlank()) userFilter = v;
                    }
                }
            }
            List<ChatSummary> chats = ctx.sessions().listChats(200, userFilter);
            List<Map<String, Object>> rows = new ArrayList<>();
            for (ChatSummary s : chats) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("chatId", s.chatId());
                r.put("lastUpdated", s.lastUpdated());
                r.put("messageCount", s.messageCount());
                r.put("lastUserText", s.lastUserText());
                r.put("lastReply", s.lastReply());
                rows.add(r);
            }
            respondJson(ex, 200, mapper.writeValueAsString(Map.of("sessions", rows)));
            return;
        }

        // /api/sessions/{id} or /api/sessions/{id}/messages
        String tail = path.substring("/api/sessions/".length());
        boolean wantsMessages = tail.endsWith("/messages");
        String id = wantsMessages ? tail.substring(0, tail.length() - "/messages".length()) : tail;
        if (id.isEmpty()) { respondJson(ex, 400, "{\"error\":\"missing chatId\"}"); return; }

        if (wantsMessages) {
            if (!"GET".equalsIgnoreCase(method)) { respondText(ex, 405, "method not allowed"); return; }
            List<ChatMessage> msgs = ctx.sessions().recent(id, 500);
            List<Map<String, Object>> rows = new ArrayList<>();
            for (ChatMessage m : msgs) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("role", m.role().name());
                r.put("content", m.content() == null ? "" : m.content());
                if (m.hasImages()) r.put("images", m.images());
                rows.add(r);
            }
            respondJson(ex, 200, mapper.writeValueAsString(Map.of("chatId", id, "messages", rows)));
            return;
        }

        if ("DELETE".equalsIgnoreCase(method)) {
            ctx.sessions().clear(id);
            respondJson(ex, 200, "{\"ok\":true,\"chatId\":\"" + escape(id) + "\"}");
            return;
        }
        respondText(ex, 405, "method not allowed");
    }

    private void handleWorkspace(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }

        String q = ex.getRequestURI().getRawQuery();
        String relPath = ".";
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq > 0 && "path".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                    relPath = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                }
            }
        }
        Path root = ctx.workspaceDir().toAbsolutePath().normalize();
        Path target = root.resolve(relPath).normalize();
        if (!target.startsWith(root)) {
            respondJson(ex, 400, "{\"error\":\"path escapes workspace\"}");
            return;
        }
        if (!Files.exists(target)) {
            respondJson(ex, 200, mapper.writeValueAsString(Map.of("path", relPath, "entries", List.of())));
            return;
        }
        if (!Files.isDirectory(target)) {
            respondJson(ex, 400, "{\"error\":\"not a directory\"}");
            return;
        }
        List<Map<String, Object>> entries = new ArrayList<>();
        try (Stream<Path> s = Files.list(target)) {
            s.sorted((a, b) -> {
                boolean da = Files.isDirectory(a), db = Files.isDirectory(b);
                if (da != db) return da ? -1 : 1;
                return a.getFileName().toString().compareToIgnoreCase(b.getFileName().toString());
            }).forEach(p -> {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("name", p.getFileName().toString());
                r.put("type", Files.isDirectory(p) ? "dir" : "file");
                try { r.put("size", Files.isDirectory(p) ? 0L : Files.size(p)); } catch (Exception e) { r.put("size", 0L); }
                try { r.put("modified", Files.getLastModifiedTime(p).toMillis()); } catch (Exception e) { r.put("modified", 0L); }
                entries.add(r);
            });
        }
        respondJson(ex, 200, mapper.writeValueAsString(Map.of(
                "root", root.toString(),
                "path", relPath,
                "entries", entries)));
    }

    /**
     * Stream a single workspace file. Used by the workspace inspector's "download" button.
     *
     * <p>{@code GET /api/workspace/file?path=relative/path.ext}
     *
     * <p>Path traversal protection: resolves under the configured workspace root and rejects
     * anything that escapes (symlinks, {@code ..}, absolute paths). Streams the bytes via
     * {@link Files#copy} — never loads the whole file into memory.
     */
    private void handleWorkspaceFile(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }

        String relPath = "";
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq > 0 && "path".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                    relPath = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                }
            }
        }
        if (relPath.isBlank()) { respondJson(ex, 400, "{\"error\":\"path required\"}"); return; }

        Path root = ctx.workspaceDir().toAbsolutePath().normalize();
        Path target = root.resolve(relPath).normalize();
        if (!target.startsWith(root)) { respondJson(ex, 400, "{\"error\":\"path escapes workspace\"}"); return; }
        if (!Files.exists(target))    { respondJson(ex, 404, "{\"error\":\"not found\"}"); return; }
        if (!Files.isRegularFile(target)) {
            respondJson(ex, 400, "{\"error\":\"not a regular file\"}");
            return;
        }

        long size = Files.size(target);
        String filename = target.getFileName().toString();
        String contentType = guessContentType(filename);

        ex.getResponseHeaders().add("Content-Type", contentType);
        ex.getResponseHeaders().add("Content-Disposition",
                "attachment; filename=\"" + filename.replace("\"", "") + "\"");
        ex.getResponseHeaders().add("Cache-Control", "no-store");
        ex.sendResponseHeaders(200, size);
        try (OutputStream os = ex.getResponseBody()) {
            Files.copy(target, os);
        }
    }

    private static String guessContentType(String name) {
        int dot = name.lastIndexOf('.');
        String ext = dot < 0 ? "" : name.substring(dot + 1).toLowerCase();
        return switch (ext) {
            case "txt", "md", "log"     -> "text/plain; charset=utf-8";
            case "html", "htm"          -> "text/html; charset=utf-8";
            case "css"                  -> "text/css; charset=utf-8";
            case "js", "mjs"            -> "application/javascript; charset=utf-8";
            case "json"                 -> "application/json; charset=utf-8";
            case "xml"                  -> "application/xml; charset=utf-8";
            case "yaml", "yml"          -> "application/yaml; charset=utf-8";
            case "csv"                  -> "text/csv; charset=utf-8";
            case "png"                  -> "image/png";
            case "jpg", "jpeg"          -> "image/jpeg";
            case "gif"                  -> "image/gif";
            case "webp"                 -> "image/webp";
            case "svg"                  -> "image/svg+xml";
            case "pdf"                  -> "application/pdf";
            case "zip"                  -> "application/zip";
            case "tar"                  -> "application/x-tar";
            case "gz", "tgz"            -> "application/gzip";
            default                     -> "application/octet-stream";
        };
    }

    private void handleChat(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        try {
            IncomingMessage in = parseInbound(ex);
            if (in == null) return;
            OutgoingMessage out = handler.handle(in);
            String body = mapper.writeValueAsString(Map.of(
                    "chatId", in.chatId(),
                    "text", out.text() == null ? "" : out.text()
            ));
            respondJson(ex, 200, body);
        } catch (Exception e) {
            LOG.warn("/api/chat failed", e);
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    private void handleChatStream(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (streamingLoop == null) { respondJson(ex, 501, "{\"error\":\"streaming not enabled\"}"); return; }
        IncomingMessage in;
        try {
            in = parseInbound(ex);
            if (in == null) return;
        } catch (Exception e) {
            respondJson(ex, 400, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
            return;
        }

        // IMPORTANT — do NOT use Content-Type: text/event-stream here.
        //
        // WebKit (Safari) applies SSE-specific timeout + closure heuristics any time it sees
        // text/event-stream, EVEN WHEN the stream is consumed via fetch + ReadableStream
        // instead of EventSource. The result was Safari's network layer closing the stream
        // ~immediately after the first chunk, surfacing in JS as
        //     TypeError: Error in input stream
        //
        // Curl + Chrome were tolerant; Safari was not. Switching to application/x-ndjson
        // (newline-delimited JSON) is semantically what we are sending (line-delimited JSON
        // events) and Safari treats it as a generic byte stream — no early closure.
        // Our JS parser does manual line splitting, so it doesn't care about the MIME type.
        ex.getResponseHeaders().add("Content-Type", "application/x-ndjson; charset=utf-8");
        ex.getResponseHeaders().add("Cache-Control", "no-store");
        ex.getResponseHeaders().add("X-Accel-Buffering", "no");
        // Explicit keep-alive + small immediate payload prevents Safari's URLSession from
        // closing the connection during the gap between headers and first byte.
        ex.getResponseHeaders().add("Connection", "keep-alive");
        ex.sendResponseHeaders(200, 0);

        try (OutputStream os = ex.getResponseBody()) {
            // ─────────────────────────────────────────────────────────────────────
            //  Async SSE writer with bounded queue.
            //
            //  Why: previously sink.emit() did a synchronous os.write() under a lock.
            //  When the receiver (browser via slow SSH tunnel) was slow, that write
            //  blocked. While blocked, parseSseStream upstream couldn't read more
            //  bytes from llama-server. llama-server's send buffer eventually filled,
            //  llama-server's HTTP keep-alive timeout fired, and it CLOSED the
            //  connection mid-stream. JDK then threw IOException("Error in input
            //  stream") and the browser saw a partial response + an error.
            //
            //  Fix: dedicated writer thread with a bounded BlockingQueue. emit()
            //  is non-blocking — events go on the queue and the producer
            //  (parseSseStream) returns immediately. The writer thread drains the
            //  queue at whatever pace the slow downstream allows. llama-server
            //  never sees backpressure.
            //
            //  Bounded queue prevents unbounded memory growth if the browser is
            //  truly dead. If queue fills up we drop oldest events (head) and
            //  log a warning — better degraded UX than OOM.
            // ─────────────────────────────────────────────────────────────────────
            final java.util.concurrent.LinkedBlockingDeque<Map<String, Object>> outbox =
                    new java.util.concurrent.LinkedBlockingDeque<>();
            final int maxQueue = 4096;
            final java.util.concurrent.atomic.AtomicBoolean clientAlive =
                    new java.util.concurrent.atomic.AtomicBoolean(true);

            // The writer thread. One pop, one write. The synchronized was only
            // needed when multiple threads contended on os; here only this thread
            // touches os, so we can drop the lock.
            final Thread writer = new Thread(() -> {
                try {
                    while (clientAlive.get() || !outbox.isEmpty()) {
                        Map<String, Object> ev = outbox.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS);
                        if (ev == null) continue;
                        String type = (String) ev.remove("__type");
                        if ("__heartbeat".equals(type)) {
                            os.write((": keepalive " + Instant.now().toEpochMilli() + "\n\n")
                                    .getBytes(StandardCharsets.UTF_8));
                        } else if ("__raw".equals(type)) {
                            os.write(((String) ev.get("__line")).getBytes(StandardCharsets.UTF_8));
                        } else {
                            sseSend(os, type, ev);
                        }
                        os.flush();
                    }
                } catch (IOException ioe) {
                    // Client gone. Stop trying to write but keep draining producer
                    // so the agent-loop thread doesn't block on emit indefinitely.
                    clientAlive.set(false);
                    outbox.clear();
                    LOG.debug("SSE writer: client gone ({})", ioe.toString());
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }, "sse-writer-" + in.chatId());
            writer.setDaemon(true);
            writer.start();

            // Enqueue helper. Non-blocking: drops the oldest queued event if
            // the queue is full so the most-recent stays. Marker key __type
            // tells the writer which payload shape this is.
            java.util.function.BiConsumer<String, Map<String, Object>> enqueue = (type, payload) -> {
                if (!clientAlive.get()) return;
                Map<String, Object> envelope = new LinkedHashMap<>(payload == null ? Map.of() : payload);
                envelope.put("__type", type);
                while (outbox.size() >= maxQueue) {
                    Map<String, Object> dropped = outbox.pollFirst();
                    if (dropped != null) {
                        LOG.warn("SSE outbox full ({}) — dropping oldest event ({})",
                                maxQueue, dropped.get("__type"));
                    }
                }
                outbox.offer(envelope);
            };

            // Initial Meta
            enqueue.accept("Meta", new LinkedHashMap<>(Map.of("chatId", in.chatId())));

            // Heartbeat: enqueue a comment frame every ~10s. Cheap (just one queue
            // insert per tick) and keeps the SSH tunnel non-idle even if the
            // writer thread is currently blocked on a slow os.write.
            final java.util.concurrent.ScheduledExecutorService heartbeat =
                    java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
                        Thread t = new Thread(r, "sse-heartbeat-" + in.chatId());
                        t.setDaemon(true);
                        return t;
                    });
            heartbeat.scheduleAtFixedRate(() -> {
                if (!clientAlive.get()) return;
                enqueue.accept("__heartbeat", new LinkedHashMap<>());
            }, 10, 10, java.util.concurrent.TimeUnit.SECONDS);

            AgentEventSink sink = event -> {
                if (!clientAlive.get()) return;
                enqueue.accept(eventTypeName(event), new LinkedHashMap<>(eventToMap(event)));
            };

            try { streamingLoop.runStreaming(in, sink); }
            catch (Exception e) {
                LOG.warn("streaming loop failed", e);
                enqueue.accept("Error",
                        new LinkedHashMap<>(Map.of("message", String.valueOf(e.getMessage()))));
            } finally {
                heartbeat.shutdownNow();
            }

            // Final [DONE] sentinel. Use the raw lane so the writer emits exactly
            // the legacy "data: [DONE]\n\n" the JS client expects.
            enqueue.accept("__raw",
                    new LinkedHashMap<>(Map.of("__line", "data: [DONE]\n\n")));

            // Signal writer to stop after the queue drains.
            clientAlive.set(false);
            try {
                writer.join(java.util.concurrent.TimeUnit.SECONDS.toMillis(15));
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // ============================ Chat-executor endpoints ===========================
    //
    // The new model: a chat is server-owned (ChatExecution running in a virtual thread).
    // Browsers POST messages and SUBSCRIBE to events independently. This means closing the
    // tab doesn't stop the chat, and reopening replays everything you missed via the ring
    // buffer. Multiple tabs/devices can subscribe to the same chat at once.
    //
    // It also fixes the Safari fetch+ReadableStream "Error in input stream" bug: events go
    // out via the EventSource API, which Safari handles correctly.
    // =================================================================================

    /**
     * Enqueue a user message into a chat's input queue. Returns immediately with the chatId.
     * Body: {@code {chatId, userId, text, images[]}}. Images are data URIs (same shape as
     * the legacy /api/chat/stream endpoint).
     */
    private void handleChatEnqueue(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null || ctx.chatExecutor() == null) {
            respondJson(ex, 501, "{\"error\":\"chat executor not wired\"}");
            return;
        }
        try {
            IncomingMessage in = parseInbound(ex);
            if (in == null) return;
            ChatExecution exec = ctx.chatExecutor().getOrCreate(in.chatId());
            ChatExecution.EnqueueResult er = exec.enqueue(in);
            switch (er) {
                case QUEUE_FULL -> {
                    respondJson(ex, 429, mapper.writeValueAsString(Map.of(
                            "ok", false,
                            "error", "chat queue is full — wait for the current run to finish, "
                                   + "or call /api/chat/stop to cancel.",
                            "chatId", in.chatId(),
                            "queueDepth", exec.queueDepth())));
                    return;
                }
                case CLOSED -> {
                    respondJson(ex, 410, mapper.writeValueAsString(Map.of(
                            "ok", false,
                            "error", "chat closed",
                            "chatId", in.chatId())));
                    return;
                }
                case ACCEPTED -> { /* fall through */ }
            }
            respondJson(ex, 200, mapper.writeValueAsString(Map.of(
                    "ok", true,
                    "chatId", in.chatId(),
                    "queueDepth", exec.queueDepth(),
                    "status", exec.status().name())));
        } catch (Exception e) {
            LOG.warn("/api/chat/messages failed", e);
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /**
     * EventSource subscribe endpoint. Long-lived response. Browser uses
     * {@code new EventSource('/api/chat/events?chatId=X&since=N')}.
     *
     * <p>Replays buffered events with {@code seq > since} from the ring, then tails live events
     * forever (until the client disconnects). Heartbeat comments every 5s keep the SSH tunnel
     * non-idle.
     */
    private void handleChatEvents(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.chatExecutor() == null) {
            respondJson(ex, 501, "{\"error\":\"chat executor not wired\"}");
            return;
        }
        String chatIdParam = "";
        long since = -1;
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq <= 0) continue;
                String k = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                String v = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                if ("chatId".equals(k)) chatIdParam = v;
                else if ("since".equals(k)) {
                    try { since = Long.parseLong(v); } catch (NumberFormatException ignored) {}
                }
            }
        }
        if (chatIdParam.isBlank()) { respondJson(ex, 400, "{\"error\":\"chatId required\"}"); return; }
        // Capture as effectively-final for lambdas / inner threads below.
        final String chatId = chatIdParam;
        ChatExecution exec = ctx.chatExecutor().getOrCreate(chatId);

        // Standard SSE Content-Type — EventSource REQUIRES text/event-stream. The earlier
        // Safari bug (text/event-stream + fetch+ReadableStream) doesn't apply here because
        // EventSource is the API Safari expects to see this MIME with.
        ex.getResponseHeaders().add("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().add("Cache-Control", "no-store");
        ex.getResponseHeaders().add("X-Accel-Buffering", "no");
        ex.sendResponseHeaders(200, 0);

        // Async write infrastructure — same pattern as before so the producer never blocks
        // on a slow consumer. Subscriber adds events to outbox; writer thread drains.
        final java.util.concurrent.LinkedBlockingDeque<String> outbox = new java.util.concurrent.LinkedBlockingDeque<>();
        final java.util.concurrent.atomic.AtomicBoolean clientAlive = new java.util.concurrent.atomic.AtomicBoolean(true);
        final int maxQueue = 8192;

        try (OutputStream os = ex.getResponseBody()) {
            // Hello frame so the EventSource fires `onopen` immediately and Safari's URLSession
            // sees data before any timeout.
            outbox.offer(": connected " + Instant.now().toEpochMilli() + "\n\n");
            // Always tell the client what they're connected to and where to resume from.
            outbox.offer(formatSseEvent("Meta",
                    Map.of("chatId", chatId, "status", exec.status().name(), "latestSeq", exec.latestSeq())));

            // Replay buffered events newer than `since`.
            for (ChatEvent ce : exec.eventsSince(since)) {
                outbox.offer(formatSseChatEvent(ce));
            }

            Thread writer = new Thread(() -> {
                try {
                    while (clientAlive.get() || !outbox.isEmpty()) {
                        String frame = outbox.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS);
                        if (frame == null) continue;
                        os.write(frame.getBytes(StandardCharsets.UTF_8));
                        os.flush();
                    }
                } catch (IOException ioe) {
                    clientAlive.set(false);
                    outbox.clear();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }, "sse-writer-" + chatId);
            writer.setDaemon(true);
            writer.start();

            // Live subscriber.
            EventSubscriber subscriber = ce -> {
                if (!clientAlive.get()) return;
                while (outbox.size() >= maxQueue) outbox.pollFirst();
                outbox.offer(formatSseChatEvent(ce));
            };
            ChatExecution.Subscription sub = exec.subscribe(subscriber);

            // 5s heartbeat — Safari URLSession is happier with frequent traffic.
            java.util.concurrent.ScheduledExecutorService heartbeat =
                    java.util.concurrent.Executors.newSingleThreadScheduledExecutor(r -> {
                        Thread t = new Thread(r, "sse-hb-" + chatId); t.setDaemon(true); return t;
                    });
            heartbeat.scheduleAtFixedRate(() -> {
                if (!clientAlive.get()) return;
                outbox.offer(": keepalive " + Instant.now().toEpochMilli() + "\n\n");
            }, 5, 5, java.util.concurrent.TimeUnit.SECONDS);

            // Block until the client disconnects (writer thread sets clientAlive=false on IOException).
            try {
                while (clientAlive.get()) {
                    Thread.sleep(200);
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            } finally {
                heartbeat.shutdownNow();
                sub.unsubscribe();
                clientAlive.set(false);
                try { writer.join(java.util.concurrent.TimeUnit.SECONDS.toMillis(5)); }
                catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            }
        }
    }

    /** Format a {@link ChatEvent} as one SSE frame (named event with seq id, JSON body). */
    private String formatSseChatEvent(ChatEvent ce) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("seq", ce.seq());
            body.put("ts", ce.timestampMs());
            body.put("type", eventTypeName(ce.event()));
            body.putAll(eventToMap(ce.event()));
            return "id: " + ce.seq() + "\n"
                    + "event: " + eventTypeName(ce.event()) + "\n"
                    + "data: " + mapper.writeValueAsString(body) + "\n\n";
        } catch (Exception e) {
            return ": event-format-failed " + ce.seq() + "\n\n";
        }
    }

    private String formatSseEvent(String type, Map<String, Object> body) {
        try {
            Map<String, Object> envelope = new LinkedHashMap<>(body == null ? Map.of() : body);
            envelope.put("type", type);
            return "event: " + type + "\n" + "data: " + mapper.writeValueAsString(envelope) + "\n\n";
        } catch (Exception e) {
            return ": event-format-failed\n\n";
        }
    }

    /**
     * Stop the currently-running turn for a chat. Returns whether something was actually stopped.
     * Body: {@code {chatId, dropQueue?: bool}}. If {@code dropQueue} is true, also drops queued
     * messages (otherwise just the active turn is interrupted).
     */
    private void handleChatStop(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.chatExecutor() == null) {
            respondJson(ex, 501, "{\"error\":\"chat executor not wired\"}");
            return;
        }
        try {
            JsonNode req;
            try (InputStream is = ex.getRequestBody()) { req = mapper.readTree(is); }
            String chatId = req.path("chatId").asText("");
            boolean dropQueue = req.path("dropQueue").asBoolean(false);
            if (chatId.isBlank()) { respondJson(ex, 400, "{\"error\":\"chatId required\"}"); return; }
            ChatExecution exec = ctx.chatExecutor().get(chatId);
            if (exec == null) {
                respondJson(ex, 404, "{\"error\":\"no such chat execution\"}");
                return;
            }
            boolean stopped = dropQueue ? exec.stopAndClearQueue() : exec.stopCurrent();
            respondJson(ex, 200, mapper.writeValueAsString(Map.of(
                    "ok", true, "stopped", stopped, "status", exec.status().name())));
        } catch (Exception e) {
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /** Lightweight state probe. {@code GET /api/chat/state?chatId=X}. */
    private void handleChatState(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.chatExecutor() == null) {
            respondJson(ex, 501, "{\"error\":\"chat executor not wired\"}");
            return;
        }
        String chatId = "";
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq > 0 && "chatId".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                    chatId = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                }
            }
        }
        if (chatId.isBlank()) { respondJson(ex, 400, "{\"error\":\"chatId required\"}"); return; }
        ChatExecution exec = ctx.chatExecutor().get(chatId);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("chatId", chatId);
        body.put("known", exec != null);
        if (exec == null) {
            body.put("status", "UNKNOWN");
            body.put("queueDepth", 0);
            body.put("latestSeq", 0);
        } else {
            body.put("status", exec.status().name());
            body.put("queueDepth", exec.queueDepth());
            body.put("latestSeq", exec.latestSeq());
        }
        respondJson(ex, 200, mapper.writeValueAsString(body));
    }

    /**
     * Manual condense entry point.
     *
     * <p>Request: {@code POST /api/chat/condense} with JSON
     * <pre>{ "chatId": "...", "userId": "web-user", "mode": "full"|"partial_from"|"partial_up_to",
     *        "pivotIndex": 12 }</pre>
     * Pivot index is required for partial modes; ignored for full.
     *
     * <p>Bypasses the auto-condense circuit breaker (manual user intent overrides protection)
     * but still increments the failure counter on error so back-to-back failures eventually trip.
     */
    private void handleChatCondense(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.condense() == null) {
            respondJson(ex, 501, "{\"error\":\"condense engine not wired\"}");
            return;
        }
        try {
            JsonNode req;
            try (InputStream is = ex.getRequestBody()) { req = mapper.readTree(is); }
            String chatId = req.path("chatId").asText("");
            String userId = req.path("userId").asText("web-user");
            String modeStr = req.path("mode").asText("full").toLowerCase();
            int pivot = req.path("pivotIndex").asInt(0);
            if (chatId.isBlank()) { respondJson(ex, 400, "{\"error\":\"chatId required\"}"); return; }

            CondenseMode mode = switch (modeStr) {
                case "partial_from"  -> CondenseMode.PARTIAL_FROM;
                case "partial_up_to" -> CondenseMode.PARTIAL_UP_TO;
                default              -> CondenseMode.FULL;
            };
            CondenseRequest cr = new CondenseRequest(chatId, userId, mode, pivot, CondenseRequest.Trigger.MANUAL);
            CondenseResult result = ctx.condense().condense(cr);

            Map<String, Object> body = new LinkedHashMap<>();
            body.put("ok", result.ok());
            body.put("mode", result.mode().name());
            body.put("trigger", result.trigger().name());
            body.put("messagesBefore", result.messagesBefore());
            body.put("messagesAfter", result.messagesAfter());
            body.put("tokensBefore", result.tokensBefore());
            body.put("tokensAfter", result.tokensAfter());
            body.put("durationMs", result.durationMs());
            body.put("reinjectedFiles", result.reinjectedFiles());
            body.put("reinjectedSkills", result.reinjectedSkills());
            if (!result.ok()) body.put("error", result.error() == null ? "" : result.error());
            // Include the formatted summary so the UI can render a "what just got condensed" toast.
            if (result.formattedSummary() != null) body.put("summary", result.formattedSummary());
            respondJson(ex, result.ok() ? 200 : 500, mapper.writeValueAsString(body));
        } catch (Exception e) {
            LOG.warn("/api/chat/condense failed", e);
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /**
     * Per-chat context stats. Tells the UI: how full is the context window, and should the user
     * condense soon? Cheap to compute (single SQL query + heuristic token counts), so we let the
     * UI poll it after every turn.
     *
     * <p>{@code GET /api/chat/stats?chatId=...}
     */
    // ═════════════════════════════════════════════════════════════════════════
    // Voice proxy — forwards to the FastAPI voice sidecar on :7780
    // ═════════════════════════════════════════════════════════════════════════
    private static final String VOICE_BASE = System.getenv().getOrDefault(
            "NIZO_VOICE_URL", "http://127.0.0.1:7780");
    /** HTTP/1.1 explicitly — Java's HttpClient defaults to HTTP/2 with an Upgrade header,
     *  which uvicorn rejects with "Unsupported upgrade request" and never reads the body. */
    private static final java.net.http.HttpClient VOICE_HTTP = java.net.http.HttpClient.newBuilder()
            .connectTimeout(java.time.Duration.ofSeconds(5))
            .version(java.net.http.HttpClient.Version.HTTP_1_1)
            .build();

    /** Sidecar /voices — returns XTTS-v2 speaker list + supported languages. */
    private void handleVoiceVoices(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        try {
            // /voices triggers XTTS to load on first call (60-90s). Generous timeout.
            java.net.http.HttpResponse<String> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/voices"))
                            .timeout(java.time.Duration.ofSeconds(120)).GET().build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            respondJson(ex, r.statusCode(), r.body());
        } catch (Exception e) {
            respondJson(ex, 503, "{\"speakers\":[],\"languages\":[],\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    private void handleVoiceHealth(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        try {
            java.net.http.HttpResponse<String> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/health"))
                            .timeout(java.time.Duration.ofSeconds(3)).GET().build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            respondJson(ex, r.statusCode(), r.body());
        } catch (Exception e) {
            respondJson(ex, 503, "{\"ok\":false,\"error\":\"sidecar unreachable: " + escape(e.getMessage()) + "\"}");
        }
    }

    /** STT — accept an audio upload (multipart), forward to sidecar /transcribe verbatim. */
    private void handleVoiceTranscribe(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        // Read incoming multipart body in full and re-POST it to the sidecar.
        // We don't need to parse it — the sidecar's FastAPI handler will.
        String contentType = ex.getRequestHeaders().getFirst("Content-Type");
        if (contentType == null || !contentType.toLowerCase().contains("multipart/")) {
            respondJson(ex, 400, "{\"error\":\"multipart/form-data required\"}"); return;
        }
        byte[] body;
        try (InputStream is = ex.getRequestBody()) { body = is.readAllBytes(); }
        try {
            java.net.http.HttpResponse<String> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/transcribe"))
                            .timeout(java.time.Duration.ofSeconds(120))
                            .header("Content-Type", contentType)
                            .POST(java.net.http.HttpRequest.BodyPublishers.ofByteArray(body))
                            .build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            respondJson(ex, r.statusCode(), r.body());
        } catch (Exception e) {
            LOG.warn("/api/voice/transcribe proxy failed", e);
            respondJson(ex, 502, "{\"error\":\"sidecar error: " + escape(e.getMessage()) + "\"}");
        }
    }

    /** TTS — accept JSON, forward to sidecar /speak, stream WAV bytes back. */
    private void handleVoiceSpeak(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        byte[] body;
        try (InputStream is = ex.getRequestBody()) { body = is.readAllBytes(); }
        try {
            java.net.http.HttpResponse<byte[]> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/speak"))
                            .timeout(java.time.Duration.ofSeconds(180))
                            .header("Content-Type", "application/json")
                            .POST(java.net.http.HttpRequest.BodyPublishers.ofByteArray(body))
                            .build(),
                    java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            int sc = r.statusCode();
            String ct = r.headers().firstValue("Content-Type").orElse("application/octet-stream");
            byte[] respBody = r.body();
            ex.getResponseHeaders().add("Content-Type", ct);
            ex.sendResponseHeaders(sc, respBody.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(respBody); }
        } catch (Exception e) {
            LOG.warn("/api/voice/speak proxy failed", e);
            respondJson(ex, 502, "{\"error\":\"sidecar error: " + escape(e.getMessage()) + "\"}");
        }
    }

    /** Music gen — accept JSON {prompt, duration_sec?, size?}, forward to sidecar /compose,
     *  stream the WAV bytes back. Long timeout because medium-size MusicGen takes 30-90s
     *  per minute of audio on our hardware. */
    private void handleMusicCompose(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        byte[] body;
        try (InputStream is = ex.getRequestBody()) { body = is.readAllBytes(); }
        try {
            // Timeout must exceed worst-case render time or the proxy 502s while the
            // sidecar is still working (June 2026: a 1-min YuE song took 301.1s against
            // a 5-min timeout — missed by ONE second; the finished mp3 never reached
            // the browser). YuE ≈ 3 min per song-minute + model (re)load; MusicGen
            // large can take a few minutes too.
            boolean isYue = false;
            int nSegments = 2;
            try {
                JsonNode req = mapper.readTree(body);
                isYue = "yue".equalsIgnoreCase(req.path("engine").asText(""));
                nSegments = Math.max(2, Math.min(8, req.path("n_segments").asInt(2)));
            } catch (Exception ignore) {}
            java.time.Duration timeout = isYue
                    ? java.time.Duration.ofMinutes(Math.max(20, nSegments * 3L * 3 / 2 + 8))
                    : java.time.Duration.ofMinutes(10);
            java.net.http.HttpResponse<byte[]> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/compose"))
                            .timeout(timeout)
                            .header("Content-Type", "application/json")
                            .POST(java.net.http.HttpRequest.BodyPublishers.ofByteArray(body))
                            .build(),
                    java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            int sc = r.statusCode();
            String ct = r.headers().firstValue("Content-Type").orElse("application/octet-stream");
            byte[] respBody = r.body();
            ex.getResponseHeaders().add("Content-Type", ct);
            ex.sendResponseHeaders(sc, respBody.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(respBody); }
        } catch (Exception e) {
            LOG.warn("/api/music/compose proxy failed", e);
            respondJson(ex, 502, "{\"error\":\"sidecar error: " + escape(e.getMessage()) + "\"}");
        }
    }

    /** Async music compose — POST kicks off job, returns {jobId, status}. */
    private void handleMusicComposeAsync(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        byte[] body;
        try (InputStream is = ex.getRequestBody()) { body = is.readAllBytes(); }
        try {
            java.net.http.HttpResponse<byte[]> r = VOICE_HTTP.send(
                    java.net.http.HttpRequest.newBuilder(java.net.URI.create(VOICE_BASE + "/compose-async"))
                            .timeout(java.time.Duration.ofSeconds(10))
                            .header("Content-Type", "application/json")
                            .POST(java.net.http.HttpRequest.BodyPublishers.ofByteArray(body))
                            .build(),
                    java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(r.statusCode(), r.body().length);
            try (OutputStream os = ex.getResponseBody()) { os.write(r.body()); }
        } catch (Exception e) {
            LOG.warn("/api/music/compose-async proxy failed", e);
            respondJson(ex, 502, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /** Music jobs — GET /api/music/jobs lists; GET/DELETE /api/music/jobs/{id} for one. */
    private void handleMusicJobs(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        String path = ex.getRequestURI().getPath();
        // Strip /api/music/jobs prefix; leftover is empty (list) or /<id>
        String tail = path.replaceFirst("^/api/music/jobs", "");
        String method = ex.getRequestMethod().toUpperCase();
        try {
            String url = VOICE_BASE + "/jobs" + tail;
            String qs = ex.getRequestURI().getQuery();
            if (qs != null && !qs.isEmpty()) url += "?" + qs;
            var reqBuilder = java.net.http.HttpRequest.newBuilder(java.net.URI.create(url))
                    .timeout(java.time.Duration.ofSeconds(15));
            if (method.equals("DELETE")) reqBuilder.DELETE();
            java.net.http.HttpResponse<byte[]> r = VOICE_HTTP.send(reqBuilder.build(),
                    java.net.http.HttpResponse.BodyHandlers.ofByteArray());
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(r.statusCode(), r.body().length);
            try (OutputStream os = ex.getResponseBody()) { os.write(r.body()); }
        } catch (Exception e) {
            LOG.warn("/api/music/jobs proxy failed", e);
            respondJson(ex, 502, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /**
     * GET /api/ticker/search?q=APP&limit=10 — typeahead for the ticker input on the
     * Stocks tab. Backs the input by Yahoo Finance's free public search endpoint, which
     * covers every major exchange: NASDAQ, NYSE, NSE (.NS), BSE (.BO), ASX (.AX), LSE
     * (.L), Tokyo (.T), etc. Returns each match with {symbol, name, exchange, type}.
     *
     * <p>Yahoo's endpoint: {@code https://query2.finance.yahoo.com/v1/finance/search?q=…}.
     * We don't ship an API key; the endpoint is open and ungated. If Yahoo rate-limits
     * we fall back to an empty result so the input stays usable.
     */
    private void handleTickerSearch(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        String q = "", raw = ex.getRequestURI().getRawQuery();
        int limit = 10;
        if (raw != null) {
            for (String pair : raw.split("&")) {
                int eq = pair.indexOf('=');
                if (eq <= 0) continue;
                String k = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                String v = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                if ("q".equals(k)) q = v.trim();
                else if ("limit".equals(k)) {
                    try { limit = Math.max(1, Math.min(Integer.parseInt(v), 25)); }
                    catch (NumberFormatException ignore) {}
                }
            }
        }
        if (q.isEmpty() || q.length() > 64) { respondJson(ex, 200, "{\"results\":[]}"); return; }

        // Hit Yahoo. They return JSON with a "quotes" array of {symbol, shortname, exchDisp,
        // quoteType, ...}. We re-emit a trimmed shape the UI can consume directly.
        String yahooUrl = "https://query2.finance.yahoo.com/v1/finance/search?q="
                + java.net.URLEncoder.encode(q, java.nio.charset.StandardCharsets.UTF_8)
                + "&quotesCount=" + limit + "&newsCount=0";
        try {
            // Yahoo URL is hardcoded — only the user-supplied query string varies, which is
            // URL-encoded above. SSRF guard not needed here. The 5MB response cap below
            // bounds the body so a misbehaving upstream can't OOM us.
            java.net.URI uri = java.net.URI.create(yahooUrl);
            java.net.http.HttpRequest req = java.net.http.HttpRequest.newBuilder(uri)
                    .timeout(java.time.Duration.ofSeconds(6))
                    .header("User-Agent", "Mozilla/5.0 nizo-ticker-search")
                    .header("Accept", "application/json")
                    .GET().build();
            java.net.http.HttpResponse<String> resp = SHARED_HTTP.send(req, java.net.http.HttpResponse.BodyHandlers.ofString());
            if (resp.body() != null && resp.body().length() > 5 * 1024 * 1024) {
                respondJson(ex, 200, "{\"results\":[]}");
                return;
            }
            int written = 0;
            StringBuilder sb = new StringBuilder("{\"results\":[");
            if (resp.statusCode() / 100 == 2) {
                com.fasterxml.jackson.databind.JsonNode root = MAPPER.readTree(resp.body());
                com.fasterxml.jackson.databind.JsonNode quotes = root.path("quotes");
                for (com.fasterxml.jackson.databind.JsonNode quote : quotes) {
                    String type = quote.path("quoteType").asText("");
                    // Only surface investable types — skip cryptos, futures, mutual funds and
                    // anything else the deterministic pipeline can't price-fetch.
                    if (!("EQUITY".equals(type) || "ETF".equals(type) || "INDEX".equals(type))) continue;
                    String symbol = quote.path("symbol").asText("");
                    if (symbol.isBlank()) continue;
                    String name = quote.path("shortname").asText("");
                    if (name.isBlank()) name = quote.path("longname").asText("");
                    String exchange = quote.path("exchDisp").asText("");
                    if (written > 0) sb.append(',');
                    sb.append("{\"symbol\":\"").append(jsonStr(symbol)).append("\",")
                      .append("\"name\":\"").append(jsonStr(name)).append("\",")
                      .append("\"exchange\":\"").append(jsonStr(exchange)).append("\",")
                      .append("\"type\":\"").append(jsonStr(type)).append("\"}");
                    if (++written >= limit) break;
                }
            }
            // Tier-2 fallback: Yahoo failed or returned no investable results → try Finnhub.
            // Finnhub's free tier (60 req/min, no daily cap) covers the same exchanges as Yahoo
            // (NSE, BSE, ASX, LSE, etc.) but only fires when FINNHUB_TOKEN env var is set, so
            // deployments without one keep working with Yahoo-only behaviour.
            if (written == 0) {
                int finnhubWritten = appendFinnhubTickerMatches(sb, q, limit);
                written += finnhubWritten;
            }
            sb.append("]}");
            ex.getResponseHeaders().add("Cache-Control", "public, max-age=300");
            respondJson(ex, 200, sb.toString());
        } catch (Exception e) {
            LOG.warn("ticker search failed for q={}: {}", q, e.toString());
            respondJson(ex, 200, "{\"results\":[]}");
        }
    }

    /** Fall back to Finnhub /search when Yahoo's ticker endpoint returns 429 / empty.
     *  Returns the count appended to {@code sb}; 0 when Finnhub disabled / no matches. */
    private int appendFinnhubTickerMatches(StringBuilder sb, String q, int limit) {
        String token = System.getenv("FINNHUB_TOKEN");
        if (token == null || token.isBlank()) return 0;
        String url = "https://finnhub.io/api/v1/search?q="
                + java.net.URLEncoder.encode(q, java.nio.charset.StandardCharsets.UTF_8)
                + "&token=" + java.net.URLEncoder.encode(token.trim(), java.nio.charset.StandardCharsets.UTF_8);
        try {
            java.net.http.HttpRequest req = java.net.http.HttpRequest.newBuilder(java.net.URI.create(url))
                    .timeout(java.time.Duration.ofSeconds(6))
                    .header("User-Agent", "nizo-ticker-search")
                    .header("Accept", "application/json")
                    .GET().build();
            java.net.http.HttpResponse<String> resp = SHARED_HTTP.send(req, java.net.http.HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() / 100 != 2) {
                LOG.warn("Finnhub ticker search HTTP {} for q={}", resp.statusCode(), q);
                return 0;
            }
            com.fasterxml.jackson.databind.JsonNode root = MAPPER.readTree(resp.body());
            com.fasterxml.jackson.databind.JsonNode results = root.path("result");
            if (!results.isArray()) return 0;
            int written = 0;
            for (com.fasterxml.jackson.databind.JsonNode r : results) {
                if (written >= limit) break;
                String symbol = r.path("symbol").asText("");
                String displaySymbol = r.path("displaySymbol").asText(symbol);
                String desc = r.path("description").asText("");
                String type = r.path("type").asText("");
                if (symbol.isBlank() && displaySymbol.isBlank()) continue;
                // Finnhub types like "Common Stock" / "ETF" / "Index" — normalize to UI's enum.
                String normalizedType = "EQUITY";
                if (type.toLowerCase().contains("etf")) normalizedType = "ETF";
                else if (type.toLowerCase().contains("index")) normalizedType = "INDEX";
                else if (!type.toLowerCase().contains("stock") && !type.isEmpty()) continue; // skip warrants etc.
                // Exchange isn't directly in /search — symbol suffix gives a hint.
                String exchange = exchangeFromSymbol(displaySymbol.isBlank() ? symbol : displaySymbol);
                if (sb.charAt(sb.length() - 1) == '[') {
                    // first entry
                } else {
                    sb.append(',');
                }
                sb.append("{\"symbol\":\"").append(jsonStr(displaySymbol.isBlank() ? symbol : displaySymbol)).append("\",")
                  .append("\"name\":\"").append(jsonStr(desc)).append("\",")
                  .append("\"exchange\":\"").append(jsonStr(exchange)).append("\",")
                  .append("\"type\":\"").append(jsonStr(normalizedType)).append("\"}");
                written++;
            }
            return written;
        } catch (Exception e) {
            LOG.warn("Finnhub ticker search failed for q={}: {}", q, e.toString());
            return 0;
        }
    }

    /** Map a Yahoo-style symbol suffix to a display exchange name. */
    private static String exchangeFromSymbol(String s) {
        if (s == null) return "";
        int dot = s.lastIndexOf('.');
        if (dot < 0) return "US";
        String suffix = s.substring(dot + 1).toUpperCase();
        return switch (suffix) {
            case "NS" -> "NSE";
            case "BO" -> "BSE";
            case "AX" -> "ASX";
            case "L"  -> "London";
            case "T"  -> "Tokyo";
            case "HK" -> "HKEX";
            case "TO" -> "Toronto";
            case "PA" -> "Paris";
            case "F"  -> "Frankfurt";
            default   -> suffix;
        };
    }

    private static final java.net.http.HttpClient SHARED_HTTP =
            java.net.http.HttpClient.newBuilder()
                    .connectTimeout(java.time.Duration.ofSeconds(5))
                    .followRedirects(java.net.http.HttpClient.Redirect.NORMAL)
                    .build();
    private static final com.fasterxml.jackson.databind.ObjectMapper MAPPER = new com.fasterxml.jackson.databind.ObjectMapper();

    /** GET /api/stock/history?limit=50 — list completed stock reports for the
     *  current user (single-user for now; future: ?userId=...). Each row has
     *  the metadata + a snippet of the report so the Library can render
     *  without a second round-trip. */
    private void handleStockHistory(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.stockReports() == null) {
            respondJson(ex, 200, "{\"reports\":[],\"note\":\"stock-reports store not wired\"}");
            return;
        }
        // Parse ?limit=&userId= without a fancy helper.
        int limit = 50;
        String userId = ai.nizo.agent.cache.StockReportStore.DEFAULT_USER;
        String raw = ex.getRequestURI().getRawQuery();
        if (raw != null) {
            for (String pair : raw.split("&")) {
                int eq = pair.indexOf('=');
                if (eq <= 0) continue;
                String k = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                String v = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                if ("limit".equals(k)) {
                    try { limit = Math.max(1, Math.min(Integer.parseInt(v), 200)); }
                    catch (NumberFormatException ignore) {}
                } else if ("userId".equals(k) && !v.isBlank()) {
                    userId = v;
                }
            }
        }
        List<ai.nizo.agent.cache.StockReportStore.Report> reports = ctx.stockReports().recent(userId, limit);
        StringBuilder sb = new StringBuilder("{\"reports\":[");
        for (int i = 0; i < reports.size(); i++) {
            if (i > 0) sb.append(',');
            ai.nizo.agent.cache.StockReportStore.Report r = reports.get(i);
            String full = r.finalText() == null ? "" : r.finalText();
            String snippet = extractStockSnippet(full);
            String verdict = extractStockVerdict(full);
            sb.append("{")
              .append("\"chatId\":\"").append(jsonStr(r.chatId())).append("\",")
              .append("\"ticker\":\"").append(jsonStr(r.ticker())).append("\",")
              .append("\"createdAt\":").append(r.createdAt()).append(',')
              .append("\"channel\":\"").append(jsonStr(r.channel())).append("\",")
              .append("\"iters\":").append(r.iters()).append(',')
              .append("\"tools\":").append(r.tools()).append(',')
              .append("\"elapsedMs\":").append(r.elapsedMs()).append(',')
              .append("\"prompt\":\"").append(jsonStr(r.prompt())).append("\",")
              .append("\"snippet\":\"").append(jsonStr(snippet)).append("\",")
              .append("\"verdict\":\"").append(jsonStr(verdict)).append("\",")
              .append("\"length\":").append(full.length())
              .append("}");
        }
        sb.append("]}");
        respondJson(ex, 200, sb.toString());
    }

    /** GET /api/stock/report/{chatId} — full row including finalText.
     *  DELETE /api/stock/report/{chatId} — removes from cache. */
    private void handleStockReport(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null || ctx.stockReports() == null) {
            respondJson(ex, 503, "{\"error\":\"stock-reports store not wired\"}");
            return;
        }
        String path = ex.getRequestURI().getPath();    // e.g. /api/stock/report/stock-amzn-12345
        String prefix = "/api/stock/report/";
        if (!path.startsWith(prefix) || path.length() <= prefix.length()) {
            respondJson(ex, 400, "{\"error\":\"missing chatId in path\"}");
            return;
        }
        String chatId = path.substring(prefix.length());
        // strip any trailing /
        if (chatId.endsWith("/")) chatId = chatId.substring(0, chatId.length() - 1);

        if ("DELETE".equalsIgnoreCase(ex.getRequestMethod())) {
            boolean ok = ctx.stockReports().delete(chatId);
            respondJson(ex, ok ? 200 : 404, "{\"deleted\":" + ok + "}");
            return;
        }
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        java.util.Optional<ai.nizo.agent.cache.StockReportStore.Report> opt = ctx.stockReports().get(chatId);
        if (opt.isEmpty()) { respondJson(ex, 404, "{\"error\":\"not found\"}"); return; }
        ai.nizo.agent.cache.StockReportStore.Report r = opt.get();
        String json = "{"
                + "\"chatId\":\"" + jsonStr(r.chatId()) + "\","
                + "\"ticker\":\"" + jsonStr(r.ticker()) + "\","
                + "\"createdAt\":" + r.createdAt() + ","
                + "\"channel\":\"" + jsonStr(r.channel()) + "\","
                + "\"iters\":" + r.iters() + ","
                + "\"tools\":" + r.tools() + ","
                + "\"elapsedMs\":" + r.elapsedMs() + ","
                + "\"stopReason\":\"" + jsonStr(r.stopReason()) + "\","
                + "\"prompt\":\"" + jsonStr(r.prompt()) + "\","
                + "\"finalText\":\"" + jsonStr(r.finalText()) + "\""
                + "}";
        respondJson(ex, 200, json);
    }

    /**
     * POST /api/stock/rerun-stage
     * Body: {"chatId": "stock-aapl-1234", "stage": "fundamentals_analyst"}
     *
     * <p>Re-runs ONE sub-agent (skill_stock_<stage>) for the same ticker, then patches the
     * master report's section in-place. The new section is identified by the heading prefix
     * defined in the orchestrator SKILL.md ("### 3. Fundamentals", etc.). Returns the patched
     * full report so the front-end can re-render.
     *
     * <p>This is the "user can rerun a single stage and see the report update" feature.
     * Cache stays useful for the unchanged sections; only the requested stage is re-fetched.
     */

    /** POST /api/notify/vpn-event — VPN clients (papa, brother, etc.) ping this when
     *  their tunnel comes up or goes down. We always log; if {@code TELEGRAM_BOT_TOKEN}
     *  + {@code TELEGRAM_NOTIFY_CHAT_ID} env vars are set, we also forward to Telegram
     *  so kislay gets a message on his phone.
     *
     *  <p>Body: {@code {"user":"brother","event":"up","ip":"10.10.0.5","host":"BrosBook"}}.
     *  No auth — the endpoint is reachable only from the VPN itself (192.168.4.0/22),
     *  so being on the tunnel IS the credential. */
    private void handleNotifyVpnEvent(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            respondText(ex, 405, "method not allowed"); return;
        }
        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        // Parse the small JSON ourselves — keep this dependency-free.
        String user  = jsonField(body, "user");
        String event = jsonField(body, "event");
        String ip    = jsonField(body, "ip");
        String host  = jsonField(body, "host");
        String fromAddr = ex.getRemoteAddress() == null ? "?" : ex.getRemoteAddress().toString();
        LOG.info("vpn-event: user={} event={} ip={} host={} from={}",
                user, event, ip, host, fromAddr);

        String tgToken  = System.getenv("TELEGRAM_BOT_TOKEN");
        String tgChatId = System.getenv("TELEGRAM_NOTIFY_CHAT_ID");
        boolean sent = false;
        if (tgToken != null && !tgToken.isBlank() && tgChatId != null && !tgChatId.isBlank()) {
            try {
                String text = "🔐 *VPN " + ("up".equalsIgnoreCase(event) ? "connected" : "disconnected")
                            + "*\n• user: `" + nz(user)  + "`"
                            + "\n• ip: `"   + nz(ip)    + "`"
                            + "\n• host: `" + nz(host)  + "`";
                String tgBody = "{\"chat_id\":\"" + jsonStr(tgChatId) + "\","
                              + "\"text\":\""    + jsonStr(text)    + "\","
                              + "\"parse_mode\":\"Markdown\"}";
                java.net.http.HttpClient http = java.net.http.HttpClient.newBuilder()
                        .connectTimeout(Duration.ofSeconds(5)).build();
                java.net.http.HttpResponse<String> resp = http.send(
                        java.net.http.HttpRequest.newBuilder(
                                URI.create("https://api.telegram.org/bot" + tgToken + "/sendMessage"))
                                .timeout(Duration.ofSeconds(10))
                                .header("Content-Type", "application/json")
                                .POST(java.net.http.HttpRequest.BodyPublishers.ofString(tgBody))
                                .build(),
                        java.net.http.HttpResponse.BodyHandlers.ofString());
                sent = resp.statusCode() / 100 == 2;
                if (!sent) LOG.warn("telegram notify failed: status={} body={}",
                        resp.statusCode(), resp.body().substring(0, Math.min(resp.body().length(), 200)));
            } catch (Exception e) {
                LOG.warn("telegram notify exception: {}", e.toString());
            }
        }
        respondJson(ex, 200, "{\"ok\":true,\"telegram\":" + sent + "}");
    }

    /** Minimal JSON-string-field extractor. Body is small + caller-controlled; we don't
     *  pull in jackson for one POST endpoint. Returns "" if absent. */
    private static String jsonField(String json, String key) {
        if (json == null || json.isBlank()) return "";
        java.util.regex.Matcher m = java.util.regex.Pattern.compile(
                "\"" + java.util.regex.Pattern.quote(key) + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\""
        ).matcher(json);
        if (!m.find()) return "";
        return m.group(1).replace("\\\"", "\"").replace("\\\\", "\\");
    }

    private static String nz(String s) { return (s == null || s.isBlank()) ? "?" : s; }

    private void handleStockRerunStage(HttpExchange ex) throws IOException {
        if (rejectIfUnauthenticated(ex)) return;
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.stockReports() == null || ctx.tools() == null) {
            respondJson(ex, 503, "{\"error\":\"stocks/rerun not wired\"}");
            return;
        }
        try {
            JsonNode body = readBoundedJson(ex);
            if (body == null) return; // 413 already written
            String chatId = body.path("chatId").asText("").trim();
            String stage = body.path("stage").asText("").trim().toLowerCase();
            if (chatId.isEmpty() || stage.isEmpty()) {
                respondJson(ex, 400, "{\"error\":\"chatId and stage required\"}");
                return;
            }
            // Map UI-friendly stage names to skill_stock_* tool names. Accepts both forms.
            String toolName = stage.startsWith("skill_") ? stage : ("skill_stock_" + stage);
            java.util.Optional<ai.nizo.api.tool.Tool> toolOpt = ctx.tools().byName(toolName);
            if (toolOpt.isEmpty()) {
                respondJson(ex, 404, "{\"error\":\"unknown stage tool: " + escape(toolName) + "\"}");
                return;
            }
            java.util.Optional<ai.nizo.agent.cache.StockReportStore.Report> reportOpt = ctx.stockReports().get(chatId);
            if (reportOpt.isEmpty()) {
                respondJson(ex, 404, "{\"error\":\"no cached report for chatId\"}");
                return;
            }
            ai.nizo.agent.cache.StockReportStore.Report report = reportOpt.get();
            String ticker = report.ticker();

            // Run the sub-skill in-line. SubAgentSkillTool returns ToolResult.ok with the
            // section markdown. We don't pipe its inner ToolCallStart events back to a sink
            // — this is a one-shot synchronous request — so the UI shows a spinner and
            // updates when we respond.
            String args = "{\"input\":\"" + escape(ticker) + " — rerun ONLY this stage's section. "
                    + "Today's date is " + java.time.LocalDate.now() + ".\"}";
            ai.nizo.api.tool.ToolResult result;
            try {
                result = toolOpt.get().execute(args);
            } catch (Exception e) {
                respondJson(ex, 500, "{\"error\":\"sub-agent threw: " + escape(e.getMessage()) + "\"}");
                return;
            }
            if (!result.ok()) {
                respondJson(ex, 502, "{\"error\":\"sub-agent failed: " + escape(result.content()) + "\"}");
                return;
            }
            String newSection = result.content();

            // Patch the master report. Heuristic: the orchestrator SKILL.md numbers each
            // section ("### 3. Fundamentals + Buffett scorecard", etc.). We find the heading
            // matching the stage and replace its content up to the next "### " heading.
            String patched = patchMasterReportSection(report.finalText(), stage, newSection);
            // Save the patched version back to the store so reload keeps the new section.
            ai.nizo.agent.cache.StockReportStore.Report updated =
                    new ai.nizo.agent.cache.StockReportStore.Report(
                            report.chatId(), report.userId(), report.ticker(),
                            System.currentTimeMillis(),
                            patched, report.prompt(), report.channel(),
                            report.iters(), report.tools(), report.elapsedMs(),
                            report.stopReason());
            try { ctx.stockReports().save(updated); } catch (Exception ignored) {}

            respondJson(ex, 200, "{"
                    + "\"ok\":true,"
                    + "\"chatId\":\"" + jsonStr(chatId) + "\","
                    + "\"stage\":\"" + jsonStr(stage) + "\","
                    + "\"newSection\":\"" + jsonStr(newSection) + "\","
                    + "\"finalText\":\"" + jsonStr(patched) + "\""
                    + "}");
        } catch (Exception e) {
            LOG.warn("rerun-stage failed", e);
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    /**
     * Replace one section of the master report with new content. Sections are matched by the
     * stage name appearing in a {@code ###} heading (case-insensitive). If no matching
     * heading exists, the new section is appended at the end.
     */
    private static String patchMasterReportSection(String fullText, String stage, String newSection) {
        if (fullText == null) fullText = "";
        // Map stage → heading keywords we look for. Order matters when multiple words could
        // collide (e.g. "analyst estimates" vs "fundamentals analyst") — match the most
        // specific stage first.
        String headingHint = switch (stage) {
            case "fundamentals_analyst",  "fundamentals" -> "fundamental";
            case "analyst_estimates",     "estimates"    -> "analyst estimates";
            case "news_analyst",          "news"         -> "news";
            case "sentiment_analyst",     "sentiment"    -> "sentiment";
            case "technical_analyst",     "technical"    -> "technical";
            case "bull_researcher",       "bull"         -> "bull";
            case "bear_researcher",       "bear"         -> "bear";
            case "trader",                "verdict"      -> "verdict";
            default -> stage;
        };
        String[] lines = fullText.split("\n", -1);
        int startIdx = -1, endIdx = lines.length;
        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];
            if (line.startsWith("### ") && line.toLowerCase().contains(headingHint.toLowerCase())) {
                startIdx = i;
                for (int j = i + 1; j < lines.length; j++) {
                    if (lines[j].startsWith("### ")) { endIdx = j; break; }
                }
                break;
            }
        }
        if (startIdx == -1) {
            // No matching section — append to the end with a clear divider.
            return fullText.trim() + "\n\n---\n\n### Rerun: " + stage + "\n\n" + newSection.trim() + "\n";
        }
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < startIdx; i++) { out.append(lines[i]).append('\n'); }
        out.append(lines[startIdx]).append('\n')           // keep the heading line
                .append("\n_(updated " + java.time.LocalDate.now() + ")_\n\n")
                .append(newSection.trim()).append("\n\n");
        for (int i = endIdx; i < lines.length; i++) { out.append(lines[i]).append('\n'); }
        return out.toString();
    }

    /** Tool/skill usage telemetry. Returns aggregate per-name stats + a recent invocation
     *  list (most recent first) tagged with chatId so the UI can deep-link. */
    private void handleUsage(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }
        if (ctx == null || ctx.usage() == null) {
            respondJson(ex, 200, "{\"stats\":{},\"recent\":[],\"note\":\"usage tracker not wired\"}");
            return;
        }
        ai.nizo.tools.registry.UsageTracker u = ctx.usage();

        // Optional ?limit=<n> for the recent list (default 100, max 500)
        int limit = 100;
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq > 0 && "limit".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                    try { limit = Math.max(1, Math.min(500,
                            Integer.parseInt(URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8))));
                    } catch (NumberFormatException ignored) {}
                }
            }
        }

        Map<String, Object> stats = new LinkedHashMap<>();
        for (Map.Entry<String, ai.nizo.tools.registry.UsageTracker.Stats> e : u.snapshot().entrySet()) {
            ai.nizo.tools.registry.UsageTracker.Stats s = e.getValue();
            stats.put(e.getKey(), Map.of(
                    "count", s.count(),
                    "errors", s.errors(),
                    "lastUsed", s.lastUsed(),
                    "avgDurationMs", s.avgDurationMs()));
        }
        List<Map<String, Object>> recent = new ArrayList<>();
        for (ai.nizo.tools.registry.UsageTracker.Usage rec : u.recent(limit)) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("toolName", rec.toolName());
            m.put("chatId", rec.chatId() == null ? "" : rec.chatId());
            m.put("timestamp", rec.timestamp());
            m.put("ok", rec.ok());
            m.put("durationMs", rec.durationMs());
            m.put("argsPreview", rec.argsPreview() == null ? "" : rec.argsPreview());
            recent.add(m);
        }
        Map<String, Object> resp = Map.of("stats", stats, "recent", recent);
        respondJson(ex, 200, mapper.writeValueAsString(resp));
    }

    private void handleChatStats(HttpExchange ex) throws IOException {
        if (ctx == null) { respondJson(ex, 501, "{\"error\":\"context not wired\"}"); return; }
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) { respondText(ex, 405, "method not allowed"); return; }

        String chatId = "";
        String q = ex.getRequestURI().getRawQuery();
        if (q != null) {
            for (String pair : q.split("&")) {
                int eq = pair.indexOf('=');
                if (eq > 0 && "chatId".equals(URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8))) {
                    chatId = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                }
            }
        }

        int contextWindow = CondenseConstants.effectiveContextWindow();
        int condenseThreshold = CondenseConstants.autoCondenseThreshold();
        int tokens = 0;
        int messageCount = 0;
        if (!chatId.isBlank()) {
            List<ChatMessage> msgs = ctx.sessions().recent(chatId, 10_000);
            messageCount = msgs.size();
            tokens = TokenEstimator.estimateMessages(msgs);
            // Add the system prompt + tool schemas — that's what the LLM actually sees per turn.
            tokens += TokenEstimator.estimateTools(ctx.tools().toolDefs());
        }

        // Recommendation: thresholds are relative to the auto-condense trigger so the UI hint
        // matches the engine's behavior. "consider" → 60%, "soon" → 80%, "now" → ≥100%.
        double thresholdPct = condenseThreshold > 0 ? (tokens * 100.0 / condenseThreshold) : 0;
        String rec;
        if (thresholdPct >= 100)      rec = "now";
        else if (thresholdPct >= 80)  rec = "soon";
        else if (thresholdPct >= 60)  rec = "consider";
        else                          rec = "ok";

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("chatId", chatId);
        body.put("messageCount", messageCount);
        body.put("tokens", tokens);
        body.put("contextWindow", contextWindow);
        body.put("condenseThreshold", condenseThreshold);
        body.put("usagePct", contextWindow > 0 ? Math.round(tokens * 1000.0 / contextWindow) / 10.0 : 0.0);
        body.put("thresholdPct", Math.round(thresholdPct * 10.0) / 10.0);
        body.put("recommendation", rec);
        body.put("model", ctx.modelName());
        respondJson(ex, 200, mapper.writeValueAsString(body));
    }

    // ============================ MCP =====================================

    /**
     * MCP routes (the workbench uses these to surface and onboard servers):
     * <pre>
     *   GET    /api/mcp                     — list servers + tool counts + status
     *   POST   /api/mcp                     — add + hot-start a stdio server
     *   DELETE /api/mcp/{name}              — stop subprocess + drop tools + persist removal
     *   POST   /api/mcp/{name}/restart      — hot-restart a flaky server
     * </pre>
     *
     * <p>Hot-start writes to {@code ~/.nizo/mcp.json} and registers tools live — no agent
     * restart required. The new tools are visible to the agent on its next turn.
     */
    private void handleMcp(HttpExchange ex) throws IOException {
        // Highest-risk endpoint — POST spawns subprocesses with user-supplied command/args/env.
        // Token + Origin gate enforced before any body parsing.
        if (rejectIfUnauthenticated(ex)) return;
        if (ctx == null || ctx.mcpPool() == null || ctx.mcpConfig() == null) {
            respondJson(ex, 501, "{\"error\":\"mcp not wired\"}");
            return;
        }
        String path = ex.getRequestURI().getPath();
        String method = ex.getRequestMethod().toUpperCase();

        try {
            if (path.equals("/api/mcp") || path.equals("/api/mcp/")) {
                if ("GET".equals(method))  { mcpList(ex); return; }
                if ("POST".equals(method)) { mcpAdd(ex);  return; }
                respondText(ex, 405, "method not allowed");
                return;
            }
            String tail = path.substring("/api/mcp/".length());
            // /{name} or /{name}/restart
            int slash = tail.indexOf('/');
            String name = slash < 0 ? tail : tail.substring(0, slash);
            String sub  = slash < 0 ? "" : tail.substring(slash + 1);
            if (name.isEmpty()) { respondJson(ex, 400, "{\"error\":\"missing name\"}"); return; }

            if (sub.equals("restart") && "POST".equals(method)) { mcpRestart(ex, name); return; }
            if (sub.isEmpty() && "DELETE".equals(method))       { mcpRemove(ex, name);  return; }
            respondText(ex, 405, "method not allowed");
        } catch (Exception e) {
            LOG.warn("/api/mcp failed: {}", e.toString(), e);
            respondJson(ex, 500, "{\"error\":\"" + escape(e.getMessage()) + "\"}");
        }
    }

    private void mcpList(HttpExchange ex) throws IOException {
        var pool = ctx.mcpPool();
        var cfg = ctx.mcpConfig();
        var counts = pool.toolCounts();
        var failures = pool.failures();

        // Build a per-server map of tools registered locally (filtered from the live registry).
        Map<String, List<Map<String, Object>>> toolsByServer = new LinkedHashMap<>();
        for (Tool t : ctx.tools().all()) {
            if (t instanceof McpClientTool m) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("localName", m.name());
                entry.put("remoteName", m.remoteName());
                entry.put("description", m.description());
                toolsByServer.computeIfAbsent(m.serverName(), k -> new ArrayList<>()).add(entry);
            }
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        for (var s : cfg.list()) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("name", s.name());
            r.put("transport", s.transport().name());
            r.put("disabled", s.disabled());
            if (s.transport() == McpServerConfig.Transport.STDIO) {
                r.put("command", s.command());
                r.put("args", s.args());
                r.put("env", s.env().keySet());      // names only — never leak token VALUES via API
            } else {
                r.put("url", s.url());
            }
            String status;
            if (s.disabled())                     status = "disabled";
            else if (counts.containsKey(s.name())) status = "ok";
            else if (failures.containsKey(s.name())) status = "failed";
            else                                   status = "unknown";
            r.put("status", status);
            r.put("toolCount", counts.getOrDefault(s.name(), 0));
            if (failures.containsKey(s.name())) r.put("error", failures.get(s.name()));
            r.put("tools", toolsByServer.getOrDefault(s.name(), List.of()));
            rows.add(r);
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("configPath", cfg.path() == null ? null : cfg.path().toString());
        body.put("servers", rows);
        body.put("totalMcpTools", toolsByServer.values().stream().mapToInt(List::size).sum());
        respondJson(ex, 200, mapper.writeValueAsString(body));
    }

    /**
     * Add a stdio MCP server and hot-start it. Body: {@code {name, command, args[], env{}}}.
     * On success the server's tools are registered immediately and the agent picks them up
     * on its next turn.
     */
    private void mcpAdd(HttpExchange ex) throws IOException {
        JsonNode req;
        try (InputStream is = ex.getRequestBody()) { req = mapper.readTree(is); }
        String name = req.path("name").asText("").trim();
        String command = req.path("command").asText("").trim();
        if (name.isEmpty() || command.isEmpty()) {
            respondJson(ex, 400, "{\"error\":\"name and command required\"}");
            return;
        }
        if (ctx.mcpPool().isRunning(name) || ctx.mcpConfig().servers().containsKey(name)) {
            respondJson(ex, 409, "{\"error\":\"server already exists: " + escape(name) + "\"}");
            return;
        }
        List<String> args = new ArrayList<>();
        for (JsonNode a : req.path("args")) if (a.isTextual()) args.add(a.asText());
        Map<String, String> env = new LinkedHashMap<>();
        if (req.path("env").isObject()) {
            req.path("env").fields().forEachRemaining(e -> {
                if (e.getValue().isTextual()) env.put(e.getKey(), e.getValue().asText());
            });
        }

        McpServerConfig newCfg = McpServerConfig.stdio(name, command, args, env);

        // Persist FIRST so we don't end up with a running subprocess that survives a crash without
        // any record of why it exists.
        ctx.mcpConfig().put(newCfg);
        try { ctx.mcpConfig().save(); }
        catch (Exception e) {
            ctx.mcpConfig().remove(name);
            respondJson(ex, 500, "{\"error\":\"could not write config: " + escape(e.getMessage()) + "\"}");
            return;
        }

        // Now start the subprocess and merge tools into the live registry.
        try {
            List<Tool> newTools = ctx.mcpPool().startOne(newCfg);
            int added = 0;
            for (Tool t : newTools) {
                try { ctx.tools().add(t); added++; }
                catch (Exception e) { LOG.warn("tool add failed for {}: {}", t.name(), e.toString()); }
            }
            Map<String, Object> body = Map.of(
                    "ok", true,
                    "name", name,
                    "toolCount", added,
                    "tools", newTools.stream().map(Tool::name).toList());
            respondJson(ex, 200, mapper.writeValueAsString(body));
        } catch (Exception e) {
            // Roll back the config write so the workbench shows reality.
            ctx.mcpConfig().remove(name);
            try { ctx.mcpConfig().save(); } catch (Exception ignore) {}
            respondJson(ex, 500, "{\"error\":\"start failed: " + escape(e.getMessage()) + "\"}");
        }
    }

    private void mcpRemove(HttpExchange ex, String name) throws IOException {
        String prefix = ctx.mcpPool().stopOne(name);
        int removed = (prefix != null) ? ctx.tools().removeByPrefix(prefix) : 0;
        ctx.mcpConfig().remove(name);
        try { ctx.mcpConfig().save(); }
        catch (Exception e) {
            respondJson(ex, 500, "{\"error\":\"config save failed: " + escape(e.getMessage()) + "\"}");
            return;
        }
        respondJson(ex, 200, mapper.writeValueAsString(Map.of("ok", true, "name", name, "toolsRemoved", removed)));
    }

    private void mcpRestart(HttpExchange ex, String name) throws IOException {
        var cfgEntry = ctx.mcpConfig().servers().get(name);
        if (cfgEntry == null) {
            respondJson(ex, 404, "{\"error\":\"no such server: " + escape(name) + "\"}");
            return;
        }
        // Stop + drop existing tools first
        String prefix = ctx.mcpPool().stopOne(name);
        if (prefix != null) ctx.tools().removeByPrefix(prefix);

        try {
            List<Tool> newTools = ctx.mcpPool().startOne(cfgEntry);
            int added = 0;
            for (Tool t : newTools) {
                try { ctx.tools().add(t); added++; }
                catch (Exception e) { LOG.warn("tool add failed: {}", e.toString()); }
            }
            respondJson(ex, 200, mapper.writeValueAsString(Map.of(
                    "ok", true, "name", name, "toolCount", added,
                    "tools", newTools.stream().map(Tool::name).toList())));
        } catch (Exception e) {
            respondJson(ex, 500, "{\"error\":\"restart failed: " + escape(e.getMessage()) + "\"}");
        }
    }

    // ============================ helpers =================================

    private IncomingMessage parseInbound(HttpExchange ex) throws IOException {
        JsonNode req;
        try (InputStream is = ex.getRequestBody()) { req = mapper.readTree(is); }
        String text = req.path("text").asText("");
        String chatId = req.path("chatId").asText("");
        String userId = req.path("userId").asText("web-user");
        if (chatId.isBlank()) chatId = "web-" + sessionSeq.incrementAndGet();
        List<String> images = new ArrayList<>();
        for (JsonNode img : req.path("images")) if (img.isTextual()) images.add(img.asText());
        if (text.isBlank() && images.isEmpty()) {
            respondJson(ex, 400, "{\"error\":\"text or image required\"}");
            return null;
        }
        // Optional: mode ("voice" vs "text") and language (ISO 639-1) — tighten the system
        // prompt when the request came from speech so the spoken reply is crisp + correct lang.
        String mode = req.path("mode").asText("text");
        String language = req.path("language").asText("");
        return new IncomingMessage(userId, chatId, text, images, CHANNEL_NAME, mode, language);
    }

    private void sseSend(OutputStream os, String type, Map<String, Object> payload) throws IOException {
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("type", type);
        envelope.putAll(payload);
        String json = mapper.writeValueAsString(envelope);
        os.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    private static String eventTypeName(AgentEvent e) { return e.getClass().getSimpleName(); }

    private static Map<String, Object> eventToMap(AgentEvent e) {
        return switch (e) {
            case AgentEvent.TokenChunk t       -> Map.of("iteration", t.iteration(), "token", t.text());
            case AgentEvent.ThinkingChunk t    -> Map.of("iteration", t.iteration(), "token", t.text());
            case AgentEvent.ToolCallStart s    -> Map.of("iteration", s.iteration(), "callId", s.callId(),
                                                          "toolName", s.toolName(), "arguments", s.argumentsJson() == null ? "" : s.argumentsJson());
            case AgentEvent.ToolCallResult r   -> Map.of("iteration", r.iteration(), "callId", r.callId(),
                                                          "toolName", r.toolName(), "ok", r.ok(),
                                                          "content", r.content() == null ? "" : r.content(),
                                                          "durationMs", r.durationMs());
            case AgentEvent.FinalReply f       -> Map.of("iteration", f.iteration(), "text", f.text() == null ? "" : f.text(),
                                                          "promptTokens", f.promptTokens(), "completionTokens", f.completionTokens(),
                                                          "stopReason", f.stopReason() == null ? "" : f.stopReason());
            case AgentEvent.Warning w          -> Map.of("iteration", w.iteration(), "message", w.message() == null ? "" : w.message());
            case AgentEvent.ReminderFired r    -> Map.of("iteration", r.iteration(),
                                                          "taskId", r.taskId() == null ? "" : r.taskId(),
                                                          "chatId", r.chatId() == null ? "" : r.chatId(),
                                                          "prompt", r.prompt() == null ? "" : r.prompt());
        };
    }

    /**
     * Auth gate. Call at the top of every state-changing or sensitive handler. Returns
     * {@code true} if the request is rejected (response written, caller should return).
     *
     * <p>Two checks in order:
     * <ol>
     *   <li>{@code Origin} header on POST/PUT/DELETE: any non-loopback origin → 403.
     *       (Browsers always send Origin on cross-origin POSTs; this kills CSRF dead.)</li>
     *   <li>Token: header {@code X-Nizo-Token} or cookie {@code nizo_token}. Missing/wrong → 401.</li>
     * </ol>
     */
    private boolean rejectIfUnauthenticated(HttpExchange ex) throws IOException {
        if (auth == null) {
            // start() should have set this; if not, treat as misconfigured rather than wide open.
            respondJson(ex, 503, "{\"error\":\"auth not initialized\"}");
            return true;
        }
        String method = ex.getRequestMethod() == null ? "" : ex.getRequestMethod().toUpperCase();
        boolean mutating = "POST".equals(method) || "PUT".equals(method) || "DELETE".equals(method);
        if (mutating && !WebAuth.originIsAcceptable(ex)) {
            respondJson(ex, 403, "{\"error\":\"forbidden origin\"}");
            return true;
        }
        if (!auth.isAuthenticated(ex)) {
            ex.getResponseHeaders().add("WWW-Authenticate", "X-Nizo-Token");
            respondJson(ex, 401,
                    "{\"error\":\"missing or invalid token\","
                  + "\"hint\":\"reload http://" + escape(config.host()) + ":" + config.port() + "/ to refresh the cookie, "
                  + "or read ~/.nizo/web-token and send it as the X-Nizo-Token header\"}");
            return true;
        }
        return false;
    }

    /**
     * Read a JSON body subject to the configured size limit. Returns {@code null} after writing
     * a 413 response, in which case the caller should return immediately.
     */
    private JsonNode readBoundedJson(HttpExchange ex) throws IOException {
        if (!BoundedBodyReader.enforceContentLength(ex, maxBodyBytes)) return null;
        try (InputStream is = BoundedBodyReader.wrap(ex.getRequestBody(), maxBodyBytes)) {
            return mapper.readTree(is);
        } catch (BoundedBodyReader.BodyTooLargeException tooBig) {
            BoundedBodyReader.writeTooLarge(ex, tooBig.limit(), -1);
            return null;
        }
    }

    /**
     * Read a raw byte body subject to the configured size limit. Returns {@code null} after
     * writing a 413, otherwise the bytes.
     */
    private byte[] readBoundedBytes(HttpExchange ex) throws IOException {
        if (!BoundedBodyReader.enforceContentLength(ex, maxBodyBytes)) return null;
        try {
            return BoundedBodyReader.readAllBounded(ex, maxBodyBytes);
        } catch (BoundedBodyReader.BodyTooLargeException tooBig) {
            BoundedBodyReader.writeTooLarge(ex, tooBig.limit(), -1);
            return null;
        }
    }

    private static void respondText(HttpExchange ex, int status, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "text/plain; charset=utf-8");
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }

    private static void respondJson(HttpExchange ex, int status, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(b); }
    }

    private static String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ");
    }

    /** Pull a useful preview snippet out of a master report — skips the LLM's
     *  preamble ("Now I have all the data..."), finds the first markdown header
     *  or paragraph after it, and returns up to ~320 chars. Falls back to the
     *  raw first chunk if no good anchor is found. */
    private static String extractStockSnippet(String full) {
        if (full == null || full.isEmpty()) return "";
        // Anchor on the first H1/H2 (e.g. "# MSFT — Master Investment Research Report")
        int h = -1;
        int n = full.length();
        for (int i = 0; i < n - 2 && i < 1500; i++) {
            char c = full.charAt(i);
            if (c == '#') {
                int newlineBefore = (i == 0) ? 0 : full.lastIndexOf('\n', i - 1);
                if (newlineBefore == i - 1 || i == 0) { h = i; break; }
            }
        }
        int from = (h >= 0) ? h : 0;
        int to = Math.min(full.length(), from + 320);
        return full.substring(from, to);
    }

    /** Heuristic to pluck a one-word verdict (BUY / SELL / HOLD / NEUTRAL) out of
     *  the master report. Searches the FULL text (not just the snippet) — these
     *  reports always have explicit "Rating:" or "Verdict:" lines but they appear
     *  ~70% through the doc. */
    private static String extractStockVerdict(String full) {
        if (full == null || full.isEmpty()) return "";
        // Try "Rating: STRONG BUY" / "Verdict: HOLD" labels first.
        java.util.regex.Matcher labeled = java.util.regex.Pattern.compile(
                "(?i)\\b(?:rating|verdict|recommendation)\\s*[:\\-]?\\*?\\*?\\s*(?:`)?(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|NEUTRAL|ACCUMULATE|REDUCE)(?:`)?",
                java.util.regex.Pattern.CASE_INSENSITIVE
        ).matcher(full);
        if (labeled.find()) return labeled.group(1).toUpperCase();
        // Fallback: any standalone uppercase BUY/SELL/HOLD inside ** bold ** markers.
        java.util.regex.Matcher bold = java.util.regex.Pattern.compile(
                "\\*\\*(STRONG BUY|STRONG SELL|BUY|SELL|HOLD|NEUTRAL|ACCUMULATE|REDUCE)\\*\\*"
        ).matcher(full);
        if (bold.find()) return bold.group(1).toUpperCase();
        return "";
    }

    /** Proper JSON string escaper — preserves newlines/tabs as their JSON escapes,
     *  unlike {@link #escape(String)} which collapses them. Use for any field
     *  that the client renders as Markdown / preformatted text. */
    private static String jsonStr(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder(s.length() + 16);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\': sb.append("\\\\"); break;
                case '"':  sb.append("\\\""); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                case '\b': sb.append("\\b");  break;
                case '\f': sb.append("\\f");  break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        return sb.toString();
    }
}
