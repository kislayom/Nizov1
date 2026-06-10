package ai.nizo.memory.server;

import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.util.Json;
import ai.nizo.memory.api.MemoryDtos;
import ai.nizo.memory.compact.CompactionService;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * Stand-alone HTTP façade in front of any {@link MemoryService} implementation.
 *
 * <p>Routes:
 * <pre>
 *   POST  /v1/memory/items         — remember (userId in body)
 *   POST  /v1/memory/facts         — learn a semantic fact (userId in body)
 *   POST  /v1/memory/recall        — hybrid BM25 + vector recall (userId in body)
 *   POST  /v1/memory/consolidate   — force a consolidation pass (userId in body)
 *   POST  /v1/memory/compact       — context compaction (userId in body)
 *   GET   /v1/memory/stats         — per-tier counts (userId via ?userId= query param)
 *   GET   /v1/healthz              — liveness probe
 * </pre>
 *
 * <p>JSON in, JSON out. Zero external web-framework dependencies — uses the
 * JDK's built-in {@link HttpServer}, which is sufficient for single-host
 * deployments and trivially proxyable behind anything serious.
 */
public final class MemoryHttpServer {

    private static final Logger log = LoggerFactory.getLogger(MemoryHttpServer.class);

    private final MemoryService memory;
    private final CompactionService compaction;
    private final EmbeddingClient embedder;
    private final ExtractionService extraction;
    private final HttpServer server;
    private final long startedAt = System.currentTimeMillis();

    public MemoryHttpServer(MemoryService memory, int port, int threads) throws IOException {
        this(memory, null, null, null, port, threads);
    }

    public MemoryHttpServer(MemoryService memory,
                            CompactionService compaction,
                            EmbeddingClient embedder,
                            int port, int threads) throws IOException {
        this(memory, compaction, embedder, null, port, threads);
    }

    public MemoryHttpServer(MemoryService memory,
                            CompactionService compaction,
                            EmbeddingClient embedder,
                            ExtractionService extraction,
                            int port, int threads) throws IOException {
        this.memory = memory;
        this.compaction = compaction;
        this.embedder = embedder;
        this.extraction = extraction;
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        this.server.setExecutor(Executors.newFixedThreadPool(threads, r -> {
            Thread t = new Thread(r, "nizo-memory-http");
            t.setDaemon(true);
            return t;
        }));
        route("/v1/memory/items",        "POST", this::handleRemember);
        route("/v1/memory/facts",        "POST", this::handleLearnFact);
        route("/v1/memory/recall",       "POST", this::handleRecall);
        route("/v1/memory/consolidate",  "POST", this::handleConsolidate);
        route("/v1/memory/compact",      "POST", this::handleCompact);
        route("/v1/memory/stats",        "GET",  this::handleStats);
        route("/v1/memory/embedder-info","GET",  this::handleEmbedderInfo);
        route("/v1/memory/extract",      "POST", this::handleExtract);
        route("/v1/memory/inspect",      "GET",  this::handleInspect);
        route("/v1/memory/forget",       "POST", this::handleForgetAbout);
        route("/v1/memory/pin",          "POST", this::handlePin);
        route("/v1/memory/reconfirm",    "POST", this::handleReconfirm);
        route("/v1/memory/import",       "POST", this::handleImportFacts);
        route("/v1/memory/forget-user",  "POST", this::handleForgetUser);
        route("/v1/healthz",             "GET",  this::handleHealth);
        route("/v1/memory/health",       "GET",  this::handleHealth);   // G1 alias
        route("/v1/memory/reflect",      "POST", this::handleReflect);  // G30
        route("/v1/memory/surface",      "POST", this::handleSurface);  // Active Memory
        route("/v1/memory/index",        "GET",  this::handleCanonicalIndex); // Phase C
    }

    public void start() {
        server.start();
        log.info("Nizo memory service listening on port {}", server.getAddress().getPort());
    }

    public void stop() { server.stop(0); }

    public int port() { return server.getAddress().getPort(); }

    // ---------- routing plumbing ----------

    private void route(String path, String method, HttpHandler delegate) {
        server.createContext(path, ex -> {
            try {
                if (!method.equalsIgnoreCase(ex.getRequestMethod())) {
                    send(ex, 405, new MemoryDtos.ErrorResponseDto("method not allowed"));
                    return;
                }
                delegate.handle(ex);
            } catch (com.fasterxml.jackson.core.JsonProcessingException jpe) {
                // G22/G24/G32 — Jackson errors are client problems (malformed
                // JSON, unknown/unaccepted fields, wrong types). Return 400
                // with a short error, not 500 with a class name + stack trace.
                log.debug("{} {} — malformed request: {}", method, path, jpe.getOriginalMessage());
                try {
                    send(ex, 400, new MemoryDtos.ErrorResponseDto(
                            "malformed request: " + jpe.getOriginalMessage()));
                } catch (IOException ignored) {}
            } catch (BadRequestException bre) {
                // G3/G5/G23/G26/G35 — uniform client-error path.
                log.debug("{} {} — bad request: {}", method, path, bre.getMessage());
                try { send(ex, 400, new MemoryDtos.ErrorResponseDto(bre.getMessage())); }
                catch (IOException ignored) {}
            } catch (Exception e) {
                log.warn("{} {} failed", method, path, e);
                try { send(ex, 500, new MemoryDtos.ErrorResponseDto("internal error")); }
                catch (IOException ignored) { /* connection already gone */ }
            }
        });
    }

    /** Client-error signal the route wrapper converts to HTTP 400. */
    static final class BadRequestException extends RuntimeException {
        BadRequestException(String msg) { super(msg); }
    }

    // ---------- handlers ----------

    private void handleRemember(HttpExchange ex) throws IOException {
        MemoryDtos.RememberRequestDto req = parse(ex, MemoryDtos.RememberRequestDto.class);
        if (req.content() == null || req.content().isBlank()) {
            throw new BadRequestException("content required");
        }
        String userId = validUserId(req.userId());
        String id = memory.remember(userId, req.content(),
                req.tags() == null ? Map.of() : req.tags(), req.source());
        send(ex, 201, new MemoryDtos.IdResponse(id));
    }

    private void handleLearnFact(HttpExchange ex) throws IOException {
        MemoryDtos.LearnFactRequestDto req = parse(ex, MemoryDtos.LearnFactRequestDto.class);
        if (req.content() == null || req.content().isBlank()) {
            throw new BadRequestException("content required");
        }
        String userId = validUserId(req.userId());
        double conf = req.confidence() == null ? 0.9 : req.confidence();
        String id = memory.learnFact(userId, req.content(), req.source(), conf);
        send(ex, 201, new MemoryDtos.IdResponse(id));
    }

    private void handleRecall(HttpExchange ex) throws IOException {
        MemoryDtos.RecallRequestDto req = parse(ex, MemoryDtos.RecallRequestDto.class);
        // G23 — require userId and non-blank query (was silently returning
        // empty on missing fields).
        String userId = validUserId(req.userId());
        if (req.query() == null || req.query().isBlank()) {
            throw new BadRequestException("query required");
        }
        RecallRequest r = new RecallRequest(
                userId,
                req.query(),
                req.tokenBudget() == null ? 1200 : req.tokenBudget(),
                req.tiers(),
                req.requiredTags() == null ? Map.of() : req.requiredTags(),
                req.minConfidence() == null ? 0.0 : req.minConfidence());
        List<MemoryItem> items = memory.recall(r);
        List<MemoryDtos.MemoryItemDto> dtos = new ArrayList<>(items.size());
        for (MemoryItem m : items) dtos.add(MemoryDtos.MemoryItemDto.fromDomain(m));
        send(ex, 200, new MemoryDtos.RecallResponseDto(dtos));
    }

    private void handleConsolidate(HttpExchange ex) throws IOException {
        // Accept optional userId in body; default to "default".
        Map<?, ?> body = parse(ex, Map.class);
        String userId = "default";
        if (body != null && body.get("userId") instanceof String u) {
            userId = u;
        }
        // Force legacy consolidate via system property, scoped to this call.
        String prev = System.getProperty("nizo.memory.legacy-consolidate");
        System.setProperty("nizo.memory.legacy-consolidate", "true");
        try { memory.consolidate(userId); }
        finally {
            if (prev == null) System.clearProperty("nizo.memory.legacy-consolidate");
            else System.setProperty("nizo.memory.legacy-consolidate", prev);
        }
        send(ex, 200, Map.of("ok", true, "note", "legacy consolidation; prefer /v1/memory/reflect"));
    }

    /** G30 — customer-triggered reflection. Delegates to legacy consolidate for
     *  now so the endpoint exists; full ReflectionService wiring comes via
     *  the scheduler on startup. */
    private void handleReflect(HttpExchange ex) throws IOException {
        Map<?, ?> body = parse(ex, Map.class);
        String userId = "default";
        if (body != null && body.get("userId") instanceof String u) userId = u;
        memory.consolidate(userId);
        send(ex, 200, Map.of("ok", true, "userId", userId));
    }

    /**
     * Active Memory — pre-reply proactive surface. Agent calls this every
     * turn with the user's latest message; we decide whether to return
     * relevant facts or abstain.
     */
    private void handleSurface(HttpExchange ex) throws IOException {
        MemoryDtos.SurfaceRequestDto req = parse(ex, MemoryDtos.SurfaceRequestDto.class);
        if (req.message() == null || req.message().isBlank()) {
            throw new BadRequestException("message required");
        }
        if (req.message().length() > 20_000) {
            throw new BadRequestException(
                    "message too long: " + req.message().length() + " chars (max 20000)");
        }
        String userId = validUserId(req.userId());

        List<MemoryService.ConversationTurn> recent = req.recentTurns() == null
                ? List.of()
                : req.recentTurns().stream()
                    .map(t -> new MemoryService.ConversationTurn(t.role(), t.content()))
                    .toList();

        MemoryService.SurfaceRequest sr = new MemoryService.SurfaceRequest(
                userId,
                req.message(),
                req.mode() == null ? "balanced" : req.mode(),
                req.maxItems() == null ? 5 : req.maxItems(),
                req.maxSummaryChars() == null ? 500 : req.maxSummaryChars(),
                recent);

        MemoryService.SurfaceResult result = memory.surface(sr);

        List<MemoryDtos.SurfacedItemDto> items = result.items().stream()
                .map(m -> new MemoryDtos.SurfacedItemDto(
                        m.id(),
                        m.content(),
                        m.confidence(),
                        m.source(),
                        m.tier().name(),
                        m.tags()))
                .toList();
        send(ex, 200, new MemoryDtos.SurfaceResponseDto(
                result.surfaced(),
                result.summary(),
                items,
                result.skipReason(),
                result.mode()));
    }

    private void handleStats(HttpExchange ex) throws IOException {
        String userId = queryParam(ex, "userId");
        if (userId == null) userId = "default";
        send(ex, 200, new MemoryDtos.StatsResponseDto(memory.stats(userId)));
    }

    private void handleCompact(HttpExchange ex) throws IOException {
        if (compaction == null) {
            send(ex, 503, new MemoryDtos.ErrorResponseDto("compaction not configured"));
            return;
        }
        MemoryDtos.CompactRequestDto req = parse(ex, MemoryDtos.CompactRequestDto.class);
        List<Message> msgs = new ArrayList<>();
        if (req.messages() != null) {
            for (MemoryDtos.MessageDto m : req.messages()) {
                Message.Role role = switch (m.role() == null ? "user" : m.role().toLowerCase()) {
                    case "system"    -> Message.Role.SYSTEM;
                    case "assistant" -> Message.Role.ASSISTANT;
                    default          -> Message.Role.USER;
                };
                msgs.add(new Message(role, List.of(new Message.TextPart(m.text()))));
            }
        }
        int maxTokens = req.maxTokens() == null ? 20000 : req.maxTokens();
        CompactionService.CompactionResult r = compaction.compact("default", msgs, maxTokens);
        send(ex, 200, new MemoryDtos.CompactResponseDto(
                r.compacted(), r.summary(), r.messagesCompacted(),
                r.inputTokens(), r.outputTokens(), r.skipReason()));
    }

    private void handleEmbedderInfo(HttpExchange ex) throws IOException {
        if (embedder == null) {
            send(ex, 200, new MemoryDtos.EmbedderInfoDto("none", 0, ""));
        } else {
            String type = embedder.getClass().getSimpleName();
            send(ex, 200, new MemoryDtos.EmbedderInfoDto(type, embedder.dimensions(), ""));
        }
    }

    private void handleHealth(HttpExchange ex) throws IOException {
        send(ex, 200, new MemoryDtos.HealthResponseDto("ok",
                System.currentTimeMillis() - startedAt));
    }

    // ---------- extraction / customer-facing controls ----------

    private void handleExtract(HttpExchange ex) throws IOException {
        MemoryDtos.ExtractRequestDto req = parse(ex, MemoryDtos.ExtractRequestDto.class);
        if (req.message() == null || req.message().isBlank()) {
            throw new BadRequestException("message required");
        }
        // G26 — cap extraction message length too. The underlying prompt
        // bloats with long messages and Ollama latency spikes quadratically.
        if (req.message().length() > 20_000) {
            throw new BadRequestException(
                    "message too long: " + req.message().length() + " chars (max 20000)");
        }
        if (extraction == null) {
            send(ex, 503, new MemoryDtos.ErrorResponseDto(
                    "extraction disabled — check config (extraction.enabled + LLM backend)"));
            return;
        }
        String userId = validUserId(req.userId());
        ExtractionResult r = extraction.extract(userId, req.message());
        Set<String> cats = r.types().stream()
                .map(Enum::name).collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
        send(ex, 200, new MemoryDtos.ExtractResponseDto(r.count(), cats, r.raw()));
    }

    private void handleCanonicalIndex(HttpExchange ex) throws IOException {
        String userId = validUserId(queryParam(ex, "userId"));
        String limitStr = queryParam(ex, "limit");
        int limit;
        try {
            limit = limitStr == null ? 50 : Math.max(1, Math.min(500, Integer.parseInt(limitStr)));
        } catch (NumberFormatException e) {
            throw new BadRequestException("limit must be an integer 1..500");
        }
        List<MemoryService.IndexEntry> entries = memory.canonicalIndex(userId, limit);
        List<MemoryDtos.CanonicalIndexEntryDto> dtos = entries.stream()
                .map(e1 -> new MemoryDtos.CanonicalIndexEntryDto(
                        e1.clusterKey(),
                        e1.fact(),
                        e1.facet(),
                        e1.lastReconfirmed() == null ? null : e1.lastReconfirmed().toString()))
                .toList();
        send(ex, 200, new MemoryDtos.CanonicalIndexResponseDto(userId, dtos.size(), dtos));
    }

    private void handleInspect(HttpExchange ex) throws IOException {
        String userId = queryParam(ex, "userId");
        if (userId == null) userId = "default";
        String limitStr = queryParam(ex, "limit");
        int limit = limitStr == null ? 100 : Integer.parseInt(limitStr);
        // G20 — filter noise by default. Customer-facing inspect should show
        // what the system "knows" (SEMANTIC facts), not every raw user
        // message + extraction sub-episode. Pass ?raw=true to see everything.
        boolean showRaw = "true".equalsIgnoreCase(queryParam(ex, "raw"));
        List<MemoryItem> items = memory.inspect(userId, limit);
        List<MemoryItem> filtered = showRaw ? items : items.stream().filter(m -> {
            // Drop raw user messages and extraction-created EPISODIC sub-items
            if (m.tier() == MemoryItem.Tier.EPISODIC) {
                if ("user_message".equals(m.source())) return false;
                if ("extraction".equals(m.source())) return false;
            }
            return true;
        }).toList();
        var dtos = filtered.stream().map(MemoryDtos.MemoryItemDto::fromDomain).toList();
        send(ex, 200, new MemoryDtos.InspectResponseDto(dtos.size(), dtos));
    }

    private void handleForgetAbout(HttpExchange ex) throws IOException {
        MemoryDtos.ForgetRequestDto req = parse(ex, MemoryDtos.ForgetRequestDto.class);
        if (req.topic() == null || req.topic().isBlank()) {
            throw new BadRequestException("topic required");
        }
        String userId = validUserId(req.userId());
        int n = memory.forgetAbout(userId, req.topic());
        send(ex, 200, new MemoryDtos.ForgetResponseDto(n));
    }

    private void handlePin(HttpExchange ex) throws IOException {
        MemoryDtos.PinRequestDto req = parse(ex, MemoryDtos.PinRequestDto.class);
        if (req.factId() == null || req.factId().isBlank()) {
            throw new BadRequestException("factId required");
        }
        String userId = validUserId(req.userId());
        boolean pinned = req.pinned() == null ? true : req.pinned();
        boolean updated = memory.pin(userId, req.factId(), pinned, req.reason());
        // G28 — ghost factId is a 404, not a silent 200 false.
        if (!updated) {
            send(ex, 404, new MemoryDtos.ErrorResponseDto(
                    "factId not found: " + req.factId()));
            return;
        }
        send(ex, 200, new MemoryDtos.PinResponseDto(updated));
    }

    private void handleReconfirm(HttpExchange ex) throws IOException {
        MemoryDtos.ReconfirmRequestDto req = parse(ex, MemoryDtos.ReconfirmRequestDto.class);
        if (req.factId() == null || req.factId().isBlank()) {
            throw new BadRequestException("factId required");
        }
        String userId = validUserId(req.userId());
        boolean updated = memory.reconfirm(userId, req.factId());
        if (!updated) {
            send(ex, 404, new MemoryDtos.ErrorResponseDto(
                    "factId not found: " + req.factId()));
            return;
        }
        send(ex, 200, new MemoryDtos.PinResponseDto(updated));
    }

    /** G26 — per-call limit on how many facts a single /import can load. */
    static final int MAX_FACTS_PER_IMPORT = 500;

    private void handleImportFacts(HttpExchange ex) throws IOException {
        MemoryDtos.ImportRequestDto req = parse(ex, MemoryDtos.ImportRequestDto.class);
        if (req.facts() == null || req.facts().isEmpty()) {
            throw new BadRequestException("facts[] required");
        }
        if (req.facts().size() > MAX_FACTS_PER_IMPORT) {
            throw new BadRequestException(
                    "too many facts: " + req.facts().size()
                            + " (max " + MAX_FACTS_PER_IMPORT + " per call — paginate)");
        }
        String userId = validUserId(req.userId());
        List<MemoryService.ImportedFact> list = req.facts().stream()
                .map(f -> new MemoryService.ImportedFact(f.content(), f.tags(), f.confidence()))
                .toList();
        int n = memory.importFacts(userId, list);
        send(ex, 201, new MemoryDtos.ImportResponseDto(n));
    }

    private void handleForgetUser(HttpExchange ex) throws IOException {
        MemoryDtos.ForgetUserRequestDto req = parse(ex, MemoryDtos.ForgetUserRequestDto.class);
        String userId = validUserId(req.userId());
        int n = memory.forgetUser(userId);
        send(ex, 200, new MemoryDtos.ForgetResponseDto(n));
    }

    // ---------- helpers ----------

    /** G26 — cap request body at 1 MiB. Anything larger is rejected with 413
     *  (or 400 if client sent no Content-Length). This protects the service
     *  from trivial DoS (50K facts in one import) and accidental blowups. */
    static final int MAX_BODY_BYTES = 1 * 1024 * 1024;

    private static <T> T parse(HttpExchange ex, Class<T> type) throws IOException {
        // G25/R4-E — require JSON content-type on POSTs. We still accept
        // requests with no content-type at all (some CLIs default to this)
        // but reject explicitly-wrong types so the client gets a clear error
        // instead of silent auto-parse.
        String ct = ex.getRequestHeaders().getFirst("Content-Type");
        if (ct != null && !ct.isBlank()) {
            String low = ct.toLowerCase();
            if (!low.startsWith("application/json")) {
                throw new BadRequestException(
                        "unsupported content-type '" + ct + "' — use application/json");
            }
        }
        String cl = ex.getRequestHeaders().getFirst("Content-Length");
        if (cl != null) {
            try {
                long n = Long.parseLong(cl);
                if (n > MAX_BODY_BYTES) {
                    throw new BadRequestException(
                            "request body too large: " + n + " bytes > limit " + MAX_BODY_BYTES);
                }
            } catch (NumberFormatException ignore) {}
        }
        try (InputStream in = ex.getRequestBody()) {
            // Read with a hard cap — defends against missing/lying Content-Length.
            java.io.ByteArrayOutputStream buf = new java.io.ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int total = 0;
            int read;
            while ((read = in.read(chunk)) != -1) {
                total += read;
                if (total > MAX_BODY_BYTES) {
                    throw new BadRequestException(
                            "request body too large: exceeded " + MAX_BODY_BYTES + " bytes");
                }
                buf.write(chunk, 0, read);
            }
            byte[] body = buf.toByteArray();
            if (body.length == 0) return Json.MAPPER.readValue("{}", type);
            return Json.MAPPER.readValue(body, type);
        }
    }

    /** G3/G5/R7-A — sanity-check userId values. Null/blank/absurdly-long
     *  rejected, and the id must match a safe character whitelist so log
     *  injection / path-traversal / SQL-shaped ids don't slip through. */
    static final java.util.regex.Pattern USERID_PATTERN =
            java.util.regex.Pattern.compile("^[A-Za-z0-9._@:/-]{1,200}$");

    static String validUserId(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new BadRequestException(
                    "userId required (null / empty not permitted — use a real id)");
        }
        if (raw.length() > 200) {
            throw new BadRequestException(
                    "userId too long: " + raw.length() + " chars (max 200)");
        }
        if (!USERID_PATTERN.matcher(raw).matches()) {
            throw new BadRequestException(
                    "userId contains illegal characters — allowed: A-Z a-z 0-9 . _ - / : @");
        }
        // Defend against path-traversal in userIds that get embedded in
        // log lines or file paths.
        if (raw.contains("..") || raw.startsWith("/") || raw.startsWith(".")) {
            throw new BadRequestException(
                    "userId must not contain '..' or start with '/' or '.'");
        }
        return raw;
    }

    private static void send(HttpExchange ex, int status, Object body) throws IOException {
        byte[] bytes = Json.MAPPER.writeValueAsBytes(body);
        Headers h = ex.getResponseHeaders();
        h.set("content-type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        ex.getResponseBody().write(bytes);
        ex.close();
    }

    private static String queryParam(HttpExchange ex, String name) {
        String query = ex.getRequestURI().getQuery();
        if (query == null) return null;
        for (String pair : query.split("&")) {
            String[] kv = pair.split("=", 2);
            if (kv.length == 2 && kv[0].equals(name)) return kv[1];
        }
        return null;
    }

    /** Convenience constant for tests to reference the content type. */
    public static final String CONTENT_TYPE = "application/json; charset=" + StandardCharsets.UTF_8;
}
