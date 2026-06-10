package ai.nizo.memory.client;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.util.Http;
import ai.nizo.memory.util.Json;
import ai.nizo.memory.api.MemoryDtos;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * {@link MemoryService} adapter that forwards every call to a remote
 * {@code nizo-memory-service}. Implements the same contract as
 * {@link ai.nizo.memory.LayeredMemoryService} so callers are
 * transport-agnostic.
 *
 * <p>Failure policy: network errors surface as {@link MemoryServiceException}
 * rather than silently swallowing — the agent should prefer a clear failure
 * over made-up recalls (sticking to the "facts or nothing" principle).
 */
public final class MemoryHttpClient implements MemoryService {

    private final HttpClient http;
    private final String baseUrl;
    private final Duration timeout;

    public MemoryHttpClient(String baseUrl, Duration timeout) {
        this.baseUrl = Http.normaliseEndpoint(baseUrl);
        this.timeout = timeout;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    @Override
    public String remember(String userId, String content, Map<String, String> tags, String source) {
        MemoryDtos.IdResponse r = post("/v1/memory/items",
                new MemoryDtos.RememberRequestDto(userId, content, tags, source),
                MemoryDtos.IdResponse.class);
        return r.id();
    }

    @Override
    public String learnFact(String userId, String fact, String source, double confidence) {
        MemoryDtos.IdResponse r = post("/v1/memory/facts",
                new MemoryDtos.LearnFactRequestDto(userId, fact, source, confidence),
                MemoryDtos.IdResponse.class);
        return r.id();
    }

    @Override
    public List<MemoryItem> recall(RecallRequest req) {
        MemoryDtos.RecallRequestDto body = new MemoryDtos.RecallRequestDto(
                req.userId(), req.query(), req.tokenBudget(), req.tiers(),
                req.requiredTags(), req.minConfidence());
        MemoryDtos.RecallResponseDto resp = post("/v1/memory/recall", body,
                MemoryDtos.RecallResponseDto.class);
        List<MemoryItem> out = new ArrayList<>(resp.items().size());
        for (MemoryDtos.MemoryItemDto d : resp.items()) out.add(d.toDomain());
        return out;
    }

    @Override
    public void consolidate(String userId) {
        post("/v1/memory/consolidate", Map.of("userId", userId == null ? "default" : userId), Map.class);
    }

    @Override
    public Map<MemoryItem.Tier, Long> stats(String userId) {
        String uid = userId == null ? "default" : userId;
        MemoryDtos.StatsResponseDto resp = get("/v1/memory/stats?userId=" + uid,
                MemoryDtos.StatsResponseDto.class);
        return resp.counts();
    }

    // -------- low-level --------

    private <T> T post(String path, Object body, Class<T> type) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("content-type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.stringify(body)))
                .build();
        return send(req, type);
    }

    private <T> T get(String path, Class<T> type) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .timeout(timeout).GET().build();
        return send(req, type);
    }

    private <T> T send(HttpRequest req, Class<T> type) {
        try {
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400) {
                throw new MemoryServiceException("HTTP " + resp.statusCode() + ": " + resp.body());
            }
            if (type == Map.class || resp.body() == null || resp.body().isEmpty()) {
                return Json.parse(resp.body() == null || resp.body().isEmpty() ? "{}" : resp.body(), type);
            }
            return Json.parse(resp.body(), type);
        } catch (MemoryServiceException e) {
            throw e;
        } catch (Exception e) {
            throw new MemoryServiceException("memory service call failed: " + e.getMessage(), e);
        }
    }

    public static final class MemoryServiceException extends RuntimeException {
        public MemoryServiceException(String m) { super(m); }
        public MemoryServiceException(String m, Throwable t) { super(m, t); }
    }
}
