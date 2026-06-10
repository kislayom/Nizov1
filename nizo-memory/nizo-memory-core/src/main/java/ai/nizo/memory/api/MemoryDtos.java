package ai.nizo.memory.api;

import ai.nizo.memory.api.memory.MemoryItem;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

// Timestamps are carried as epoch-millis to avoid pulling jackson-datatype-jsr310
// into the memory service's dependency tree. Clients convert back to Instant.

/**
 * Wire-format records for the memory HTTP API. Kept deliberately flat and
 * free of framework annotations so Jackson serialises them via record
 * components without any reflection gymnastics.
 *
 * <p>Note: we do NOT send embeddings over the wire — they are an internal
 * implementation detail of the service and can be hundreds of KB per item.
 * Clients supply only {@code content}; the service computes the vector.
 *
 * <p>All request DTOs include an optional {@code userId} field. When absent,
 * the server defaults to {@code "default"}.
 */
public final class MemoryDtos {

    private MemoryDtos() {}

    public record RememberRequestDto(String userId, String content, Map<String, String> tags, String source) {}
    public record LearnFactRequestDto(String userId, String content, String source, Double confidence) {}
    public record IdResponse(String id) {}

    public record RecallRequestDto(
            String userId,
            String query,
            Integer tokenBudget,
            Set<MemoryItem.Tier> tiers,
            Map<String, String> requiredTags,
            Double minConfidence
    ) {}

    public record RecallResponseDto(List<MemoryItemDto> items) {}

    /** Same fields as {@link MemoryItem} minus {@code embedding}; timestamps are epoch-millis. */
    public record MemoryItemDto(
            String id,
            String userId,
            MemoryItem.Tier tier,
            String content,
            Map<String, String> tags,
            String source,
            double confidence,
            long createdAtMillis,
            long lastAccessedAtMillis,
            int accessCount,
            int tokens
    ) {
        public static MemoryItemDto fromDomain(MemoryItem m) {
            return new MemoryItemDto(m.id(), m.userId(), m.tier(), m.content(), m.tags(), m.source(),
                    m.confidence(),
                    m.createdAt().toEpochMilli(), m.lastAccessedAt().toEpochMilli(),
                    m.accessCount(), m.tokens());
        }

        public MemoryItem toDomain() {
            return new MemoryItem(id, userId, tier, content, null, tags, source, confidence,
                    Instant.ofEpochMilli(createdAtMillis),
                    Instant.ofEpochMilli(lastAccessedAtMillis),
                    accessCount, tokens);
        }
    }

    // ---- compaction ----
    // G31 — userId kept optional so older clients still work; compaction
    // itself is currently user-agnostic (acts on an incoming message list),
    // but we accept the field so the DTO schema matches other endpoints and
    // callers can be consistent.
    public record CompactRequestDto(String userId, List<MessageDto> messages, Integer maxTokens) {}
    public record MessageDto(String role, String text) {}
    public record CompactResponseDto(
            boolean compacted,
            String summary,
            int messagesCompacted,
            int inputTokens,
            int outputTokens,
            String skipReason
    ) {}

    // ---- verification / embedder info ----
    public record EmbedderInfoDto(String type, int dimensions, String model) {}

    public record StatsResponseDto(Map<MemoryItem.Tier, Long> counts) {}
    public record HealthResponseDto(String status, long uptimeMs) {}
    public record ErrorResponseDto(String error) {}

    // ---- extraction / customer-facing controls ----
    public record ExtractRequestDto(String userId, String message) {}
    public record ExtractResponseDto(int count, Set<String> categories, Map<String, Object> raw) {}
    public record InspectResponseDto(int total, List<MemoryItemDto> items) {}
    public record ForgetRequestDto(String userId, String topic) {}
    public record ForgetResponseDto(int deleted) {}
    public record PinRequestDto(String userId, String factId, Boolean pinned, String reason) {}
    public record PinResponseDto(boolean updated) {}
    public record ImportRequestDto(String userId, List<ImportedFactDto> facts) {}
    public record ImportedFactDto(String content, Map<String, String> tags, Double confidence) {}
    public record ImportResponseDto(int loaded) {}
    public record ReconfirmRequestDto(String userId, String factId) {}
    public record ForgetUserRequestDto(String userId) {}

    // ---- active memory ----
    // Pre-reply proactive surface: the calling agent sends the user's latest
    // message; we return the facts worth considering BEFORE the agent generates
    // its response. Bounded, abstains honestly, no mandatory LLM call.
    public record SurfaceRequestDto(
            String userId,
            String message,
            String mode,              // balanced | strict | recall-heavy | precision-heavy | preference-only
            Integer maxItems,
            Integer maxSummaryChars,
            List<SurfaceContextTurn> recentTurns   // optional: recent conversation for intent
    ) {}

    public record SurfaceContextTurn(String role, String content) {}

    public record SurfaceResponseDto(
            boolean surfaced,
            String summary,
            List<SurfacedItemDto> items,
            String skipReason,
            String mode
    ) {}

    public record SurfacedItemDto(
            String id,
            String content,
            double relevance,
            String source,
            String tier,
            Map<String, String> tags
    ) {}

    // ---- Canonical index (Phase C) ----

    /** One row of the canonical index — the top-of-prompt table of contents. */
    public record CanonicalIndexEntryDto(
            String clusterKey,
            String fact,
            String facet,
            String lastReconfirmed  // ISO-8601, nullable
    ) {}

    /** Response for GET /v1/memory/index. */
    public record CanonicalIndexResponseDto(
            String userId,
            int count,
            List<CanonicalIndexEntryDto> entries
    ) {}
}
