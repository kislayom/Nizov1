package ai.nizo.memory.reflect;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.MemoryTags;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.util.Vectors;
import ai.nizo.memory.vector.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * F1 — Reflection / summarization worker. Periodically distills older
 * EPISODIC memories into consolidated SEMANTIC facts.
 *
 * <p>The problem this solves: without reflection, EPISODIC grows
 * monotonically. A user saying "mentioned anniversary in March" produces
 * dozens of low-signal fragments that clog recall. Reflection compresses
 * those into one high-confidence semantic fact ("anniversary: March 14") and
 * marks the source episodes as processed.
 *
 * <p>Design:
 * <ul>
 *   <li>Runs on a scheduled executor — independent of request traffic.</li>
 *   <li>Per-tick, enumerates distinct users. For each user with
 *       ≥ {@code minEpisodesPerTick} unreflected episodes older than
 *       {@code minAgeHours}, runs an LLM distillation.</li>
 *   <li>Episodes are marked with {@link MemoryTags#REFLECTED_AT} on
 *       completion so they're not re-distilled on the next tick.</li>
 *   <li>Derived facts inherit {@code subject} / {@code mode} tags from
 *       majority vote of source episodes (when available).</li>
 *   <li>Near-duplicate SEMANTIC facts (cosine ≥ 0.92) are skipped —
 *       reflection reinforces, it does not bloat.</li>
 * </ul>
 *
 * <p>Not Spring — plain {@link ScheduledExecutorService}.
 */
public final class ReflectionService implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ReflectionService.class);

    private static final String REFLECTION_PROMPT = """
            You are the memory reflector for a long-term personal memory system.
            Below are EPISODIC events from a single user's recent interactions.
            Your job: extract the STABLE facts a human would remember a week
            later — preferences, long-running projects, recurring entities,
            biographical updates, beliefs, health constraints.

            Rules:
            - Output ONE fact per line, no bullets, no numbering, no quotes.
            - Each fact MUST be a complete sentence about the user (or a named
              third party).
            - Do NOT restate trivia ("user said hi"). Skip anything that
              won't still matter in a month.
            - Prefer specificity: "uses Vim as code editor" beats "likes editors".
            - If there are no stable facts worth retaining, output nothing.
            - Output facts directly. No preamble, no explanation.

            Events:
            %s
            """;

    private final MemoryService memory;
    private final SqliteMemoryStore store;
    private final VectorIndex vectorIndex;
    private final EmbeddingClient embedder;
    private final ModelClient llm;
    private final ScheduledExecutorService scheduler;

    private final Duration tickInterval;
    private final long minEpisodeAgeMillis;
    private final int minEpisodesPerTick;
    private final int maxEpisodesPerBatch;
    private final double duplicateThreshold;

    public ReflectionService(MemoryService memory,
                             SqliteMemoryStore store,
                             VectorIndex vectorIndex,
                             EmbeddingClient embedder,
                             ModelClient llm,
                             Duration tickInterval,
                             Duration minEpisodeAge,
                             int minEpisodesPerTick,
                             int maxEpisodesPerBatch,
                             double duplicateThreshold) {
        this.memory = Objects.requireNonNull(memory);
        this.store = Objects.requireNonNull(store);
        this.vectorIndex = vectorIndex;     // nullable — dedup skipped if absent
        this.embedder = embedder;           // nullable — dedup skipped if absent
        this.llm = Objects.requireNonNull(llm);
        this.tickInterval = Objects.requireNonNull(tickInterval);
        this.minEpisodeAgeMillis = minEpisodeAge.toMillis();
        this.minEpisodesPerTick = minEpisodesPerTick;
        this.maxEpisodesPerBatch = maxEpisodesPerBatch;
        this.duplicateThreshold = duplicateThreshold;
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "nizo-memory-reflection");
            t.setDaemon(true);
            return t;
        });
    }

    /** Start the reflection scheduler. First tick runs after one interval. */
    public void start() {
        long periodSec = Math.max(30, tickInterval.toSeconds());
        scheduler.scheduleAtFixedRate(this::tickSafely, periodSec, periodSec, TimeUnit.SECONDS);
        log.info("ReflectionService started — tick every {}s, minEpisodeAge={}h, minPerTick={}",
                periodSec, minEpisodeAgeMillis / 3_600_000L, minEpisodesPerTick);
    }

    /** Run one reflection tick right now. Blocking. Useful for tests. */
    public int runOnce() {
        int totalFacts = 0;
        for (String uid : store.distinctUserIds()) {
            try {
                totalFacts += reflectForUser(uid);
            } catch (RuntimeException e) {
                log.warn("Reflection failed for user {}: {}", uid, e.toString());
            }
        }
        return totalFacts;
    }

    private void tickSafely() {
        try { runOnce(); }
        catch (Throwable t) { log.warn("Reflection tick failed: {}", t.toString()); }
    }

    /** Returns the number of new SEMANTIC facts stored for this user. */
    int reflectForUser(String userId) {
        List<MemoryItem> candidates = store.olderEpisodes(userId, minEpisodeAgeMillis, maxEpisodesPerBatch * 2);
        // Filter out episodes already reflected.
        List<MemoryItem> unreflected = new ArrayList<>();
        for (MemoryItem m : candidates) {
            if (m.tags() != null && m.tags().containsKey(MemoryTags.REFLECTED_AT)) continue;
            unreflected.add(m);
            if (unreflected.size() >= maxEpisodesPerBatch) break;
        }
        if (unreflected.size() < minEpisodesPerTick) return 0;

        StringBuilder buf = new StringBuilder();
        for (MemoryItem m : unreflected) {
            String oneLine = m.content().replace('\n', ' ').trim();
            buf.append("- ").append(oneLine).append('\n');
        }
        String raw;
        try {
            raw = llm.complete(ModelRequest.of(List.of(
                    Message.user(REFLECTION_PROMPT.formatted(buf)))))
                    .text();
        } catch (RuntimeException e) {
            log.warn("LLM call failed in reflection for user={}: {}", userId, e.toString());
            return 0;
        }
        if (raw == null || raw.isBlank()) {
            // Even if the LLM produced nothing, still mark the episodes as
            // reflected so we don't keep retrying empty batches forever.
            markReflected(unreflected);
            return 0;
        }

        // Majority vote on tags.
        String majoritySubject = majorityTag(unreflected, MemoryTags.SUBJECT, MemoryTags.SUBJECT_SELF);
        String majorityMode = majorityTag(unreflected, MemoryTags.MODE, null);
        String majoritySensitivity = majorityTag(unreflected, MemoryTags.SENSITIVITY, null);

        int stored = 0;
        for (String line : raw.split("\\r?\\n")) {
            String fact = line.replaceFirst("^[-*\\d.\\s]+", "").trim();
            if (fact.length() < 8) continue;
            if (fact.length() > 400) fact = fact.substring(0, 400);
            if (isDuplicate(userId, fact)) continue;

            Map<String, String> tags = new LinkedHashMap<>();
            tags.put("kind", "reflection");
            if (majoritySubject != null) tags.put(MemoryTags.SUBJECT, majoritySubject);
            if (majorityMode != null) tags.put(MemoryTags.MODE, majorityMode);
            if (majoritySensitivity != null) tags.put(MemoryTags.SENSITIVITY, majoritySensitivity);
            // Provenance: comma-joined source episode ids, capped at ~5 to
            // keep the tag small. Full list is recoverable by scanning
            // memory for items with matching reflected_at timestamp.
            String sourceIds = unreflected.stream()
                    .limit(5)
                    .map(MemoryItem::id)
                    .reduce((a, b) -> a + "," + b)
                    .orElse("");
            if (!sourceIds.isBlank()) tags.put(MemoryTags.SOURCE_MESSAGE_ID, sourceIds);

            memory.learnFact(userId, fact, "reflection", 0.75, tags);
            stored++;
        }

        markReflected(unreflected);
        log.info("Reflection for user={} produced {} new facts from {} episodes",
                userId, stored, unreflected.size());
        return stored;
    }

    private boolean isDuplicate(String userId, String candidate) {
        if (embedder == null || vectorIndex == null) return false;
        try {
            float[] v = embedder.embed(candidate);
            if (v == null) return false;
            for (VectorIndex.Hit h : vectorIndex.topK(userId, v, 3)) {
                var maybe = store.findById(h.id());
                if (maybe.isEmpty()) continue;
                MemoryItem existing = maybe.get();
                if (existing.tier() != MemoryItem.Tier.SEMANTIC) continue;
                if (existing.embedding() == null) continue;
                double cos = Vectors.cosine(v, existing.embedding());
                if (cos >= duplicateThreshold) {
                    log.debug("Reflection skipped duplicate (cos={}): {}", cos, candidate);
                    return true;
                }
            }
        } catch (RuntimeException ignore) { /* fail open — store the fact */ }
        return false;
    }

    private void markReflected(List<MemoryItem> episodes) {
        String stamp = Instant.now().toString();
        for (MemoryItem m : episodes) {
            Map<String, String> newTags = new LinkedHashMap<>(m.tags() == null ? Map.of() : m.tags());
            newTags.put(MemoryTags.REFLECTED_AT, stamp);
            store.updateTags(m.id(), newTags);
        }
    }

    private static String majorityTag(List<MemoryItem> items, String key, String fallback) {
        Map<String, Integer> counts = new HashMap<>();
        for (MemoryItem m : items) {
            String v = m.tags() == null ? null : m.tags().get(key);
            if (v == null || v.isBlank()) continue;
            counts.merge(v, 1, Integer::sum);
        }
        if (counts.isEmpty()) return fallback;
        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(fallback);
    }

    @Override
    public void close() {
        scheduler.shutdownNow();
        try { scheduler.awaitTermination(5, TimeUnit.SECONDS); }
        catch (InterruptedException ignore) { Thread.currentThread().interrupt(); }
    }
}
