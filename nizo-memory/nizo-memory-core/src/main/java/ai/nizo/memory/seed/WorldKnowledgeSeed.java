package ai.nizo.memory.seed;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Seeds default world knowledge (common-sense heuristics) into a user's
 * PROCEDURAL memory. Called once per user — checks for an existing seed marker
 * before re-loading.
 *
 * <p>The heuristics live in {@code resources/world-knowledge-seed.yaml} and
 * cover extraction guards, terminology, currency, relationships, health,
 * finance, tech, and dates. The {@link ai.nizo.memory.extract.ExtractionPipeline}
 * pulls these as additional context for every extraction so the LLM has
 * common-sense guardrails (e.g., "iPhone user is a phone preference, not a job").
 *
 * <p>Source confidence: world heuristics are stored at confidence 0.95 with
 * source="world_knowledge". Higher than typical "extracted" facts (0.7-0.85)
 * but lower than "user_stated" (0.95) — so a user can correct a world rule
 * if needed.
 */
public final class WorldKnowledgeSeed {

    private static final Logger log = LoggerFactory.getLogger(WorldKnowledgeSeed.class);

    private static final String SEED_RESOURCE = "/world-knowledge-seed.yaml";
    private static final String CURATOR_PROMPT_RESOURCE = "/world-knowledge-curator-prompt.md";
    private static final String SEED_MARKER_PREFIX = "WORLD_KNOWLEDGE_SEEDED_";
    private static final String SEED_SOURCE = "world_knowledge";
    private static final double SEED_CONFIDENCE = 0.95;

    /**
     * Bump this when shipping new defaults — old marker will not match,
     * so the new bundle gets seeded. Old seeded items remain (with the
     * old version's source) and can be cleaned up by a separate sweeper.
     *
     * <p>v2 (this release): expanded from 39 → 320+ heuristics across 25
     * categories (geography, famous people, companies, cinema, music,
     * hobbies, lifestyle brands, science, math, history, travel, climate,
     * privacy, internet culture). Geographic balance — India / US / UK / EU
     * / China / Japan / Korea / Mexico / LATAM / SEA / Africa / ME / Nordics
     * / Eastern Europe / Oceania.
     */
    public static final String CURRENT_SEED_VERSION = "v2";

    private WorldKnowledgeSeed() {}

    /**
     * Seed the bundled default heuristics for {@code userId} if not already
     * done at the current version. Idempotent — re-running is a no-op until
     * {@link #CURRENT_SEED_VERSION} is bumped.
     *
     * @return number of heuristics loaded (0 if already seeded)
     */
    public static int seedIfNeeded(MemoryService memory, String userId) {
        return seedIfNeeded(memory, userId, CURRENT_SEED_VERSION);
    }

    /**
     * Seed at a specific version marker. Useful when shipping a fresh bundle —
     * pass a new version string to re-seed without manual cleanup.
     */
    /**
     * System property to skip seeding entirely. Set
     * {@code -Dnizo.seed.skip-world-knowledge=true} to make every
     * {@link #seedIfNeeded} call a no-op. Designed for benchmarks where
     * {@code forgetUser} runs between items: with seeding on, we re-upload
     * 326 heuristics per item which dominates wall-clock time
     * (170s/item vs 10s/item observed on LongMemEval oracle-48 with
     * extraction enabled, GPU underutilised at 17%).
     *
     * <p>Production callers leave this unset — the per-user seed only
     * happens on first ingest for that user and is then a marker-based
     * no-op forever after, so normal usage isn't impacted.
     */
    public static final String SKIP_PROPERTY = "nizo.seed.skip-world-knowledge";

    public static int seedIfNeeded(MemoryService memory, String userId, String version) {
        if (memory == null) return 0;
        if ("true".equalsIgnoreCase(System.getProperty(SKIP_PROPERTY))) {
            log.debug("World-knowledge seeding skipped via {}=true (user {})",
                    SKIP_PROPERTY, userId);
            return 0;
        }
        try {
            String marker = SEED_MARKER_PREFIX + version;
            if (hasMarker(memory, userId, marker)) {
                log.debug("World knowledge {} already seeded for user {}", version, userId);
                return 0;
            }
            List<Heuristic> heuristics = loadHeuristicsFromClasspath();
            int loaded = uploadBatch(memory, userId, heuristics, SEED_SOURCE, SEED_CONFIDENCE);
            memory.learnProcedural(userId, marker, SEED_SOURCE, SEED_CONFIDENCE);
            log.info("Seeded {} world-knowledge heuristics ({}) for user {}",
                    loaded, version, userId);
            return loaded;
        } catch (Exception e) {
            log.warn("Failed to seed world knowledge for user {}: {}", userId, e.toString());
            return 0;
        }
    }

    /**
     * Batch-upload a set of heuristics for a user. Used by Nizo (the agent
     * layer) when refreshing world knowledge from a remote source — e.g.,
     * pulling an updated YAML and pushing the delta. Caller decides the
     * source label and confidence (defaults to "world_knowledge"/0.95).
     *
     * <p>This method is idempotent at the content level — re-uploading the
     * same heuristic creates a duplicate row but the LayeredMemoryService's
     * MMR dedup will collapse near-identical items at recall time.
     *
     * @return number of heuristics uploaded
     */
    public static int uploadBatch(MemoryService memory, String userId,
                                   List<Heuristic> heuristics) {
        return uploadBatch(memory, userId, heuristics, SEED_SOURCE, SEED_CONFIDENCE);
    }

    public static int uploadBatch(MemoryService memory, String userId,
                                   List<Heuristic> heuristics,
                                   String source, double confidence) {
        if (memory == null || heuristics == null || heuristics.isEmpty()) return 0;
        int loaded = 0;
        for (Heuristic h : heuristics) {
            if (h == null || h.text() == null || h.text().isBlank()) continue;
            memory.learnProcedural(userId, h.text(), source, confidence);
            loaded++;
        }
        log.info("Uploaded batch of {} heuristics for user {} (source={})",
                loaded, userId, source);
        return loaded;
    }

    /**
     * Load heuristics from a YAML file path (for batch refresh from disk).
     * Format matches {@link #SEED_RESOURCE} bundled with the JAR.
     */
    public static List<Heuristic> loadHeuristicsFromFile(java.nio.file.Path yamlPath) throws Exception {
        try (InputStream is = java.nio.file.Files.newInputStream(yamlPath)) {
            return parseHeuristicsYaml(is);
        }
    }

    /** Load the bundled default heuristics from the classpath. */
    public static List<Heuristic> loadHeuristicsFromClasspath() throws Exception {
        try (InputStream is = WorldKnowledgeSeed.class.getResourceAsStream(SEED_RESOURCE)) {
            if (is == null) {
                log.warn("World knowledge seed not on classpath: {}", SEED_RESOURCE);
                return List.of();
            }
            return parseHeuristicsYaml(is);
        }
    }

    /**
     * Load the bundled curator system prompt. Used by Nizo (or any external
     * curator) when calling an LLM to propose updates to this corpus over
     * time. Format: Markdown with a strict JSON output contract.
     *
     * @return raw markdown prompt (empty string if resource is missing)
     */
    public static String loadCuratorPrompt() {
        try (InputStream is = WorldKnowledgeSeed.class.getResourceAsStream(CURATOR_PROMPT_RESOURCE)) {
            if (is == null) {
                log.warn("Curator prompt not on classpath: {}", CURATOR_PROMPT_RESOURCE);
                return "";
            }
            return new String(is.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) {
            log.warn("Failed to load curator prompt: {}", e.toString());
            return "";
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Heuristic> parseHeuristicsYaml(InputStream is) throws Exception {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        Map<String, Object> root = mapper.readValue(is, Map.class);
        Object raw = root.get("heuristics");
        if (!(raw instanceof List<?> list)) return List.of();
        List<Heuristic> out = new java.util.ArrayList<>();
        for (Object entry : list) {
            if (entry instanceof Map<?, ?> m) {
                String text = String.valueOf(m.get("text"));
                String category = m.get("category") == null ? "general"
                        : String.valueOf(m.get("category"));
                out.add(new Heuristic(text, category));
            }
        }
        return out;
    }

    private static boolean hasMarker(MemoryService memory, String userId, String marker) {
        try {
            var req = new RecallRequest(
                    userId, marker, 100,
                    Set.of(MemoryItem.Tier.PROCEDURAL),
                    Map.of(), 0.0);
            var hits = memory.recall(req);
            for (var item : hits) {
                if (marker.equals(item.content())) return true;
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    /** A single seeded heuristic loaded from YAML. */
    public record Heuristic(String text, String category) {}
}
