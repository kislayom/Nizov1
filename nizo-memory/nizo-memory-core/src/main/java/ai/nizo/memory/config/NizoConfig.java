package ai.nizo.memory.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * Strongly-typed view of {@code nizo-memory.yaml}.
 *
 * <p>All sub-records carry the full configuration for one concern (storage,
 * embedder, extraction, etc.). Anything you omit from the YAML falls back to
 * the matching field in {@link #defaults()}.
 *
 * <p>The Jackson YAML mapper is configured with
 * {@link com.fasterxml.jackson.databind.PropertyNamingStrategies#SNAKE_CASE},
 * so wire-protocol keys like {@code consolidate_every_n} bind to Java's
 * {@code consolidateEveryN}. Records are annotated
 * {@code @JsonIgnoreProperties(ignoreUnknown = true)} so a forward-compatible
 * config file with newer keys still loads on an older binary.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record NizoConfig(
        StorageConfig storage,
        VectorConfig vector,
        EmbedderConfig embedder,
        ExtractionConfig extraction,
        ConsolidationConfig consolidation,
        CompactionConfig compaction,
        VerifierConfig verifier,
        GraphConfig graph,
        RecallConfig recall,
        ServerConfig server,
        ReflectionConfig reflection
) {

    // --- inner records -----------------------------------------------------

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record StorageConfig(String backend, String path) {
        /**
         * Default path is {@code ~/.nizo/memory.db} — under the user's home
         * directory so the SQLite file (and every fact, graph edge, and
         * seeded heuristic in it) survives a JAR reinstall. Override in
         * {@code nizo-memory.yaml} via {@code storage.path}; tokens
         * {@code ~/} and {@code ${user.data}} are expanded by
         * {@link ai.nizo.memory.util.DataPaths#resolve(String)}.
         */
        public static StorageConfig defaults() {
            return new StorageConfig("sqlite", "${user.data}/memory.db");
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record VectorConfig(String backend) {
        public static VectorConfig defaults() {
            return new VectorConfig("inmemory");
        }
    }

    /**
     * Common bag for any Ollama endpoint. Fields are boxed so the YAML can
     * leave them out and we can fall through to the defaults baked into the
     * code that uses this config.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record OllamaConfig(
            String baseUrl,
            String model,
            Double temperature,
            Integer timeoutSeconds
    ) {
        public String baseUrlOr(String fallback) {
            return baseUrl == null || baseUrl.isBlank() ? fallback : baseUrl;
        }

        public String modelOr(String fallback) {
            return model == null || model.isBlank() ? fallback : model;
        }

        public double temperatureOr(double fallback) {
            return temperature == null ? fallback : temperature;
        }

        public int timeoutSecondsOr(int fallback) {
            return timeoutSeconds == null ? fallback : timeoutSeconds;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record OnnxConfig(String modelDir, Integer maxSeqLen) {
        public static OnnxConfig defaults() {
            return new OnnxConfig("models/all-MiniLM-L6-v2", 128);
        }

        public int maxSeqLenOr(int fallback) {
            return maxSeqLen == null ? fallback : maxSeqLen;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record EmbedderConfig(
            boolean enabled,
            String backend,
            OllamaConfig ollama,
            OnnxConfig onnx
    ) {
        public static EmbedderConfig defaults() {
            return new EmbedderConfig(
                    true,
                    "ollama",
                    new OllamaConfig("http://localhost:11434", "nomic-embed-text", null, 30),
                    OnnxConfig.defaults());
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ExtractionConfig(
            boolean enabled,
            String backend,
            OllamaConfig ollama
    ) {
        public static ExtractionConfig defaults() {
            return new ExtractionConfig(
                    true,
                    "ollama",
                    new OllamaConfig("http://localhost:11434", "qwen2.5:14b", 0.1, 60));
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ConsolidationConfig(
            boolean enabled,
            int consolidateEveryN,
            String backend,
            OllamaConfig ollama
    ) {
        public static ConsolidationConfig defaults() {
            return new ConsolidationConfig(
                    true, 12, "ollama",
                    new OllamaConfig("http://localhost:11434", "qwen2.5:14b", null, 60));
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record CompactionConfig(
            boolean enabled,
            String backend,
            OllamaConfig ollama,
            int reserveBufferTokens
    ) {
        public static CompactionConfig defaults() {
            return new CompactionConfig(
                    true, "ollama",
                    new OllamaConfig("http://localhost:11434", "qwen2.5:14b", null, 60),
                    13_000);
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record VerifierConfig(
            boolean enabled,
            String backend,
            OllamaConfig ollama,
            int maxFactsToVerify,
            double outdatedPenalty,
            double unverifiablePenalty
    ) {
        public static VerifierConfig defaults() {
            return new VerifierConfig(
                    false, "ollama",
                    new OllamaConfig("http://localhost:11434", "qwen2.5:3b", null, 30),
                    5, 0.2, 0.7);
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record GraphConfig(boolean enabled) {
        public static GraphConfig defaults() {
            return new GraphConfig(true);
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RecallConfig(
            int defaultTokenBudget,
            double confidenceFloor,
            Double minSimilarityFloor,
            Double minTopScore
    ) {
        /**
         * Defaults calibrated for nomic-embed-text (the recommended embedder):
         * <ul>
         *   <li>{@code min_similarity_floor=0.60} — raw cosine below which a
         *       vector hit is treated as noise. Nomic's "nearest neighbour is
         *       always 0.4-0.55 similar" means 0.55 lets too many weak hits
         *       through the door; 0.60 closes the 2-item leaks we saw on
         *       fully unrelated queries like "what's my LinkedIn URL".</li>
         *   <li>{@code min_top_score=0.55} — if the top-ranked result scores
         *       below this, drop the entire result (return empty).</li>
         * </ul>
         * For FakeEmbedder in tests, pass 0.01 / 0.0 via the constructors.
         */
        public static RecallConfig defaults() {
            return new RecallConfig(1200, 0.1, 0.55, 0.55);
        }

        public double minSimilarityFloorOr(double fallback) {
            return minSimilarityFloor == null ? fallback : minSimilarityFloor;
        }

        public double minTopScoreOr(double fallback) {
            return minTopScore == null ? fallback : minTopScore;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ServerConfig(int port, int threads) {
        public static ServerConfig defaults() {
            return new ServerConfig(8765, 8);
        }
    }

    /**
     * F1 — Reflection / summarization worker config. Periodically distills
     * EPISODIC events into SEMANTIC facts. Disable ({@code enabled: false})
     * to opt out.
     *
     * <ul>
     *   <li>{@code tickIntervalMinutes}: how often the worker wakes up</li>
     *   <li>{@code minEpisodeAgeHours}: only episodes older than this are
     *       eligible for reflection (gives the extraction pipeline a
     *       chance to tag them first)</li>
     *   <li>{@code minEpisodesPerTick}: skip users with fewer than this
     *       many unreflected episodes (not worth an LLM call)</li>
     *   <li>{@code maxEpisodesPerBatch}: cap per LLM call</li>
     *   <li>{@code duplicateThreshold}: cosine above which a candidate
     *       fact is considered already-known</li>
     * </ul>
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ReflectionConfig(
            boolean enabled,
            Integer tickIntervalMinutes,
            Integer minEpisodeAgeHours,
            Integer minEpisodesPerTick,
            Integer maxEpisodesPerBatch,
            Double duplicateThreshold,
            String backend,
            OllamaConfig ollama
    ) {
        public static ReflectionConfig defaults() {
            return new ReflectionConfig(
                    true, 60, 2, 6, 40, 0.92, "ollama",
                    new OllamaConfig("http://localhost:11434", "qwen2.5:14b", null, 60));
        }

        public int tickIntervalMinutesOr(int fb) {
            return tickIntervalMinutes == null ? fb : tickIntervalMinutes;
        }
        public int minEpisodeAgeHoursOr(int fb) {
            return minEpisodeAgeHours == null ? fb : minEpisodeAgeHours;
        }
        public int minEpisodesPerTickOr(int fb) {
            return minEpisodesPerTick == null ? fb : minEpisodesPerTick;
        }
        public int maxEpisodesPerBatchOr(int fb) {
            return maxEpisodesPerBatch == null ? fb : maxEpisodesPerBatch;
        }
        public double duplicateThresholdOr(double fb) {
            return duplicateThreshold == null ? fb : duplicateThreshold;
        }
    }

    // --- defaults / merging -----------------------------------------------

    /** Fully-populated config used when no YAML file is present. */
    public static NizoConfig defaults() {
        return new NizoConfig(
                StorageConfig.defaults(),
                VectorConfig.defaults(),
                EmbedderConfig.defaults(),
                ExtractionConfig.defaults(),
                ConsolidationConfig.defaults(),
                CompactionConfig.defaults(),
                VerifierConfig.defaults(),
                GraphConfig.defaults(),
                RecallConfig.defaults(),
                ServerConfig.defaults(),
                ReflectionConfig.defaults());
    }

    /**
     * Return a copy of this config with any null section filled from
     * {@link #defaults()}. Lets the YAML file omit whole sections without
     * bombing the loader downstream.
     */
    public NizoConfig withDefaults() {
        NizoConfig d = defaults();
        return new NizoConfig(
                storage      == null ? d.storage      : storage,
                vector       == null ? d.vector       : vector,
                embedder     == null ? d.embedder     : embedder,
                extraction   == null ? d.extraction   : extraction,
                consolidation== null ? d.consolidation: consolidation,
                compaction   == null ? d.compaction   : compaction,
                verifier     == null ? d.verifier     : verifier,
                graph        == null ? d.graph        : graph,
                recall       == null ? d.recall       : recall,
                server       == null ? d.server       : server,
                reflection   == null ? d.reflection   : reflection);
    }
}
