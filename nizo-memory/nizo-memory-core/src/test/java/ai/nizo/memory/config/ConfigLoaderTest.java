package ai.nizo.memory.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ConfigLoader} and {@link NizoConfig}. We exercise:
 *
 * <ul>
 *   <li>loading defaults when no file is present;</li>
 *   <li>parsing a fully-populated YAML and seeing every field round-trip
 *       through Jackson's SNAKE_CASE strategy;</li>
 *   <li>disabling every LLM-using stage at once;</li>
 *   <li>the mixed case of "embedder on, everything else off";</li>
 *   <li>the loadOrDefault swallowing parse errors.</li>
 * </ul>
 *
 * <p>None of these tests touch a real Ollama endpoint — they only validate
 * config loading and the wiring of records.
 */
class ConfigLoaderTest {

    // ----- defaults --------------------------------------------------------

    @Test
    void loadReturnsDefaultsWhenFileMissing(@TempDir Path tmp) {
        Path missing = tmp.resolve("does-not-exist.yaml");

        NizoConfig cfg = ConfigLoader.load(missing);

        assertNotNull(cfg);
        assertEquals("sqlite", cfg.storage().backend());
        assertEquals("${user.data}/memory.db", cfg.storage().path());
        assertEquals(0.55, cfg.recall().minSimilarityFloorOr(-1));
        assertEquals(0.55, cfg.recall().minTopScoreOr(-1));
        assertEquals("inmemory", cfg.vector().backend());
        assertTrue(cfg.embedder().enabled());
        assertEquals("ollama", cfg.embedder().backend());
        assertEquals("http://localhost:11434", cfg.embedder().ollama().baseUrl());
        assertEquals("nomic-embed-text", cfg.embedder().ollama().model());
        assertTrue(cfg.extraction().enabled());
        assertTrue(cfg.consolidation().enabled());
        assertEquals(12, cfg.consolidation().consolidateEveryN());
        assertTrue(cfg.compaction().enabled());
        assertEquals(13_000, cfg.compaction().reserveBufferTokens());
        assertFalse(cfg.verifier().enabled(), "verifier is OFF by default");
        assertEquals(5, cfg.verifier().maxFactsToVerify());
        assertEquals(0.2, cfg.verifier().outdatedPenalty(), 1e-9);
        assertEquals(0.7, cfg.verifier().unverifiablePenalty(), 1e-9);
        assertTrue(cfg.graph().enabled());
        assertEquals(1200, cfg.recall().defaultTokenBudget());
        assertEquals(0.1, cfg.recall().confidenceFloor(), 1e-9);
        assertEquals(8765, cfg.server().port());
        assertEquals(8, cfg.server().threads());
    }

    @Test
    void defaultsRecordHasNoNullSections() {
        NizoConfig d = NizoConfig.defaults();

        assertAll(
                () -> assertNotNull(d.storage()),
                () -> assertNotNull(d.vector()),
                () -> assertNotNull(d.embedder()),
                () -> assertNotNull(d.embedder().ollama()),
                () -> assertNotNull(d.embedder().onnx()),
                () -> assertNotNull(d.extraction()),
                () -> assertNotNull(d.extraction().ollama()),
                () -> assertNotNull(d.consolidation()),
                () -> assertNotNull(d.consolidation().ollama()),
                () -> assertNotNull(d.compaction()),
                () -> assertNotNull(d.compaction().ollama()),
                () -> assertNotNull(d.verifier()),
                () -> assertNotNull(d.verifier().ollama()),
                () -> assertNotNull(d.graph()),
                () -> assertNotNull(d.recall()),
                () -> assertNotNull(d.server())
        );
    }

    // ----- full YAML round-trip --------------------------------------------

    @Test
    void loadParsesFullYamlConfig(@TempDir Path tmp) throws IOException {
        Path yaml = tmp.resolve("config.yaml");
        Files.writeString(yaml, """
                storage:
                  backend: sqlite
                  path: /var/lib/nizo/memory.db

                vector:
                  backend: inmemory

                embedder:
                  enabled: true
                  backend: ollama
                  ollama:
                    base_url: http://ollama.example.com:11434
                    model: nomic-embed-text
                    timeout_seconds: 45
                  onnx:
                    model_dir: /opt/models/all-MiniLM-L6-v2
                    max_seq_len: 256

                extraction:
                  enabled: true
                  backend: ollama
                  ollama:
                    base_url: http://ollama.example.com:11434
                    model: qwen2.5:14b
                    temperature: 0.05
                    timeout_seconds: 90

                consolidation:
                  enabled: true
                  consolidate_every_n: 24
                  backend: ollama
                  ollama:
                    base_url: http://ollama.example.com:11434
                    model: qwen2.5:7b

                compaction:
                  enabled: true
                  backend: ollama
                  ollama:
                    base_url: http://ollama.example.com:11434
                    model: qwen2.5:7b
                  reserve_buffer_tokens: 16000

                verifier:
                  enabled: true
                  backend: ollama
                  ollama:
                    base_url: http://ollama.example.com:11434
                    model: qwen2.5:3b
                  max_facts_to_verify: 7
                  outdated_penalty: 0.15
                  unverifiable_penalty: 0.6

                graph:
                  enabled: true

                recall:
                  default_token_budget: 1800
                  confidence_floor: 0.25

                server:
                  port: 9999
                  threads: 16
                """);

        NizoConfig cfg = ConfigLoader.load(yaml);

        assertEquals("/var/lib/nizo/memory.db", cfg.storage().path());
        assertEquals("inmemory", cfg.vector().backend());

        // embedder
        assertTrue(cfg.embedder().enabled());
        assertEquals("http://ollama.example.com:11434", cfg.embedder().ollama().baseUrl());
        assertEquals(45, cfg.embedder().ollama().timeoutSeconds());
        assertEquals("/opt/models/all-MiniLM-L6-v2", cfg.embedder().onnx().modelDir());
        assertEquals(256, cfg.embedder().onnx().maxSeqLen());

        // extraction
        assertTrue(cfg.extraction().enabled());
        assertEquals("qwen2.5:14b", cfg.extraction().ollama().model());
        assertEquals(0.05, cfg.extraction().ollama().temperature(), 1e-9);
        assertEquals(90, cfg.extraction().ollama().timeoutSeconds());

        // consolidation
        assertEquals(24, cfg.consolidation().consolidateEveryN());

        // compaction
        assertEquals(16_000, cfg.compaction().reserveBufferTokens());

        // verifier
        assertTrue(cfg.verifier().enabled());
        assertEquals(7, cfg.verifier().maxFactsToVerify());
        assertEquals(0.15, cfg.verifier().outdatedPenalty(), 1e-9);
        assertEquals(0.6, cfg.verifier().unverifiablePenalty(), 1e-9);

        // graph + recall + server
        assertTrue(cfg.graph().enabled());
        assertEquals(1800, cfg.recall().defaultTokenBudget());
        assertEquals(0.25, cfg.recall().confidenceFloor(), 1e-9);
        assertEquals(9999, cfg.server().port());
        assertEquals(16, cfg.server().threads());
    }

    // ----- everything off --------------------------------------------------

    @Test
    void allLlmStagesCanBeDisabled(@TempDir Path tmp) throws IOException {
        Path yaml = tmp.resolve("off.yaml");
        Files.writeString(yaml, """
                embedder:
                  enabled: false
                  backend: none

                extraction:
                  enabled: false
                  backend: none

                consolidation:
                  enabled: false
                  consolidate_every_n: 12
                  backend: none

                compaction:
                  enabled: false
                  backend: none
                  reserve_buffer_tokens: 13000

                verifier:
                  enabled: false
                  backend: none
                  max_facts_to_verify: 5
                  outdated_penalty: 0.2
                  unverifiable_penalty: 0.7

                graph:
                  enabled: false
                """);

        NizoConfig cfg = ConfigLoader.load(yaml);

        assertFalse(cfg.embedder().enabled());
        assertFalse(cfg.extraction().enabled());
        assertFalse(cfg.consolidation().enabled());
        assertFalse(cfg.compaction().enabled());
        assertFalse(cfg.verifier().enabled());
        assertFalse(cfg.graph().enabled());
        // Sections we omitted should fall through to defaults.
        assertNotNull(cfg.storage());
        assertEquals("sqlite", cfg.storage().backend());
        assertNotNull(cfg.recall());
        assertNotNull(cfg.server());
    }

    // ----- mixed case ------------------------------------------------------

    @Test
    void enableOnlyEmbedderEverythingElseOff(@TempDir Path tmp) throws IOException {
        Path yaml = tmp.resolve("mixed.yaml");
        Files.writeString(yaml, """
                embedder:
                  enabled: true
                  backend: ollama
                  ollama:
                    base_url: http://localhost:11434
                    model: nomic-embed-text

                extraction:
                  enabled: false
                  backend: none

                consolidation:
                  enabled: false
                  consolidate_every_n: 0
                  backend: none

                compaction:
                  enabled: false
                  backend: none
                  reserve_buffer_tokens: 0

                verifier:
                  enabled: false
                  backend: none
                  max_facts_to_verify: 0
                  outdated_penalty: 0
                  unverifiable_penalty: 0
                """);

        NizoConfig cfg = ConfigLoader.load(yaml);

        assertTrue(cfg.embedder().enabled());
        assertEquals("nomic-embed-text", cfg.embedder().ollama().model());
        assertFalse(cfg.extraction().enabled());
        assertFalse(cfg.consolidation().enabled());
        assertFalse(cfg.compaction().enabled());
        assertFalse(cfg.verifier().enabled());
    }

    // ----- partial sections fill in via withDefaults() ---------------------

    @Test
    void omittedSectionsFallBackToDefaults(@TempDir Path tmp) throws IOException {
        Path yaml = tmp.resolve("partial.yaml");
        // Only override server port — everything else should use defaults.
        Files.writeString(yaml, """
                server:
                  port: 7000
                  threads: 4
                """);

        NizoConfig cfg = ConfigLoader.load(yaml);

        assertEquals(7000, cfg.server().port());
        assertEquals(4, cfg.server().threads());
        // Default sections still present
        assertNotNull(cfg.storage());
        assertNotNull(cfg.embedder());
        assertNotNull(cfg.compaction());
        assertEquals(13_000, cfg.compaction().reserveBufferTokens());
    }

    // ----- error handling --------------------------------------------------

    @Test
    void loadOrDefaultSwallowsParseErrors(@TempDir Path tmp) throws IOException {
        Path bogus = tmp.resolve("bad.yaml");
        // Invalid YAML — unclosed mapping
        Files.writeString(bogus, "embedder: { enabled: true,\n");

        // load() throws, loadOrDefault() falls back.
        assertThrows(ConfigLoader.ConfigException.class, () -> ConfigLoader.load(bogus));

        NizoConfig cfg = ConfigLoader.loadOrDefault(bogus);
        assertNotNull(cfg);
        assertTrue(cfg.embedder().enabled(),
                "loadOrDefault should fall through to defaults on parse failure");
    }

    @Test
    void loadOrDefaultReturnsDefaultsForMissingFile(@TempDir Path tmp) {
        NizoConfig cfg = ConfigLoader.loadOrDefault(tmp.resolve("nope.yaml"));
        assertNotNull(cfg);
        assertEquals(8765, cfg.server().port());
    }

    // ----- factory wiring sanity -------------------------------------------

    @Test
    void factoryCanWireMemoryFromDefaults(@TempDir Path tmp) {
        // Use a fresh DB path so we don't pollute the repo.
        NizoConfig cfg = ConfigLoader.loadOrDefault(tmp.resolve("none.yaml"));
        // Disable LLM-backed embedder so the factory call doesn't reach Ollama.
        NizoConfig adjusted = new NizoConfig(
                new NizoConfig.StorageConfig("sqlite", tmp.resolve("mem.db").toString()),
                cfg.vector(),
                new NizoConfig.EmbedderConfig(false, "none", cfg.embedder().ollama(),
                        cfg.embedder().onnx()),
                new NizoConfig.ExtractionConfig(false, "none", cfg.extraction().ollama()),
                new NizoConfig.ConsolidationConfig(false, 12, "none", cfg.consolidation().ollama()),
                new NizoConfig.CompactionConfig(false, "none", cfg.compaction().ollama(), 13000),
                new NizoConfig.VerifierConfig(false, "none", cfg.verifier().ollama(),
                        5, 0.2, 0.7),
                new NizoConfig.GraphConfig(false),
                cfg.recall(),
                cfg.server(),
                new NizoConfig.ReflectionConfig(false, 60, 2, 6, 40, 0.92, "none",
                        cfg.reflection() == null ? null : cfg.reflection().ollama()));

        var memory = ai.nizo.memory.MemoryFactory.fromConfig(adjusted);
        assertNotNull(memory);
        // remember/recall round-trip with all LLMs off (FTS path only)
        memory.remember("default", "User likes oolong tea", java.util.Map.of(), "test");
        var hits = memory.recall(
                ai.nizo.memory.api.memory.RecallRequest.of("default", "oolong", 400));
        assertFalse(hits.isEmpty(), "FTS-only recall should still find the item");
    }
}
