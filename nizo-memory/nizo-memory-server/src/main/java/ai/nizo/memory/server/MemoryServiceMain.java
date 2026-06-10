package ai.nizo.memory.server;

import ai.nizo.memory.MemoryFactory;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.compact.CompactionService;
import ai.nizo.memory.config.ConfigLoader;
import ai.nizo.memory.config.NizoConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * Runs the memory tier as an independent service.
 *
 * <p>Usage:
 * <pre>
 *   java -jar nizo-memory-server.jar [--config nizo-memory.yaml]
 * </pre>
 *
 * <p>All other knobs (port, db path, embedder, model backends) are now driven
 * by the YAML config — see {@code nizo-memory.yaml} at repo root for the full
 * schema. If {@code --config} isn't provided, the loader looks for
 * {@code nizo-memory.yaml} in the working directory; if that's also absent it
 * falls back to {@link NizoConfig#defaults()}.
 */
public final class MemoryServiceMain {

    private static final Logger log = LoggerFactory.getLogger(MemoryServiceMain.class);

    public static void main(String[] args) throws Exception {
        String configPath = "nizo-memory.yaml";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--config" -> configPath = args[++i];
                case "-h", "--help" -> {
                    printUsage();
                    return;
                }
                default -> log.warn("Unknown argument: {}", args[i]);
            }
        }

        NizoConfig config = ConfigLoader.loadOrDefault(Path.of(configPath));

        MemoryFactory.Bundle bundle = MemoryFactory.buildBundle(config);
        MemoryService memory = bundle.service();
        EmbeddingClient embedder = MemoryFactory.embedderFromConfig(config);
        CompactionService compaction = MemoryFactory.compactionFromConfig(config, memory);

        // Wire extraction if enabled in config + graph is available
        ExtractionService extraction = null;
        if (config.graph().enabled() && config.extraction() != null && config.extraction().enabled()
                && bundle.graph() != null) {
            extraction = MemoryFactory.extractionFromConfig(config, bundle.graph(), memory);
        }

        // F1: start the reflection worker if enabled. Uses the same store +
        // vector index as the main service so duplicate detection works.
        ai.nizo.memory.reflect.ReflectionService reflection =
                MemoryFactory.reflectionFromConfig(config, memory, bundle.store(),
                        bundle.index(), embedder);
        if (reflection != null) reflection.start();

        int port = config.server().port();
        int threads = Math.max(1, config.server().threads());

        MemoryHttpServer server = new MemoryHttpServer(memory, compaction, embedder, extraction, port, threads);
        server.start();

        log.info("nizo-memory-server started on port {} (db={}, threads={}, embedder={}, compaction={}, extraction={}, reflection={})",
                port,
                config.storage().path(),
                threads,
                embedder == null ? "none" : embedder.getClass().getSimpleName(),
                compaction == null ? "off" : "on",
                extraction == null ? "off" : "on",
                reflection == null ? "off" : "on");

        ai.nizo.memory.reflect.ReflectionService finalReflection = reflection;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop();
            if (finalReflection != null) finalReflection.close();
        }, "nizo-memory-shutdown"));
        Thread.currentThread().join();
    }

    private static void printUsage() {
        System.out.println("""
                Usage: nizo-memory-server [--config <path>]

                Options:
                  --config <path>   YAML config file (default: nizo-memory.yaml)
                  -h, --help        Show this message
                """);
    }
}
