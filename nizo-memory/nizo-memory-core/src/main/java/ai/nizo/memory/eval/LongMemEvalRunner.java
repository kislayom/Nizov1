package ai.nizo.memory.eval;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.MemoryFactory;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.config.ConfigLoader;
import ai.nizo.memory.config.NizoConfig;
import ai.nizo.memory.llm.OllamaModelClient;
import ai.nizo.memory.session.LlmSessionPicker;
import ai.nizo.memory.util.Json;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

/**
 * CLI entry point for running LongMemEval against a live nizo-memory stack.
 *
 * <pre>
 * java -cp nizo-memory-core.jar ai.nizo.memory.eval.LongMemEvalRunner \
 *      --config nizo-memory.yaml \
 *      --dataset ~/data/longmemeval_s.jsonl \
 *      --limit 50 \
 *      --out report.json \
 *      --answer-model qwen2.5:14b \
 *      --judge-model qwen2.5:14b
 * </pre>
 *
 * <p>Produces a JSON report to stdout (or to --out) with overall accuracy
 * and per-question-type breakdown. Intended for periodic benchmarking
 * against the paper's dataset.
 */
public final class LongMemEvalRunner {

    public static void main(String[] args) throws Exception {
        String configPath = "nizo-memory.yaml";
        String datasetPath = null;
        int limit = 0;
        String outPath = null;
        String ollamaUrl = "http://localhost:11434";
        String answerModel = "qwen2.5:14b";
        String judgeModel = "qwen2.5:14b";
        int tokenBudget = 1200;
        boolean thinking = false;
        int answererTimeoutSec = 300;  // default generous — thinking traces can add 1000-5000 tokens
        boolean sessionPicker = false;
        String pickerModel = null;     // defaults to the answer model if unset
        int pickerThreshold = 10;      // kick in only when there are > N sessions
        int pickerTopN = 5;            // narrow to this many sessions per query

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--config" -> configPath = args[++i];
                case "--dataset" -> datasetPath = args[++i];
                case "--limit" -> limit = Integer.parseInt(args[++i]);
                case "--out" -> outPath = args[++i];
                case "--ollama" -> ollamaUrl = args[++i];
                case "--answer-model" -> answerModel = args[++i];
                case "--judge-model" -> judgeModel = args[++i];
                case "--token-budget" -> tokenBudget = Integer.parseInt(args[++i]);
                case "--thinking" -> thinking = true;
                case "--answerer-timeout" -> answererTimeoutSec = Integer.parseInt(args[++i]);
                case "--session-picker" -> sessionPicker = true;
                case "--picker-model" -> pickerModel = args[++i];
                case "--picker-threshold" -> pickerThreshold = Integer.parseInt(args[++i]);
                case "--picker-top-n" -> pickerTopN = Integer.parseInt(args[++i]);
                case "-h", "--help" -> {
                    System.out.println("""
                        LongMemEvalRunner

                        Args:
                          --config <path>     nizo-memory.yaml (default: nizo-memory.yaml)
                          --dataset <path>    LongMemEval JSONL dataset (required)
                          --limit <N>         cap number of items (default: all)
                          --out <path>        write JSON report (default: stdout)
                          --ollama <url>      Ollama base URL (default: http://localhost:11434)
                          --answer-model      model for the answerer (default: qwen2.5:14b)
                          --judge-model       model for the LLM-as-judge grader (default: qwen2.5:14b)
                          --token-budget      recall token budget (default: 1200)
                          --thinking          enable Qwen3 /think chain-of-thought on the answerer
                          --answerer-timeout  answerer HTTP timeout in seconds (default: 300)
                          --session-picker    enable LLM session-picker pre-filter for recall
                          --picker-model      model for the session picker (default: answer model)
                          --picker-threshold  min distinct sessions before picker runs (default: 10)
                          --picker-top-n      number of sessions picker narrows to (default: 5)
                        """);
                    return;
                }
                default -> System.err.println("Unknown flag: " + args[i]);
            }
        }

        if (datasetPath == null) {
            System.err.println("--dataset is required. Try --help.");
            System.exit(2);
        }

        NizoConfig config = ConfigLoader.loadOrDefault(Path.of(configPath));
        MemoryFactory.Bundle bundle = MemoryFactory.buildBundle(config);
        MemoryService memory = bundle.service();
        EmbeddingClient embedder = MemoryFactory.embedderFromConfig(config);
        ExtractionService extraction = null;
        if (config.graph().enabled() && config.extraction() != null && config.extraction().enabled()
                && bundle.graph() != null) {
            extraction = MemoryFactory.extractionFromConfig(config, bundle.graph(), memory);
        }

        ModelClient answerer = new OllamaModelClient(ollamaUrl, answerModel,
                0.1, Duration.ofSeconds(answererTimeoutSec));
        ModelClient judge = new OllamaModelClient(ollamaUrl, judgeModel,
                0.0, Duration.ofSeconds(60));

        // Session-picker pre-filter (multi-session recall fix). Shares the
        // answerer's HTTP config but uses its own model at temperature 0 so
        // session selection is deterministic across runs.
        if (sessionPicker && memory instanceof LayeredMemoryService lms) {
            String pm = pickerModel == null || pickerModel.isBlank() ? answerModel : pickerModel;
            ModelClient pickerClient = new OllamaModelClient(ollamaUrl, pm,
                    0.0, Duration.ofSeconds(60));
            lms.setSessionPicker(new LlmSessionPicker(pickerClient),
                    pickerThreshold, pickerTopN);
            System.err.printf("Session picker enabled: model=%s threshold=%d topN=%d%n",
                    pm, pickerThreshold, pickerTopN);
        }

        System.err.printf("Loading dataset: %s (limit=%d)%n", datasetPath, limit);
        List<LongMemEvalHarness.Item> items =
                LongMemEvalHarness.loadDataset(Path.of(datasetPath), limit);
        System.err.printf("Running %d items against memory; embedder=%s, extraction=%s, thinking=%s, picker=%s%n",
                items.size(),
                embedder == null ? "none" : embedder.getClass().getSimpleName(),
                extraction == null ? "off" : "on",
                thinking,
                sessionPicker);

        LongMemEvalHarness harness = new LongMemEvalHarness(
                memory, extraction, answerer, judge, tokenBudget, thinking);
        long t0 = System.currentTimeMillis();
        LongMemEvalHarness.Report report = harness.run(items);
        long elapsed = System.currentTimeMillis() - t0;

        System.err.println("---");
        System.err.println(report.summary());
        System.err.printf("Elapsed: %.1fs (%.1fs/item)%n",
                elapsed / 1000.0, elapsed / 1000.0 / Math.max(1, items.size()));

        String json = Json.stringify(report);
        if (outPath != null) {
            Files.writeString(Path.of(outPath), json);
            System.err.println("Report written to " + outPath);
        } else {
            System.out.println(json);
        }
    }
}
