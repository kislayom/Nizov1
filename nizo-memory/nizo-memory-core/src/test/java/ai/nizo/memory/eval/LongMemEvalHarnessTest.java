package ai.nizo.memory.eval;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.testsupport.FakeEmbedder;
import ai.nizo.memory.testsupport.FakeModelClient;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * F16 — Proves the LongMemEval harness round-trips end-to-end against a
 * synthetic 3-item mini dataset. No Ollama required.
 *
 * <p>To run against the real dataset once downloaded:
 * <pre>
 * java -cp nizo-memory-core.jar ai.nizo.memory.eval.LongMemEvalRunner \
 *      --dataset ~/data/longmemeval_s.jsonl --limit 50 --out report.json
 * </pre>
 */
class LongMemEvalHarnessTest {

    @Test
    void harness_endToEnd_onSyntheticDataset(@TempDir Path tmp) throws Exception {
        // Build a tiny in-process memory stack.
        SqliteMemoryStore store = new SqliteMemoryStore(tmp.resolve("eval.db"));
        InMemoryVectorIndex index = new InMemoryVectorIndex();
        FakeEmbedder embedder = new FakeEmbedder(List.of(
                "alice", "acme", "bob", "surgeon", "vegetarian", "bali",
                "stripe", "company", "working", "wedding", "ceremony",
                "product", "manager", "friend"));
        LayeredMemoryService svc = new LayeredMemoryService(
                store, index, embedder, null, null, null,
                999_999, 0.0, 0.01, 0.0);

        // Answerer: returns the first recalled fact verbatim — simulates a
        // model that just quotes memory.
        FakeModelClient answerer = new FakeModelClient(prompt -> {
            int ctx = prompt.indexOf("MEMORY CONTEXT:");
            int q = prompt.indexOf("QUESTION:");
            if (ctx < 0 || q < 0) return "I don't know";
            String context = prompt.substring(ctx, q);
            // Extract first bullet
            int dash = context.indexOf("- ");
            if (dash < 0) return "I don't know";
            int nl = context.indexOf('\n', dash);
            return context.substring(dash + 2, nl < 0 ? context.length() : nl).trim();
        });

        // Judge: says YES iff predicted contains the ground-truth answer as substring.
        FakeModelClient judge = new FakeModelClient(prompt -> {
            int gt = prompt.indexOf("GROUND TRUTH:");
            int ma = prompt.indexOf("MODEL ANSWER:");
            int end = prompt.indexOf("GRADE");
            if (gt < 0 || ma < 0 || end < 0) return "NO";
            String truth = prompt.substring(gt + "GROUND TRUTH:".length(), ma).trim();
            String pred = prompt.substring(ma + "MODEL ANSWER:".length(), end).trim();
            return pred.toLowerCase().contains(truth.toLowerCase()) ? "YES" : "NO";
        });

        // Synthetic 3-item dataset written as JSONL to a temp file.
        Path jsonl = tmp.resolve("mini.jsonl");
        Files.writeString(jsonl, String.join("\n",
                "{\"question_id\":\"q1\",\"question\":\"Where does Alice work?\",\"question_type\":\"single_session_user\"," +
                "\"sessions\":[[{\"role\":\"user\",\"content\":\"Alice works at Acme as a product manager.\"}]]," +
                "\"answer\":\"Acme\"}",
                "{\"question_id\":\"q2\",\"question\":\"Is Bob vegetarian?\",\"question_type\":\"single_session_user\"," +
                "\"sessions\":[[{\"role\":\"user\",\"content\":\"Bob went vegetarian last month.\"}]]," +
                "\"answer\":\"vegetarian\"}",
                "{\"question_id\":\"q3\",\"question\":\"Where was the wedding?\",\"question_type\":\"multi_session\"," +
                "\"sessions\":[[{\"role\":\"user\",\"content\":\"The wedding ceremony was in Bali.\"}]]," +
                "\"answer\":\"Bali\"}"));

        List<LongMemEvalHarness.Item> items = LongMemEvalHarness.loadDataset(jsonl, 10);
        assertEquals(3, items.size(), "dataset loader should find 3 items");

        // Harness — extraction=null so ingest falls back to remember().
        LongMemEvalHarness harness = new LongMemEvalHarness(
                svc, null, answerer, judge, 800);

        LongMemEvalHarness.Report report = harness.run(items);
        assertEquals(3, report.total());
        assertEquals(3, report.correct(),
                "all 3 synthetic items should pass with the trivial answerer/judge");
        assertEquals(1.0, report.accuracy(), 1e-6);

        // Per-type breakdown
        assertEquals(2, report.byType().get("single_session_user").total());
        assertEquals(2, report.byType().get("single_session_user").correct());
        assertEquals(1, report.byType().get("multi_session").total());

        // Summary readable
        String summary = report.summary();
        assertTrue(summary.contains("Overall: 3/3"), "summary: " + summary);
        assertTrue(summary.contains("single_session_user"), "summary should break down by type");
    }
}
