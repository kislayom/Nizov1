package ai.nizo.memory.eval;

import ai.nizo.memory.LayeredMemoryService;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.util.Json;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * F16 — LongMemEval harness.
 *
 * <p>Runs the LongMemEval benchmark against the nizo-memory stack. For each
 * evaluation item:
 * <ol>
 *   <li>Create a fresh per-item userId (so nothing leaks across questions).</li>
 *   <li>Ingest every session turn through the extraction pipeline.</li>
 *   <li>Recall against the question text; pack top-K facts into context.</li>
 *   <li>Ask an answer LLM to answer using ONLY the recalled context.</li>
 *   <li>Grade the answer with an LLM-as-judge comparing to ground truth.</li>
 * </ol>
 *
 * <p>Emits a per-question-type accuracy breakdown.
 *
 * <p><b>Dataset format</b>: JSONL, one item per line, fields:
 * <pre>
 * {
 *   "question_id": "abc",
 *   "question": "...",
 *   "question_type": "single_session_user | temporal_reasoning | multi_session | knowledge_update | abstention",
 *   "sessions": [[{"role":"user|assistant","content":"..."}], ...],
 *   "answer": "ground truth answer text"
 * }
 * </pre>
 *
 * <p>Compatible with the {@code longmemeval_s} / {@code longmemeval_m} / {@code longmemeval_oracle}
 * format published at xiaowu0162/LongMemEval.
 */
public final class LongMemEvalHarness {

    private static final String ANSWER_PROMPT = """
            You are answering a user's question using ONLY the memory context below.
            If the context doesn't contain the answer, say "I don't know."
            Be concise — one or two sentences. No preamble.

            MEMORY CONTEXT:
            %s

            QUESTION: %s
            ANSWER:""";

    /**
     * Aggregation prompt for "how many X" / "list all Y" / "count Z" style
     * questions. Asks the LLM to ENUMERATE distinct items first, then count.
     * Without this, qwen-class models routinely undercount because they
     * skim the context and report a partial subset.
     */
    private static final String AGGREGATION_PROMPT = """
            You are answering a counting/aggregation question using ONLY the memory context below.

            STEP 1: Read every item in the context. Identify each distinct entity that
                    matches the question.
            STEP 2: List the distinct entities you found, one per line.
            STEP 3: Provide the final count as the LAST line of your answer.

            If the context contains no relevant evidence, say "I don't know."

            MEMORY CONTEXT:
            %s

            QUESTION: %s
            ANSWER:""";

    private static final String JUDGE_PROMPT = """
            You are grading whether a model's answer is correct given the ground truth.
            Reply with a single token: YES if the answer is correct (captures the same facts,
            even if worded differently), NO if it's wrong, missing, or says "I don't know"
            when the ground truth is specific.

            QUESTION: %s
            GROUND TRUTH: %s
            MODEL ANSWER: %s
            GRADE (YES/NO):""";

    private final MemoryService memory;
    private final ExtractionService extraction;     // nullable — if null, we fall back to remember()
    private final ModelClient answerer;
    private final ModelClient judge;
    private final int recallTokenBudget;
    private final boolean thinking;

    public LongMemEvalHarness(MemoryService memory,
                              ExtractionService extraction,
                              ModelClient answerer,
                              ModelClient judge,
                              int recallTokenBudget) {
        this(memory, extraction, answerer, judge, recallTokenBudget, false);
    }

    /**
     * Full constructor. When {@code thinking} is true, the answerer prompt is
     * prefixed with Qwen3's {@code /think} directive so the model emits a
     * {@code <think>…</think>} chain-of-thought before its final answer. The
     * reasoning block is stripped from the predicted text before it reaches
     * the judge so the grader only sees the conclusion.
     */
    public LongMemEvalHarness(MemoryService memory,
                              ExtractionService extraction,
                              ModelClient answerer,
                              ModelClient judge,
                              int recallTokenBudget,
                              boolean thinking) {
        this.memory = Objects.requireNonNull(memory);
        this.extraction = extraction;
        this.answerer = Objects.requireNonNull(answerer);
        this.judge = Objects.requireNonNull(judge);
        this.recallTokenBudget = Math.max(200, recallTokenBudget);
        this.thinking = thinking;
    }

    public record Item(
            String questionId,
            String question,
            String questionType,
            List<List<Map<String, String>>> sessions,
            String answer
    ) {}

    public record ItemResult(
            String questionId,
            String questionType,
            String question,
            String predicted,
            String groundTruth,
            boolean correct,
            long ingestMs,
            long recallMs,
            int recalledItems,
            List<String> recalledContent
    ) {}

    public record Report(
            int total,
            int correct,
            double accuracy,
            Map<String, TypeBreakdown> byType,
            List<ItemResult> items
    ) {
        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("Overall: %d/%d = %.1f%%%n", correct, total, accuracy * 100));
            for (var e : byType.entrySet()) {
                sb.append(String.format("  %-30s %d/%d = %.1f%%%n",
                        e.getKey(), e.getValue().correct, e.getValue().total,
                        e.getValue().accuracy() * 100));
            }
            return sb.toString();
        }
    }

    public record TypeBreakdown(int total, int correct) {
        public double accuracy() { return total == 0 ? 0.0 : (double) correct / total; }
    }

    /** Parse a LongMemEval-format JSONL file. */
    public static List<Item> loadDataset(Path jsonl, int limit) throws IOException {
        List<Item> items = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(jsonl)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                Map<String, Object> raw = Json.parseMap(line);
                items.add(toItem(raw));
                if (limit > 0 && items.size() >= limit) break;
            }
        }
        return items;
    }

    @SuppressWarnings("unchecked")
    private static Item toItem(Map<String, Object> raw) {
        String qid = str(raw.get("question_id"));
        String question = str(raw.get("question"));
        String qtype = str(raw.getOrDefault("question_type", "unknown"));
        String answer = str(raw.getOrDefault("answer", ""));
        List<List<Map<String, String>>> sessions = new ArrayList<>();
        Object sessObj = raw.get("sessions");
        if (sessObj instanceof List<?> sessList) {
            for (Object s : sessList) {
                List<Map<String, String>> session = new ArrayList<>();
                if (s instanceof List<?> turns) {
                    for (Object t : turns) {
                        if (t instanceof Map<?, ?> turn) {
                            session.add(Map.of(
                                    "role", str(turn.get("role")),
                                    "content", str(turn.get("content"))));
                        }
                    }
                }
                sessions.add(session);
            }
        }
        return new Item(qid, question, qtype, sessions, answer);
    }

    private static String str(Object o) { return o == null ? "" : o.toString(); }

    /** Run the benchmark over the given items and return a report. */
    public Report run(List<Item> items) {
        List<ItemResult> results = new ArrayList<>();
        Map<String, int[]> perType = new LinkedHashMap<>();   // [total, correct]

        for (Item item : items) {
            String uid = "longmemeval-" + item.questionId();

            // 1. Ingest all session turns. Each haystack session gets a stable
            //    per-item session_id so downstream session-filtered recall can
            //    use the picker to narrow retrieval to a handful of likely
            //    sessions instead of ranking all ~500 turns across 54 sessions.
            long tIngest = System.nanoTime();
            int turnCount = 0;
            int sessionIdx = 0;
            for (List<Map<String, String>> session : item.sessions()) {
                String sessionId = "sess-" + sessionIdx++;
                for (Map<String, String> turn : session) {
                    String role = turn.get("role");
                    String content = turn.get("content");
                    if (content == null || content.isBlank()) continue;
                    Map<String, String> tags = Map.of(
                            "role", role == null ? "user" : role,
                            ai.nizo.memory.api.memory.MemoryTags.SESSION_ID, sessionId);
                    if ("user".equalsIgnoreCase(role) && extraction != null) {
                        // User messages go through extraction → episodic + semantic.
                        try {
                            extraction.extract(uid, content);
                        } catch (RuntimeException e) {
                            // fall back to raw remember() so we don't lose the turn
                            memory.remember(uid, content, tags, "longmemeval");
                        }
                    } else {
                        memory.remember(uid, content, tags, "longmemeval");
                    }
                    turnCount++;
                }
            }
            long ingestMs = (System.nanoTime() - tIngest) / 1_000_000L;

            // 2. Recall against the question.
            long tRecall = System.nanoTime();
            List<MemoryItem> recalled = memory.recall(
                    RecallRequest.of(uid, item.question(), recallTokenBudget));
            long recallMs = (System.nanoTime() - tRecall) / 1_000_000L;

            // 3. Ask answer LLM.
            String context = recalled.stream()
                    .map(m -> "- " + m.content().replace('\n', ' '))
                    .collect(Collectors.joining("\n"));
            if (context.isEmpty()) context = "(no relevant memory)";
            String predicted;
            try {
                // Ollama exposes Qwen3 / R1-style chain-of-thought via a top-level
                // `think: true` param on /api/chat. Reasoning lands in a separate
                // `message.thinking` field which our model client drops, so the
                // returned `text()` already excludes the trace; we still defensively
                // strip any inline <think>…</think> in case a future model emits one.
                String userPrompt = ANSWER_PROMPT.formatted(context, item.question());
                ModelRequest req = thinking
                        ? new ModelRequest(List.of(Message.user(userPrompt)),
                                List.of(),
                                Map.of("think", true))
                        : ModelRequest.of(List.of(Message.user(userPrompt)));
                predicted = answerer.complete(req).text();
            } catch (RuntimeException e) {
                predicted = "(answerer failed: " + e.getMessage() + ")";
            }
            if (predicted == null) predicted = "";
            predicted = predicted.strip();
            if (thinking) predicted = stripThinkBlock(predicted);

            // 4. LLM-as-judge grade.
            boolean correct = gradeIsYes(item.question(), item.answer(), predicted);

            // Capture the recalled content (text only, capped) so an external
            // grader can compute recall@k independent of the answerer LLM.
            List<String> recalledContent = recalled.stream()
                    .map(m -> m.content().replace('\n', ' '))
                    .limit(20)
                    .toList();
            results.add(new ItemResult(
                    item.questionId(), item.questionType(), item.question(),
                    predicted, item.answer(), correct,
                    ingestMs, recallMs, recalled.size(), recalledContent));

            perType.computeIfAbsent(item.questionType(), k -> new int[2])[0]++;
            if (correct) perType.get(item.questionType())[1]++;

            // 5. Purge memory for next item so users don't contaminate each other.
            // (Skipped when nizo.bench.keep-data=true so a single-item run leaves
            //  the DB inspectable for debugging.)
            if (!"true".equalsIgnoreCase(System.getProperty("nizo.bench.keep-data"))) {
                memory.forgetUser(uid);
            }
        }

        int total = results.size();
        int totalCorrect = (int) results.stream().filter(ItemResult::correct).count();
        double acc = total == 0 ? 0.0 : (double) totalCorrect / total;
        Map<String, TypeBreakdown> breakdowns = new LinkedHashMap<>();
        for (var e : perType.entrySet()) {
            breakdowns.put(e.getKey(),
                    new TypeBreakdown(e.getValue()[0], e.getValue()[1]));
        }
        return new Report(total, totalCorrect, acc, breakdowns, results);
    }

    /**
     * Strip any leading {@code <think>…</think>} chain-of-thought block emitted
     * by a Qwen3 / DeepSeek-R1-style reasoning model. Handles multiple blocks
     * and trims whitespace so the judge only sees the final answer.
     *
     * <p>Kept intentionally tolerant: if the model emits only reasoning and
     * forgets to close the tag, we return the text after the last {@code </think>}
     * or the original text if no {@code </think>} was found.
     */
    static String stripThinkBlock(String text) {
        if (text == null || text.isEmpty()) return text == null ? "" : text;
        // Remove all <think>…</think> pairs (non-greedy, DOTALL).
        String stripped = text.replaceAll("(?s)<think>.*?</think>\\s*", "");
        // If there's still a stray </think> (unmatched opener omitted), keep only after it.
        int lastClose = stripped.lastIndexOf("</think>");
        if (lastClose >= 0) {
            stripped = stripped.substring(lastClose + "</think>".length());
        }
        return stripped.strip();
    }

    private boolean gradeIsYes(String question, String truth, String predicted) {
        try {
            String verdict = judge.complete(ModelRequest.of(List.of(
                    Message.user(JUDGE_PROMPT.formatted(question, truth, predicted)))))
                    .text();
            if (verdict == null) return false;
            return verdict.trim().toUpperCase(Locale.ROOT).startsWith("YES");
        } catch (RuntimeException e) {
            return false;
        }
    }
}
