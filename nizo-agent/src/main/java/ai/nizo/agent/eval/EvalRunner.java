package ai.nizo.agent.eval;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.llm.OpenAiCompatibleClient;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The accuracy regression harness. Drives a set of gold tasks ({@link GoldTask}) through the
 * <em>running</em> server's {@code /api/chat} endpoint — exactly the path a real user hits — and
 * scores each reply with a deterministic checker. Prints a scorecard and exits non-zero if any
 * task fails, so it slots into CI or a pre-deploy gate.
 *
 * <p>This deliberately tests the deployed system end-to-end (LLM + tools + agent loop), not a mock.
 * It is the measurable backbone for the "machine-level accuracy" goal: run it before and after a
 * change and compare the number.
 *
 * <h2>Run</h2>
 * <pre>
 *   # on the server (loopback + token available):
 *   NIZO_WEB_TOKEN=$(cat ~/.nizo/web-token) \
 *     java -cp nizo-app/target/nizo.jar ai.nizo.agent.eval.EvalRunner
 *
 *   # custom task file / endpoint:
 *   NIZO_WEB_URL=http://127.0.0.1:7777 java -cp ... ai.nizo.agent.eval.EvalRunner /path/tasks.json
 * </pre>
 *
 * <p>Env: {@code NIZO_WEB_URL} (default {@code http://127.0.0.1:7777}),
 * {@code NIZO_WEB_TOKEN} (falls back to {@code ~/.nizo/web-token}).
 */
public final class EvalRunner {

    private static final ObjectMapper M = new ObjectMapper();
    private static final Pattern NUMBER = Pattern.compile("-?\\d[\\d,]*(?:\\.\\d+)?");

    public static void main(String[] args) throws Exception {
        String base = env("NIZO_WEB_URL", "http://127.0.0.1:7777");
        String token = resolveToken();
        List<GoldTask> tasks = loadTasks(args.length > 0 ? args[0] : null);

        HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        // Judge model for free-text "judge" checks (SimpleQA-style CORRECT/INCORRECT/NOT_ATTEMPTED).
        // Defaults to the local LLM — note that grading the same model you are benchmarking is
        // self-grading, biased slightly toward leniency. Override via NIZO_LLM_URL / NIZO_LLM_MODEL.
        String judgeModel = env("NIZO_LLM_MODEL", "Qwen/Qwen3.6-27B");
        LlmClient judge = new OpenAiCompatibleClient(env("NIZO_LLM_URL", "http://127.0.0.1:8080"),
                System.getenv("NIZO_LLM_TOKEN"));
        boolean anyJudge = tasks.stream().anyMatch(t -> "judge".equalsIgnoreCase(t.check()));
        System.out.printf("Nizo eval — %d tasks against %s%s%n%n",
                tasks.size(), base, anyJudge ? "  (judge=" + judgeModel + ")" : "");

        int correct = 0, attempted = 0;
        long t0all = System.currentTimeMillis();
        for (GoldTask t : tasks) {
            long t0 = System.currentTimeMillis();
            String reply;
            try {
                reply = chat(http, base, token, "eval-" + t.id(), t.prompt());
            } catch (Exception e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                System.out.printf("  ERR   %-14s  request error: %s%n", t.id(), msg);
                attempted++;   // an agent error is an attempt that failed
                continue;
            }
            long ms = System.currentTimeMillis() - t0;
            CheckResult cr = check(t, reply, judge, judgeModel);
            if (cr.pass) correct++;
            if (cr.attempted) attempted++;
            System.out.printf("  %-4s  %-14s  %s  [%dms]%n",
                    cr.pass ? "PASS" : (cr.attempted ? "FAIL" : "N/A "), t.id(), cr.detail, ms);
            if (!cr.pass) System.out.println("          reply: " + oneLine(reply, 200));
        }
        double secs = (System.currentTimeMillis() - t0all) / 1000.0;
        int n = Math.max(1, tasks.size());
        System.out.printf("%nCorrect: %d/%d (%.1f%%)  in %.1fs%n", correct, tasks.size(), 100.0 * correct / n, secs);
        if (anyJudge) {
            int notAttempted = tasks.size() - attempted;
            System.out.printf("Attempted: %d/%d   Not-attempted: %d   Correct-given-attempted: %.1f%%%n",
                    attempted, tasks.size(), notAttempted,
                    attempted == 0 ? 0.0 : 100.0 * correct / attempted);
        }
        System.exit(correct == tasks.size() ? 0 : 1);
    }

    private static String chat(HttpClient http, String base, String token, String chatId, String prompt)
            throws Exception {
        String body = M.writeValueAsString(Map.of("chatId", chatId, "text", prompt));
        HttpRequest req = HttpRequest.newBuilder(URI.create(base + "/api/chat"))
                .timeout(Duration.ofMinutes(5))
                .header("Content-Type", "application/json")
                .header("X-Nizo-Token", token)
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200)
            throw new RuntimeException("HTTP " + resp.statusCode() + ": " + oneLine(resp.body(), 140));
        return M.readTree(resp.body()).path("text").asText("");
    }

    private record CheckResult(boolean pass, boolean attempted, String detail) {}

    private static CheckResult check(GoldTask t, String reply, LlmClient judge, String judgeModel) {
        switch (t.check() == null ? "" : t.check().toLowerCase()) {
            case "numeric" -> {
                double exp = Double.parseDouble(t.expected());
                Double best = closestNumber(reply, exp);
                if (best == null) return new CheckResult(false, true, "no number in reply; expected " + exp);
                boolean ok = Math.abs(best - exp) <= Math.max(t.tolerance(), 1e-9);
                return new CheckResult(ok, true,
                        (ok ? "= " : "got " + trim(best) + ", want ") + trim(exp) + " ±" + trim(t.tolerance()));
            }
            case "contains" -> {
                StringBuilder miss = new StringBuilder();
                String hay = reply.toLowerCase();
                for (String need : t.expected().split("\\|\\|"))
                    if (!hay.contains(need.trim().toLowerCase())) miss.append('"').append(need.trim()).append("\" ");
                boolean ok = miss.length() == 0;
                return new CheckResult(ok, true, ok ? "contains all" : "missing: " + miss.toString().trim());
            }
            case "regex" -> {
                boolean ok = Pattern.compile(t.expected(), Pattern.CASE_INSENSITIVE | Pattern.DOTALL)
                        .matcher(reply).find();
                return new CheckResult(ok, true, ok ? "matched" : "no match for /" + t.expected() + "/");
            }
            case "judge" -> {
                return judgeAnswer(judge, judgeModel, t, reply);
            }
            default -> {
                return new CheckResult(false, true, "unknown check type: " + t.check());
            }
        }
    }

    /** SimpleQA-style LLM grading: CORRECT / INCORRECT / NOT_ATTEMPTED. */
    private static CheckResult judgeAnswer(LlmClient judge, String model, GoldTask t, String reply) {
        String user = "Question: " + t.prompt()
                + "\nGold answer: " + t.expected()
                + "\nPredicted answer: " + oneLine(reply, 1500);
        ChatRequest req = ChatRequest.of(model, List.of(ChatMessage.system(JUDGE_SYSTEM), ChatMessage.user(user)))
                .withExtraBody(Map.of("chat_template_kwargs", Map.of("enable_thinking", false)));
        String verdict;
        try { verdict = judge.chat(req).content(); }
        catch (Exception e) { return new CheckResult(false, true, "judge error: " + e.getClass().getSimpleName()); }
        return switch (firstLetter(verdict)) {
            case 'A' -> new CheckResult(true, true, "CORRECT");
            case 'C' -> new CheckResult(false, false, "NOT_ATTEMPTED (gold: " + oneLine(t.expected(), 50) + ")");
            default  -> new CheckResult(false, true, "INCORRECT (gold: " + oneLine(t.expected(), 50) + ")");
        };
    }

    private static char firstLetter(String s) {
        if (s == null) return '?';
        for (int i = 0; i < s.length(); i++) {
            char c = Character.toUpperCase(s.charAt(i));
            if (c == 'A' || c == 'B' || c == 'C') return c;
        }
        return '?';
    }

    private static final String JUDGE_SYSTEM = """
        You grade whether a PREDICTED answer correctly answers a QUESTION, given the GOLD answer.
        Reply with exactly ONE letter:
        A = CORRECT: the prediction states the gold answer or a clearly equivalent form (extra correct detail is fine).
        B = INCORRECT: the prediction misses the gold answer, contradicts it, or differs materially.
        C = NOT_ATTEMPTED: the prediction declines, says it does not know or could not find it, or gives no concrete answer.
        Output ONLY the letter A, B, or C.
        """;

    /** The number in {@code s} closest to {@code target} — robust against restated input numbers. */
    private static Double closestNumber(String s, double target) {
        Matcher m = NUMBER.matcher(s);
        Double best = null;
        double bestDist = Double.MAX_VALUE;
        while (m.find()) {
            try {
                double v = Double.parseDouble(m.group().replace(",", ""));
                double d = Math.abs(v - target);
                if (d < bestDist) { bestDist = d; best = v; }
            } catch (NumberFormatException ignored) { /* skip non-numbers like a bare "-" */ }
        }
        return best;
    }

    private static List<GoldTask> loadTasks(String pathArg) throws Exception {
        byte[] bytes;
        if (pathArg != null) {
            bytes = Files.readAllBytes(Path.of(pathArg));
        } else {
            try (InputStream in = EvalRunner.class.getResourceAsStream("/eval/gold-tasks.json")) {
                if (in == null) throw new IllegalStateException("/eval/gold-tasks.json not on classpath");
                bytes = in.readAllBytes();
            }
        }
        return Arrays.asList(M.readValue(bytes, GoldTask[].class));
    }

    private static String resolveToken() throws Exception {
        String t = System.getenv("NIZO_WEB_TOKEN");
        if (t != null && !t.isBlank()) return t.trim();
        Path p = Path.of(System.getProperty("user.home"), ".nizo", "web-token");
        return Files.exists(p) ? Files.readString(p).trim() : "";
    }

    private static String env(String k, String d) {
        String v = System.getenv(k);
        return (v == null || v.isBlank()) ? d : v.trim();
    }

    private static String trim(double d) {
        return (d == Math.rint(d)) ? String.valueOf((long) d) : String.valueOf(d);
    }

    private static String oneLine(String s, int max) {
        if (s == null) return "";
        String x = s.replaceAll("\\s+", " ").trim();
        return x.length() <= max ? x : x.substring(0, max) + "…";
    }

    private EvalRunner() {}
}
