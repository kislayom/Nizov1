package ai.nizo.agent.eval;

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
        System.out.printf("Nizo eval — %d gold tasks against %s%n%n", tasks.size(), base);

        int pass = 0;
        long t0all = System.currentTimeMillis();
        for (GoldTask t : tasks) {
            long t0 = System.currentTimeMillis();
            String reply;
            try {
                reply = chat(http, base, token, "eval-" + t.id(), t.prompt());
            } catch (Exception e) {
                String msg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
                System.out.printf("  FAIL  %-14s  request error: %s%n", t.id(), msg);
                continue;
            }
            long ms = System.currentTimeMillis() - t0;
            CheckResult cr = check(t, reply);
            if (cr.pass) pass++;
            System.out.printf("  %-4s  %-14s  %s  [%dms]%n", cr.pass ? "PASS" : "FAIL", t.id(), cr.detail, ms);
            if (!cr.pass) System.out.println("          reply: " + oneLine(reply, 220));
        }
        double secs = (System.currentTimeMillis() - t0all) / 1000.0;
        System.out.printf("%nScore: %d/%d (%.0f%%) in %.1fs%n",
                pass, tasks.size(), 100.0 * pass / Math.max(1, tasks.size()), secs);
        System.exit(pass == tasks.size() ? 0 : 1);
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

    private record CheckResult(boolean pass, String detail) {}

    private static CheckResult check(GoldTask t, String reply) {
        switch (t.check() == null ? "" : t.check()) {
            case "numeric" -> {
                double exp = Double.parseDouble(t.expected());
                Double best = closestNumber(reply, exp);
                if (best == null) return new CheckResult(false, "no number in reply; expected " + exp);
                boolean ok = Math.abs(best - exp) <= Math.max(t.tolerance(), 1e-9);
                return new CheckResult(ok,
                        (ok ? "= " : "got " + trim(best) + ", want ") + trim(exp) + " ±" + trim(t.tolerance()));
            }
            case "contains" -> {
                StringBuilder miss = new StringBuilder();
                String hay = reply.toLowerCase();
                for (String need : t.expected().split("\\|\\|"))
                    if (!hay.contains(need.trim().toLowerCase())) miss.append('"').append(need.trim()).append("\" ");
                boolean ok = miss.length() == 0;
                return new CheckResult(ok, ok ? "contains all" : "missing: " + miss.toString().trim());
            }
            case "regex" -> {
                boolean ok = Pattern.compile(t.expected(), Pattern.CASE_INSENSITIVE | Pattern.DOTALL)
                        .matcher(reply).find();
                return new CheckResult(ok, ok ? "matched" : "no match for /" + t.expected() + "/");
            }
            default -> {
                return new CheckResult(false, "unknown check type: " + t.check());
            }
        }
    }

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
