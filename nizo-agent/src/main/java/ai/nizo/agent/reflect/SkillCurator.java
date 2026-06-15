package ai.nizo.agent.reflect;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Closes the self-learning loop: {@link ReflectionEngine} <em>writes</em> skills, this
 * <em>grades and prunes</em> them on a schedule (the Hermes pattern — "writes, grades, and prunes").
 *
 * <p>Without curation a self-authoring agent's skill library only ever grows: vacuous one-off
 * skills, near-duplicates, and notes-overfit-to-a-single-conversation accumulate and dilute the
 * catalogue the model has to read on every turn. This periodically asks the model to grade each
 * <em>reflection-authored</em> skill (0-10) for how reusable and non-redundant it is, and reversibly
 * retires the genuinely-bad ones.
 *
 * <h2>Safety</h2>
 * <ul>
 *   <li><b>Only touches reflection-authored skills</b> — identified by the
 *       {@code authored by reflection} marker {@link ReflectionEngine} writes. Hand-built and
 *       bundled skills (stock_*, india_*, …) are never graded or moved.</li>
 *   <li><b>Reversible retirement</b> — a retired skill is <em>moved</em> to
 *       {@code skills/.retired/<name>-<epoch>/}, not deleted. The loader walks only depth 2, so the
 *       dot-dir drops out of the active set, but the file is right there to restore.</li>
 *   <li><b>Grace period</b> — a freshly-authored skill is left alone until it is at least
 *       {@code graceHours} old, so a skill never gets retired before it has had a chance to be used.</li>
 *   <li><b>Conservative threshold</b> — default retire-at-or-below 2/10, i.e. only skills the grader
 *       considers genuinely vacuous, overfit, or duplicative. LLM grades are noisy; the reversible
 *       move + grace period keep a single bad grade from being costly.</li>
 * </ul>
 *
 * <p>Disable entirely with {@code NIZO_SKILL_CURATOR=off}. Tunables:
 * {@code NIZO_SKILL_CURATOR_MIN_SCORE} (retire at-or-below; default 2),
 * {@code NIZO_SKILL_CURATOR_GRACE_HOURS} (default 24),
 * {@code NIZO_SKILL_CURATOR_INITIAL_DELAY_SEC} (default 120),
 * {@code NIZO_SKILL_CURATOR_PERIOD_HOURS} (default 24).
 */
public final class SkillCurator {

    private static final Logger LOG = LoggerFactory.getLogger(SkillCurator.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long PER_CALL_TIMEOUT_MS = 120_000L;
    private static final String REFLECTION_MARKER = "authored by reflection";
    private static final Pattern CREATED_DATE = Pattern.compile(
            "authored by reflection from chat .*? on (\\d{4}-\\d{2}-\\d{2})");
    private static final Pattern FIRST_JSON_OBJECT = Pattern.compile("\\{.*?\\}", Pattern.DOTALL);

    private final LlmClient llm;
    private final String model;
    private final Path skillsDir;
    private final int retireAtOrBelow;
    private final long graceHours;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private ScheduledExecutorService scheduler;

    public SkillCurator(LlmClient llm, String model, Path skillsDir) {
        this.llm = llm;
        this.model = model;
        this.skillsDir = skillsDir;
        this.retireAtOrBelow = (int) envLong("NIZO_SKILL_CURATOR_MIN_SCORE", 2);
        this.graceHours = envLong("NIZO_SKILL_CURATOR_GRACE_HOURS", 24);
    }

    public static boolean enabled() {
        String v = System.getenv("NIZO_SKILL_CURATOR");
        return v == null || !(v.equalsIgnoreCase("off") || v.equals("0") || v.equalsIgnoreCase("false"));
    }

    /** Start the periodic grade-and-prune. Idempotent-ish; call once at bootstrap. */
    public void start() {
        long initialDelay = envLong("NIZO_SKILL_CURATOR_INITIAL_DELAY_SEC", 120);
        long periodHours = envLong("NIZO_SKILL_CURATOR_PERIOD_HOURS", 24);
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "nizo-skill-curator");
            t.setDaemon(true);
            return t;
        });
        scheduler.scheduleAtFixedRate(this::safeRun, initialDelay, periodHours * 3600, TimeUnit.SECONDS);
        LOG.info("skill curator ENABLED (retire<= {}/10, grace {}h, first run in {}s, every {}h)",
                retireAtOrBelow, graceHours, initialDelay, periodHours);
    }

    public void stop() {
        if (scheduler != null) scheduler.shutdownNow();
    }

    private void safeRun() {
        if (!running.compareAndSet(false, true)) {
            LOG.debug("skill curator already running, skipping this tick");
            return;
        }
        try {
            runOnce();
        } catch (Exception e) {
            LOG.warn("skill curator run failed: {}", e.toString());
        } finally {
            running.set(false);
        }
    }

    /** One grade-and-prune pass. Public so it can be triggered manually / from a test. Returns retired count. */
    public int runOnce() {
        if (!Files.isDirectory(skillsDir)) return 0;
        List<Path> candidates = new ArrayList<>();
        List<String> allSkillNames = new ArrayList<>();
        try (Stream<Path> dirs = Files.list(skillsDir)) {
            for (Path dir : (Iterable<Path>) dirs::iterator) {
                String dname = dir.getFileName().toString();
                if (dname.startsWith(".") || !Files.isDirectory(dir)) continue;   // skip .retired etc.
                Path md = dir.resolve("SKILL.md");
                if (!Files.isRegularFile(md)) continue;
                allSkillNames.add(dname);
                String body = safeRead(md);
                if (body != null && body.contains(REFLECTION_MARKER)) candidates.add(dir);
            }
        } catch (IOException e) {
            LOG.warn("skill curator could not list {}: {}", skillsDir, e.toString());
            return 0;
        }
        if (candidates.isEmpty()) {
            LOG.info("skill curator: no reflection-authored skills to grade ({} skills total)", allSkillNames.size());
            return 0;
        }

        int graded = 0, retired = 0, skippedNew = 0;
        for (Path dir : candidates) {
            String name = dir.getFileName().toString();
            Path md = dir.resolve("SKILL.md");
            String body = safeRead(md);
            if (body == null) continue;
            if (ageHours(body, md) < graceHours) { skippedNew++; continue; }

            Grade g = grade(name, body, allSkillNames);
            graded++;
            if (g == null) {
                LOG.warn("skill curator: '{}' could not be graded (LLM error), leaving in place", name);
                continue;
            }
            LOG.info("skill curator graded '{}' = {}/10 ({})", name, g.score, g.reason);
            if (g.score <= retireAtOrBelow) {
                if (retire(dir, g)) retired++;
            }
        }
        LOG.info("skill curator pass done: {} graded, {} retired, {} too-new-to-grade",
                graded, retired, skippedNew);
        return retired;
    }

    record Grade(int score, String reason) {}

    private Grade grade(String name, String body, List<String> allSkillNames) {
        List<String> others = new ArrayList<>(allSkillNames);
        others.remove(name);
        String user = "Skill name: " + name + "\n\n"
                + "Other skills already in the library: " + (others.isEmpty() ? "(none)" : String.join(", ", others)) + "\n\n"
                + "--- SKILL.md ---\n" + (body.length() > 8000 ? body.substring(0, 8000) : body);
        ChatRequest req = ChatRequest.of(model, List.of(ChatMessage.system(GRADER_SYSTEM), ChatMessage.user(user)))
                .withExtraBody(Map.of("chat_template_kwargs", Map.of("enable_thinking", false)));
        ChatResponse resp;
        try {
            resp = CompletableFuture.supplyAsync(() -> llm.chat(req))
                    .get(PER_CALL_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            return null;
        }
        return parseGrade(resp == null ? null : resp.content());
    }

    static Grade parseGrade(String content) {
        if (content == null || content.isBlank()) return null;
        Matcher m = FIRST_JSON_OBJECT.matcher(content);
        String json = m.find() ? m.group() : content;
        try {
            JsonNode n = MAPPER.readTree(json);
            if (!n.has("score")) return null;
            int score = Math.max(0, Math.min(10, n.path("score").asInt()));
            String reason = n.path("reason").asText("").trim();
            if (reason.length() > 200) reason = reason.substring(0, 200);
            return new Grade(score, reason.isBlank() ? "(no reason given)" : reason);
        } catch (Exception e) {
            return null;
        }
    }

    /** Move a skill dir into {@code .retired/} (reversible). Returns true on success. */
    private boolean retire(Path dir, Grade g) {
        try {
            Path retiredRoot = skillsDir.resolve(".retired");
            Files.createDirectories(retiredRoot);
            Path dest = retiredRoot.resolve(dir.getFileName().toString() + "-" + Instant.now().getEpochSecond());
            Files.move(dir, dest, StandardCopyOption.ATOMIC_MOVE);
            LOG.info("skill curator RETIRED '{}' (score {}/10: {}) -> {}",
                    dir.getFileName(), g.score, g.reason, dest);
            return true;
        } catch (IOException e) {
            LOG.warn("skill curator could not retire {}: {}", dir, e.toString());
            return false;
        }
    }

    private static long ageHours(String body, Path md) {
        Matcher m = CREATED_DATE.matcher(body);
        if (m.find()) {
            try {
                LocalDate d = LocalDate.parse(m.group(1));
                Instant created = d.atStartOfDay(ZoneId.systemDefault()).toInstant();
                return Duration.between(created, Instant.now()).toHours();
            } catch (Exception ignored) { /* fall through to mtime */ }
        }
        try {
            return Duration.between(md.toFile().lastModified() == 0
                    ? Instant.now() : Instant.ofEpochMilli(md.toFile().lastModified()), Instant.now()).toHours();
        } catch (Exception e) {
            return Long.MAX_VALUE; // unknown age → treat as old enough to grade
        }
    }

    private static String safeRead(Path p) {
        try { return Files.readString(p, StandardCharsets.UTF_8); }
        catch (IOException e) { return null; }
    }

    private static long envLong(String key, long dflt) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) return dflt;
        try { return Long.parseLong(v.trim()); } catch (NumberFormatException e) { return dflt; }
    }

    private static final String GRADER_SYSTEM = """
        You grade a SELF-AUTHORED agent skill for whether it earns a place in the library. The skill
        was written automatically by the agent after a past task. Score it 0-10 on reusability:
        - 8-10: a specific, actionable, reusable PROCEDURE that will help on many future tasks.
        - 4-7: useful but narrow, or partially overlapping another skill.
        - 0-3: vacuous, overfit to one past conversation (just restates a single answer), or
          duplicative of another skill in the library.
        Penalise redundancy with the other skills listed. Reward concrete, transferable how-to content.
        Reply with EXACTLY one JSON object and nothing else:
        {"score": <integer 0-10>, "reason": "<one short sentence>"}
        """;
}
