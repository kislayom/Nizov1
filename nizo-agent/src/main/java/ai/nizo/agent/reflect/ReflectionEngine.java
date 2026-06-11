package ai.nizo.agent.reflect;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.memory.UserFactStore;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.regex.Pattern;

/**
 * Post-task Reflective Phase — the self-learning loop (project phase 11, and the
 * headline capability of Hermes-class agents: "when the agent solves a hard problem,
 * it writes a reusable skill so it never forgets how").
 *
 * <p>After a turn that took real work (≥ {@code minTools} tool calls, clean stop),
 * {@link #maybeReflect} runs ONE asynchronous LLM pass over a compact trajectory
 * transcript and lets the model pick exactly one outcome:
 * <ul>
 *   <li>{@code new_skill} / {@code update_skill} — write {@code ~/.nizo/skills/<name>/SKILL.md}
 *       (agentskills.io format, same writer conventions as SaveSkillTool)</li>
 *   <li>{@code user_fact} — durable observation about the user → {@link UserFactStore}</li>
 *   <li>{@code nothing} — most turns; the bar for writing is deliberately high</li>
 * </ul>
 *
 * <p>Design rails:
 * <ul>
 *   <li>Never blocks the user turn — runs on a daemon virtual thread.</li>
 *   <li>Single-flight: if a reflection is already running, the new one is DROPPED
 *       (not queued). Reflection is best-effort by design.</li>
 *   <li>Stock-pipeline chats are excluded — that flow is deterministic already.</li>
 *   <li>Skill names are sandboxed to {@code [a-z0-9_]+} and bodies size-capped, so a
 *       confused model can't write outside the skills dir or dump a novel.</li>
 *   <li>Disabled with {@code NIZO_REFLECT=0}; tool threshold via {@code NIZO_REFLECT_MIN_TOOLS}.</li>
 * </ul>
 */
public final class ReflectionEngine {

    private static final Logger LOG = LoggerFactory.getLogger(ReflectionEngine.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAFE_NAME = Pattern.compile("^[a-z0-9_]{3,48}$");
    private static final int MAX_BODY_CHARS = 6_000;
    private static final int MAX_TRAJECTORY_CHARS = 12_000;

    /** One compact line per tool call, recorded by AgentLoop during the turn. */
    public record Trajectory(
            String chatId,
            String userId,
            String channel,
            String userText,
            List<String> toolLines,   // "web_search {query=...} → ok 245ms · 8 results"
            String finalPreview,      // first ~600 chars of the final reply
            int iterations,
            int toolCalls,
            long elapsedMs,
            String stopReason
    ) {}

    private final LlmClient llm;
    private final String model;
    private final Path skillsDir;
    private final UserFactStore userFacts;   // nullable
    private final boolean enabled;
    private final int minTools;
    private final Semaphore inFlight = new Semaphore(1);

    public ReflectionEngine(LlmClient llm, String model, Path skillsDir, UserFactStore userFacts) {
        this.llm = llm;
        this.model = model;
        this.skillsDir = skillsDir;
        this.userFacts = userFacts;
        this.enabled = !"0".equals(System.getenv("NIZO_REFLECT"));
        int mt = 5;
        try {
            String v = System.getenv("NIZO_REFLECT_MIN_TOOLS");
            if (v != null && !v.isBlank()) mt = Integer.parseInt(v.trim());
        } catch (NumberFormatException ignore) {}
        this.minTools = Math.max(1, mt);
        LOG.info("reflection engine {} (minTools={})", enabled ? "ENABLED" : "disabled", minTools);
    }

    /** Cheap gate + async dispatch. Called by AgentLoop after every completed turn. */
    public void maybeReflect(Trajectory t) {
        if (!enabled || t == null) return;
        if (t.toolCalls() < minTools) return;
        if (!"stop".equals(t.stopReason()) && !"length".equals(t.stopReason())) return;
        if (t.chatId() != null && t.chatId().startsWith("stock-")) return;  // deterministic pipeline
        if (t.finalPreview() == null || t.finalPreview().isBlank()) return;
        if (!inFlight.tryAcquire()) {
            LOG.debug("reflection skipped for {} — one already in flight", t.chatId());
            return;
        }
        Thread.ofVirtual().name("reflect-" + t.chatId()).start(() -> {
            try {
                reflect(t);
            } catch (Exception e) {
                LOG.warn("reflection failed for {}: {}", t.chatId(), e.toString());
            } finally {
                inFlight.release();
            }
        });
    }

    private void reflect(Trajectory t) throws Exception {
        long t0 = System.currentTimeMillis();
        String prompt = buildPrompt(t);
        // enable_thinking=false is load-bearing: with thinking on, Qwen3.6 burns the
        // entire completion budget inside <think> and returns 0 content chars
        // (verified June 2026 on the first live reflection — "no parseable verdict").
        ChatRequest req = ChatRequest.of(model, List.of(
                ChatMessage.system(REFLECTION_SYSTEM),
                ChatMessage.user(prompt)
        )).withMaxTokens(2_500)
          .withExtraBody(java.util.Map.of("chat_template_kwargs",
                  java.util.Map.of("enable_thinking", false)));
        ChatResponse resp = llm.chat(req);
        String content = resp.content() == null ? "" : resp.content().trim();
        JsonNode verdict = parseVerdict(content);
        if (verdict == null) {
            LOG.info("reflection for {} produced no parseable verdict ({} chars)", t.chatId(), content.length());
            return;
        }
        String action = verdict.path("action").asText("nothing");
        switch (action) {
            case "new_skill", "update_skill" -> applySkill(t, verdict, action);
            case "user_fact" -> applyFact(t, verdict);
            default -> LOG.info("reflection[{}]: nothing to keep ({})",
                    t.chatId(), verdict.path("reason").asText(""));
        }
        LOG.info("reflection for {} done in {}ms (action={})",
                t.chatId(), System.currentTimeMillis() - t0, action);
    }

    private void applySkill(Trajectory t, JsonNode v, String action) throws Exception {
        String name = v.path("name").asText("").trim();
        String desc = v.path("description").asText("").trim();
        String whenToUse = v.path("when_to_use").asText("").trim();
        String body = v.path("body").asText("");
        if (!SAFE_NAME.matcher(name).matches()) {
            LOG.warn("reflection[{}]: rejected skill name '{}'", t.chatId(), name);
            return;
        }
        if (desc.isEmpty() || body.isBlank()) {
            LOG.warn("reflection[{}]: skill '{}' missing description/body", t.chatId(), name);
            return;
        }
        if (body.length() > MAX_BODY_CHARS) body = body.substring(0, MAX_BODY_CHARS);

        Path dir = skillsDir.resolve(name);
        Path file = dir.resolve("SKILL.md");
        boolean exists = Files.exists(file);
        if ("new_skill".equals(action) && exists) {
            // The model thought it was new but it isn't — treat as update, which is safe
            // because our skills are single-file and the new body supersedes.
            LOG.info("reflection[{}]: '{}' already exists, treating as update", t.chatId(), name);
        }
        Files.createDirectories(dir);
        String md = """
                ---
                name: %s
                description: %s
                when_to_use: %s
                tags: [learned, reflection]
                ---

                %s

                <!-- authored by reflection from chat %s on %s -->
                """.formatted(name, desc,
                whenToUse.isEmpty() ? desc : whenToUse,
                body.strip(),
                t.chatId(), java.time.LocalDate.now());
        Files.writeString(file, md, StandardCharsets.UTF_8);
        LOG.info("reflection[{}]: {} skill '{}' ({} chars) — available as skill_{} after restart",
                t.chatId(), exists ? "updated" : "created", name, md.length(), name);
    }

    private void applyFact(Trajectory t, JsonNode v) {
        String fact = v.path("fact").asText("").trim();
        if (fact.isEmpty() || userFacts == null) return;
        if (fact.length() > 500) fact = fact.substring(0, 500);
        long id = userFacts.remember(t.userId(), fact, "reflection");
        LOG.info("reflection[{}]: remembered fact #{} for {}: {}", t.chatId(), id, t.userId(),
                fact.length() > 80 ? fact.substring(0, 80) + "…" : fact);
    }

    private String buildPrompt(Trajectory t) {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("## Task transcript\n\n");
        sb.append("User (").append(t.userId()).append(" via ").append(t.channel()).append(") asked:\n");
        sb.append("> ").append(truncate(t.userText(), 600)).append("\n\n");
        sb.append("Agent ran ").append(t.toolCalls()).append(" tool calls over ")
          .append(t.iterations()).append(" iterations (").append(t.elapsedMs() / 1000).append("s):\n");
        int budget = MAX_TRAJECTORY_CHARS;
        for (String line : t.toolLines()) {
            String l = "- " + truncate(line, 220) + "\n";
            if (budget - l.length() < 0) { sb.append("- …(truncated)\n"); break; }
            sb.append(l);
            budget -= l.length();
        }
        sb.append("\nFinal reply (preview):\n> ").append(truncate(t.finalPreview(), 600)).append('\n');
        return sb.toString();
    }

    private static String truncate(String s, int n) {
        if (s == null) return "";
        s = s.replaceAll("\\s+", " ").trim();
        return s.length() <= n ? s : s.substring(0, n) + "…";
    }

    private static JsonNode parseVerdict(String content) {
        // The model is instructed to emit exactly one JSON object; be tolerant of
        // fences and prose around it.
        int start = content.indexOf('{');
        int end = content.lastIndexOf('}');
        if (start < 0 || end <= start) return null;
        try {
            return MAPPER.readTree(content.substring(start, end + 1));
        } catch (Exception e) {
            return null;
        }
    }

    private static final String REFLECTION_SYSTEM = """
            You are the reflection module of a personal agent. You just watched the agent
            complete a task. Decide if anything from this run is worth keeping permanently.

            The bar is HIGH. Most runs teach nothing durable — answer "nothing" for those.

            Keep something ONLY if:
            - new_skill: the run worked out a REPEATABLE multi-step procedure the user will
              plausibly ask for again (a workflow, a report format, a sequence of tools with
              known pitfalls). Generic one-off Q&A is NOT a skill.
            - update_skill: the run revealed a correction/improvement to an existing learned
              procedure (you'll know from the transcript if a skill was followed and failed).
            - user_fact: the run revealed a durable fact about the USER themselves
              (preference, recurring context, environment) that future runs should know.
              Trivia about the world is NOT a user fact.

            Reply with EXACTLY ONE JSON object, no prose, one of:
            {"action":"nothing","reason":"<one line>"}
            {"action":"user_fact","fact":"<one sentence, third person>","reason":"<one line>"}
            {"action":"new_skill","name":"<snake_case>","description":"<one paragraph>",
             "when_to_use":"<trigger phrasing>",
             "body":"<markdown: numbered steps, exact tool names + arguments that worked,
                      pitfalls observed in THIS run, verification step>",
             "reason":"<one line>"}
            {"action":"update_skill", ...same fields as new_skill...}

            Skill bodies must be grounded in what ACTUALLY happened in the transcript —
            real tool names, real arguments, real failure modes. Never invent steps.
            """;
}
