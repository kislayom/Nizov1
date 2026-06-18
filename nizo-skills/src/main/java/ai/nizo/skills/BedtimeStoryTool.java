package ai.nizo.skills;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Personalised, multi-voice bedtime stories rendered to audio. Two stages:
 * <ol>
 *   <li><b>Script (LLM):</b> Qwen writes a calming, age-appropriate story as a STRUCTURED SCRIPT —
 *       title, a cast with a voice assigned per character (the narrator is the cloned parent voice),
 *       and ordered segments {speaker, text, sfx?} + a music mood prompt.</li>
 *   <li><b>Render (sidecar):</b> POST the script to {@code /narrate-story} — per-segment multi-voice
 *       TTS (Kokoro voices + XTTS clone) mixed with a gentle ducked MusicGen bed → one mp3.</li>
 * </ol>
 * Saves the mp3 into the workspace and returns a markdown audio reference the chat plays inline.
 *
 * <p>v1 renders synchronously — keep stories short (a couple of minutes) until the async/two-part
 * split lands. Forces HTTP/1.1 to the uvicorn sidecar.
 */
public final class BedtimeStoryTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(BedtimeStoryTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final ExecutorService TIMEOUT_EXEC =
            Executors.newCachedThreadPool(r -> { Thread t = new Thread(r, "bedtime-llm"); t.setDaemon(true); return t; });

    /** Curated soothing voice palette offered to the LLM for casting (Kokoro ids + the clone token). */
    private static final String VOICE_PALETTE =
        "NARRATOR must be \"clone\" (the parent's cloned voice). Character voice ids to choose from: "
      + "af_heart (warm gentle female), af_bella (bright young female — good for child characters), "
      + "af_nicole (soft whisper female), af_sarah (calm female), am_michael (deep warm male — big animals), "
      + "am_adam (friendly male), am_puck (playful male — small creatures), bf_emma (British female), "
      + "bm_george (British male). Pick a voice that fits each character's size/personality.";

    private final LlmClient llm;
    private final String model;
    private final Path workspace;
    private final Path voicesDir;
    private final String sidecarUrl;
    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1).connectTimeout(Duration.ofSeconds(10)).build();

    public BedtimeStoryTool(LlmClient llm, String model, Path workspace, Path voicesDir) {
        this(llm, model, workspace, voicesDir, System.getenv().getOrDefault("NIZO_VOICE_URL", "http://127.0.0.1:7780"));
    }

    public BedtimeStoryTool(LlmClient llm, String model, Path workspace, Path voicesDir, String sidecarUrl) {
        this.llm = llm; this.model = model; this.workspace = workspace; this.voicesDir = voicesDir;
        this.sidecarUrl = sidecarUrl.endsWith("/") ? sidecarUrl.substring(0, sidecarUrl.length() - 1) : sidecarUrl;
    }

    @Override public String name() { return "bedtime_story"; }

    @Override
    public String description() {
        return "Create a personalised, multi-voice bedtime STORY as narrated audio (different voices "
             + "per character, a gentle music bed, soothing pace). Use when the user asks for a bedtime "
             + "story / story for a child. Renders to an audio file — takes a minute or two. Returns a "
             + "markdown audio reference; include it VERBATIM in your reply so the child can press play.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {"type":"object","properties":{
              "child":{"type":"string","description":"The child's name (personalises the hero)."},
              "about":{"type":"string","description":"What it should be about — theme, characters, setting, a lesson."},
              "minutes":{"type":"number","description":"Target length in minutes (default 2; keep small for now)."},
              "language":{"type":"string","description":"ISO code, default 'en'."}
            },"required":["about"]}""";
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String about = args.path("about").asText("").trim();
        if (about.isEmpty()) return ToolResult.error("'about' is required — what should the story be about?");
        String child = args.path("child").asText("").trim();
        double minutes = Math.max(0.5, Math.min(6.0, args.path("minutes").asDouble(2.0)));
        String language = args.path("language").asText("en").trim();
        int words = (int) Math.round(minutes * 150);   // ~150 wpm narration

        // ── 1. LLM writes the structured script ──
        String sys = """
            You are a gentle bedtime-storyteller for a young child. Write a SOOTHING, age-appropriate story:
            nothing scary, kind characters, a soft and happy ending that winds down sleepily. Calm rhythm,
            short sentences, a little wonder. %s

            Output ONLY a JSON object, no prose, no markdown fences, with EXACTLY this shape:
            {
              "title": "string",
              "musicPrompt": "a short text prompt for a gentle instrumental bed, e.g. 'soft music-box lullaby, warm and slow'",
              "characters": [{"name":"NARRATOR","voice":"clone"}, {"name":"LION","voice":"am_michael"}],
              "segments": [{"speaker":"NARRATOR","text":"..."}, {"speaker":"LION","text":"..."}]
            }
            %s
            Aim for about %d words total across the segments. Use the child's name if given. Keep each segment
            one or two sentences. You MAY add "sfx" to a segment (e.g. {"sfx":"gentle stream"}) for a vivid
            calm moment, sparingly. The final 2-3 segments should be very calm and sleepy.
            """.formatted(child.isEmpty() ? "" : "The hero is named " + child + ".", VOICE_PALETTE, words);

        String user = "Write the bedtime story. It is about: " + about
                    + (child.isEmpty() ? "" : "\nHero's name: " + child);

        ChatResponse resp = chatWithTimeout(ChatRequest.of(model, List.of(ChatMessage.system(sys), ChatMessage.user(user)))
                .withExtraBody(Map.of("chat_template_kwargs", Map.of("enable_thinking", false))), 180);
        String raw = resp.content() == null ? "" : resp.content().strip();
        JsonNode script = extractJson(raw);
        if (script == null || !script.has("segments") || !script.path("segments").isArray()
                || script.path("segments").isEmpty()) {
            return ToolResult.error("could not produce a valid story script (LLM output was not parseable JSON).");
        }
        String title = script.path("title").asText("A Bedtime Story");

        // Personalised narrator: if this user recorded a voice sample, drive the cloned narrator
        // (NARRATOR is already cast as "clone" in the palette). Same-host path → the sidecar reads it.
        com.fasterxml.jackson.databind.node.ObjectNode scriptObj =
                script instanceof com.fasterxml.jackson.databind.node.ObjectNode o
                        ? o : (com.fasterxml.jackson.databind.node.ObjectNode) MAPPER.valueToTree(script);
        String uid = ai.nizo.api.tool.UserContext.current();
        String safeUid = (uid == null || uid.isBlank() ? "web-user" : uid).replaceAll("[^A-Za-z0-9_-]", "");
        Path clone = voicesDir.resolve(safeUid + ".wav");
        if (Files.exists(clone)) {
            scriptObj.put("voiceSampleWav", clone.toString());
            LOG.info("bedtime_story: cloned narrator voice for {}", safeUid);
        }
        script = scriptObj;

        // Long stories (5-6 min) are too long to render synchronously — split into two halves as an
        // async job: Part 1 returns as soon as it's ready and plays while Part 2 finishes.
        if (minutes > 3.0 || script.path("segments").size() > 8) {
            return renderAsyncSplit(title, script);
        }

        // ── 2. Render via the sidecar (short story, synchronous) ──
        HttpResponse<String> r;
        try {
            r = http.send(HttpRequest.newBuilder(URI.create(sidecarUrl + "/narrate-story"))
                    .timeout(Duration.ofMinutes(20))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(script), StandardCharsets.UTF_8))
                    .build(), HttpResponse.BodyHandlers.ofString());
        } catch (java.net.ConnectException e) {
            return ToolResult.error("voice sidecar not reachable at " + sidecarUrl + " — is nizo-voice running?");
        }
        if (r.statusCode() / 100 != 2) {
            String d = r.body() == null ? "" : r.body();
            return ToolResult.error("story render failed (HTTP " + r.statusCode() + "): " + d.substring(0, Math.min(300, d.length())));
        }
        JsonNode out = MAPPER.readTree(r.body());
        String b64 = out.path("audio_b64").asText("");
        if (b64.isEmpty()) return ToolResult.error("renderer returned no audio");
        byte[] mp3 = Base64.getDecoder().decode(b64);

        Path genDir = workspace.resolve("gen");
        Files.createDirectories(genDir);
        String fname = "story-" + UUID.randomUUID().toString().substring(0, 8) + ".mp3";
        Files.write(genDir.resolve(fname), mp3);

        double dur = out.path("durationSec").asDouble(0);
        int segs = out.path("segments").asInt(0);
        String url = "/api/workspace/file?path=gen/" + fname;
        LOG.info("bedtime_story '{}' -> {} ({} segments, {}s, {} bytes)", title, fname, segs, dur, mp3.length);
        // Verbatim render — the music-bed gen pauses Qwen, so don't make a final synthesis call.
        return ToolResult.ok(DeterministicStockOrchestratorTool.VERBATIM_MARKER
                + "Here's the story 🌙\n\n**" + title + "** (~" + Math.round(dur) + "s, "
                + segs + " parts).\n\n"
                + "![" + title.replace("]", " ") + "](" + url + ")\n\nPress play. Sweet dreams ✨");
    }

    /** Two-part async render: kick off the job, return Part 1 as soon as it's ready, and write Part 2
     *  on a background thread when the job finishes. The chat shows Part 1 immediately + a Part 2
     *  placeholder that the UI swaps for a player once {@code gen/story-<id>-b.mp3} appears. */
    private ToolResult renderAsyncSplit(String title, JsonNode script) throws Exception {
        HttpResponse<String> r;
        try {
            r = http.send(HttpRequest.newBuilder(URI.create(sidecarUrl + "/narrate-story-async"))
                    .timeout(Duration.ofSeconds(15)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(script), StandardCharsets.UTF_8))
                    .build(), HttpResponse.BodyHandlers.ofString());
        } catch (java.net.ConnectException e) {
            return ToolResult.error("voice sidecar not reachable — is nizo-voice running?");
        }
        if (r.statusCode() / 100 != 2) return ToolResult.error("story render failed to start: " + r.body());
        String jobId = MAPPER.readTree(r.body()).path("jobId").asText("");
        if (jobId.isEmpty()) return ToolResult.error("renderer returned no jobId");

        Path genDir = workspace.resolve("gen");
        Files.createDirectories(genDir);
        String id = UUID.randomUUID().toString().substring(0, 8);
        final String aName = "story-" + id + "-a.mp3";
        final String bName = "story-" + id + "-b.mp3";

        JsonNode job = pollJob(jobId, "partA_done", 300);   // wait up to 5 min for Part 1
        if (job == null) return ToolResult.error("Part 1 took too long to render — try a shorter story.");
        byte[] aMp3 = Base64.getDecoder().decode(job.path("partA").path("audio_b64").asText(""));
        if (aMp3.length == 0) return ToolResult.error("Part 1 produced no audio");
        Files.write(genDir.resolve(aName), aMp3);

        // Background: poll to completion, then write Part 2 so the UI's file-poll can pick it up.
        TIMEOUT_EXEC.submit(() -> {
            try {
                JsonNode done = pollJob(jobId, "done", 900);
                if (done != null && done.has("partB")) {
                    byte[] bMp3 = Base64.getDecoder().decode(done.path("partB").path("audio_b64").asText(""));
                    if (bMp3.length > 0) {
                        Files.write(genDir.resolve(bName), bMp3);
                        LOG.info("bedtime_story Part 2 written: {}", bName);
                    }
                }
            } catch (Exception e) { LOG.warn("bedtime_story Part 2 failed: {}", e.toString()); }
        });

        String alt = title.replace("]", " ");
        LOG.info("bedtime_story '{}' async two-part: part1={} part2={}", title, aName, bName);
        // Render verbatim (no final LLM call): the background Part 2 render pauses Qwen, so an extra
        // synthesis call here would race a paused model. The tool result IS the reply.
        return ToolResult.ok(DeterministicStockOrchestratorTool.VERBATIM_MARKER
                + "Here's the story 🌙\n\n**" + title + "** — narrated in two parts, in your voice.\n\n"
                + "![" + alt + " · Part 1](/api/workspace/file?path=gen/" + aName + ")\n\n"
                + "[STORY-PART-2 file=gen/" + bName + "]\n\nPart 1 plays now; Part 2 appears when it finishes rendering. Sweet dreams ✨");
    }

    /** Poll GET /jobs/{id} until status is {@code untilStatus} or "done" (or "failed"→null). */
    private JsonNode pollJob(String jobId, String untilStatus, int maxSeconds) {
        long deadline = System.currentTimeMillis() + maxSeconds * 1000L;
        while (System.currentTimeMillis() < deadline) {
            try {
                HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create(sidecarUrl + "/jobs/" + jobId))
                        .timeout(Duration.ofSeconds(10)).GET().build(), HttpResponse.BodyHandlers.ofString());
                if (r.statusCode() / 100 == 2) {
                    JsonNode j = MAPPER.readTree(r.body());
                    String st = j.path("status").asText("");
                    if ("failed".equals(st)) return null;
                    if ("done".equals(st) || untilStatus.equals(st)) return j;
                }
            } catch (Exception ignore) { /* transient — keep polling */ }
            try { Thread.sleep(4000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); return null; }
        }
        return null;
    }

    /** Pull the first balanced {...} JSON object out of an LLM reply (tolerates fences/prose). */
    static JsonNode extractJson(String s) {
        if (s == null) return null;
        int start = s.indexOf('{');
        if (start < 0) return null;
        int depth = 0; boolean inStr = false; boolean esc = false;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (inStr) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') inStr = false;
            } else {
                if (c == '"') inStr = true;
                else if (c == '{') depth++;
                else if (c == '}') { depth--; if (depth == 0) {
                    try { return MAPPER.readTree(s.substring(start, i + 1)); }
                    catch (Exception e) { return null; }
                } }
            }
        }
        return null;
    }

    private ChatResponse chatWithTimeout(ChatRequest req, int seconds) throws Exception {
        Future<ChatResponse> f = TIMEOUT_EXEC.submit(() -> llm.chat(req));
        try {
            return f.get(seconds, TimeUnit.SECONDS);
        } catch (Exception e) {
            f.cancel(true);
            throw e;
        }
    }
}
