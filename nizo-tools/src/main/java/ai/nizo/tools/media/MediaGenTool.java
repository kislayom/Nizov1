package ai.nizo.tools.media;

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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Text-to-image / text-to-video generation. Calls the voice/media sidecar (FLUX.1-schnell for image,
 * LTX-Video for video — both run subprocess-isolated under {@code llama_paused()} so the full GPU is
 * free), decodes the returned base64, saves it into the workspace, and hands back a markdown media
 * reference the chat renders inline ({@code ![alt](/api/workspace/file?path=gen/…)}).
 *
 * <p>One class, two modes: {@code image_generate} and {@code video_generate}.
 *
 * <p>Forces HTTP/1.1 — the sidecar is uvicorn (HTTP/1.1) and the shared HTTP/2 client silently drops
 * POST bodies over cleartext h2c (the documented BrowserTool bug).
 */
public final class MediaGenTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(MediaGenTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public enum Mode { IMAGE, VIDEO }

    private final Mode mode;
    private final Path workspace;
    private final String sidecarUrl;
    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    public MediaGenTool(Mode mode, Path workspace) {
        this(mode, workspace, System.getenv().getOrDefault("NIZO_VOICE_URL", "http://127.0.0.1:7780"));
    }

    public MediaGenTool(Mode mode, Path workspace, String sidecarUrl) {
        this.mode = mode;
        this.workspace = workspace;
        this.sidecarUrl = sidecarUrl.endsWith("/") ? sidecarUrl.substring(0, sidecarUrl.length() - 1) : sidecarUrl;
    }

    @Override public String name() { return mode == Mode.IMAGE ? "image_generate" : "video_generate"; }

    @Override
    public String description() {
        if (mode == Mode.IMAGE) {
            return "Generate an IMAGE from a text prompt (FLUX.1-schnell, runs locally on the GPU). "
                 + "Use when the user asks you to draw, create, design, or imagine a picture/logo/scene. "
                 + "Takes ~20-40s. Returns a markdown image reference — include it VERBATIM in your reply "
                 + "so the user sees the picture.";
        }
        return "Generate a short VIDEO clip from a text prompt (LTX-Video, runs locally on the GPU). "
             + "Use when the user asks for a video/animation/clip. Produces a few-second clip and takes "
             + "SEVERAL MINUTES. Returns a markdown reference (.mp4) — include it VERBATIM in your reply.";
    }

    @Override
    public String parametersJsonSchema() {
        if (mode == Mode.IMAGE) {
            return """
                {"type":"object","properties":{
                  "prompt":{"type":"string","description":"What to draw — be vivid and specific."},
                  "width":{"type":"integer","description":"px, default 1024"},
                  "height":{"type":"integer","description":"px, default 1024"},
                  "seed":{"type":"integer","description":"optional seed for reproducibility"}
                },"required":["prompt"]}""";
        }
        return """
            {"type":"object","properties":{
              "prompt":{"type":"string","description":"What the video should show."},
              "width":{"type":"integer","description":"px (multiple of 32), default 704"},
              "height":{"type":"integer","description":"px (multiple of 32), default 480"},
              "frames":{"type":"integer","description":"frame count (8k+1), default 97 (~4s)"}
            },"required":["prompt"]}""";
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String prompt = args.path("prompt").asText("").trim();
        if (prompt.isEmpty()) return ToolResult.error("prompt is required");

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("prompt", prompt);
        if (args.has("width"))  body.put("width", args.get("width").asInt());
        if (args.has("height")) body.put("height", args.get("height").asInt());
        if (mode == Mode.IMAGE) {
            if (args.has("seed")) body.put("seed", args.get("seed").asInt());
        } else {
            if (args.has("frames")) body.put("frames", args.get("frames").asInt());
        }

        String endpoint = sidecarUrl + (mode == Mode.IMAGE ? "/generate-image" : "/generate-video");
        // Image ~5 min ceiling (24 GB load + gen); video ~25 min (load + 40-step diffusion).
        Duration timeout = Duration.ofMinutes(mode == Mode.IMAGE ? 6 : 25);

        HttpResponse<String> resp;
        try {
            resp = http.send(HttpRequest.newBuilder(URI.create(endpoint))
                    .timeout(timeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body), StandardCharsets.UTF_8))
                    .build(), HttpResponse.BodyHandlers.ofString());
        } catch (java.net.ConnectException e) {
            return ToolResult.error("media sidecar not reachable at " + sidecarUrl
                    + " — is nizo-voice running? (" + e.getMessage() + ")");
        }
        if (resp.statusCode() / 100 != 2) {
            String detail = resp.body() == null ? "" : resp.body();
            if (detail.length() > 400) detail = detail.substring(0, 400);
            return ToolResult.error((mode == Mode.IMAGE ? "image" : "video") + " generation failed (HTTP "
                    + resp.statusCode() + "): " + detail);
        }

        JsonNode out = MAPPER.readTree(resp.body());
        String b64 = out.path(mode == Mode.IMAGE ? "image_b64" : "video_b64").asText("");
        if (b64.isEmpty()) return ToolResult.error("sidecar returned no media payload");
        byte[] bytes = Base64.getDecoder().decode(b64);

        String ext = mode == Mode.IMAGE ? "png" : "mp4";
        Path genDir = workspace.resolve("gen");
        Files.createDirectories(genDir);
        String fname = (mode == Mode.IMAGE ? "img-" : "vid-") + UUID.randomUUID().toString().substring(0, 8) + "." + ext;
        Files.write(genDir.resolve(fname), bytes);

        String url = "/api/workspace/file?path=" + "gen/" + fname;
        String alt = prompt.replace("]", " ").replace("\n", " ");
        if (alt.length() > 120) alt = alt.substring(0, 120);
        // Same ![alt](url) syntax for both; the UI renders <video> when the url ends in .mp4.
        String kind = mode == Mode.IMAGE ? "image" : "video";
        LOG.info("{} generated {} bytes -> {}", name(), bytes.length, fname);
        return ToolResult.ok("Generated " + kind + " for: \"" + prompt + "\"\n\n"
                + "![" + alt + "](" + url + ")\n\n"
                + "(Include the line above verbatim in your reply so the user sees the " + kind + ".)");
    }
}
