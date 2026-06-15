package ai.nizo.tools.vision;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Locale;

/**
 * Look at an image and answer a question about it — the agent's "eyes". Reads an image from the
 * workspace, sends it to the vision-capable model as a data URI, and returns the model's text.
 *
 * <p>Pairs with the browser tool's {@code screenshot} action (saved into the workspace): the agent
 * can screenshot a page it can't parse from the DOM, then {@code image_analyze} it — a visual web
 * fallback. Also handles photos, charts, scans, diagrams the user drops in the workspace.
 *
 * <p>Path-guarded to the workspace ({@code toRealPath} + prefix check, same as the file tools) so it
 * can't read images outside it.
 */
public final class ImageAnalyzeTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long MAX_BYTES = 12L * 1024 * 1024;   // 12 MB — Qwen3-VL handles large images

    private final LlmClient llm;
    private final String model;
    private final Path workspace;

    public ImageAnalyzeTool(LlmClient llm, String model, Path workspace) {
        this.llm = llm;
        this.model = model;
        this.workspace = workspace;
    }

    @Override public String name() { return "image_analyze"; }

    @Override
    public String description() {
        return "Look at an image in the workspace and answer a question about it (the model is "
                + "vision-capable). Use for photos, charts, scanned documents, diagrams — and as a "
                + "VISUAL FALLBACK for the browser: take a browser 'screenshot' (it saves into the "
                + "workspace), then analyze it here when the page's DOM is hard to read. Args: "
                + "path (workspace-relative image file), question (what to find out).";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "path":     { "type": "string", "description": "Workspace-relative path to the image (png/jpg/webp/gif)." },
                "question": { "type": "string", "description": "What to find out about the image. Default: describe it in detail." }
              },
              "required": ["path"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        JsonNode a;
        try { a = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson); }
        catch (Exception e) { return ToolResult.error("bad arguments JSON: " + e.getMessage()); }

        String rel = a.path("path").asText("").trim();
        if (rel.isEmpty()) return ToolResult.error("path is required (a workspace-relative image file)");
        String question = a.path("question").asText("").trim();
        if (question.isEmpty()) question = "Describe this image in detail. Transcribe any visible text.";

        Path img = workspace.resolve(rel).normalize();
        if (!Files.isRegularFile(img)) return ToolResult.error("no such image: " + rel);
        try {
            if (!img.toRealPath().startsWith(workspace.toRealPath())) {
                return ToolResult.error("path escapes the workspace");
            }
        } catch (Exception e) {
            return ToolResult.error("cannot resolve image path: " + e.getMessage());
        }

        byte[] bytes;
        try {
            long sz = Files.size(img);
            if (sz > MAX_BYTES) return ToolResult.error("image too large (" + sz + " bytes; max " + MAX_BYTES + ")");
            bytes = Files.readAllBytes(img);
        } catch (Exception e) {
            return ToolResult.error("could not read image: " + e.getMessage());
        }

        String dataUri = "data:" + mime(rel) + ";base64," + Base64.getEncoder().encodeToString(bytes);
        ChatRequest req = ChatRequest.of(model, List.of(ChatMessage.userWithImages(question, List.of(dataUri))));
        try {
            ChatResponse resp = llm.chat(req);
            String text = resp == null ? "" : (resp.content() == null ? "" : resp.content().strip());
            return text.isBlank() ? ToolResult.error("the vision model returned no description")
                                  : ToolResult.ok(text);
        } catch (Exception e) {
            return ToolResult.error("vision call failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    static String mime(String name) {
        String n = name.toLowerCase(Locale.ROOT);
        if (n.endsWith(".png")) return "image/png";
        if (n.endsWith(".jpg") || n.endsWith(".jpeg")) return "image/jpeg";
        if (n.endsWith(".webp")) return "image/webp";
        if (n.endsWith(".gif")) return "image/gif";
        return "image/png";
    }
}
