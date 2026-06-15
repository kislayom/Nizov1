package ai.nizo.tools.vision;

import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ImageAnalyzeToolTest {

    /** Captures the request so we can assert the image was passed; returns a canned description. */
    private static final class CaptureLlm implements LlmClient {
        ChatRequest last;
        final String reply;
        CaptureLlm(String reply) { this.reply = reply; }
        @Override public ChatResponse chat(ChatRequest r) {
            last = r;
            return new ChatResponse(reply, List.of(), "stop", ChatResponse.Usage.EMPTY);
        }
    }

    @Test
    void readsImageAndPassesItToVisionModel(@TempDir Path tmp) throws Exception {
        Path ws = Files.createDirectories(tmp.resolve("ws"));
        Files.write(ws.resolve("shot.png"), new byte[]{(byte) 0x89, 'P', 'N', 'G', 1, 2, 3});
        CaptureLlm llm = new CaptureLlm("A red square on white.");
        ImageAnalyzeTool t = new ImageAnalyzeTool(llm, "m", ws);

        ToolResult r = t.execute("{\"path\":\"shot.png\",\"question\":\"what is shown?\"}");
        assertTrue(r.ok(), r.content());
        assertEquals("A red square on white.", r.content());

        // The model received exactly one image, as a png data URI.
        List<String> imgs = llm.last.messages().get(0).images();
        assertEquals(1, imgs.size());
        assertTrue(imgs.get(0).startsWith("data:image/png;base64,"), imgs.get(0).substring(0, 30));
    }

    @Test
    void missingFileIsAnError(@TempDir Path tmp) throws Exception {
        Path ws = Files.createDirectories(tmp.resolve("ws"));
        ImageAnalyzeTool t = new ImageAnalyzeTool(new CaptureLlm("x"), "m", ws);
        ToolResult r = t.execute("{\"path\":\"nope.png\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("no such image"), r.content());
    }

    @Test
    void pathEscapeRejected(@TempDir Path tmp) throws Exception {
        Path ws = Files.createDirectories(tmp.resolve("ws"));
        Files.write(tmp.resolve("outside.png"), new byte[]{1, 2, 3});   // sibling of ws
        ImageAnalyzeTool t = new ImageAnalyzeTool(new CaptureLlm("x"), "m", ws);
        ToolResult r = t.execute("{\"path\":\"../outside.png\"}");
        assertFalse(r.ok(), "must not read outside the workspace: " + r.content());
        assertTrue(r.content().toLowerCase().contains("escape"), r.content());
    }

    @Test
    void mimeFromExtension() {
        assertEquals("image/png", ImageAnalyzeTool.mime("a/b.png"));
        assertEquals("image/jpeg", ImageAnalyzeTool.mime("x.JPG"));
        assertEquals("image/webp", ImageAnalyzeTool.mime("y.webp"));
    }
}
