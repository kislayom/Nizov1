package ai.nizo.tools.file;

import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Symlink-escape and traversal tests for {@link FileReadTool}. These exercise the boundary
 * established by {@link WorkspacePaths#resolveSafe} — a normalize()-only check would let any
 * of these through.
 */
class FileReadToolTest {

    @Test
    void readsRegularFileInsideWorkspace(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.writeString(workspace.resolve("hello.txt"), "world");

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"hello.txt\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("world"));
    }

    @Test
    void rejectsSymlinkPointingOutsideWorkspace(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Path outside = tmp.resolve("secret.txt");
        Files.writeString(outside, "TOP-SECRET");

        // Try to skip the test on filesystems that can't make symlinks.
        Path link = workspace.resolve("escape");
        try {
            Files.createSymbolicLink(link, outside);
        } catch (UnsupportedOperationException | IOException e) {
            return; // skip — no symlink support
        }

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"escape\"}");
        assertFalse(r.ok(), "should reject symlink that escapes workspace");
        assertTrue(r.content().toLowerCase().contains("outside workspace")
                        || r.content().toLowerCase().contains("symlink"),
                "expected explicit symlink/traversal error, got: " + r.content());
        assertFalse(r.content().contains("TOP-SECRET"),
                "must not leak file contents in the error message");
    }

    @Test
    void allowsWorkspaceInternalSymlink(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.writeString(workspace.resolve("real.txt"), "actual");

        Path link = workspace.resolve("alias.txt");
        try {
            Files.createSymbolicLink(link, workspace.resolve("real.txt"));
        } catch (UnsupportedOperationException | IOException e) {
            return; // skip
        }

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"alias.txt\"}");
        assertTrue(r.ok(), "internal symlink should be allowed: " + r.content());
        assertTrue(r.content().contains("actual"));
    }

    @Test
    void rejectsLexicalTraversal(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.writeString(tmp.resolve("victim"), "leaked");

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"../victim\"}");
        assertFalse(r.ok(), "../ traversal must be rejected");
        assertFalse(r.content().contains("leaked"));
    }

    @Test
    void rejectsAbsolutePath(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileReadTool t = new FileReadTool(workspace);

        ToolResult r = t.execute("{\"path\":\"/etc/passwd\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("absolute")
                        || r.content().toLowerCase().contains("outside workspace"),
                "expected absolute-path rejection, got: " + r.content());
    }

    @Test
    void rejectsSymlinkChainExitingViaSubdirectory(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace.resolve("sub"));
        Path outside = tmp.resolve("secrets-dir");
        Files.createDirectories(outside);
        Files.writeString(outside.resolve("k.txt"), "KEY");

        Path link = workspace.resolve("sub").resolve("hop");
        try {
            Files.createSymbolicLink(link, outside);
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"sub/hop/k.txt\"}");
        assertFalse(r.ok(), "symlink directory in middle of path must be rejected");
        assertFalse(r.content().contains("KEY"));
    }

    @Test
    void missingFileGivesNotFound(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"nope.txt\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("not found"));
    }

    @Test
    void emptyPathRejected(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"));
    }

    @Test
    void truncatesAtMaxBytes(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        StringBuilder big = new StringBuilder();
        for (int i = 0; i < 100; i++) big.append("0123456789");
        Files.writeString(workspace.resolve("big.txt"), big.toString());

        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"big.txt\",\"max_bytes\":10}");
        assertTrue(r.ok());
        assertTrue(r.content().contains("(truncated)"));
        // Body after the header should be exactly 10 bytes.
        int idx = r.content().indexOf("\n---\n");
        String body = r.content().substring(idx + 5);
        assertEquals("0123456789", body);
    }

    @Test
    void readingDirectoryFails(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace.resolve("subdir"));
        FileReadTool t = new FileReadTool(workspace);
        ToolResult r = t.execute("{\"path\":\"subdir\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("not a regular file"));
    }
}
