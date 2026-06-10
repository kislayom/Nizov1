package ai.nizo.tools.file;

import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class FileWriteToolTest {

    @Test
    void writesNewFile(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"a.txt\",\"content\":\"hello\"}");
        assertTrue(r.ok(), r.content());
        assertEquals("hello", Files.readString(workspace.resolve("a.txt")));
    }

    @Test
    void createsMissingParentsForNonexistentPath(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"deep/nested/new.txt\",\"content\":\"x\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(Files.exists(workspace.resolve("deep/nested/new.txt")));
        assertEquals("x", Files.readString(workspace.resolve("deep/nested/new.txt")));
    }

    @Test
    void rejectsWriteThroughSymlinkPointingOutsideWorkspace(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Path outsideDir = tmp.resolve("outside");
        Files.createDirectories(outsideDir);

        // workspace/escape -> outside  (a symlinked dir inside workspace pointing out)
        Path link = workspace.resolve("escape");
        try {
            Files.createSymbolicLink(link, outsideDir);
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"escape/pwned.txt\",\"content\":\"P\"}");
        assertFalse(r.ok(), "write through escape symlink must be rejected");
        assertFalse(Files.exists(outsideDir.resolve("pwned.txt")),
                "no file should have been created outside workspace");
    }

    @Test
    void rejectsLeafSymlinkPointingOutside(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Path outside = tmp.resolve("victim.txt");
        Files.writeString(outside, "ORIGINAL");

        Path link = workspace.resolve("alias.txt");
        try {
            Files.createSymbolicLink(link, outside);
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"alias.txt\",\"content\":\"REPLACED\"}");
        assertFalse(r.ok(), "writing through escape-symlink must be rejected");
        // Original outside the workspace must be untouched.
        assertEquals("ORIGINAL", Files.readString(outside),
                "external file must be untouched after rejected write");
    }

    @Test
    void allowsWorkspaceInternalSymlinkWrite(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace.resolve("real"));
        Path link = workspace.resolve("alias");
        try {
            Files.createSymbolicLink(link, workspace.resolve("real"));
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"alias/inside.txt\",\"content\":\"ok\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(Files.exists(workspace.resolve("real/inside.txt")));
    }

    @Test
    void rejectsLexicalTraversal(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"../escape.txt\",\"content\":\"x\"}");
        assertFalse(r.ok());
        assertFalse(Files.exists(tmp.resolve("escape.txt")));
    }

    @Test
    void rejectsAbsolutePath(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"" + tmp.resolve("oops.txt").toAbsolutePath().toString().replace("\\","\\\\") + "\",\"content\":\"x\"}");
        assertFalse(r.ok());
        assertFalse(Files.exists(tmp.resolve("oops.txt")));
    }

    @Test
    void appendModeAppends(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        assertTrue(t.execute("{\"path\":\"log.txt\",\"content\":\"line1\\n\"}").ok());
        assertTrue(t.execute("{\"path\":\"log.txt\",\"content\":\"line2\\n\",\"append\":true}").ok());
        assertEquals("line1\nline2\n", Files.readString(workspace.resolve("log.txt")));
    }

    @Test
    void emptyPathRejected(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileWriteTool t = new FileWriteTool(workspace);
        ToolResult r = t.execute("{\"path\":\"\",\"content\":\"x\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("required"));
    }
}
