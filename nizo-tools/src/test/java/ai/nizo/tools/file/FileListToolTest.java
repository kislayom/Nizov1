package ai.nizo.tools.file;

import ai.nizo.api.tool.ToolResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class FileListToolTest {

    @Test
    void listsWorkspaceRoot(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace.resolve("subdir"));
        Files.writeString(workspace.resolve("a.txt"), "x");

        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\".\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("a.txt"));
        assertTrue(r.content().contains("subdir/"));
    }

    @Test
    void emptyPathTreatedAsRoot(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.writeString(workspace.resolve("only.txt"), "x");
        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("only.txt"));
    }

    @Test
    void rejectsLexicalTraversal(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.createDirectories(tmp.resolve("outside"));
        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"../outside\"}");
        assertFalse(r.ok());
    }

    @Test
    void rejectsSymlinkDirectoryToOutside(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Path outside = tmp.resolve("classified");
        Files.createDirectories(outside);
        Files.writeString(outside.resolve("doc.txt"), "x");

        Path link = workspace.resolve("peek");
        try {
            Files.createSymbolicLink(link, outside);
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"peek\"}");
        assertFalse(r.ok(), "listing through escape-symlink must be rejected");
        assertFalse(r.content().contains("doc.txt"),
                "must not enumerate files outside the workspace");
    }

    @Test
    void allowsWorkspaceInternalSymlinkDir(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace.resolve("real"));
        Files.writeString(workspace.resolve("real").resolve("inside.txt"), "x");
        Path link = workspace.resolve("alias");
        try {
            Files.createSymbolicLink(link, workspace.resolve("real"));
        } catch (UnsupportedOperationException | IOException e) {
            return;
        }

        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"alias\"}");
        assertTrue(r.ok(), r.content());
        assertTrue(r.content().contains("inside.txt"));
    }

    @Test
    void rejectsAbsolutePath(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"/etc\"}");
        assertFalse(r.ok());
    }

    @Test
    void notFoundDir(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"missing\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("not found"));
    }

    @Test
    void notADirectory(@TempDir Path tmp) throws Exception {
        Path workspace = tmp.resolve("ws");
        Files.createDirectories(workspace);
        Files.writeString(workspace.resolve("file.txt"), "x");
        FileListTool t = new FileListTool(workspace);
        ToolResult r = t.execute("{\"path\":\"file.txt\"}");
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("not a directory"));
    }
}
