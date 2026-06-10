package ai.nizo.tools.file;

import ai.nizo.api.condense.FileCache;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Read a UTF-8 text file from inside the agent workspace.
 *
 * <p>Path is resolved relative to {@code NIZO_HOME/workspace}; absolute paths and traversal
 * out of the workspace are rejected. Symlinks are fully resolved via
 * {@link Path#toRealPath} before the boundary check, so a symlink inside the workspace
 * pointing at e.g. {@code /etc/passwd} is rejected.
 *
 * <p>Successful reads are recorded into the supplied {@link FileCache} so the condense engine
 * can re-attach them after summarization. Pass {@link FileCache#NOOP} when re-injection isn't
 * wanted (e.g. unit tests).
 */
public final class FileReadTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int MAX_BYTES = 256 * 1024;
    /** Conservative chars-per-token estimate; matches TokenEstimator without depending on it. */
    private static final double CHARS_PER_TOKEN = 3.5;

    private final Path workspace;
    private final FileCache fileCache;

    public FileReadTool(Path workspace) { this(workspace, FileCache.NOOP); }

    public FileReadTool(Path workspace, FileCache fileCache) {
        try {
            Files.createDirectories(workspace);
            this.workspace = WorkspacePaths.resolveWorkspaceReal(workspace);
        } catch (IOException e) {
            throw new RuntimeException("could not initialize file workspace " + workspace, e);
        }
        this.fileCache = fileCache == null ? FileCache.NOOP : fileCache;
    }

    @Override public String name() { return "file_read"; }

    @Override
    public String description() {
        return "Read a UTF-8 text file from the agent workspace. Returns up to 256 KB. "
                + "Use 'path' relative to the workspace, e.g. 'notes/2026-04-30.md'.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "path":     { "type": "string", "description": "Workspace-relative file path." },
                "max_bytes":{ "type": "integer", "description": "Optional cap, default 262144." }
              },
              "required": ["path"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String relPath = args.path("path").asText("").trim();
        if (relPath.isEmpty()) return ToolResult.error("path is required");
        int max = args.path("max_bytes").asInt(MAX_BYTES);
        if (max <= 0) max = MAX_BYTES;

        Path target;
        try {
            target = WorkspacePaths.resolveSafe(workspace, relPath);
        } catch (SecurityException e) {
            return ToolResult.error(e.getMessage());
        }
        if (!Files.exists(target)) return ToolResult.error("not found: " + relPath);
        if (!Files.isRegularFile(target)) return ToolResult.error("not a regular file");
        long size = Files.size(target);
        byte[] bytes = Files.readAllBytes(target);
        boolean truncated = bytes.length > max;
        String body = new String(bytes, 0, Math.min(bytes.length, max), StandardCharsets.UTF_8);
        String header = "path=" + relPath + " size=" + size + (truncated ? " (truncated)" : "") + "\n---\n";

        // Record into the file cache so a future condense can re-inject this file. Stores the
        // raw body (not the header), since the agent doesn't need the bookkeeping line on rehydrate.
        int tokens = (int) Math.ceil(body.length() / CHARS_PER_TOKEN);
        try { fileCache.record(relPath, body, tokens); }
        catch (Exception ignore) { /* never let cache bookkeeping break a read */ }

        return ToolResult.ok(header + body);
    }
}
