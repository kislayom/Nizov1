package ai.nizo.tools.file;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Write a UTF-8 text file into the agent workspace. Creates parent directories as needed.
 *
 * <p>Path is resolved relative to {@code NIZO_HOME/workspace}; absolute paths and traversal
 * out of the workspace are rejected. Symlinks are resolved via {@link Path#toRealPath} on the
 * closest existing ancestor before the boundary check, so a symlink directory inside the
 * workspace pointing at {@code /etc} is rejected before any write happens.
 */
public final class FileWriteTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final Path workspace;

    public FileWriteTool(Path workspace) {
        try {
            Files.createDirectories(workspace);
            this.workspace = WorkspacePaths.resolveWorkspaceReal(workspace);
        } catch (IOException e) {
            throw new RuntimeException("could not initialize file workspace " + workspace, e);
        }
    }

    @Override public String name() { return "file_write"; }

    @Override
    public String description() {
        return "Write text content to a file in the agent workspace. By default overwrites. "
                + "Set 'append' to true to append instead. Use this for notes, drafts, scratch files.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "path":    { "type": "string", "description": "Workspace-relative path." },
                "content": { "type": "string", "description": "UTF-8 text body." },
                "append":  { "type": "boolean", "description": "Append instead of overwrite (default false)." }
              },
              "required": ["path", "content"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String relPath = args.path("path").asText("").trim();
        if (relPath.isEmpty()) return ToolResult.error("path is required");
        String content = args.path("content").asText("");
        boolean append = args.path("append").asBoolean(false);

        Path target;
        try {
            target = WorkspacePaths.resolveSafe(workspace, relPath);
        } catch (SecurityException e) {
            return ToolResult.error(e.getMessage());
        }
        if (target.getParent() != null) Files.createDirectories(target.getParent());

        byte[] bytes = content.getBytes(StandardCharsets.UTF_8);
        if (append) {
            Files.write(target, bytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } else {
            Files.write(target, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }
        return ToolResult.ok("wrote " + bytes.length + " bytes to " + relPath
                + (append ? " (append)" : ""));
    }
}
