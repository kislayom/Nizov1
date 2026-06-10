package ai.nizo.tools.file;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * List files in a workspace-relative directory.
 *
 * <p>Path is resolved relative to the workspace; absolute paths and traversal out of the
 * workspace are rejected. Symlinks are fully resolved via {@link Path#toRealPath} before the
 * boundary check.
 */
public final class FileListTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final Path workspace;

    public FileListTool(Path workspace) {
        try {
            Files.createDirectories(workspace);
            this.workspace = WorkspacePaths.resolveWorkspaceReal(workspace);
        } catch (IOException e) {
            throw new RuntimeException("could not initialize file workspace " + workspace, e);
        }
    }

    @Override public String name() { return "file_list"; }

    @Override
    public String description() {
        return "List files and directories under a workspace-relative path. Returns one entry per line "
                + "with size and type. Use 'path' = '.' for the workspace root.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "path": { "type": "string", "description": "Workspace-relative directory. Default '.'." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String relPath = args.path("path").asText(".").trim();
        if (relPath.isEmpty() || relPath.equals(".") || relPath.equals("./")) relPath = "";

        Path target;
        try {
            target = WorkspacePaths.resolveSafe(workspace, relPath);
        } catch (SecurityException e) {
            return ToolResult.error(e.getMessage());
        }
        if (!Files.exists(target)) return ToolResult.error("not found: " + relPath);
        if (!Files.isDirectory(target)) return ToolResult.error("not a directory");

        StringBuilder sb = new StringBuilder();
        sb.append("listing of: ").append(relPath.isEmpty() ? "." : relPath).append("\n---\n");
        try (Stream<Path> s = Files.list(target)) {
            s.sorted(Comparator.comparing(p -> p.getFileName().toString()))
             .forEach(p -> {
                String name = p.getFileName().toString();
                try {
                    if (Files.isDirectory(p)) {
                        sb.append("DIR  ").append(name).append("/\n");
                    } else {
                        long sz = Files.size(p);
                        sb.append(String.format("%-4s %10d  %s%n", "FILE", sz, name));
                    }
                } catch (Exception e) {
                    sb.append("ERR  ").append(name).append(" (").append(e.getMessage()).append(")\n");
                }
            });
        }
        return ToolResult.ok(sb.toString().trim());
    }
}
