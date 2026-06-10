package ai.nizo.skills;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.regex.Pattern;

/**
 * Lets the agent author and persist a new filesystem skill.
 *
 * <p>The agent calls this with {@code name}, {@code description}, and {@code body} (markdown
 * instructions). We write {@code <skillsDir>/<name>/SKILL.md} with proper frontmatter so the
 * skill becomes discoverable by {@link SkillLoader} on the next start.
 *
 * <p>Hermes calls this "self-improvement" — the agent records what worked so it can do it
 * faster next time. Pair with reflection.
 */
public final class SaveSkillTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern SAFE_NAME = Pattern.compile("^[a-z0-9_]+$");

    private final Path skillsDir;

    public SaveSkillTool(Path skillsDir) {
        this.skillsDir = skillsDir;
    }

    @Override public String name() { return "save_skill"; }

    @Override
    public String description() {
        return "Save a reusable skill to disk. Use this when you've worked out a procedure "
                + "the user will likely repeat (e.g. 'morning briefing', 'review my codebase'). "
                + "name must be snake_case. The body should be a short markdown plan the next "
                + "agent run can follow. Skill becomes available as 'skill_<name>' after restart.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "name":         { "type": "string", "description": "snake_case identifier" },
                "description":  { "type": "string", "description": "one-paragraph 'what this skill is'" },
                "when_to_use":  { "type": "string", "description": "trigger phrasing — when the agent should invoke this skill" },
                "body":         { "type": "string", "description": "markdown body — the plan / instructions" },
                "tags":         { "type": "array",  "items": {"type": "string"} }
              },
              "required": ["name", "description", "body"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String name = args.path("name").asText("").trim();
        String desc = args.path("description").asText("").trim();
        String whenToUse = args.path("when_to_use").asText("").trim();
        String body = args.path("body").asText("");
        if (name.isEmpty()) return ToolResult.error("name is required");
        if (!SAFE_NAME.matcher(name).matches()) return ToolResult.error("name must be snake_case ([a-z0-9_]+)");
        if (desc.isEmpty()) return ToolResult.error("description is required");
        if (body.isBlank()) return ToolResult.error("body is required");

        StringBuilder tagsLine = new StringBuilder();
        if (args.has("tags") && args.get("tags").isArray()) {
            tagsLine.append("[");
            boolean first = true;
            for (JsonNode t : args.get("tags")) {
                if (!t.isTextual()) continue;
                if (!first) tagsLine.append(", ");
                tagsLine.append("\"").append(t.asText().replace("\"", "\\\"")).append("\"");
                first = false;
            }
            tagsLine.append("]");
        } else {
            tagsLine.append("[]");
        }

        Path skillDir = skillsDir.resolve(name);
        Files.createDirectories(skillDir);
        Path file = skillDir.resolve("SKILL.md");

        StringBuilder content = new StringBuilder();
        content.append("---\n");
        content.append("name: ").append(name).append("\n");
        content.append("description: ").append(desc.replace("\n", " ")).append("\n");
        if (!whenToUse.isEmpty()) {
            content.append("when_to_use: ").append(whenToUse.replace("\n", " ")).append("\n");
        }
        content.append("tags: ").append(tagsLine).append("\n");
        content.append("---\n\n");
        content.append(body);
        if (!body.endsWith("\n")) content.append("\n");

        Files.write(file, content.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        return ToolResult.ok("saved skill '" + name + "' → " + file
                + " (will be available as 'skill_" + name + "' after restart)");
    }
}
