package ai.nizo.skills;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;

/**
 * Wraps a filesystem skill as a callable {@link Tool}. The "execution" of a skill is to
 * return the skill's body text — the agent loop then uses that text as guidance for the
 * model on how to perform the task. Inspired by Hermes' procedural-memory pattern.
 */
public final class FilesystemSkillTool implements Tool {

    private final SkillManifest manifest;

    public FilesystemSkillTool(SkillManifest manifest) {
        this.manifest = manifest;
    }

    @Override public String name() { return "skill_" + manifest.name(); }

    @Override
    public String description() {
        String d = manifest.modelDescription();
        return (d == null || d.isBlank())
                ? "Run the '" + manifest.name() + "' skill (filesystem-loaded)."
                : d;
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "input": { "type": "string", "description": "Optional context to pass to the skill." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Skill: ").append(manifest.name()).append("\n");
        if (!manifest.description().isBlank()) sb.append("_").append(manifest.description()).append("_\n");
        sb.append("\n");
        sb.append(manifest.body());
        sb.append("\n\n_(call other tools as needed to carry out the steps above.)_");
        return ToolResult.ok(sb.toString());
    }

    public SkillManifest manifest() { return manifest; }
}
