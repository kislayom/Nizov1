package ai.nizo.skills;

import java.nio.file.Path;
import java.util.List;

/**
 * Parsed contents of a single skill's {@code SKILL.md} file (agentskills.io-style).
 *
 * <p>SKILL.md uses YAML frontmatter for metadata and a markdown body for the skill content
 * (instructions, prompts, code snippets) that gets surfaced to the model when the skill is
 * invoked.
 *
 * <pre>
 *   ---
 *   name: morning_briefing
 *   description: Generate a morning briefing of weather, market news, and goals.
 *   when_to_use: User asks for a morning briefing or 'how is my day looking'.
 *   tags: [routine, daily]
 *   ---
 *
 *   # Morning Briefing
 *
 *   Steps:
 *   1. Call current_time to anchor the date.
 *   2. Call web_search for "[city] weather today" and "stocks today".
 *   3. Compose a 5-bullet briefing.
 *   ...
 * </pre>
 */
public record SkillManifest(
        String name,
        String description,
        String whenToUse,
        List<String> tags,
        boolean agent,
        String body,
        Path source
) {
    public SkillManifest {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("skill name required");
        description = description == null ? "" : description;
        whenToUse   = whenToUse == null ? "" : whenToUse;
        tags = tags == null ? List.of() : List.copyOf(tags);
        body = body == null ? "" : body;
    }

    /** Backwards-compat constructor for callers that don't set the agent flag. */
    public SkillManifest(String name, String description, String whenToUse,
                         List<String> tags, String body, Path source) {
        this(name, description, whenToUse, tags, false, body, source);
    }

    public String modelDescription() {
        StringBuilder sb = new StringBuilder();
        if (!description.isBlank()) sb.append(description);
        if (!whenToUse.isBlank()) {
            if (sb.length() > 0) sb.append(" ");
            sb.append("Use when: ").append(whenToUse);
        }
        return sb.toString();
    }
}
