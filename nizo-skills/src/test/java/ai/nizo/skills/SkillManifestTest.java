package ai.nizo.skills;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the {@code agent: true} frontmatter parsing that decides whether a skill
 * runs as a passive instruction-return ({@link FilesystemSkillTool}) or as a real
 * sub-agent loop ({@link SubAgentSkillTool}).
 */
class SkillManifestTest {

    private final SkillLoader loader = new SkillLoader();

    @Test
    void agentTrue_setsAgentFlag(@TempDir Path tmp) throws Exception {
        Path skill = writeSkill(tmp, "fundamentals_analyst", """
                ---
                name: fundamentals_analyst
                description: Deep financial dive
                tags: [finance, sub-skill]
                agent: true
                ---

                # Body
                Do research.
                """);
        SkillManifest m = loader.parse(skill);
        assertNotNull(m);
        assertEquals("fundamentals_analyst", m.name());
        assertTrue(m.agent(), "agent: true frontmatter should set agent=true");
    }

    @Test
    void agentFalse_setsAgentFlagFalse(@TempDir Path tmp) throws Exception {
        Path skill = writeSkill(tmp, "orchestrator", """
                ---
                name: orchestrator
                tags: [routine]
                agent: false
                ---

                # Body
                """);
        SkillManifest m = loader.parse(skill);
        assertFalse(m.agent());
    }

    @Test
    void agentMissing_defaultsToFalse(@TempDir Path tmp) throws Exception {
        Path skill = writeSkill(tmp, "passive", """
                ---
                name: passive
                tags: [routine]
                ---

                # Body
                """);
        SkillManifest m = loader.parse(skill);
        assertFalse(m.agent(),
                "Skills without an agent: line should default to passive (FilesystemSkillTool)");
    }

    @Test
    void agentTrueWithVariousCasing_recognized(@TempDir Path tmp) throws Exception {
        // YAML truthiness — accept "true", "yes", "1"
        for (String val : List.of("true", "True", "TRUE", "yes", "Yes", "1")) {
            Path skill = writeSkill(tmp.resolve(val), "s", "---\nname: s\nagent: " + val + "\n---\nbody\n");
            assertTrue(loader.parse(skill).agent(),
                    "agent: " + val + " should be truthy");
        }
    }

    @Test
    void agentRandomString_isFalsy(@TempDir Path tmp) throws Exception {
        Path skill = writeSkill(tmp, "s", "---\nname: s\nagent: maybe\n---\nbody\n");
        assertFalse(loader.parse(skill).agent());
    }

    @Test
    void allSevenStockSubSkillsAreAgents(@TempDir Path tmp) throws Exception {
        // Mirror what's actually shipped in deploy/server/skills/
        for (String name : List.of(
                "stock_fundamentals_analyst", "stock_news_analyst",
                "stock_sentiment_analyst", "stock_technical_analyst",
                "stock_bull_researcher", "stock_bear_researcher", "stock_trader")) {
            Path skill = writeSkill(tmp.resolve(name), name,
                    "---\nname: " + name + "\nagent: true\n---\nbody\n");
            SkillManifest m = loader.parse(skill);
            assertTrue(m.agent(), name + " should be flagged as agent");
        }
    }

    @Test
    void orchestratorSkillIsNotAgent(@TempDir Path tmp) throws Exception {
        // The orchestrator (stock_analysis) must NOT be an agent — otherwise it'd recurse.
        Path skill = writeSkill(tmp, "stock_analysis",
                "---\nname: stock_analysis\ntags: [orchestrator]\n---\nbody\n");
        SkillManifest m = loader.parse(skill);
        assertFalse(m.agent(),
                "Orchestrator skill must not be flagged as agent (would cause recursion)");
    }

    private static Path writeSkill(Path parent, String dirName, String content) throws IOException {
        Path dir = parent.resolve(dirName);
        Files.createDirectories(dir);
        Path file = dir.resolve("SKILL.md");
        Files.writeString(file, content);
        return file;
    }
}
