package ai.nizo.agent.reflect;

import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Offline tests for {@link SkillCurator}: it must grade ONLY reflection-authored skills, reversibly
 * retire the bad ones, leave good ones and hand-built ones alone, and never touch a skill still
 * inside its grace period.
 */
class SkillCuratorTest {

    /** Stub model: scores a skill 1 if its name contains "bad", else 8. Records which skills it saw. */
    private static final class GradingLlm implements LlmClient {
        final List<String> graded = new ArrayList<>();
        private static final Pattern NAME = Pattern.compile("Skill name: (\\S+)");
        @Override public ChatResponse chat(ChatRequest req) {
            String user = req.messages().get(req.messages().size() - 1).content();
            Matcher m = NAME.matcher(user == null ? "" : user);
            String name = m.find() ? m.group(1) : "?";
            graded.add(name);
            int score = name.contains("bad") ? 1 : 8;
            return new ChatResponse("{\"score\": " + score + ", \"reason\": \"test grade for " + name + "\"}",
                    List.of(), "stop", ChatResponse.Usage.EMPTY);
        }
    }

    private static void writeSkill(Path skillsDir, String name, boolean reflectionAuthored, String createdDate)
            throws IOException {
        Path dir = skillsDir.resolve(name);
        Files.createDirectories(dir);
        StringBuilder md = new StringBuilder();
        md.append("---\nname: ").append(name).append("\ndescription: a ").append(name)
          .append("\ntags: [learned, reflection]\n---\n\nProcedure body for ").append(name).append(".\n");
        if (reflectionAuthored) {
            md.append("\n<!-- authored by reflection from chat test on ").append(createdDate).append(" -->\n");
        }
        Files.writeString(dir.resolve("SKILL.md"), md.toString());
    }

    @Test
    void retiresBadKeepsGoodSkipsBuiltinAndNew(@TempDir Path skillsDir) throws Exception {
        String old = "2026-01-01";                       // well past the 24h grace (today is 2026-06)
        String today = LocalDate.now().toString();
        writeSkill(skillsDir, "bad-skill", true, old);    // reflection-authored, old, will score 1
        writeSkill(skillsDir, "good-skill", true, old);   // reflection-authored, old, will score 8
        writeSkill(skillsDir, "builtin-helper", false, null); // hand-built — no marker, must be ignored
        writeSkill(skillsDir, "new-bad-skill", true, today);  // reflection-authored but inside grace

        GradingLlm llm = new GradingLlm();
        SkillCurator curator = new SkillCurator(llm, "test-model", skillsDir);
        int retired = curator.runOnce();

        assertEquals(1, retired, "exactly the one old low-scoring skill should be retired");

        // bad-skill moved out of the active set, into .retired/
        assertFalse(Files.exists(skillsDir.resolve("bad-skill")), "bad-skill should have been moved out");
        Path retiredRoot = skillsDir.resolve(".retired");
        assertTrue(Files.isDirectory(retiredRoot), ".retired/ should exist");
        try (var s = Files.list(retiredRoot)) {
            assertTrue(s.anyMatch(p -> p.getFileName().toString().startsWith("bad-skill-")),
                    "retired copy of bad-skill should be present (reversible)");
        }

        // good, builtin, and the too-new skill all remain active
        assertTrue(Files.exists(skillsDir.resolve("good-skill")), "good-skill must remain");
        assertTrue(Files.exists(skillsDir.resolve("builtin-helper")), "hand-built skill must never be touched");
        assertTrue(Files.exists(skillsDir.resolve("new-bad-skill")), "skill inside grace must be left alone");

        // The grader was asked only about the two old reflection-authored skills.
        assertTrue(llm.graded.contains("bad-skill"));
        assertTrue(llm.graded.contains("good-skill"));
        assertFalse(llm.graded.contains("builtin-helper"), "must not grade hand-built skills");
        assertFalse(llm.graded.contains("new-bad-skill"), "must not grade skills inside grace");
    }

    @Test
    void noReflectionSkillsIsANoop(@TempDir Path skillsDir) throws Exception {
        writeSkill(skillsDir, "builtin-a", false, null);
        writeSkill(skillsDir, "builtin-b", false, null);
        GradingLlm llm = new GradingLlm();
        assertEquals(0, new SkillCurator(llm, "m", skillsDir).runOnce());
        assertTrue(llm.graded.isEmpty(), "no LLM calls when there is nothing to grade");
    }

    @Test
    void parseGradeToleratesFencesAndProse() {
        assertEquals(2, SkillCurator.parseGrade("```json\n{\"score\": 2, \"reason\": \"vacuous\"}\n```").score());
        assertEquals(7, SkillCurator.parseGrade("Here is my grade: {\"score\":7,\"reason\":\"ok\"} done").score());
        assertEquals(10, SkillCurator.parseGrade("{\"score\": 99, \"reason\": \"clamped\"}").score(),
                "score is clamped to 0..10");
        assertNull(SkillCurator.parseGrade("no json here"));
        assertNull(SkillCurator.parseGrade(""));
    }
}
