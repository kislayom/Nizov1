package ai.nizo.skills;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Discover skills under a directory. A skill is a folder containing a {@code SKILL.md}
 * with YAML frontmatter (name, description, optional tags) followed by a markdown body.
 */
public final class SkillLoader {

    private static final Logger LOG = LoggerFactory.getLogger(SkillLoader.class);
    private static final Pattern FRONTMATTER = Pattern.compile(
            "(?s)\\A\\s*---\\s*\\n(.*?)\\n---\\s*\\n(.*)");

    /**
     * Soft cap on SKILL.md body length. Bodies longer than this are flagged with a WARN at load
     * time so the operator can see they're spending unnecessary tokens on every sub-agent
     * invocation. Doesn't reject — just signals.
     */
    static final int BODY_WARN_THRESHOLD_CHARS = 8_000;

    public List<SkillManifest> discover(Path root) {
        List<SkillManifest> out = new ArrayList<>();
        if (!Files.isDirectory(root)) return out;

        try (Stream<Path> stream = Files.walk(root, 2)) {
            stream.filter(p -> p.getFileName().toString().equalsIgnoreCase("SKILL.md"))
                    .forEach(file -> {
                        try {
                            SkillManifest m = parse(file);
                            if (m != null) {
                                out.add(m);
                                LOG.info("loaded skill: {} ({})", m.name(), file);
                            }
                        } catch (Exception e) {
                            LOG.warn("skipping bad skill {}: {}", file, e.toString());
                        }
                    });
        } catch (Exception e) {
            LOG.warn("skill scan failed at {}: {}", root, e.toString());
        }
        return out;
    }

    public SkillManifest parse(Path file) throws Exception {
        String text = Files.readString(file);
        Matcher m = FRONTMATTER.matcher(text);
        if (!m.find()) {
            // No frontmatter — synthesise from filename and body.
            String stem = file.getParent() == null ? "skill" : file.getParent().getFileName().toString();
            warnIfBodyTooLong(file, text);
            return new SkillManifest(stem, "", "", List.of(), false, text, file);
        }
        String header = m.group(1);
        String body = m.group(2);

        String name = headerValue(header, "name");
        if (name == null || name.isBlank()) {
            name = file.getParent() == null ? "skill" : file.getParent().getFileName().toString();
        }
        String desc = headerValue(header, "description");
        String whenToUse = headerValue(header, "when_to_use");
        List<String> tags = headerList(header, "tags");
        boolean agent = headerBoolean(header, "agent", false);
        warnIfBodyTooLong(file, body);
        return new SkillManifest(
                name,
                desc == null ? "" : desc,
                whenToUse == null ? "" : whenToUse,
                tags,
                agent,
                body,
                file);
    }

    /**
     * Soft-warn if the SKILL.md body is unusually long. Big bodies cost tokens on every
     * sub-agent invocation and usually mean the skill author dumped extraneous detail in
     * the playbook instead of keeping it tight. Doesn't reject the skill.
     */
    private static void warnIfBodyTooLong(Path file, String body) {
        if (body == null) return;
        if (body.length() > BODY_WARN_THRESHOLD_CHARS) {
            LOG.warn("skill body is {} chars (>{} threshold) — consider trimming for token cost: {}",
                    body.length(), BODY_WARN_THRESHOLD_CHARS, file);
        }
    }

    private static boolean headerBoolean(String header, String key, boolean defaultValue) {
        String v = headerValue(header, key);
        if (v == null) return defaultValue;
        v = v.trim().toLowerCase();
        return v.equals("true") || v.equals("yes") || v.equals("1");
    }

    private static String headerValue(String header, String key) {
        Pattern kv = Pattern.compile("(?im)^\\s*" + Pattern.quote(key) + "\\s*:\\s*(.+?)\\s*$");
        Matcher m = kv.matcher(header);
        if (!m.find()) return null;
        String v = m.group(1).trim();
        if ((v.startsWith("\"") && v.endsWith("\"")) || (v.startsWith("'") && v.endsWith("'"))) {
            v = v.substring(1, v.length() - 1);
        }
        return v;
    }

    private static List<String> headerList(String header, String key) {
        Pattern kv = Pattern.compile("(?im)^\\s*" + Pattern.quote(key) + "\\s*:\\s*\\[(.*?)\\]\\s*$");
        Matcher m = kv.matcher(header);
        if (!m.find()) return List.of();
        String inner = m.group(1).trim();
        if (inner.isEmpty()) return List.of();
        List<String> out = new ArrayList<>();
        for (String t : inner.split(",")) {
            String tt = t.trim();
            if ((tt.startsWith("\"") && tt.endsWith("\"")) || (tt.startsWith("'") && tt.endsWith("'"))) {
                tt = tt.substring(1, tt.length() - 1);
            }
            if (!tt.isEmpty()) out.add(tt);
        }
        return out;
    }
}
