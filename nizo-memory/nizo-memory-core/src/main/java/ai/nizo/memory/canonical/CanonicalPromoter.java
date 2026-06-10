package ai.nizo.memory.canonical;

import ai.nizo.memory.api.memory.MemoryTags;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Decides whether an incoming semantic fact is foundational enough to earn a
 * seat at the user's canonical table — the small set of high-confidence
 * statements that the recall pipeline treats as always-on (index in the
 * system prompt, boosted RRF weight, sensitivity-gate bypass on topic match).
 *
 * <h2>Promotion policy</h2>
 * An item is promoted when <em>any</em> of the following is true:
 * <ul>
 *   <li><b>Explicit pin.</b> {@code pinned=true} — the user has flagged the
 *       fact as permanently relevant; canonical is a superset of pinned.</li>
 *   <li><b>Definitional shape.</b> The content matches a stable-claim pattern
 *       ({@code "my X is Y"}, {@code "I'm a/an X"}, {@code "I always Z"}) and
 *       is not hedged ({@code hypothetical} unset).</li>
 *   <li><b>Foundational facet.</b> {@code facet} is
 *       {@link MemoryTags#FACET_IDENTITY}, {@link MemoryTags#FACET_PROFILE},
 *       or {@link MemoryTags#FACET_HEALTH} with confidence ≥ 0.8.</li>
 *   <li><b>Reinforced belief.</b> {@code mention_count} ≥ 3 — the user has
 *       restated the same fact at least three times.</li>
 * </ul>
 *
 * <p>Hedged / hypothetical statements are <em>never</em> promoted, regardless
 * of other signals — saying "I'm thinking about switching to vegetarianism"
 * is not the same as "I'm vegetarian".
 *
 * <p>Promotion is pure: the promoter returns a new tags map (or the same
 * reference when no change is needed); it never mutates the input.
 */
public final class CanonicalPromoter {

    /** Definitional-shape patterns. All case-insensitive, anchored at start. */
    private static final Pattern[] DEFINITIONAL = {
            // "my X is Y", "my wife's name is ..."
            Pattern.compile("^\\s*my\\s+[\\w'\\s]{1,40}\\s+is\\s+",
                    Pattern.CASE_INSENSITIVE),
            // "I am a doctor", "I'm allergic to peanuts"
            Pattern.compile("^\\s*(?:i\\s+am|i'm)\\s+",
                    Pattern.CASE_INSENSITIVE),
            // "I have a daughter", "I have diabetes"
            Pattern.compile("^\\s*i\\s+have\\s+",
                    Pattern.CASE_INSENSITIVE),
            // "I live in Sydney"
            Pattern.compile("^\\s*i\\s+live\\s+in\\s+",
                    Pattern.CASE_INSENSITIVE),
            // "I always X", "I never Y"
            Pattern.compile("^\\s*i\\s+(?:always|never)\\s+",
                    Pattern.CASE_INSENSITIVE),
            // "I work at X", "I work as Y"
            Pattern.compile("^\\s*i\\s+work\\s+(?:at|as|for)\\s+",
                    Pattern.CASE_INSENSITIVE),
    };

    private final double confidenceThreshold;
    private final int mentionThreshold;

    public CanonicalPromoter() {
        this(0.8, 3);
    }

    public CanonicalPromoter(double confidenceThreshold, int mentionThreshold) {
        this.confidenceThreshold = confidenceThreshold;
        this.mentionThreshold = Math.max(1, mentionThreshold);
    }

    /**
     * Inspect {@code content} plus its existing {@code tags} and return a tag
     * map (possibly the same reference) that marks the item canonical if it
     * qualifies. The returned map always contains every original entry; no
     * user-set tags are dropped.
     *
     * @param content      the fact text
     * @param tags         existing tags (nullable)
     * @param confidence   item confidence (0..1)
     * @return tags, with {@link MemoryTags#CANONICAL} and
     *         {@link MemoryTags#CLUSTER_KEY} added if promotion fires
     */
    public Map<String, String> maybePromote(String content,
                                             Map<String, String> tags,
                                             double confidence) {
        if (content == null || content.isBlank()) {
            return tags == null ? Map.of() : tags;
        }
        Map<String, String> in = tags == null ? Map.of() : tags;
        // Already canonical — nothing to do (the caller may re-upsert with
        // the same item, and we must be idempotent).
        if ("true".equals(in.get(MemoryTags.CANONICAL))) return in;
        // Hard veto: hypothetical / hedged statements never become canonical.
        if ("true".equals(in.get(MemoryTags.HYPOTHETICAL))) return in;

        boolean pinned = "true".equals(in.get(MemoryTags.PINNED));
        boolean definitional = matchesDefinitional(content);
        String facet = in.getOrDefault(MemoryTags.FACET, "");
        boolean foundationalFacet =
                (MemoryTags.FACET_IDENTITY.equals(facet)
                        || MemoryTags.FACET_PROFILE.equals(facet)
                        || MemoryTags.FACET_HEALTH.equals(facet))
                && confidence >= confidenceThreshold;
        int mentionCount = parseInt(in.get(MemoryTags.MENTION_COUNT), 1);
        boolean reinforced = mentionCount >= mentionThreshold;

        if (!(pinned || definitional || foundationalFacet || reinforced)) {
            return in;  // doesn't qualify
        }

        Map<String, String> out = new LinkedHashMap<>(in);
        out.put(MemoryTags.CANONICAL, "true");
        if (!out.containsKey(MemoryTags.CLUSTER_KEY)) {
            out.put(MemoryTags.CLUSTER_KEY, deriveClusterKey(content, facet));
        }
        return out;
    }

    /** True when {@code content} looks like a stable, definitional claim. */
    public static boolean matchesDefinitional(String content) {
        if (content == null) return false;
        for (Pattern p : DEFINITIONAL) {
            Matcher m = p.matcher(content);
            if (m.find()) return true;
        }
        return false;
    }

    /**
     * Best-effort cluster key derivation. Canonical items with the same
     * cluster key represent the same underlying attribute, so a newer value
     * supersedes the older one.
     *
     * <p>The key is {@code facet:slot} where the slot is either extracted
     * from the sentence (e.g. "my <b>wife</b> is ..." → slot "wife") or
     * falls back to {@code "self"}.
     */
    public static String deriveClusterKey(String content, String facet) {
        String f = (facet == null || facet.isBlank()) ? MemoryTags.FACET_OTHER : facet;
        String slot = extractSlot(content);
        return f + ":" + (slot == null ? "self" : slot);
    }

    /** Pull the slot noun from "my X is ...", otherwise null. */
    private static String extractSlot(String content) {
        if (content == null) return null;
        Matcher m = Pattern.compile("^\\s*my\\s+(\\w+)",
                Pattern.CASE_INSENSITIVE).matcher(content);
        if (m.find()) {
            return m.group(1).toLowerCase();
        }
        // "I have a daughter" / "I live in Sydney" — slot = first noun after the verb.
        Matcher m2 = Pattern.compile("^\\s*i\\s+(?:have|live\\s+in|work\\s+at|work\\s+as|work\\s+for)\\s+(?:a\\s+|an\\s+|the\\s+)?(\\w+)",
                Pattern.CASE_INSENSITIVE).matcher(content);
        if (m2.find()) {
            return m2.group(1).toLowerCase();
        }
        return null;
    }

    private static int parseInt(String s, int fallback) {
        if (s == null) return fallback;
        try { return Integer.parseInt(s.trim()); } catch (NumberFormatException e) { return fallback; }
    }
}
