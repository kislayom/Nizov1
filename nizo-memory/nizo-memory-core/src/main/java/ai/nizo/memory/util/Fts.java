package ai.nizo.memory.util;

/** SQLite FTS5 helpers. */
public final class Fts {

    private Fts() {}

    /**
     * Turn free-form user input into a safe FTS5 MATCH expression.
     * Non-alphanumerics become spaces, short tokens are dropped. Each
     * remaining token is expanded into both its original form and a
     * stem+prefix-wildcard form so pluralization and light inflection
     * don't break recall (G11/G12/G13).
     *
     * <p>Example transforms:
     * <ul>
     *   <li>{@code "flights"} → {@code flights OR flight*}</li>
     *   <li>{@code "running"} → {@code running OR run*}</li>
     *   <li>{@code "trips"} → {@code trips OR trip*}</li>
     *   <li>{@code "Tokyo"} → {@code Tokyo*} (short enough — just prefix)</li>
     * </ul>
     *
     * <p>We can't quote tokens AND use {@code *} prefix in FTS5 (quotes disable
     * the prefix operator), so characters are sanitised with a character
     * whitelist instead of quoting.
     */
    public static String sanitiseMatch(String raw) {
        if (raw == null) return "\"\"";
        StringBuilder cleaned = new StringBuilder(raw.length());
        for (int i = 0; i < raw.length(); i++) {
            char c = raw.charAt(i);
            // FTS5-safe whitelist. {@code -} is DELIBERATELY excluded:
            // FTS5 parses {@code UK-based} as {@code UK NOT based} (or as a
            // column-name operator, depending on context), which fails
            // queries silently. Same for {@code :} (column separator),
            // {@code +} (required term), {@code "}/{@code (}/{@code )}/
            // {@code ^}/{@code *} (operators we don't want from user input —
            // we add the {@code *} prefix wildcard ourselves below).
            //
            // Letters, digits, spaces, and underscores survive. Everything
            // else, including hyphens, becomes a space — so "UK-based"
            // tokenizes as {@code UK} + {@code based} and the query works.
            cleaned.append(Character.isLetterOrDigit(c) || c == ' ' || c == '_' ? c : ' ');
        }
        String[] terms = cleaned.toString().trim().split("\\s+");
        StringBuilder out = new StringBuilder();
        java.util.Set<String> emitted = new java.util.HashSet<>();
        for (String t : terms) {
            if (t.length() < 2) continue;
            String original = t;
            String stem = simpleStem(t);
            // Emit original (length ≥ 2). We used to require length ≥ 3
            // which silently dropped acronyms like "UK", "US", "AI", "ML",
            // "BTC", "ETH", "GPT", "RAM" — exactly the words a user is
            // most likely to query. They're indexed in FTS5 (default min
            // token length = 1); the only reason to drop them was avoiding
            // overly-broad prefix wildcards, which we handle separately
            // (no {@code *} for short tokens, see below).
            if (emitted.add(original.toLowerCase())) {
                if (out.length() > 0) out.append(" OR ");
                out.append(original);
            }
            // Emit stem+wildcard ONLY when stem ≥ 3 chars (to avoid e.g.
            // "in*" matching half the corpus). Plural / inflection only
            // matters for content words anyway.
            if (stem.length() >= 3 && emitted.add(stem.toLowerCase() + "*")) {
                if (out.length() > 0) out.append(" OR ");
                out.append(stem).append('*');
            }
            // G12/G13 — small synonym expansion so "travel" finds "flight",
            // "fruit" finds "apples", "goals" finds "aspiration", etc.
            for (String syn : SYNONYMS.getOrDefault(stem.toLowerCase(), java.util.List.of())) {
                if (emitted.add(syn + "*")) {
                    if (out.length() > 0) out.append(" OR ");
                    out.append(syn).append('*');
                }
            }
        }
        return out.length() == 0 ? "\"\"" : out.toString();
    }

    /**
     * Tiny curated synonym table — covers the most common
     * question↔content mismatches a personal-memory system sees. Deliberately
     * conservative; over-broad synonyms pollute recall worse than missing ones.
     * Keys are <em>stems</em> (see {@link #simpleStem}).
     */
    private static final java.util.Map<String, java.util.List<String>> SYNONYMS =
            java.util.Map.ofEntries(
                    // Travel
                    java.util.Map.entry("travel", java.util.List.of("trip", "flight", "journey", "vacation", "holiday")),
                    java.util.Map.entry("trip",   java.util.List.of("travel", "flight", "journey", "vacation")),
                    java.util.Map.entry("flight", java.util.List.of("travel", "trip", "journey")),
                    java.util.Map.entry("vacat",  java.util.List.of("travel", "trip", "holiday", "flight")),
                    java.util.Map.entry("holiday",java.util.List.of("travel", "trip", "vacation", "flight")),
                    // Food / diet — enriched for Active Memory bridging
                    // ("dinner tonight" → "vegetarian" / "Thai curry")
                    java.util.Map.entry("food",   java.util.List.of("meal", "cuisine", "dish", "diet", "eat", "dinner", "lunch", "breakfast")),
                    java.util.Map.entry("meal",   java.util.List.of("food", "cuisine", "dinner", "lunch", "breakfast")),
                    java.util.Map.entry("dinner", java.util.List.of("food", "meal", "cuisine", "eat")),
                    java.util.Map.entry("lunch",  java.util.List.of("food", "meal", "cuisine", "eat")),
                    java.util.Map.entry("breakfast", java.util.List.of("food", "meal", "cuisine", "eat")),
                    java.util.Map.entry("snack",  java.util.List.of("food", "eat")),
                    java.util.Map.entry("restaurant", java.util.List.of("food", "cuisine", "meal", "dine")),
                    java.util.Map.entry("cuisine", java.util.List.of("food", "meal", "dish")),
                    java.util.Map.entry("fruit",  java.util.List.of("apple", "banana", "orange", "mango")),
                    java.util.Map.entry("diet",   java.util.List.of("food", "meal", "vegetarian", "vegan")),
                    java.util.Map.entry("vegetarian", java.util.List.of("vegan", "diet", "food")),
                    // Work / career
                    java.util.Map.entry("job",    java.util.List.of("work", "career", "occupation", "employer", "role")),
                    java.util.Map.entry("work",   java.util.List.of("job", "career", "occupation", "employer")),
                    java.util.Map.entry("career", java.util.List.of("job", "work", "occupation", "role")),
                    java.util.Map.entry("role",   java.util.List.of("job", "work", "position", "occupation")),
                    java.util.Map.entry("employ", java.util.List.of("job", "work", "employer", "company")),
                    // Goals / plans / pending work — kept tight to avoid
                    // over-match (R2-1 caught "my travel plans" matching
                    // commitments and resolutions). "plan" no longer pulls
                    // in commitment/pending; those have their own synonyms.
                    java.util.Map.entry("goal",   java.util.List.of("aspiration", "target", "objective")),
                    java.util.Map.entry("task",   java.util.List.of("followup", "todo", "reminder")),
                    java.util.Map.entry("pend",   java.util.List.of("waiting", "deferred")),
                    // Health
                    java.util.Map.entry("health", java.util.List.of("medical", "wellness", "fitness")),
                    java.util.Map.entry("allerg", java.util.List.of("allergy", "intolerance", "reaction")),
                    java.util.Map.entry("medic",  java.util.List.of("medication", "prescription", "health")),
                    // Family — symmetric so "wife"/"husband" queries find
                    // "spouse"-tagged facts and vice versa.
                    java.util.Map.entry("spouse", java.util.List.of("wife", "husband", "partner")),
                    java.util.Map.entry("wife",   java.util.List.of("spouse", "partner")),
                    java.util.Map.entry("husband",java.util.List.of("spouse", "partner")),
                    java.util.Map.entry("partner",java.util.List.of("spouse", "wife", "husband")),
                    java.util.Map.entry("parent", java.util.List.of("mother", "father", "mom", "dad")),
                    java.util.Map.entry("mother", java.util.List.of("parent", "mom")),
                    java.util.Map.entry("father", java.util.List.of("parent", "dad")),
                    java.util.Map.entry("mom",    java.util.List.of("parent", "mother")),
                    java.util.Map.entry("dad",    java.util.List.of("parent", "father")),
                    java.util.Map.entry("kid",    java.util.List.of("child", "son", "daughter")),
                    java.util.Map.entry("son",    java.util.List.of("child", "kid")),
                    java.util.Map.entry("daughter",java.util.List.of("child", "kid")),
                    java.util.Map.entry("child",  java.util.List.of("kid", "son", "daughter")),
                    java.util.Map.entry("family", java.util.List.of("wife", "husband", "spouse", "child", "son", "daughter", "mother", "father", "parent", "sibling", "brother", "sister")),
                    java.util.Map.entry("sibling",java.util.List.of("brother", "sister")),
                    java.util.Map.entry("brother",java.util.List.of("sibling")),
                    java.util.Map.entry("sister", java.util.List.of("sibling")),
                    // Hobbies / interests — meta-synonyms so "my hobbies" finds
                    // "likes chess", "plays football", "enjoys reading" etc.
                    java.util.Map.entry("hobby",  java.util.List.of("like", "play", "enjoy", "practice", "activity", "pastime", "interest", "prefer", "love")),
                    java.util.Map.entry("pastime",java.util.List.of("like", "play", "enjoy", "hobby", "interest")),
                    java.util.Map.entry("activity",java.util.List.of("like", "play", "enjoy", "hobby", "interest")),
                    java.util.Map.entry("interest", java.util.List.of("like", "enjoy", "hobby", "passion")),
                    java.util.Map.entry("passion", java.util.List.of("like", "enjoy", "hobby", "interest")),
                    java.util.Map.entry("enjoy", java.util.List.of("like", "hobby", "interest")),
                    // Contacts
                    java.util.Map.entry("holding",java.util.List.of("hold", "own", "portfolio", "position")),
                    java.util.Map.entry("own",    java.util.List.of("hold", "have", "possess"))
            );

    /**
     * Lightweight stemmer — strips common English suffixes recursively so
     * "flights" → "flight" → "flight", "holdings" → "holding" → "hold",
     * "runnings" → "running" → "run". Deliberately crude; FTS5 prefix
     * matching picks up the rest.
     */
    static String simpleStem(String t) {
        String s = t.toLowerCase();
        // Apply one suffix strip at a time, up to a few passes, to catch
        // compound inflections like "holdings" (plural of "holding").
        for (int i = 0; i < 3; i++) {
            String next = stripOneSuffix(s);
            if (next.equals(s)) break;
            s = next;
        }
        return s;
    }

    private static String stripOneSuffix(String s) {
        if (s.endsWith("ies") && s.length() > 4) return s.substring(0, s.length() - 3) + "y";
        if (s.endsWith("ses") && s.length() > 4) return s.substring(0, s.length() - 2);
        if (s.endsWith("es")  && s.length() > 3) return s.substring(0, s.length() - 2);
        if (s.endsWith("s")   && s.length() > 3 && !s.endsWith("ss")) return s.substring(0, s.length() - 1);
        if (s.endsWith("ing") && s.length() > 5) {
            String core = s.substring(0, s.length() - 3);
            if (core.length() >= 2 && core.charAt(core.length() - 1) == core.charAt(core.length() - 2)
                    && !isVowel(core.charAt(core.length() - 1))) {
                return core.substring(0, core.length() - 1);
            }
            return core;
        }
        if (s.endsWith("ed") && s.length() > 4) return s.substring(0, s.length() - 2);
        return s;
    }

    private static boolean isVowel(char c) {
        return "aeiou".indexOf(c) >= 0;
    }
}
