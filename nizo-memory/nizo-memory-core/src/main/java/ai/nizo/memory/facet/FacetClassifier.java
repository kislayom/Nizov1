package ai.nizo.memory.facet;

import ai.nizo.memory.api.memory.MemoryTags;

import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Semantic facet classification — what <em>kind</em> of answer a fact provides
 * and what <em>kind</em> of answer a query is asking for.
 *
 * <p>The facet dimension is orthogonal to the memory tier (SEMANTIC / EPISODIC /
 * etc.) and to the source-priority tag. It answers: <em>does this fact address
 * the question's intent, or merely identify the entity the question is about?</em>
 *
 * <p>Used by {@code LayeredMemoryService.surface()} in precision-heavy mode to
 * filter out facts that share entity tokens with the query but address a
 * different facet. Example:
 *
 * <pre>
 *   Query:   "Is my wife home tonight?"    facet = SCHEDULE
 *   Fact:    "Priya is a cardiologist..."  facet = PROFILE
 *   Compat:  false  → abstain (entity-only match, no facet answer)
 * </pre>
 *
 * <p>Heuristic: keyword/regex patterns, not LLM. Deterministic. False negatives
 * (facet=OTHER) fall back to the legacy entity-marker heuristic; false positives
 * are possible but bounded by the keyword lists being tight.
 */
public final class FacetClassifier {

    private FacetClassifier() {}

    // Compatibility matrix: for a given query facet, which content facets
    // count as "answers"? A fact tagged with any of the allowed content
    // facets for the query's facet is considered on-topic.
    //
    // Asymmetric on purpose: an IDENTITY question accepts RELATIONSHIP /
    // PROFILE content (they identify the person), but a SCHEDULE question
    // does NOT accept PROFILE content (knowing her profession doesn't tell
    // you where she is tonight).
    private static final java.util.Map<String, Set<String>> COMPAT = java.util.Map.ofEntries(
            java.util.Map.entry(MemoryTags.FACET_IDENTITY,
                    Set.of(MemoryTags.FACET_IDENTITY, MemoryTags.FACET_RELATIONSHIP,
                           MemoryTags.FACET_PROFILE)),
            java.util.Map.entry(MemoryTags.FACET_RELATIONSHIP,
                    Set.of(MemoryTags.FACET_RELATIONSHIP, MemoryTags.FACET_IDENTITY,
                           MemoryTags.FACET_PROFILE)),
            java.util.Map.entry(MemoryTags.FACET_PROFILE,
                    Set.of(MemoryTags.FACET_PROFILE, MemoryTags.FACET_IDENTITY,
                           MemoryTags.FACET_LOCATION)),
            java.util.Map.entry(MemoryTags.FACET_PREFERENCE,
                    Set.of(MemoryTags.FACET_PREFERENCE, MemoryTags.FACET_ROUTINE)),
            java.util.Map.entry(MemoryTags.FACET_HEALTH,
                    Set.of(MemoryTags.FACET_HEALTH)),
            java.util.Map.entry(MemoryTags.FACET_LOCATION,
                    Set.of(MemoryTags.FACET_LOCATION, MemoryTags.FACET_PROFILE)),
            java.util.Map.entry(MemoryTags.FACET_SCHEDULE,
                    Set.of(MemoryTags.FACET_SCHEDULE, MemoryTags.FACET_ROUTINE)),
            java.util.Map.entry(MemoryTags.FACET_GOAL,
                    Set.of(MemoryTags.FACET_GOAL)),
            java.util.Map.entry(MemoryTags.FACET_COMMITMENT,
                    Set.of(MemoryTags.FACET_COMMITMENT, MemoryTags.FACET_GOAL)),
            java.util.Map.entry(MemoryTags.FACET_EVENT,
                    Set.of(MemoryTags.FACET_EVENT)),
            java.util.Map.entry(MemoryTags.FACET_FINANCE,
                    Set.of(MemoryTags.FACET_FINANCE)),
            java.util.Map.entry(MemoryTags.FACET_ROUTINE,
                    Set.of(MemoryTags.FACET_ROUTINE, MemoryTags.FACET_SCHEDULE,
                           MemoryTags.FACET_PREFERENCE))
    );

    /** Compatibility check used by precision-heavy mode. */
    public static boolean isCompatible(String queryFacet, String contentFacet) {
        if (queryFacet == null || contentFacet == null) return true;
        if (MemoryTags.FACET_OTHER.equals(queryFacet)
                || MemoryTags.FACET_OTHER.equals(contentFacet)) {
            return true;   // don't filter when classifier was unsure
        }
        Set<String> allowed = COMPAT.get(queryFacet);
        if (allowed == null) return true;
        return allowed.contains(contentFacet);
    }

    // ─── Query classification ──────────────────────────────────────────────

    private static final List<java.util.Map.Entry<String, Pattern>> QUERY_PATTERNS = List.of(
            // Schedule / location-right-now
            java.util.Map.entry(MemoryTags.FACET_SCHEDULE, Pattern.compile(
                    "\\b(is|are)\\s+\\w+\\s+(home|here|around|at\\s+home|in|out)\\b" +
                    "|\\b(tonight|tomorrow|today|right\\s+now|this\\s+(morning|afternoon|evening))\\b" +
                    "|\\bwhat\\s+time\\b|\\bwhen\\s+(will|does|is)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Location (persistent, not right-now)
            java.util.Map.entry(MemoryTags.FACET_LOCATION, Pattern.compile(
                    "\\bwhere\\s+(do|does|am\\s+I|is\\s+my|are\\s+you|does\\s+(he|she|they)\\s+live)\\b" +
                    "|\\b(based\\s+in|live\\s+in|home\\s+(city|address))\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Identity — who/what is X
            java.util.Map.entry(MemoryTags.FACET_IDENTITY, Pattern.compile(
                    "\\bwho\\s+(is|am\\s+I)\\b|\\bwhat\\s+is\\s+(my|the)\\s+(name|nickname)\\b" +
                    "|\\btell\\s+me\\s+about\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Health / allergies / meds
            java.util.Map.entry(MemoryTags.FACET_HEALTH, Pattern.compile(
                    "\\b(allerg\\w*|allergic|allergy|epipen|peanut|shellfish|gluten|lactose)\\b" +
                    "|\\b(medication|prescription|condition|diabet\\w*|asthm\\w*|blood\\s+type)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Preference
            java.util.Map.entry(MemoryTags.FACET_PREFERENCE, Pattern.compile(
                    "\\b(do\\s+I\\s+(like|love|prefer)|my\\s+favou?rite|what\\s+do\\s+I\\s+(like|prefer))\\b" +
                    "|\\b(what\\s+.+\\s+(recommend|suggest)\\s+for\\s+me)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Goal / aspiration
            java.util.Map.entry(MemoryTags.FACET_GOAL, Pattern.compile(
                    "\\bmy\\s+goals?\\b|\\bwhat\\s+am\\s+I\\s+(working\\s+on|trying\\s+to)\\b" +
                    "|\\btraining\\s+for\\b|\\baspire\\b|\\blong[- ]term\\s+plan\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Commitment / waiting / pending
            java.util.Map.entry(MemoryTags.FACET_COMMITMENT, Pattern.compile(
                    "\\bwhat\\s+(did\\s+I\\s+commit|am\\s+I\\s+waiting|is\\s+pending)\\b" +
                    "|\\bmy\\s+(follow[- ]ups?|pending\\s+tasks?|commitments?|todos?)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Event (past happenings)
            java.util.Map.entry(MemoryTags.FACET_EVENT, Pattern.compile(
                    "\\bwhen\\s+did\\s+I\\b|\\bwhat\\s+happened\\b|\\bremember\\s+when\\b" +
                    "|\\bhow\\s+long\\s+ago\\b|\\bdid\\s+I\\s+attend\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Finance
            java.util.Map.entry(MemoryTags.FACET_FINANCE, Pattern.compile(
                    "\\b(how\\s+much\\s+(do|did)\\s+I\\s+(earn|make|pay|owe|save|spend))\\b" +
                    "|\\b(my\\s+(salary|income|net\\s+worth|portfolio|investments?|holdings?|savings?|budget))\\b" +
                    "|\\b(stock|etf|mutual\\s+fund|crypto|bitcoin|nifty|sensex|sip|ppf|epf|nps)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Relationship (family / friends)
            java.util.Map.entry(MemoryTags.FACET_RELATIONSHIP, Pattern.compile(
                    "\\bwho\\s+is\\s+my\\s+(wife|husband|spouse|partner|mother|mom|father|dad|son|daughter|child|kid|brother|sister|sibling|friend|colleague|boss|mentor)\\b" +
                    "|\\btell\\s+me\\s+about\\s+my\\s+(family|wife|husband|spouse|partner|mom|dad|mother|father|child|kids|children|brother|sister|sibling)\\b" +
                    "|\\bmy\\s+(family\\s+members?|immediate\\s+family|relatives?)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Routine / habit
            java.util.Map.entry(MemoryTags.FACET_ROUTINE, Pattern.compile(
                    "\\b(usually|every\\s+(day|week|month|morning|evening)|routinely|regularly|habit)\\b",
                    Pattern.CASE_INSENSITIVE))
    );

    /**
     * Classify what facet of answer the query is asking for. Returns
     * {@link MemoryTags#FACET_OTHER} when no pattern matches — in which case
     * precision-heavy falls through to the legacy entity-marker heuristic.
     *
     * <p>Convenience wrapper over {@link #classifyQueryAll}: returns the
     * first matching facet. Prefer {@link #classifyQueryAll} when you need
     * multi-facet matching (e.g. "Thai place for 2 tonight" has both
     * PREFERENCE and SCHEDULE intents).
     */
    public static String classifyQuery(String query) {
        Set<String> all = classifyQueryAll(query);
        return all.isEmpty() ? MemoryTags.FACET_OTHER : all.iterator().next();
    }

    /**
     * Multi-label facet classification. Queries often carry more than one
     * facet intent — "Thai place for 2 tonight" is both a PREFERENCE
     * question (what cuisine the user likes) and a SCHEDULE question (when
     * — tonight). Returning the full set lets the facet-compat gate check
     * ANY intent, which is the right semantics for permissive filtering.
     *
     * <p>Returns an empty set when no pattern matches.
     */
    public static Set<String> classifyQueryAll(String query) {
        if (query == null || query.isBlank()) return Set.of();
        Set<String> facets = new java.util.LinkedHashSet<>();
        for (var e : QUERY_PATTERNS) {
            if (e.getValue().matcher(query).find()) facets.add(e.getKey());
        }
        return facets;
    }

    /**
     * Multi-facet compatibility: a content facet passes if it's compatible
     * with ANY of the query's facets. Used by precision-heavy mode.
     */
    public static boolean isCompatibleAny(Set<String> queryFacets, String contentFacet) {
        if (queryFacets == null || queryFacets.isEmpty()) return true;
        for (String qf : queryFacets) {
            if (isCompatible(qf, contentFacet)) return true;
        }
        return false;
    }

    // ─── Content inference ─────────────────────────────────────────────────

    private static final List<java.util.Map.Entry<String, Pattern>> CONTENT_PATTERNS = List.of(
            // Health (check early — strongest keywords)
            java.util.Map.entry(MemoryTags.FACET_HEALTH, Pattern.compile(
                    "\\b(allerg\\w*|epipen|peanut\\s+aller|shellfish\\s+aller|lactose|gluten|asthm\\w*|" +
                    "diabet\\w*|medication|prescription|blood\\s+pressure|cholesterol|bronchitis|" +
                    "pneumonia|migraines?|seizures?|epilep\\w*|insulin|anaphyla)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Relationship (named person + relation role)
            java.util.Map.entry(MemoryTags.FACET_RELATIONSHIP, Pattern.compile(
                    "\\b(is\\s+the\\s+user'?s\\s+(wife|husband|spouse|partner|mother|mom|father|dad|" +
                    "son|daughter|child|kid|brother|sister|sibling|friend|colleague|boss|mentor|co-?founder))\\b" +
                    "|\\b(user'?s\\s+(wife|husband|spouse|mother|mom|father|dad|son|daughter|child|kid|" +
                    "brother|sister|sibling|friend|colleague|boss))\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Goal
            java.util.Map.entry(MemoryTags.FACET_GOAL, Pattern.compile(
                    "^\\s*Goal\\s*\\(|\\bUser\\s+wants\\s+to\\b|\\baspiration\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Commitment
            java.util.Map.entry(MemoryTags.FACET_COMMITMENT, Pattern.compile(
                    "^\\s*(Commitment|Follow[- ]up|Deferral)\\s*\\(|\\bUser\\s+committed\\s+to\\b" +
                    "|\\bwaiting\\s+for\\b|\\bpending\\s+(decision|action)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Resolution → event
            java.util.Map.entry(MemoryTags.FACET_EVENT, Pattern.compile(
                    "^\\s*Resolution\\s*\\(|\\bUser\\s+decided\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Finance
            java.util.Map.entry(MemoryTags.FACET_FINANCE, Pattern.compile(
                    "\\b(investment|investor|portfolio|holdings?|salary|income|earn(ed|ing)?|" +
                    "SIP|PPF|EPF|NPS|401k|mutual\\s+fund|stock|ETF|crypto|bitcoin|Nifty|Sensex|" +
                    "\\$[0-9]|₹[0-9]|[0-9]+\\s*(lakh|crore|usd|inr|eur))\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Preference (lots of hits — put later)
            java.util.Map.entry(MemoryTags.FACET_PREFERENCE, Pattern.compile(
                    "\\b(User\\s+(loves|likes|prefers|uses|enjoys|favou?rites?))\\b" +
                    "|\\b(vegetarian|vegan|iPhone\\s+user|Android\\s+user|Pixel\\s+user|Vim\\s+user|" +
                    "morning\\s+person|night\\s+owl)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Profile (bio facts)
            java.util.Map.entry(MemoryTags.FACET_PROFILE, Pattern.compile(
                    "\\b(works\\s+at|job\\s+role|occupation|employer|company|industry|" +
                    "name\\s+is|nickname|birthday|timezone|based\\s+in|live\\s+in|from)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Location
            java.util.Map.entry(MemoryTags.FACET_LOCATION, Pattern.compile(
                    "\\b(lives?\\s+in|based\\s+in|home\\s+(city|address)|moved\\s+to)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Routine
            java.util.Map.entry(MemoryTags.FACET_ROUTINE, Pattern.compile(
                    "\\b(every\\s+(day|week|month|morning|evening|weekday|weekend)|" +
                    "usually|daily|weekly|monthly|goes\\s+to\\s+\\w+\\s+every)\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Schedule (less common in stored content)
            java.util.Map.entry(MemoryTags.FACET_SCHEDULE, Pattern.compile(
                    "\\b(tonight|tomorrow|this\\s+(morning|afternoon|evening)|in\\s+\\d+\\s+(hour|minute)|at\\s+\\d{1,2}(:\\d{2})?\\s*(am|pm))\\b",
                    Pattern.CASE_INSENSITIVE)),
            // Identity (fallback — name/nickname patterns)
            java.util.Map.entry(MemoryTags.FACET_IDENTITY, Pattern.compile(
                    "^\\s*User'?s\\s+name\\s+is\\b|\\bnickname\\s+is\\b",
                    Pattern.CASE_INSENSITIVE))
    );

    /**
     * Infer a facet for stored content. Called at import / extraction time
     * when no explicit facet tag is provided. Conservative: unmatched content
     * becomes {@link MemoryTags#FACET_OTHER} (permissive for precision-heavy).
     */
    public static String inferContent(String content) {
        if (content == null || content.isBlank()) return MemoryTags.FACET_OTHER;
        for (var e : CONTENT_PATTERNS) {
            if (e.getValue().matcher(content).find()) return e.getKey();
        }
        return MemoryTags.FACET_OTHER;
    }

    /**
     * Normalise any facet string (case-insensitive) to the canonical
     * lowercase form, returning {@link MemoryTags#FACET_OTHER} for unknown
     * values.
     */
    public static String normalize(String raw) {
        if (raw == null) return MemoryTags.FACET_OTHER;
        String s = raw.toLowerCase(Locale.ROOT).trim();
        return switch (s) {
            case "identity", "preference", "health", "relationship", "location",
                 "schedule", "goal", "commitment", "event", "finance", "routine",
                 "profile" -> s;
            default -> MemoryTags.FACET_OTHER;
        };
    }
}
