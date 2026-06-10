package ai.nizo.memory.api.memory;

/**
 * Tag-key conventions used across the memory pipeline.
 *
 * <p>The {@link MemoryItem#tags} map is intentionally schema-free, but several
 * keys carry semantic meaning that the recall pipeline and extraction pipeline
 * read. Centralising the names here keeps producers and consumers aligned.
 *
 * <p>All values are strings; the conventions are documented per constant.
 */
public final class MemoryTags {

    /**
     * Who the fact is about. Critical for safety: "my mom has a peanut
     * allergy" must not become "user has a peanut allergy".
     *
     * <p>Values:
     * <ul>
     *   <li>{@code "self"} — fact is about the user (default if unset)</li>
     *   <li>{@code "other:<name-or-role>"} — fact is about a third party
     *       (e.g. {@code "other:mom"}, {@code "other:Sarah"})</li>
     * </ul>
     */
    public static final String SUBJECT = "subject";

    public static final String SUBJECT_SELF = "self";
    public static final String SUBJECT_OTHER_PREFIX = "other:";

    /**
     * Privacy / disclosure tier. Recall filters by default exclude
     * {@code SENSITIVE} and {@code CRITICAL} unless the query lexically
     * touches the topic.
     *
     * <p>Values: {@code "PUBLIC"}, {@code "PERSONAL"} (default),
     * {@code "SENSITIVE"} (mental health, finances, relationships),
     * {@code "CRITICAL"} (safety: allergies, medications, conditions).
     */
    public static final String SENSITIVITY = "sensitivity";

    public static final String SENS_PUBLIC = "PUBLIC";
    public static final String SENS_PERSONAL = "PERSONAL";
    public static final String SENS_SENSITIVE = "SENSITIVE";
    public static final String SENS_CRITICAL = "CRITICAL";

    /**
     * Life-mode / persona context. Lets the agent recall in the right
     * lane: a work prompt shouldn't surface kid's pediatrician notes.
     *
     * <p>Values: {@code "work"}, {@code "personal"} (default),
     * {@code "health"}, {@code "finance"}, {@code "social"}.
     */
    public static final String MODE = "mode";

    public static final String MODE_WORK = "work";
    public static final String MODE_PERSONAL = "personal";
    public static final String MODE_HEALTH = "health";
    public static final String MODE_FINANCE = "finance";
    public static final String MODE_SOCIAL = "social";

    /**
     * Hypothetical / hedged statement. Triggered when the user said
     * "considering / thinking about / might / maybe / I'd like to". These
     * facts must NOT be promoted to confirmed PROFILE / PREFERENCE
     * statements. They get a recall-time penalty and a "tentative" framing.
     *
     * <p>Values: {@code "true"} (presence implies hedging).
     */
    public static final String HYPOTHETICAL = "hypothetical";

    /**
     * User-pinned. Pinned facts are always recalled and get a strong
     * scoring boost. Ideal for facts the user explicitly marked as
     * important ("always remember my mother's blood type").
     *
     * <p>Values: {@code "true"} when pinned.
     */
    public static final String PINNED = "pinned";

    /** Optional reason the user pinned this. Free-form. */
    public static final String PIN_REASON = "pin_reason";

    /**
     * ISO-8601 expiry timestamp (event date for one-time events,
     * deferral horizon for DEFERRAL items). Items past their expiry
     * are demoted at recall time so a Friday flight isn't still treated
     * as upcoming on Monday.
     */
    public static final String EXPIRES_AT = "expires_at";

    /**
     * Number of times this fact (or near-duplicate) has been heard from
     * the user. Single-mention facts (count=1) are tentative; multi-mention
     * facts (count≥2) get a recall boost — they're reinforced beliefs.
     */
    public static final String MENTION_COUNT = "mention_count";

    /**
     * ISO-8601 timestamp of the last time this preference was implicitly
     * or explicitly reconfirmed by the user. Old preferences without a
     * recent reconfirmation get a confidence decay.
     */
    public static final String LAST_RECONFIRMED = "last_reconfirmed";

    /**
     * Embedder model + version that produced the stored embedding. When
     * the configured embedder differs from this tag, recall falls back
     * to FTS for the item rather than comparing incompatible vectors.
     */
    public static final String EMBEDDER_VERSION = "embedder_version";

    /**
     * For RELATIONSHIP / family kinship items, the original term the
     * user used (for cultural fidelity): "amma", "naani", "abuela".
     * Recall queries can hit either the canonical or the original.
     */
    public static final String CULTURAL_TERM = "cultural_term";

    /**
     * Provenance: the id of the EPISODIC memory item that this SEMANTIC
     * fact was derived from. Lets the system answer "why do you think X
     * about me?" by fetching the original message by id.
     *
     * <p>Set by {@code ExtractionPipeline} when it first stores the raw
     * user message as EPISODIC, then threads the episode id into every
     * fact/relationship/preference it extracts from that message.
     *
     * <p>Empty / unset means provenance is unknown (e.g. imported facts,
     * user-stated direct writes).
     */
    public static final String SOURCE_MESSAGE_ID = "source_message_id";

    /**
     * Free-form quoted excerpt from the source message that triggered this
     * fact. Kept short (truncated to ~140 chars) so inspect views can render
     * "because you said: '...'" without another DB round-trip. The authoritative
     * link is {@link #SOURCE_MESSAGE_ID}; this is a convenience copy.
     */
    public static final String SOURCE_EXCERPT = "source_excerpt";

    /**
     * ISO-8601 timestamp when the reflection worker processed this EPISODIC
     * item. Presence means the episode has already been distilled into
     * SEMANTIC facts and must not be re-processed.
     */
    public static final String REFLECTED_AT = "reflected_at";

    /**
     * Semantic facet of a fact — what *kind* of answer it provides. Used by
     * precision-heavy recall modes to filter out facts that identify the
     * right entity but don't address the question's intent.
     *
     * <p>Example: a query asking {@code "is my wife home tonight?"} has a
     * facet of {@link #FACET_SCHEDULE}; a stored fact {@code "Priya is a
     * cardiologist"} has facet {@link #FACET_PROFILE}. The facets are
     * incompatible, so precision-heavy abstains rather than returning an
     * identity-only match.
     *
     * <p>Values: one of {@code FACET_IDENTITY / PREFERENCE / HEALTH /
     * RELATIONSHIP / LOCATION / SCHEDULE / GOAL / COMMITMENT / EVENT /
     * FINANCE / ROUTINE / OTHER}.
     */
    public static final String FACET = "facet";

    public static final String FACET_IDENTITY     = "identity";
    public static final String FACET_PREFERENCE   = "preference";
    public static final String FACET_HEALTH       = "health";
    public static final String FACET_RELATIONSHIP = "relationship";
    public static final String FACET_LOCATION     = "location";
    public static final String FACET_SCHEDULE     = "schedule";
    public static final String FACET_GOAL         = "goal";
    public static final String FACET_COMMITMENT   = "commitment";
    public static final String FACET_EVENT        = "event";
    public static final String FACET_FINANCE      = "finance";
    public static final String FACET_ROUTINE      = "routine";
    public static final String FACET_PROFILE      = "profile";
    public static final String FACET_OTHER        = "other";

    /**
     * Conversation/session identifier. Groups items that arrived together in
     * one logical session (one chat tab, one phone call, one daily log, one
     * LongMemEval haystack-session). Session-filtered recall uses this to
     * restrict retrieval to a small subset of sessions picked by the
     * {@link ai.nizo.memory.session.SessionPicker} — the main lever against
     * distractor-heavy multi-session workloads.
     *
     * <p>Free-form; recommended formats: UUID, ISO-8601 timestamp, or a stable
     * hash of the session start. Items with no {@code session_id} tag are
     * treated as belonging to a single default session and are always
     * visible to recall regardless of filter.
     */
    public static final String SESSION_ID = "session_id";

    /**
     * Canonical fact marker. Value {@code "true"} designates a SEMANTIC item
     * that has been promoted to the user's "index" — definitional facts like
     * identity, confirmed preferences, safety-critical health info. Canonical
     * items get a high-weight RRF channel, bypass the default sensitivity
     * gate when their topic is queried, and appear in the
     * {@code canonicalIndex()} table of contents surfaced at the top of the
     * system prompt.
     *
     * <p>Promotion is decided by
     * {@code ai.nizo.memory.canonical.CanonicalPromoter} based on content
     * shape (definitional patterns), facet, mention count, or explicit
     * pinning — see its Javadoc for the policy matrix.
     */
    public static final String CANONICAL = "canonical";

    /**
     * Cluster key for canonical facts. Namespaced {@code facet:slot} form so
     * conflicting statements about the same underlying attribute can resolve
     * to a single canonical row. Examples: {@code "identity:self"},
     * {@code "identity:spouse"}, {@code "preference:food"},
     * {@code "health:allergy"}, {@code "location:home"}.
     *
     * <p>Only meaningful when {@link #CANONICAL} is set. A new canonical fact
     * with the same {@code (userId, cluster_key)} as an existing one signals
     * an update, not a duplicate — the consumer should treat the newest as
     * authoritative.
     */
    public static final String CLUSTER_KEY = "cluster_key";

    private MemoryTags() {}
}
