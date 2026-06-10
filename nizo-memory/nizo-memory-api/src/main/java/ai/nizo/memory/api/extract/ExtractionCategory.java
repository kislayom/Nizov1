package ai.nizo.memory.api.extract;

/**
 * Categories of facts and intents that the extraction pipeline can detect
 * in user messages.
 *
 * <p>Each category maps to a different downstream handler: some produce
 * graph nodes, others create follow-up reminders or update preference
 * profiles.
 */
public enum ExtractionCategory {

    /** Biographical / demographic facts about the user. */
    PROFILE,

    /** Named relationships (family, colleagues, friends). */
    RELATIONSHIP,

    /** Expressed likes, dislikes, or habitual choices. */
    PREFERENCE,

    /** Calendared or one-off events (past or future). */
    EVENT,

    /** Stated objectives, aspirations, or targets. */
    GOAL,

    /** Action items the user or the assistant should track. */
    FOLLOW_UP,

    /** Promises implied by context but not explicitly stated. */
    IMPLICIT_COMMITMENT,

    /** Interest in a specific financial instrument or asset class. */
    INVESTMENT_INTEREST,

    /** User explicitly defers a decision or action to a later time. */
    DEFERRAL,

    /** Closure of a previously open follow-up, goal, or commitment. */
    RESOLUTION
}
