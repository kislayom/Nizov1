package ai.nizo.memory.api.extract;

/**
 * Extracts structured facts, intents, and relationships from user messages.
 *
 * <p>The extraction pipeline inspects a single message and returns an
 * {@link ExtractionResult} containing every detectable category. Downstream
 * consumers (graph writers, reminder schedulers, preference updaters)
 * subscribe to the categories they care about.
 *
 * <p>Implementations may delegate to an LLM, a rule engine, or a hybrid
 * of both. The interface deliberately hides that choice.
 */
public interface ExtractionService {

    /**
     * Analyses a user message and returns all extractable facts and intents.
     *
     * @param userId  owner; extractions are scoped per user for context
     * @param message the raw user message text
     * @return extraction result; never {@code null} -- returns
     *         {@link ExtractionResult#empty()} when nothing is found
     */
    ExtractionResult extract(String userId, String message);

    /**
     * Quick classifier: returns {@code true} if the message is purely a
     * command or system directive with no extractable personal content
     * (e.g. "show my portfolio", "clear chat"). Callers can skip the
     * heavier {@link #extract} path for such messages.
     *
     * @param message the raw user message text
     * @return {@code true} if extraction can be skipped
     */
    boolean isCommandOnly(String message);
}
