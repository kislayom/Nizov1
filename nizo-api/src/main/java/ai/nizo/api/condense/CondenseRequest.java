package ai.nizo.api.condense;

/**
 * Inbound request to condense a chat's history.
 *
 * @param chatId      target conversation
 * @param userId      whose conversation; passed through to hooks and re-injection so per-user
 *                    skills/files can be re-attached
 * @param mode        full vs. partial (see {@link CondenseMode})
 * @param pivotIndex  required for partial modes; ignored for {@link CondenseMode#FULL}.
 *                    Index into the chat's loaded message list (0-based).
 * @param trigger     why this fired — useful in logs and for surfacing to the UI
 */
public record CondenseRequest(
        String chatId,
        String userId,
        CondenseMode mode,
        int pivotIndex,
        Trigger trigger
) {
    public CondenseRequest {
        if (chatId == null || chatId.isBlank()) throw new IllegalArgumentException("chatId required");
        if (userId == null || userId.isBlank()) throw new IllegalArgumentException("userId required");
        if (mode == null) throw new IllegalArgumentException("mode required");
        if (mode != CondenseMode.FULL && pivotIndex < 0) {
            throw new IllegalArgumentException("partial mode requires non-negative pivotIndex");
        }
    }

    public static CondenseRequest full(String chatId, String userId, Trigger trigger) {
        return new CondenseRequest(chatId, userId, CondenseMode.FULL, 0, trigger);
    }

    public static CondenseRequest partialFrom(String chatId, String userId, int pivot, Trigger trigger) {
        return new CondenseRequest(chatId, userId, CondenseMode.PARTIAL_FROM, pivot, trigger);
    }

    public static CondenseRequest partialUpTo(String chatId, String userId, int pivot, Trigger trigger) {
        return new CondenseRequest(chatId, userId, CondenseMode.PARTIAL_UP_TO, pivot, trigger);
    }

    /** Why a condense fired. Drives log lines and UI toasts. */
    public enum Trigger {
        /** Token budget check at start of query loop. */
        AUTO,
        /** User typed {@code /condense} or hit the UI button. */
        MANUAL,
        /** LLM returned a prompt-too-long error and we're falling back. */
        REACTIVE
    }
}
