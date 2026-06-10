package ai.nizo.agent.session;

/**
 * Lightweight metadata for one conversation. Drives the sessions sidebar in the UI.
 *
 * @param chatId       stable identifier
 * @param lastUpdated  epoch-ms of the most recent message in this chat
 * @param messageCount total messages across all roles
 * @param lastUserText preview of the most recent user message (first ~80 chars)
 * @param lastReply    preview of the most recent assistant reply (first ~80 chars)
 */
public record ChatSummary(
        String chatId,
        long lastUpdated,
        long messageCount,
        String lastUserText,
        String lastReply
) {}
