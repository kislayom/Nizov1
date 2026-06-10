package ai.nizo.agent.session;

import ai.nizo.api.llm.ChatMessage;

import java.util.List;

/**
 * Persistent rolling history per chat. Implementations may live in-memory, in SQLite, or
 * be backed by the nizo-memory layer.
 *
 * <p>Contract: {@link #append} is atomic; concurrent appends to the same chatId serialize.
 */
public interface SessionStore {

    /** Load the last N messages for a chat (oldest first), capped at {@code limit}. */
    List<ChatMessage> recent(String chatId, int limit);

    /** Append a single message. Implementations should persist before returning. */
    void append(String chatId, ChatMessage message);

    /**
     * Append with an owner tag. {@code userId} marks which user the chat belongs to
     * (web identity picker / Telegram user id). Default ignores the tag for
     * implementations that don't track ownership.
     */
    default void append(String chatId, ChatMessage message, String userId) {
        append(chatId, message);
    }

    /** Wipe a single chat's history. */
    void clear(String chatId);

    /**
     * Atomically replace the entire history for {@code chatId} with {@code messages}. Used by the
     * condense engine after it has synthesized a summary + re-injection set. Default implementation
     * is {@code clear} + per-message {@code append}; SQL-backed stores should override to do this
     * in a single transaction.
     */
    default void replaceHistory(String chatId, List<ChatMessage> messages) {
        clear(chatId);
        if (messages == null) return;
        for (ChatMessage m : messages) append(chatId, m);
    }

    /** Best-effort estimate of count (for stats/logging). May be 0 if not tracked. */
    default long size(String chatId) { return 0L; }

    /** All known chats, newest first. Default returns empty (not all impls track this). */
    default List<ChatSummary> listChats(int limit) { return List.of(); }

    /**
     * Chats owned by {@code userId}, newest first. Legacy rows written before ownership
     * tracking (user_id NULL) are attributed to the owner identity {@code "web-user"} so
     * pre-existing history stays with Kislay. Default ignores the filter.
     */
    default List<ChatSummary> listChats(int limit, String userId) { return listChats(limit); }
}
