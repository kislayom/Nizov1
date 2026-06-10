package ai.nizo.agent.session;

import ai.nizo.api.llm.ChatMessage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/** Process-local non-durable session store. Useful for tests and ephemeral CLI runs. */
public final class InMemorySessionStore implements SessionStore {

    private final ConcurrentHashMap<String, List<ChatMessage>> store = new ConcurrentHashMap<>();

    @Override
    public void append(String chatId, ChatMessage message) {
        store.computeIfAbsent(chatId, k -> Collections.synchronizedList(new ArrayList<>())).add(message);
    }

    @Override
    public List<ChatMessage> recent(String chatId, int n) {
        List<ChatMessage> all = store.get(chatId);
        if (all == null) return List.of();
        synchronized (all) {
            int from = Math.max(0, all.size() - Math.max(0, n));
            return List.copyOf(all.subList(from, all.size()));
        }
    }

    @Override
    public long size(String chatId) {
        List<ChatMessage> all = store.get(chatId);
        return all == null ? 0 : all.size();
    }

    @Override
    public void clear(String chatId) {
        store.remove(chatId);
    }

    @Override
    public List<ChatSummary> listChats(int limit) {
        List<ChatSummary> out = new ArrayList<>();
        for (var e : store.entrySet()) {
            List<ChatMessage> msgs = e.getValue();
            synchronized (msgs) {
                String lastUser = "";
                String lastReply = "";
                for (int i = msgs.size() - 1; i >= 0; i--) {
                    ChatMessage m = msgs.get(i);
                    if (lastReply.isEmpty() && m.role() == ai.nizo.api.llm.Role.ASSISTANT && m.content() != null) {
                        lastReply = m.content().length() > 80 ? m.content().substring(0, 80) : m.content();
                    } else if (lastUser.isEmpty() && m.role() == ai.nizo.api.llm.Role.USER && m.content() != null) {
                        lastUser = m.content().length() > 80 ? m.content().substring(0, 80) : m.content();
                    }
                    if (!lastUser.isEmpty() && !lastReply.isEmpty()) break;
                }
                out.add(new ChatSummary(e.getKey(), System.currentTimeMillis(), msgs.size(), lastUser, lastReply));
            }
        }
        out.sort((a, b) -> Long.compare(b.lastUpdated(), a.lastUpdated()));
        return out.size() > limit ? out.subList(0, limit) : out;
    }
}
