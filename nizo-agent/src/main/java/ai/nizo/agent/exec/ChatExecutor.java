package ai.nizo.agent.exec;

import ai.nizo.agent.loop.AgentLoop;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Registry of {@link ChatExecution}s. One instance per Bootstrap. Channels (web, telegram, CLI)
 * funnel user messages through here.
 *
 * <p>Threading: chat workers run on a virtual-thread executor — cheap to have one permanently
 * blocked-on-take per chat, even with 100s of chats. Total cost is essentially the inbox + ring
 * + a parked virtual thread (a few KB each).
 */
public final class ChatExecutor implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(ChatExecutor.class);

    private final AgentLoop agent;
    private final ExecutorService workers;
    private final ConcurrentHashMap<String, ChatExecution> chats = new ConcurrentHashMap<>();

    public ChatExecutor(AgentLoop agent) {
        this.agent = agent;
        this.workers = Executors.newThreadPerTaskExecutor(
                Thread.ofVirtual().name("chat-exec-", 0).factory());
    }

    /** Returns the existing execution for {@code chatId}, or creates a fresh one. */
    public ChatExecution getOrCreate(String chatId) {
        if (chatId == null || chatId.isBlank()) throw new IllegalArgumentException("chatId required");
        return chats.computeIfAbsent(chatId, id -> {
            LOG.debug("creating chat execution: {}", id);
            return new ChatExecution(id, agent, workers);
        });
    }

    /** Get without creating. Returns {@code null} if no chat with this id exists. */
    public ChatExecution get(String chatId) {
        return chats.get(chatId);
    }

    public Collection<ChatExecution> all() {
        return List.copyOf(chats.values());
    }

    public Map<String, ChatExecution.Status> statusByChat() {
        Map<String, ChatExecution.Status> out = new ConcurrentHashMap<>();
        chats.forEach((id, exec) -> out.put(id, exec.status()));
        return out;
    }

    /** Drop a chat from the registry — use when a session is deleted. */
    public void remove(String chatId) {
        ChatExecution exec = chats.remove(chatId);
        if (exec != null) exec.close();
    }

    @Override
    public void close() {
        chats.values().forEach(ChatExecution::close);
        chats.clear();
        workers.shutdownNow();
    }
}
