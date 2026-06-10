package ai.nizo.api.memory;

import java.time.Instant;
import java.util.List;

/**
 * User-scoped fact storage. Survives across conversations, channels, restarts.
 *
 * <p>Distinct from session history (per-chat transcript). This is a per-user "what does Nizo
 * know about this person" layer that the agent reads at the start of every turn and writes to
 * whenever it learns something durable.
 *
 * <p>Lives in {@code nizo-api} (not {@code nizo-agent}) so {@code nizo-tools} can depend on it
 * without a Maven cycle.
 */
public interface UserFactStore {

    record Fact(long id, String userId, String content, String source, Instant createdAt) {}

    long remember(String userId, String content, String source);

    List<Fact> recall(String userId, int limit);

    List<Fact> search(String userId, String query, int limit);

    boolean forget(long id);

    int forgetMatching(String userId, String query);

    int forgetAll(String userId);

    long count(String userId);
}
