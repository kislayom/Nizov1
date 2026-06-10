package ai.nizo.agent.exec;

/**
 * One consumer of a {@link ChatExecution}'s event stream. Typically a SSE writer in
 * {@code WebChannel} — but anything that wants live events (Telegram channel, log sink,
 * test harness) implements this.
 *
 * <p>Subscribers should be cheap; a misbehaving subscriber must not block the producer.
 * {@link ChatExecution} catches and ignores any thrown exception.
 */
public interface EventSubscriber {
    void onEvent(ChatEvent ev);

    /** Called when the chat's executor is closing — last chance to flush. */
    default void onEnd() {}
}
