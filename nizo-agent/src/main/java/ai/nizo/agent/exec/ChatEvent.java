package ai.nizo.agent.exec;

import ai.nizo.api.agent.AgentEvent;

/**
 * One {@link AgentEvent} stamped with a per-chat sequence number and timestamp.
 *
 * <p>The sequence is monotonically increasing per {@link ChatExecution} and is what reconnecting
 * subscribers use to "catch up" — they pass {@code ?since=N} on resubscribe and we replay
 * everything in the ring buffer with {@code seq > N}, then continue tailing live events.
 */
public record ChatEvent(long seq, AgentEvent event, long timestampMs) {}
