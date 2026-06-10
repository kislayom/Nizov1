package ai.nizo.api.chat;

/**
 * Channel-agnostic message handler. Channels (Telegram, REST, CLI) build {@link IncomingMessage}s
 * and call {@link #handle(IncomingMessage)}. The agent layer implements this.
 */
@FunctionalInterface
public interface ChatHandler {
    OutgoingMessage handle(IncomingMessage in);
}
