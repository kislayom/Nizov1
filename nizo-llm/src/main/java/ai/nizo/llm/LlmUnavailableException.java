package ai.nizo.llm;

/**
 * Thrown by {@link FailoverLlmClient} when every configured provider has exhausted its transport
 * retries — i.e. the model genuinely can't be reached right now (llama-server down, still warming
 * up, or paused for a YuE music render). Distinct from a model/request error, which propagates
 * immediately without failover. {@code RuntimeException} so it fits the {@code LlmClient.chat}
 * signature; the message is user-facing-friendly so the agent loop can surface it as-is.
 */
public class LlmUnavailableException extends RuntimeException {
    public LlmUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
