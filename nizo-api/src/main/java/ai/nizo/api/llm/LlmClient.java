package ai.nizo.api.llm;

public interface LlmClient {
    ChatResponse chat(ChatRequest request);

    default void streamChat(ChatRequest request, ChatStreamHandler handler) {
        throw new UnsupportedOperationException("streaming not implemented for " + getClass().getSimpleName());
    }
}
