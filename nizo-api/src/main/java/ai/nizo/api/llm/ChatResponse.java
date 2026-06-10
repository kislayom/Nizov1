package ai.nizo.api.llm;

import java.util.List;

public record ChatResponse(
        String content,
        List<ToolCall> toolCalls,
        String finishReason,
        Usage usage
) {
    public ChatResponse {
        toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
    }

    public boolean hasToolCalls() {
        return !toolCalls.isEmpty();
    }

    public record Usage(int promptTokens, int completionTokens, int totalTokens) {
        public static final Usage EMPTY = new Usage(0, 0, 0);
    }
}
