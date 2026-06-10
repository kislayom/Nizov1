package ai.nizo.api.llm;

public record ToolCall(String id, String name, String argumentsJson) {
}
