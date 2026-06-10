package ai.nizo.memory.api.model;

import java.util.Map;

/** Structured tool invocation requested by a model. */
public record ToolCall(String id, String name, Map<String, Object> arguments) {}
