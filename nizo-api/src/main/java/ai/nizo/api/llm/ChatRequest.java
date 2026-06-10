package ai.nizo.api.llm;

import java.util.List;
import java.util.Map;

public record ChatRequest(
        String model,
        List<ChatMessage> messages,
        List<ToolDef> tools,
        Double temperature,
        Integer maxTokens,
        boolean stream,
        /** Arbitrary extra JSON fields merged into the request body. Used for
         *  provider-specific knobs (e.g. Qwen3.6's {@code chat_template_kwargs.enable_thinking:false}
         *  to force content output when the model is stuck in reasoning_content-only loops). */
        Map<String, Object> extraBody
) {
    public ChatRequest {
        if (model == null || model.isBlank()) throw new IllegalArgumentException("model required");
        if (messages == null || messages.isEmpty()) throw new IllegalArgumentException("messages required");
        messages = List.copyOf(messages);
        tools = tools == null ? List.of() : List.copyOf(tools);
        extraBody = extraBody == null ? Map.of() : Map.copyOf(extraBody);
    }

    public static ChatRequest of(String model, List<ChatMessage> messages) {
        return new ChatRequest(model, messages, List.of(), null, null, false, Map.of());
    }

    public ChatRequest withTools(List<ToolDef> tools) {
        return new ChatRequest(model, messages, tools, temperature, maxTokens, stream, extraBody);
    }

    public ChatRequest withTemperature(double t) {
        return new ChatRequest(model, messages, tools, t, maxTokens, stream, extraBody);
    }

    public ChatRequest withMaxTokens(int n) {
        return new ChatRequest(model, messages, tools, temperature, n, stream, extraBody);
    }

    public ChatRequest withStream(boolean s) {
        return new ChatRequest(model, messages, tools, temperature, maxTokens, s, extraBody);
    }

    public ChatRequest withExtraBody(Map<String, Object> extras) {
        return new ChatRequest(model, messages, tools, temperature, maxTokens, stream, extras);
    }
}
