package ai.nizo.api.llm;

import java.util.List;

/**
 * One chat message. Text-only by default; supply {@code images} (data URIs) to send a
 * multimodal user message — the LLM client will pack those into the OpenAI
 * {@code [{type:"text"…}, {type:"image_url"…}]} content array.
 *
 * <p>Image data URIs look like {@code "data:image/jpeg;base64,…"}. Only honored on USER messages.
 */
public record ChatMessage(
        Role role,
        String content,
        List<String> images,
        List<ToolCall> toolCalls,
        String toolCallId,
        String name
) {
    public ChatMessage {
        if (role == null) throw new IllegalArgumentException("role required");
        images = images == null ? List.of() : List.copyOf(images);
        toolCalls = toolCalls == null ? List.of() : List.copyOf(toolCalls);
    }

    public boolean hasImages() {
        return images != null && !images.isEmpty();
    }

    public static ChatMessage system(String content) {
        return new ChatMessage(Role.SYSTEM, content, List.of(), List.of(), null, null);
    }

    public static ChatMessage user(String content) {
        return new ChatMessage(Role.USER, content, List.of(), List.of(), null, null);
    }

    public static ChatMessage userWithImages(String content, List<String> images) {
        return new ChatMessage(Role.USER, content, images, List.of(), null, null);
    }

    public static ChatMessage assistant(String content) {
        return new ChatMessage(Role.ASSISTANT, content, List.of(), List.of(), null, null);
    }

    public static ChatMessage assistantToolCalls(List<ToolCall> toolCalls) {
        return new ChatMessage(Role.ASSISTANT, null, List.of(), toolCalls, null, null);
    }

    public static ChatMessage toolResult(String toolCallId, String content) {
        return new ChatMessage(Role.TOOL, content, List.of(), List.of(), toolCallId, null);
    }
}
