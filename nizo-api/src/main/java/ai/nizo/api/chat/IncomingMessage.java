package ai.nizo.api.chat;

import java.util.List;

/**
 * One inbound message from any channel (Telegram, REST, CLI).
 *
 * @param userId    stable per-channel user identifier (e.g. Telegram user id as string)
 * @param chatId    stable per-channel conversation/chat identifier (often == userId for 1:1 chats)
 * @param text      message text (may be empty when only images are attached)
 * @param images    base64 data URIs ("data:image/...;base64,...") to forward to the LLM
 * @param channel   originating channel name (e.g. "telegram"), useful for logging/policy
 * @param mode      "voice" if the input came from speech (transcribed). The agent loop tightens
 *                  brevity + drops markdown for voice mode so the spoken reply is crisp.
 *                  Default: "text".
 * @param language  ISO 639-1 code of the spoken/typed language (e.g. "hi", "es", "en"). Used
 *                  to bias the reply language. Empty/null = let the model decide.
 */
public record IncomingMessage(
        String userId,
        String chatId,
        String text,
        List<String> images,
        String channel,
        String mode,
        String language
) {
    public IncomingMessage {
        if (userId == null || userId.isBlank()) throw new IllegalArgumentException("userId required");
        if (chatId == null || chatId.isBlank()) throw new IllegalArgumentException("chatId required");
        text = text == null ? "" : text;
        images = images == null ? List.of() : List.copyOf(images);
        if (channel == null || channel.isBlank()) throw new IllegalArgumentException("channel required");
        mode = (mode == null || mode.isBlank()) ? "text" : mode;
        language = language == null ? "" : language;
    }

    /** Back-compat constructor — defaults mode="text", language="". */
    public IncomingMessage(String userId, String chatId, String text, List<String> images, String channel) {
        this(userId, chatId, text, images, channel, "text", "");
    }

    public boolean hasImages() {
        return !images.isEmpty();
    }

    public boolean isVoice() {
        return "voice".equalsIgnoreCase(mode);
    }
}
