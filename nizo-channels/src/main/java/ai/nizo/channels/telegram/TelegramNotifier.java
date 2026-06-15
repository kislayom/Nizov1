package ai.nizo.channels.telegram;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Fire-and-forget Telegram push via the Bot API — used to deliver scheduled reminders proactively
 * (so a reminder reaches the user even when they aren't looking at the web UI).
 *
 * <p>No-op unless {@code TELEGRAM_BOT_TOKEN} is set. Target resolution: if the schedule's chat id
 * looks like a Telegram chat (numeric), push there; otherwise fall back to
 * {@code TELEGRAM_NOTIFY_CHAT_ID} (the single user's Telegram chat) so web-originated reminders
 * still reach them. Uses a plain form-encoded POST (no JSON escaping headaches) and HTTP/1.1.
 */
public final class TelegramNotifier {

    private static final Logger LOG = LoggerFactory.getLogger(TelegramNotifier.class);
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10))
            .build();
    private static final int MAX_LEN = 4000;   // Telegram hard-caps a message at 4096 chars

    private TelegramNotifier() {}

    /** Push {@code text} to the right Telegram chat for {@code chatId}. Returns true if sent. */
    public static boolean push(String chatId, String text) {
        String token = System.getenv("TELEGRAM_BOT_TOKEN");
        if (token == null || token.isBlank() || text == null || text.isBlank()) return false;
        String target = resolveTarget(chatId, System.getenv("TELEGRAM_NOTIFY_CHAT_ID"));
        if (target == null) return false;
        try {
            String body = "chat_id=" + URLEncoder.encode(target, StandardCharsets.UTF_8)
                    + "&text=" + URLEncoder.encode(truncate(text), StandardCharsets.UTF_8)
                    + "&disable_web_page_preview=true";
            HttpRequest req = HttpRequest.newBuilder(
                            URI.create("https://api.telegram.org/bot" + token.trim() + "/sendMessage"))
                    .timeout(Duration.ofSeconds(20))
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> r = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
            boolean ok = r.statusCode() / 100 == 2;
            if (!ok) LOG.warn("telegram push failed: status={} body={}", r.statusCode(),
                    r.body().length() > 160 ? r.body().substring(0, 160) : r.body());
            return ok;
        } catch (Exception e) {
            LOG.warn("telegram push exception: {}", e.toString());
            return false;
        }
    }

    /** A numeric chat id is a Telegram chat → use it; else fall back to the configured notify id (or null). */
    static String resolveTarget(String chatId, String notifyChatId) {
        if (chatId != null && chatId.trim().matches("-?\\d{4,}")) return chatId.trim();
        return (notifyChatId != null && !notifyChatId.isBlank()) ? notifyChatId.trim() : null;
    }

    private static String truncate(String s) {
        return s.length() <= MAX_LEN ? s : s.substring(0, MAX_LEN) + "…";
    }
}
