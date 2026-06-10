package ai.nizo.channels.telegram;

import ai.nizo.api.chat.ChatHandler;
import ai.nizo.api.chat.IncomingMessage;
import ai.nizo.api.chat.OutgoingMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.telegram.telegrambots.client.okhttp.OkHttpTelegramClient;
import org.telegram.telegrambots.longpolling.TelegramBotsLongPollingApplication;
import org.telegram.telegrambots.longpolling.util.LongPollingSingleThreadUpdateConsumer;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.api.objects.message.Message;
import org.telegram.telegrambots.meta.generics.TelegramClient;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Telegram long-polling channel. Wraps telegrambots 9.0.0.
 *
 * <p>Forwards each text/photo Update to the supplied {@link ChatHandler} and replies with the
 * handler's {@link OutgoingMessage}. Photos are downloaded, base64-encoded, and passed through
 * as data URIs in {@link IncomingMessage#images()}.
 */
public final class TelegramChannel implements LongPollingSingleThreadUpdateConsumer, AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(TelegramChannel.class);
    private static final String CHANNEL_NAME = "telegram";

    private final TelegramConfig config;
    private final ChatHandler handler;
    private final TelegramClient client;
    private final HttpClient http;
    private final TelegramBotsLongPollingApplication app;
    private boolean started;

    public TelegramChannel(TelegramConfig config, ChatHandler handler) {
        this.config = config;
        this.handler = handler;
        this.client = new OkHttpTelegramClient(config.botToken());
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        this.app = new TelegramBotsLongPollingApplication();
    }

    public synchronized void start() throws Exception {
        if (started) return;
        app.registerBot(config.botToken(), this);
        started = true;
        LOG.info("Telegram bot registered as @{}", config.botUsername());
    }

    @Override
    public synchronized void close() throws Exception {
        if (!started) return;
        app.close();
        started = false;
        LOG.info("Telegram bot stopped");
    }

    @Override
    public void consume(Update update) {
        try {
            if (!update.hasMessage()) return;
            Message msg = update.getMessage();
            if (msg.getFrom() == null) return;

            String userId = String.valueOf(msg.getFrom().getId());
            String chatId = String.valueOf(msg.getChatId());
            String text = msg.hasText() ? msg.getText() : (msg.getCaption() == null ? "" : msg.getCaption());

            List<String> images = new ArrayList<>();
            // TODO(vision): re-enable photo download once we lock the telegrambots 9.x PhotoSize/file API.
            // if (msg.hasPhoto()) { ... base64 download ... }

            IncomingMessage in = new IncomingMessage(userId, chatId, text, images, CHANNEL_NAME);

            OutgoingMessage out;
            try {
                out = handler.handle(in);
            } catch (RuntimeException e) {
                LOG.warn("handler failed", e);
                out = OutgoingMessage.of("Sorry — something went wrong handling that message.");
            }

            String reply = out.text();
            if (reply == null || reply.isBlank()) return;

            for (String chunk : splitForTelegram(reply)) {
                client.execute(SendMessage.builder()
                        .chatId(msg.getChatId())
                        .text(chunk)
                        .build());
            }
        } catch (Exception e) {
            LOG.error("consume failed", e);
        }
    }

    /** Telegram caps a single message at 4096 chars. Split on paragraph boundary if longer. */
    private static List<String> splitForTelegram(String text) {
        final int LIMIT = 4000;
        if (text.length() <= LIMIT) return List.of(text);
        List<String> chunks = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            int end = Math.min(text.length(), i + LIMIT);
            if (end < text.length()) {
                int nl = text.lastIndexOf('\n', end);
                if (nl > i + LIMIT / 2) end = nl;
            }
            chunks.add(text.substring(i, end));
            i = end;
        }
        return chunks;
    }
}
