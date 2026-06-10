package ai.nizo.channels.telegram;

public record TelegramConfig(String botToken, String botUsername) {
    public static TelegramConfig fromEnv() {
        String token = System.getenv("TELEGRAM_BOT_TOKEN");
        String name  = System.getenv().getOrDefault("TELEGRAM_BOT_USERNAME", "nizo_bot");
        if (token == null || token.isBlank()) {
            throw new IllegalStateException("TELEGRAM_BOT_TOKEN env var not set");
        }
        return new TelegramConfig(token, name);
    }
}
