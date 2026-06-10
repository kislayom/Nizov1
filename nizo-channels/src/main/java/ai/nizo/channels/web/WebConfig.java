package ai.nizo.channels.web;

public record WebConfig(String host, int port) {
    public static WebConfig fromEnv() {
        String host = System.getenv().getOrDefault("NIZO_WEB_HOST", "127.0.0.1");
        int port    = Integer.parseInt(System.getenv().getOrDefault("NIZO_WEB_PORT", "7777"));
        return new WebConfig(host, port);
    }
}
