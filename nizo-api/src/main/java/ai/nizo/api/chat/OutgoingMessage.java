package ai.nizo.api.chat;

public record OutgoingMessage(String text) {
    public OutgoingMessage {
        if (text == null) text = "";
    }

    public static OutgoingMessage of(String text) { return new OutgoingMessage(text); }
}
