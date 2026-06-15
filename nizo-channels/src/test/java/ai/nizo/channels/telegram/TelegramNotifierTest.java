package ai.nizo.channels.telegram;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Target-resolution logic for {@link TelegramNotifier} (the network send isn't unit-tested). */
class TelegramNotifierTest {

    @Test
    void numericChatIdGoesToThatTelegramChat() {
        assertEquals("123456789", TelegramNotifier.resolveTarget("123456789", "999"));
        assertEquals("-1001234567", TelegramNotifier.resolveTarget("-1001234567", null), "group ids are negative");
    }

    @Test
    void nonNumericChatFallsBackToNotifyId() {
        assertEquals("999", TelegramNotifier.resolveTarget("web-1jje1ltg", "999"));
        assertEquals("999", TelegramNotifier.resolveTarget("web-user", " 999 "));
    }

    @Test
    void noTargetWhenNeitherUsable() {
        assertNull(TelegramNotifier.resolveTarget("web-user", null));
        assertNull(TelegramNotifier.resolveTarget("web-user", ""));
        assertNull(TelegramNotifier.resolveTarget(null, null));
    }
}
