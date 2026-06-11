package ai.nizo.api.tool;

/**
 * Thread-local userId, set by the agent loop before invoking a tool, cleared after.
 * Tools that need user-scoped state (memory, preferences) read it via {@link #current()}.
 *
 * <p>Lives in {@code nizo-api} (not {@code nizo-tools}) so the agent module can use it
 * without depending on {@code nizo-tools}.
 */
public final class UserContext {

    private static final ThreadLocal<String> CURRENT = new ThreadLocal<>();
    private static final ThreadLocal<String> CHAT = new ThreadLocal<>();

    private UserContext() {}

    public static void set(String userId) { CURRENT.set(userId); }
    public static void clear()             { CURRENT.remove(); CHAT.remove(); }
    public static String current()         { return CURRENT.get(); }

    /** Originating chatId — lets long-running tools (deep_work) deliver results back
     *  to the conversation that started them. Set/cleared alongside the userId. */
    public static void setChat(String chatId) { CHAT.set(chatId); }
    public static String currentChat()        { return CHAT.get(); }

    public static String requireUserId() {
        String u = CURRENT.get();
        if (u == null || u.isBlank()) {
            throw new IllegalStateException(
                    "no userId in context — UserContext.set() must be called by the agent loop before tool execution");
        }
        return u;
    }
}
