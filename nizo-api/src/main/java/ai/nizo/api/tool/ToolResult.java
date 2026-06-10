package ai.nizo.api.tool;

/**
 * Outcome of one {@link Tool#execute(String)} call.
 *
 * <p>{@code content} is sent back to the model as the body of the {@code tool} role message.
 * Keep it model-friendly: prefer plain text or short markdown; avoid raw HTML or huge JSON.
 */
public record ToolResult(boolean ok, String content) {

    public static ToolResult ok(String content) {
        return new ToolResult(true, content == null ? "" : content);
    }

    public static ToolResult error(String message) {
        return new ToolResult(false, "ERROR: " + (message == null ? "unknown" : message));
    }
}
