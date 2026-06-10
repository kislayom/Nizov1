package ai.nizo.tools.memory;

import ai.nizo.api.memory.UserFactStore;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Tool the agent calls when it learns a durable fact about the user (name, role, location,
 * preferences, projects, etc.). Persists user-scoped, survives across conversations.
 *
 * <p>The {@code userId} is passed in via the per-call {@link UserContext} (set by the
 * agent loop just before invocation). Tools in this package access it through
 * {@link UserContext#current()} on the calling thread.
 */
public final class MemoryRememberTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final UserFactStore store;

    public MemoryRememberTool(UserFactStore store) {
        this.store = store;
    }

    @Override public String name() { return "memory_remember"; }

    @Override
    public String description() {
        return "Save a durable fact about the user (name, role, location, preferences, ongoing projects, "
                + "team members, etc.). Use ONLY for facts that should persist across conversations — "
                + "not for per-message context. Phrase as a self-contained sentence in third person, "
                + "e.g. 'User's name is Kislay.', 'User is an SDM at AWS.', 'User prefers concise answers.'";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "fact":   { "type": "string", "description": "The durable fact, third-person sentence." },
                "source": { "type": "string", "description": "Optional source tag (default 'agent')." }
              },
              "required": ["fact"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String fact = args.path("fact").asText("").trim();
        if (fact.isEmpty()) return ToolResult.error("fact is required");
        String source = args.path("source").asText("agent");
        String userId = UserContext.requireUserId();
        long id = store.remember(userId, fact, source);
        if (id < 0) return ToolResult.error("storage failed");
        return ToolResult.ok("remembered fact #" + id + " for user '" + userId + "': " + fact);
    }
}
