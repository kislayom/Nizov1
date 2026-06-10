package ai.nizo.tools.memory;

import ai.nizo.api.memory.UserFactStore;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Forget facts about the current user. Required to honor "forget about X" requests.
 */
public final class MemoryForgetTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final UserFactStore store;

    public MemoryForgetTool(UserFactStore store) { this.store = store; }

    @Override public String name() { return "memory_forget"; }

    @Override
    public String description() {
        return "Forget stored facts. Pass 'fact_id' to drop a single fact, OR 'query' to drop all "
                + "facts whose content matches that substring (case-insensitive). Use when the user "
                + "says 'forget about X' or corrects a previously stored fact.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "fact_id": { "type": "integer", "description": "Forget this single fact id." },
                "query":   { "type": "string",  "description": "Forget all facts containing this substring." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String userId = UserContext.requireUserId();

        if (args.has("fact_id") && args.get("fact_id").isNumber()) {
            long id = args.get("fact_id").asLong();
            return store.forget(id) ? ToolResult.ok("forgot fact #" + id) : ToolResult.error("no fact #" + id);
        }
        String query = args.path("query").asText("").trim();
        if (query.isEmpty()) return ToolResult.error("either fact_id or query is required");
        int n = store.forgetMatching(userId, query);
        return ToolResult.ok("forgot " + n + " fact(s) matching '" + query + "'");
    }
}
