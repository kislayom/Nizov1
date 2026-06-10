package ai.nizo.tools.memory;

import ai.nizo.api.memory.UserFactStore;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Look up what the agent already knows about the current user.
 *
 * <p>The agent loop already injects all known facts into the system prompt at the start of
 * every turn — but this tool exists for explicit lookups ("what do you remember about my
 * preferences?") and for substring-filtered recall ("what do you know about my work?").
 */
public final class MemoryRecallTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final DateTimeFormatter STAMP =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm").withZone(java.time.ZoneId.systemDefault());

    private final UserFactStore store;

    public MemoryRecallTool(UserFactStore store) { this.store = store; }

    @Override public String name() { return "memory_recall"; }

    @Override
    public String description() {
        return "Retrieve durable facts you've previously stored about the current user. "
                + "Pass an optional 'query' substring to filter; otherwise returns the most recent facts.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "query": { "type": "string", "description": "Optional substring filter." },
                "limit": { "type": "integer", "description": "Max facts (default 50, max 200)." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String query = args.path("query").asText("").trim();
        int limit = clamp(args.path("limit").asInt(50), 1, 200);
        String userId = UserContext.requireUserId();

        List<UserFactStore.Fact> facts = query.isEmpty()
                ? store.recall(userId, limit)
                : store.search(userId, query, limit);

        if (facts.isEmpty()) {
            return ToolResult.ok(query.isEmpty()
                    ? "(no facts stored yet for user '" + userId + "')"
                    : "(no facts matching '" + query + "' for user '" + userId + "')");
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Facts about user '").append(userId).append("' (").append(facts.size()).append("):\n");
        for (UserFactStore.Fact f : facts) {
            sb.append("- ").append(f.content())
              .append("  _(#").append(f.id()).append(", ").append(STAMP.format(f.createdAt())).append(")_\n");
        }
        return ToolResult.ok(sb.toString());
    }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
}
