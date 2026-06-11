package ai.nizo.agent.deepwork;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Chat-facing entry point for {@link DeepWorkEngine}. Late-bound to the engine because
 * the tool must be registered while the registry is still being built (and the engine
 * needs the finished registry) — Bootstrap calls {@link #bind} after both exist.
 */
public final class DeepWorkTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private volatile DeepWorkEngine engine;

    public void bind(DeepWorkEngine engine) { this.engine = engine; }

    @Override public String name() { return "deep_work"; }

    @Override
    public String description() {
        return "Start a LONG-RUNNING background job for complex tasks: deep multi-source analysis, "
                + "research that needs many steps, comparisons across several entities, anything the "
                + "user says can 'take its time'. The job plans concrete steps, executes them one at "
                + "a time with tools, VERIFIES each step's result against tool evidence, survives "
                + "restarts, and posts the finished deliverable into this chat. Use this instead of "
                + "trying to do a big job inline; tell the user the job id and the plan it returns.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "goal":        { "type": "string", "description": "What to accomplish, with all specifics the user gave (entities, criteria, constraints)." },
                "deliverable": { "type": "string", "description": "Optional: shape of the final output (e.g. 'ranked memo with a table and verdict')." }
              },
              "required": ["goal"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        if (engine == null) return ToolResult.error("deep-work engine not wired");
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            String goal = args.path("goal").asText("").trim();
            if (goal.isEmpty()) return ToolResult.error("goal is required");
            String deliverable = args.path("deliverable").asText("").trim();
            String userId = UserContext.current() == null ? "web-user" : UserContext.current();
            String chatId = UserContext.currentChat() == null ? "deepwork" : UserContext.currentChat();
            return ToolResult.ok(engine.start(userId, chatId, goal, deliverable));
        } catch (Exception e) {
            return ToolResult.error("deep_work failed to start: " + e.getMessage());
        }
    }
}
