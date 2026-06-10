package ai.nizo.api.tool;

/**
 * One callable capability surfaced to the model.
 *
 * <p>Tools are registered into a {@link ToolRegistry} and exposed to the LLM via
 * {@link ai.nizo.api.llm.ChatRequest#tools()} on every turn. When the model returns a
 * {@code tool_calls} entry, the agent loop dispatches it back here.
 *
 * <p>Implementations should be:
 * <ul>
 *   <li><b>Stateless</b> across calls (state lives in the agent or the session)</li>
 *   <li><b>Deterministic-ish</b> for the same arguments — caching is the caller's choice</li>
 *   <li><b>Bounded</b> — apply your own timeouts; the agent enforces an outer one too</li>
 *   <li><b>Safe to fail</b> — return a {@link ToolResult} with {@code ok=false} rather than throwing
 *       wherever possible. Throws are caught by the loop and surfaced to the model as errors.</li>
 * </ul>
 */
public interface Tool {

    /**
     * Stable, model-visible identifier. Snake_case. No spaces. Examples: {@code web_search},
     * {@code web_fetch}, {@code current_time}.
     */
    String name();

    /**
     * One-paragraph description for the model. Should describe <em>when</em> to use the tool,
     * not just <em>what</em> it does. Concise — every token here is paid on every turn.
     */
    String description();

    /**
     * JSON Schema for the parameters object. Must be a valid JSON object string with
     * {@code "type": "object"}, {@code "properties": {...}}, and ideally {@code "required": [...]}.
     */
    String parametersJsonSchema();

    /**
     * Execute the tool with the model's arguments (a JSON object string matching the schema above).
     */
    ToolResult execute(String argumentsJson) throws Exception;
}
