package ai.nizo.mcp.client;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.mcp.McpClient;
import ai.nizo.mcp.protocol.McpToolDescriptor;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Adapter wrapping one remote MCP tool as a native {@link Tool}. Once registered, the agent
 * loop calls it like any built-in tool — streaming, condense, hooks, all unchanged.
 *
 * <p><b>Naming convention</b>: local name is {@code <serverName>__<remoteToolName>}, e.g.
 * {@code github__create_issue}. This prevents collisions when two MCP servers expose tools of
 * the same name (common — many servers ship a {@code list} or {@code search} tool). The remote
 * server only ever sees its own bare tool name.
 *
 * <p><b>Result conversion</b>: MCP returns {@code result.content[]} as an array of typed parts
 * ({@code text}, {@code image}, {@code resource}). For v1 we concatenate text parts; image/resource
 * parts are summarized as a placeholder so the model gets a deterministic string back. The
 * {@code isError} flag flips us to {@link ToolResult#error(String)} so the agent's normal
 * error-handling path (retry / explain) kicks in.
 */
public final class McpClientTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Separator between server name and tool name in the registered tool's local id. */
    public static final String NAME_SEP = "__";

    private final McpClient client;
    private final String serverName;
    private final String remoteName;
    private final String localName;
    private final String description;
    private final String parametersJsonSchema;

    public McpClientTool(McpClient client, McpToolDescriptor td) {
        this.client = client;
        this.serverName = client.serverName();
        this.remoteName = td.name();
        this.localName = serverName + NAME_SEP + remoteName;
        // Prefix the description with the server origin so the model can reason about provenance.
        this.description = "[mcp:" + serverName + "] " + td.description();
        this.parametersJsonSchema = td.inputSchemaJson();
    }

    @Override public String name() { return localName; }
    @Override public String description() { return description; }
    @Override public String parametersJsonSchema() { return parametersJsonSchema; }

    /** Server this tool came from — useful for inspection / UI. */
    public String serverName() { return serverName; }
    /** Remote tool name the server itself uses. */
    public String remoteName() { return remoteName; }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = (argumentsJson == null || argumentsJson.isBlank())
                ? MAPPER.createObjectNode()
                : MAPPER.readTree(argumentsJson);

        JsonNode result = client.callTool(remoteName, args);

        // result.content is the canonical MCP shape: [{type:"text", text:"..."}, ...]
        StringBuilder body = new StringBuilder();
        JsonNode content = result.path("content");
        if (content.isArray()) {
            for (JsonNode part : content) {
                String type = part.path("type").asText("");
                switch (type) {
                    case "text" -> body.append(part.path("text").asText());
                    case "image" -> body.append("[image content (").append(
                            part.path("mimeType").asText("unknown")).append(") omitted]");
                    case "resource" -> body.append("[resource ")
                            .append(part.path("resource").path("uri").asText("?"))
                            .append(" — fetch separately]");
                    default -> body.append(part.toString());
                }
                body.append('\n');
            }
        } else if (content.isTextual()) {
            // Some servers return a plain string. Tolerated.
            body.append(content.asText());
        } else if (!content.isMissingNode() && !content.isNull()) {
            body.append(content.toString());
        }

        String text = body.toString().stripTrailing();
        boolean isError = result.path("isError").asBoolean(false);
        return isError ? ToolResult.error(text) : ToolResult.ok(text);
    }
}
