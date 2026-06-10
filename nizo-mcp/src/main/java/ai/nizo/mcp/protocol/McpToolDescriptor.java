package ai.nizo.mcp.protocol;

/**
 * One tool advertised by an MCP server.
 *
 * @param name             tool name as the server knows it (e.g. {@code "create_issue"})
 * @param description      human-readable description for the model
 * @param inputSchemaJson  JSON Schema describing the tool's parameters (passed straight to the LLM)
 */
public record McpToolDescriptor(String name, String description, String inputSchemaJson) {
    public McpToolDescriptor {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("name required");
        if (description == null) description = "";
        if (inputSchemaJson == null || inputSchemaJson.isBlank()) {
            inputSchemaJson = "{\"type\":\"object\",\"properties\":{}}";
        }
    }
}
