package ai.nizo.mcp;

import ai.nizo.mcp.protocol.McpToolDescriptor;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.List;

/**
 * Connection to one MCP server. Implementations are transport-specific
 * ({@code StdioMcpClient} for subprocess pipes, future {@code HttpSseMcpClient} for hosted servers).
 *
 * <p>Lifecycle: {@link #start()} → {@link #listTools()} / {@link #callTool(String, JsonNode)} → {@link #close()}.
 * {@link #start()} performs the {@code initialize} handshake; calling it twice is a no-op.
 *
 * <p>Thread safety: implementations must allow concurrent {@link #callTool} invocations from
 * multiple threads. Each call gets a unique JSON-RPC id and is correlated independently.
 */
public interface McpClient extends AutoCloseable {

    /** Server name used for logging and tool namespacing. */
    String serverName();

    /** Open the transport and run the {@code initialize} handshake. Idempotent. */
    void start() throws McpException;

    /** True after a successful {@link #start()} and before {@link #close()}. */
    boolean isReady();

    /** Fetch the server's tool catalogue. Cached after first call. */
    List<McpToolDescriptor> listTools() throws McpException;

    /**
     * Invoke a tool. {@code argumentsJson} is the JSON object the agent emitted from the LLM
     * — passed through unmodified except for parsing.
     *
     * @return the {@code result} block of the JSON-RPC response, with the canonical
     *         {@code content[]} array and optional {@code isError} flag.
     */
    JsonNode callTool(String toolName, JsonNode argumentsJson) throws McpException;

    @Override
    void close();
}
