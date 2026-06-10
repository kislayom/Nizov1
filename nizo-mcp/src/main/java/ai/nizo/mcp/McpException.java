package ai.nizo.mcp;

/** Thrown when an MCP exchange fails (transport error, JSON-RPC error response, init failure, etc.). */
public final class McpException extends RuntimeException {
    private final int code;

    public McpException(String message) { super(message); this.code = 0; }
    public McpException(String message, Throwable cause) { super(message, cause); this.code = 0; }
    public McpException(int code, String message) { super(message); this.code = code; }

    /** JSON-RPC error code (-32700, -32600, etc.) when applicable; 0 otherwise. */
    public int code() { return code; }
}
