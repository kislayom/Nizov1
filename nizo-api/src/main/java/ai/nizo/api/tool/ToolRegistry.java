package ai.nizo.api.tool;

import ai.nizo.api.llm.ToolDef;

import java.util.List;
import java.util.Optional;

/**
 * Registry of available {@link Tool}s. Initial population happens at agent startup; some
 * implementations also support hot add/remove (e.g. the workbench adding an MCP server at
 * runtime). Mutation methods default to {@link UnsupportedOperationException} so immutable
 * implementations stay safe.
 *
 * <p>Read methods are snapshot-style: callers always see a consistent view, even if another
 * thread is mutating the registry concurrently.
 */
public interface ToolRegistry {

    /** All registered tools, in registration order. */
    List<Tool> all();

    /** Lookup by exact {@link Tool#name()}. */
    Optional<Tool> byName(String name);

    /** Snapshot of {@link ToolDef}s suitable for {@link ai.nizo.api.llm.ChatRequest#tools()}. */
    List<ToolDef> toolDefs();

    /**
     * Add a tool at runtime. Throws {@link IllegalStateException} on duplicate name.
     * Default impl throws — only mutable registries support this.
     */
    default void add(Tool tool) {
        throw new UnsupportedOperationException("registry is immutable");
    }

    /**
     * Remove a tool by name. Returns {@code true} if it existed.
     * Default impl throws — only mutable registries support this.
     */
    default boolean remove(String name) {
        throw new UnsupportedOperationException("registry is immutable");
    }

    /**
     * Remove every tool whose name starts with {@code prefix} (typically used to drop all tools
     * from one MCP server when it's stopped — they share a {@code "<server>__"} prefix).
     * @return number of tools removed
     */
    default int removeByPrefix(String prefix) {
        throw new UnsupportedOperationException("registry is immutable");
    }
}
