package ai.nizo.tools.registry;

import ai.nizo.api.llm.ToolDef;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Default {@link ToolRegistry}: insertion-ordered, in-memory, mutable + thread-safe.
 *
 * <p>Originally immutable after build; relaxed to support hot add/remove of MCP server tools at
 * runtime (workbench can connect a new MCP server and its tools must show up to the next turn
 * without restarting the agent).
 *
 * <p>Concurrency: every mutator and reader takes the same {@code synchronized} on {@code this}.
 * Read methods snapshot into immutable {@link List}/{@link Map} copies so callers never observe
 * a partial mutation. Mid-turn writes are safe — the agent loop already holds a snapshot of
 * {@code toolDefs()} in its current request.
 */
public final class InMemoryToolRegistry implements ToolRegistry {

    private final LinkedHashMap<String, Tool> byName = new LinkedHashMap<>();

    private InMemoryToolRegistry() {}

    private InMemoryToolRegistry(LinkedHashMap<String, Tool> seed) {
        this.byName.putAll(seed);
    }

    @Override
    public synchronized List<Tool> all() {
        return List.copyOf(byName.values());
    }

    @Override
    public synchronized Optional<Tool> byName(String name) {
        return Optional.ofNullable(byName.get(name));
    }

    @Override
    public synchronized List<ToolDef> toolDefs() {
        List<ToolDef> d = new ArrayList<>(byName.size());
        for (Tool t : byName.values()) {
            d.add(new ToolDef(t.name(), t.description(), t.parametersJsonSchema()));
        }
        return List.copyOf(d);
    }

    @Override
    public synchronized void add(Tool tool) {
        if (tool == null) throw new IllegalArgumentException("tool null");
        if (tool.name() == null || tool.name().isBlank()) throw new IllegalArgumentException("tool name blank");
        if (byName.put(tool.name(), tool) != null) {
            // We restored the previous value via put; but we want strict semantics — duplicate is
            // a programmer error. Roll back so the failing add doesn't silently overwrite.
            byName.remove(tool.name());
            throw new IllegalStateException("duplicate tool name: " + tool.name());
        }
    }

    @Override
    public synchronized boolean remove(String name) {
        return byName.remove(name) != null;
    }

    @Override
    public synchronized int removeByPrefix(String prefix) {
        if (prefix == null || prefix.isEmpty()) return 0;
        int n = 0;
        var it = byName.keySet().iterator();
        while (it.hasNext()) {
            if (it.next().startsWith(prefix)) { it.remove(); n++; }
        }
        return n;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private final LinkedHashMap<String, Tool> byName = new LinkedHashMap<>();

        public Builder add(Tool t) {
            if (t == null) throw new IllegalArgumentException("tool null");
            if (t.name() == null || t.name().isBlank()) throw new IllegalArgumentException("tool name blank");
            if (byName.put(t.name(), t) != null) {
                throw new IllegalStateException("duplicate tool name: " + t.name());
            }
            return this;
        }

        public Builder addAll(Iterable<? extends Tool> ts) {
            for (Tool t : ts) add(t);
            return this;
        }

        public InMemoryToolRegistry build() {
            return new InMemoryToolRegistry(byName);
        }
    }
}
