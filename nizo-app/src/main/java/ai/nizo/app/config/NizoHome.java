package ai.nizo.app.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Locates and lazily creates the per-user Nizo home directory.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code NIZO_HOME} environment variable (absolute path)</li>
 *   <li>{@code $HOME/.nizo}</li>
 * </ol>
 *
 * <p>Standard subdirectories (created on first call):
 * <pre>
 *   ~/.nizo/
 *   ├── sessions.db        (SQLite — agent session history)
 *   ├── memory.db          (SQLite — nizo-memory store, future)
 *   ├── workspace/         (shell + file tools chroot here)
 *   ├── skills/            (filesystem skills, agentskills.io-style)
 *   ├── logs/              (rolling logs)
 *   └── config.yaml        (user config — optional)
 * </pre>
 */
public final class NizoHome {

    private static final Logger LOG = LoggerFactory.getLogger(NizoHome.class);

    private final Path root;

    public NizoHome(Path root) {
        this.root = root.toAbsolutePath().normalize();
        ensure(this.root);
        ensure(workspace());
        ensure(skillsDir());
        ensure(voicesDir());
        ensure(logsDir());
        LOG.info("NIZO_HOME = {}", this.root);
    }

    public static NizoHome resolve() {
        String env = System.getenv("NIZO_HOME");
        Path root = (env != null && !env.isBlank())
                ? Paths.get(env)
                : Paths.get(System.getProperty("user.home"), ".nizo");
        return new NizoHome(root);
    }

    public Path root()         { return root; }
    public Path sessionsDb()   { return root.resolve("sessions.db"); }
    public Path memoryDb()     { return root.resolve("memory.db"); }
    public Path stockReportsDb() { return root.resolve("stock_reports.db"); }
    public Path workspace()    { return root.resolve("workspace"); }
    public Path skillsDir()    { return root.resolve("skills"); }
    public Path voicesDir()    { return root.resolve("voices"); }   // stored XTTS clone reference wavs
    public Path logsDir()      { return root.resolve("logs"); }
    public Path configFile()   { return root.resolve("config.yaml"); }
    /** Claude-Desktop / OpenClaw-compatible MCP server registry. */
    public Path mcpConfigFile(){ return root.resolve("mcp.json"); }

    private static void ensure(Path p) {
        try { Files.createDirectories(p); }
        catch (IOException e) { throw new RuntimeException("could not create " + p, e); }
    }
}
