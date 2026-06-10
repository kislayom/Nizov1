package ai.nizo.memory.util;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Resolves user-data paths so the SQLite store survives JAR reinstalls.
 *
 * <p>The memory store must live <em>outside</em> the install directory —
 * otherwise an upgrade or uninstall wipes every fact the user ever taught.
 * This helper resolves config paths to a stable location under the user's
 * home directory, expands {@code ~/} and the magic {@code ${user.data}}
 * token, and creates parent directories on demand.
 *
 * <p>Token reference (use any of these in {@code storage.path}):
 * <ul>
 *   <li>{@code ~/something/db.sqlite} — expands to {@code $HOME/something/db.sqlite}</li>
 *   <li>{@code ${user.data}/db.sqlite} — expands to the platform-appropriate
 *       user-data directory (see {@link #defaultDataDir()})</li>
 *   <li>any absolute path — used verbatim</li>
 *   <li>any plain relative path — resolved against CWD (NOT recommended for
 *       production; included for tests and embedded scenarios)</li>
 * </ul>
 *
 * <p>The default data directory is {@code ~/.nizo}. We deliberately do NOT
 * use the OS-specific conventions ({@code ~/Library/Application Support} on
 * macOS, {@code %APPDATA%} on Windows, XDG on Linux) — a single dotted dir
 * in {@code $HOME} is universally understood, easy to back up, and easy to
 * spot for users who want to inspect their data.
 */
public final class DataPaths {

    /** Magic token resolved to {@link #defaultDataDir()} when present in a path. */
    public static final String USER_DATA_TOKEN = "${user.data}";

    /** Default sub-directory under {@code $HOME}. */
    public static final String DEFAULT_DIR_NAME = ".nizo";

    private DataPaths() {}

    /**
     * Default data directory: {@code $HOME/.nizo}. Used to expand the
     * {@link #USER_DATA_TOKEN} token and to back the default storage path.
     */
    public static Path defaultDataDir() {
        return Path.of(System.getProperty("user.home"), DEFAULT_DIR_NAME);
    }

    /** Default database file inside {@link #defaultDataDir()}. */
    public static Path defaultDatabaseFile() {
        return defaultDataDir().resolve("memory.db");
    }

    /**
     * Resolve a config path string into an absolute, materialised
     * {@link Path}. Expands {@code ~/}, the {@link #USER_DATA_TOKEN} token,
     * and creates parent directories if they don't exist.
     *
     * @param raw path string from config (must not be null/blank)
     * @return resolved path; parent directory is guaranteed to exist
     * @throws IllegalArgumentException if {@code raw} is null/blank
     * @throws RuntimeException if parent dir creation fails
     */
    public static Path resolve(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException("path must not be null/blank");
        }
        String s = raw.trim();
        if (s.contains(USER_DATA_TOKEN)) {
            s = s.replace(USER_DATA_TOKEN, defaultDataDir().toString());
        }
        if (s.startsWith("~/") || s.equals("~")) {
            s = System.getProperty("user.home") + s.substring(1);
        }
        Path p = Paths.get(s).toAbsolutePath().normalize();
        ensureParentExists(p);
        return p;
    }

    /** Create the parent directory if it doesn't exist; no-op if it does. */
    public static void ensureParentExists(Path file) {
        Path parent = file.getParent();
        if (parent == null) return;
        try {
            Files.createDirectories(parent);
        } catch (Exception e) {
            throw new RuntimeException(
                    "Failed to create parent directory " + parent + ": " + e.getMessage(), e);
        }
    }
}
