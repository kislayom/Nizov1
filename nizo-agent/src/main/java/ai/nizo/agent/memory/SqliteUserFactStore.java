package ai.nizo.agent.memory;

import ai.nizo.agent.store.SchemaMigrator;
import ai.nizo.api.memory.UserFactStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * SQLite-backed {@link UserFactStore}. Schema:
 *
 * <pre>
 *   CREATE TABLE user_facts (
 *     id         INTEGER PRIMARY KEY AUTOINCREMENT,
 *     user_id    TEXT    NOT NULL,
 *     content    TEXT    NOT NULL,
 *     source     TEXT,
 *     created_at INTEGER NOT NULL
 *   );
 *   CREATE INDEX idx_user_facts_user ON user_facts(user_id, created_at DESC);
 * </pre>
 */
public final class SqliteUserFactStore implements UserFactStore {

    private static final Logger LOG = LoggerFactory.getLogger(SqliteUserFactStore.class);

    private final Path dbPath;

    public SqliteUserFactStore(Path dbPath) {
        this.dbPath = dbPath;
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            // PRAGMAs persist for the DB; set them once in a connection of our own.
            try (Connection c = open(); Statement st = c.createStatement()) {
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
            }
            // Run schema migrations through SchemaMigrator. Adding columns later requires
            // bumping a new migration entry; SchemaMigrator gates on the schema_version row
            // so v1 is a no-op on already-migrated DBs.
            new SchemaMigrator(adapt(this::open), "user_facts").migrate(MIGRATIONS);
            LOG.info("user-facts store ready at {}", dbPath);
        } catch (Exception e) {
            throw new RuntimeException("failed to open user-facts store at " + dbPath + ": " + e.getMessage(), e);
        }
    }

    private static final List<SchemaMigrator.Migration> MIGRATIONS = List.of(
            SchemaMigrator.of(1, "user_facts baseline", """
                CREATE TABLE IF NOT EXISTS user_facts (
                  id         INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id    TEXT    NOT NULL,
                  content    TEXT    NOT NULL,
                  source     TEXT,
                  created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_facts_user
                    ON user_facts(user_id, created_at DESC);
                """)
    );

    private Connection open() throws SQLException {
        return DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
    }

    /** Tiny DataSource adapter so we can pass {@code this::open} to SchemaMigrator. */
    private static DataSource adapt(SqlConnSupplier supplier) {
        return new DataSourceAdapter(supplier);
    }
    @FunctionalInterface private interface SqlConnSupplier { Connection get() throws SQLException; }
    private static final class DataSourceAdapter implements DataSource {
        private final SqlConnSupplier s;
        DataSourceAdapter(SqlConnSupplier s) { this.s = s; }
        @Override public Connection getConnection() throws SQLException { return s.get(); }
        @Override public Connection getConnection(String u, String p) throws SQLException { return s.get(); }
        @Override public java.io.PrintWriter getLogWriter() { return null; }
        @Override public void setLogWriter(java.io.PrintWriter out) {}
        @Override public void setLoginTimeout(int seconds) {}
        @Override public int getLoginTimeout() { return 0; }
        @Override public java.util.logging.Logger getParentLogger() { return java.util.logging.Logger.getLogger("user_facts"); }
        @Override public <T> T unwrap(Class<T> iface) { return null; }
        @Override public boolean isWrapperFor(Class<?> iface) { return false; }
    }

    @Override
    public long remember(String userId, String content, String source) {
        if (userId == null || userId.isBlank() || content == null || content.isBlank()) return -1;
        String sql = "INSERT INTO user_facts (user_id, content, source, created_at) VALUES (?, ?, ?, ?)";
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS)) {
            ps.setString(1, userId);
            ps.setString(2, content.trim());
            ps.setString(3, source == null ? "agent" : source);
            ps.setLong(4, Instant.now().toEpochMilli());
            ps.executeUpdate();
            try (ResultSet keys = ps.getGeneratedKeys()) {
                return keys.next() ? keys.getLong(1) : -1;
            }
        } catch (Exception e) {
            LOG.warn("remember failed for {}: {}", userId, e.toString());
            return -1;
        }
    }

    @Override
    public List<Fact> recall(String userId, int limit) {
        String sql = "SELECT id, user_id, content, source, created_at FROM user_facts " +
                "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?";
        return query(sql, userId, null, limit);
    }

    @Override
    public List<Fact> search(String userId, String query, int limit) {
        String sql = "SELECT id, user_id, content, source, created_at FROM user_facts " +
                "WHERE user_id = ? AND lower(content) LIKE lower(?) " +
                "ORDER BY created_at DESC LIMIT ?";
        return query(sql, userId, query, limit);
    }

    @Override
    public boolean forget(long id) {
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement("DELETE FROM user_facts WHERE id = ?")) {
            ps.setLong(1, id);
            return ps.executeUpdate() > 0;
        } catch (Exception e) {
            LOG.warn("forget {} failed: {}", id, e.toString());
            return false;
        }
    }

    @Override
    public int forgetMatching(String userId, String query) {
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement(
                     "DELETE FROM user_facts WHERE user_id = ? AND lower(content) LIKE lower(?)")) {
            ps.setString(1, userId);
            ps.setString(2, "%" + (query == null ? "" : query) + "%");
            return ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("forgetMatching failed for {}: {}", userId, e.toString());
            return 0;
        }
    }

    @Override
    public int forgetAll(String userId) {
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement("DELETE FROM user_facts WHERE user_id = ?")) {
            ps.setString(1, userId);
            return ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("forgetAll failed for {}: {}", userId, e.toString());
            return 0;
        }
    }

    @Override
    public long count(String userId) {
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT COUNT(*) FROM user_facts WHERE user_id = ?")) {
            ps.setString(1, userId);
            try (ResultSet rs = ps.executeQuery()) { return rs.next() ? rs.getLong(1) : 0; }
        } catch (Exception e) {
            return 0;
        }
    }

    // ---- internals ----

    private List<Fact> query(String sql, String userId, String like, int limit) {
        List<Fact> out = new ArrayList<>();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, userId);
            int idx = 2;
            if (like != null) ps.setString(idx++, "%" + like + "%");
            ps.setInt(idx, Math.max(1, limit));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(new Fact(
                            rs.getLong("id"),
                            rs.getString("user_id"),
                            rs.getString("content"),
                            rs.getString("source"),
                            Instant.ofEpochMilli(rs.getLong("created_at"))));
                }
            }
        } catch (Exception e) {
            LOG.warn("query failed: {}", e.toString());
        }
        return out;
    }
}
