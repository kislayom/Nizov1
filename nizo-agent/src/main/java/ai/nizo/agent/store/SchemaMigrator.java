package ai.nizo.agent.store;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Real schema migration framework for our SQLite stores.
 *
 * <p><b>Why this exists.</b> Every store before this used {@code CREATE TABLE IF NOT EXISTS}
 * only. Adding a column to that DDL silently no-ops on upgrade because the table already
 * exists with the old shape — then the next {@code INSERT} fails with "no such column."
 * Each store now declares an ordered list of migrations and the migrator applies the new
 * ones in transaction order while tracking a per-namespace version row.
 *
 * <p><b>Namespacing.</b> One {@code schema_version} table holds one row per namespace
 * (e.g. {@code "sessions"}, {@code "stock_reports"}, {@code "user_facts"}). All three
 * stores can share a single DB file safely because their version counters are
 * independent.
 *
 * <p><b>Idempotence.</b> Calling {@link #migrate(List)} a second time with the same list
 * is a no-op: each migration is gated by {@code version > current}.
 *
 * <p><b>Failure semantics.</b> Each migration runs in its own transaction. If
 * {@link Migration#apply(Connection)} throws, we rollback that migration, log, and rethrow
 * a {@link RuntimeException} — the {@code schema_version} row is NOT bumped, so a fix-and-
 * retry will pick up exactly where we left off.
 */
public final class SchemaMigrator {

    private static final Logger LOG = LoggerFactory.getLogger(SchemaMigrator.class);

    /**
     * One ordered migration. Implementations should be idempotent against partial replay
     * where reasonable (e.g. use {@code CREATE TABLE IF NOT EXISTS}, {@code CREATE INDEX IF
     * NOT EXISTS}) but the migrator's per-migration transactional gate is what actually
     * guarantees we don't double-apply on a clean upgrade.
     */
    public interface Migration {
        int version();
        String description();
        void apply(Connection c) throws SQLException;
    }

    /**
     * Convenience factory for the common case: a migration is a name + a SQL block.
     * The SQL is split on {@code ;} (naive but adequate for DDL — no triggers/procedures here)
     * and each statement is executed.
     */
    public static Migration of(int version, String description, String sql) {
        return new Migration() {
            @Override public int version() { return version; }
            @Override public String description() { return description; }
            @Override public void apply(Connection c) throws SQLException {
                try (Statement st = c.createStatement()) {
                    for (String stmt : sql.split(";")) {
                        String trimmed = stmt.trim();
                        if (trimmed.isEmpty()) continue;
                        st.execute(trimmed);
                    }
                }
            }
        };
    }

    private final DataSource ds;
    private final String namespace;

    public SchemaMigrator(DataSource ds, String namespace) {
        this.ds = Objects.requireNonNull(ds, "ds");
        if (namespace == null || namespace.isBlank()) {
            throw new IllegalArgumentException("namespace is required");
        }
        this.namespace = namespace;
        ensureVersionTable();
    }

    /**
     * Apply all migrations in {@code migrations} whose version is greater than the current
     * recorded version for this namespace. Migrations are sorted by version before any
     * comparison, so callers can hand the list in any order.
     *
     * @return the number of migrations applied this call
     */
    public int migrate(List<Migration> migrations) {
        if (migrations == null || migrations.isEmpty()) return 0;
        // Defensive copy + sort + duplicate-version detection. A stale developer rebase
        // can easily land two v3s; better to fail loudly than to apply one and silently
        // skip the other.
        List<Migration> sorted = new ArrayList<>(migrations);
        sorted.sort(Comparator.comparingInt(Migration::version));
        for (int i = 1; i < sorted.size(); i++) {
            if (sorted.get(i).version() == sorted.get(i - 1).version()) {
                throw new IllegalStateException(
                        "duplicate migration version " + sorted.get(i).version()
                                + " in namespace " + namespace);
            }
            if (sorted.get(i).version() <= 0) {
                throw new IllegalStateException(
                        "migration versions must be > 0 (got " + sorted.get(i).version()
                                + " in namespace " + namespace + ")");
            }
        }

        int currentVersion = currentVersion();
        int applied = 0;
        for (Migration m : sorted) {
            if (m.version() <= currentVersion) continue;
            applyOne(m);
            applied++;
        }
        if (applied > 0) {
            LOG.info("schema [{}] migrated {} step(s) -> v{}", namespace,
                    applied, currentVersion());
        }
        return applied;
    }

    /** Current recorded version for this namespace (0 if no row yet). Public for tests. */
    public int currentVersion() {
        try (Connection c = ds.getConnection();
             PreparedStatement ps = c.prepareStatement(
                     "SELECT version FROM schema_version WHERE namespace = ?")) {
            ps.setString(1, namespace);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getInt(1) : 0;
            }
        } catch (SQLException e) {
            throw new RuntimeException("failed to read schema_version for " + namespace, e);
        }
    }

    private void ensureVersionTable() {
        try (Connection c = ds.getConnection(); Statement st = c.createStatement()) {
            st.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    namespace  TEXT PRIMARY KEY,
                    version    INTEGER NOT NULL,
                    applied_at INTEGER NOT NULL
                )
                """);
        } catch (SQLException e) {
            throw new RuntimeException("failed to create schema_version table", e);
        }
    }

    private void applyOne(Migration m) {
        LOG.info("schema [{}] applying v{} — {}", namespace, m.version(), m.description());
        try (Connection c = ds.getConnection()) {
            c.setAutoCommit(false);
            try {
                m.apply(c);
                upsertVersion(c, m.version());
                c.commit();
            } catch (Exception e) {
                try { c.rollback(); } catch (SQLException ignore) { /* nothing useful to do */ }
                throw new RuntimeException(
                        "migration failed for namespace=" + namespace + " v" + m.version()
                                + " (" + m.description() + "): " + e.getMessage(), e);
            } finally {
                try { c.setAutoCommit(true); } catch (SQLException ignore) { }
            }
        } catch (SQLException e) {
            throw new RuntimeException("failed to open connection for migration " + m.version(), e);
        }
    }

    private void upsertVersion(Connection c, int newVersion) throws SQLException {
        try (PreparedStatement ps = c.prepareStatement("""
                INSERT INTO schema_version (namespace, version, applied_at)
                VALUES (?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    version    = excluded.version,
                    applied_at = excluded.applied_at
                """)) {
            ps.setString(1, namespace);
            ps.setInt(2, newVersion);
            ps.setLong(3, Instant.now().toEpochMilli());
            ps.executeUpdate();
        }
    }
}
