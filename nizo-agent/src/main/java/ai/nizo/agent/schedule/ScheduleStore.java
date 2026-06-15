package ai.nizo.agent.schedule;

import ai.nizo.agent.store.SchemaMigrator;
import ai.nizo.agent.store.SqliteDataSource;
import ai.nizo.scheduler.ScheduleKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** SQLite persistence for {@link ScheduledTask}. Survives restarts; the engine queries {@link #due} each tick. */
public final class ScheduleStore {

    private static final Logger LOG = LoggerFactory.getLogger(ScheduleStore.class);
    private final SqliteDataSource ds;

    public ScheduleStore(Path dbPath) {
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            this.ds = new SqliteDataSource(dbPath);
            try (Connection c = ds.getConnection(); Statement st = c.createStatement()) {
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
            }
            new SchemaMigrator(ds, "scheduler").migrate(List.of(
                    SchemaMigrator.of(1, "scheduler baseline", """
                        CREATE TABLE IF NOT EXISTS scheduled_tasks (
                          id           TEXT PRIMARY KEY,
                          kind         TEXT NOT NULL,
                          spec         TEXT NOT NULL,
                          prompt       TEXT NOT NULL,
                          chat_id      TEXT,
                          user_id      TEXT,
                          next_fire_ms INTEGER NOT NULL,
                          last_fire_ms INTEGER NOT NULL DEFAULT 0,
                          enabled      INTEGER NOT NULL DEFAULT 1,
                          created_ms   INTEGER NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS ix_sched_due ON scheduled_tasks(enabled, next_fire_ms);
                        CREATE INDEX IF NOT EXISTS ix_sched_user ON scheduled_tasks(user_id, created_ms DESC);
                        """)
            ));
            LOG.info("scheduler store ready at {}", dbPath);
        } catch (Exception e) {
            throw new RuntimeException("failed to open scheduler store at " + dbPath, e);
        }
    }

    public void add(ScheduledTask t) {
        String sql = """
            INSERT INTO scheduled_tasks (id, kind, spec, prompt, chat_id, user_id, next_fire_ms,
                                         last_fire_ms, enabled, created_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """;
        try (Connection c = ds.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, t.id()); ps.setString(2, t.kind().name()); ps.setString(3, t.spec());
            ps.setString(4, t.prompt()); ps.setString(5, t.chatId()); ps.setString(6, t.userId());
            ps.setLong(7, t.nextFireMs()); ps.setLong(8, t.lastFireMs());
            ps.setInt(9, t.enabled() ? 1 : 0); ps.setLong(10, t.createdMs());
            ps.executeUpdate();
        } catch (Exception e) {
            throw new RuntimeException("schedule add failed: " + e.getMessage(), e);
        }
    }

    /** Enabled tasks whose next firing is at or before {@code nowMs}. */
    public List<ScheduledTask> due(long nowMs) {
        return query("SELECT * FROM scheduled_tasks WHERE enabled=1 AND next_fire_ms<=? ORDER BY next_fire_ms",
                ps -> ps.setLong(1, nowMs));
    }

    public List<ScheduledTask> listForUser(String userId) {
        return query("SELECT * FROM scheduled_tasks WHERE enabled=1 AND (user_id=? OR ? IS NULL) ORDER BY next_fire_ms",
                ps -> { ps.setString(1, userId); ps.setString(2, userId); });
    }

    public Optional<ScheduledTask> get(String id) {
        List<ScheduledTask> r = query("SELECT * FROM scheduled_tasks WHERE id=?", ps -> ps.setString(1, id));
        return r.isEmpty() ? Optional.empty() : Optional.of(r.get(0));
    }

    /** Record a firing: update last/next fire times. */
    public void markFired(String id, long lastMs, long nextMs) {
        exec("UPDATE scheduled_tasks SET last_fire_ms=?, next_fire_ms=? WHERE id=?",
                ps -> { ps.setLong(1, lastMs); ps.setLong(2, nextMs); ps.setString(3, id); });
    }

    public void setEnabled(String id, boolean enabled) {
        exec("UPDATE scheduled_tasks SET enabled=? WHERE id=?",
                ps -> { ps.setInt(1, enabled ? 1 : 0); ps.setString(2, id); });
    }

    public boolean delete(String id) {
        try (Connection c = ds.getConnection();
             PreparedStatement ps = c.prepareStatement("DELETE FROM scheduled_tasks WHERE id=?")) {
            ps.setString(1, id);
            return ps.executeUpdate() > 0;
        } catch (Exception e) {
            throw new RuntimeException("schedule delete failed: " + e.getMessage(), e);
        }
    }

    // ── helpers ─────────────────────────────────────────────────────────────
    private interface Binder { void bind(PreparedStatement ps) throws Exception; }

    private List<ScheduledTask> query(String sql, Binder b) {
        List<ScheduledTask> out = new ArrayList<>();
        try (Connection c = ds.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            b.bind(ps);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(map(rs));
            }
        } catch (Exception e) {
            throw new RuntimeException("schedule query failed: " + e.getMessage(), e);
        }
        return out;
    }

    private void exec(String sql, Binder b) {
        try (Connection c = ds.getConnection(); PreparedStatement ps = c.prepareStatement(sql)) {
            b.bind(ps);
            ps.executeUpdate();
        } catch (Exception e) {
            throw new RuntimeException("schedule exec failed: " + e.getMessage(), e);
        }
    }

    private static ScheduledTask map(ResultSet rs) throws Exception {
        return new ScheduledTask(
                rs.getString("id"), ScheduleKind.valueOf(rs.getString("kind")), rs.getString("spec"),
                rs.getString("prompt"), rs.getString("chat_id"), rs.getString("user_id"),
                rs.getLong("next_fire_ms"), rs.getLong("last_fire_ms"),
                rs.getInt("enabled") == 1, rs.getLong("created_ms"));
    }
}
