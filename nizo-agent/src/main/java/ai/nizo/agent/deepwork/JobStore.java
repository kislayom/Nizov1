package ai.nizo.agent.deepwork;

import ai.nizo.agent.store.SchemaMigrator;
import ai.nizo.agent.store.SqliteDataSource;
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

/**
 * SQLite persistence for long-running "deep work" jobs. Every state transition is
 * checkpointed here so a job survives nizo-app restarts and box reboots — the worker
 * resumes from the last completed step, never re-doing verified work.
 *
 * <p>Two tables: one row per job, one row per plan step. Steps carry their own
 * status/attempts/result so progress is inspectable mid-flight (job_status tool, UI).
 */
public final class JobStore {

    private static final Logger LOG = LoggerFactory.getLogger(JobStore.class);

    public enum JobStatus { PLANNING, RUNNING, DONE, FAILED, CANCELLED }
    public enum StepStatus { PENDING, RUNNING, PASS, FAIL }

    public record Job(String id, String userId, String chatId, String goal,
                      String deliverable, String status, int currentStep,
                      long createdAt, long updatedAt, String finalText, String error) {}

    public record Step(String jobId, int idx, String title, String status,
                       int attempts, String resultSummary, String verifyNote, long updatedAt) {}

    private final SqliteDataSource ds;

    public JobStore(Path dbPath) {
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            this.ds = new SqliteDataSource(dbPath);
            try (Connection c = ds.getConnection(); Statement st = c.createStatement()) {
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
            }
            new SchemaMigrator(ds, "deep_work").migrate(List.of(
                    SchemaMigrator.of(1, "deep_work baseline", """
                        CREATE TABLE IF NOT EXISTS deep_jobs (
                          id          TEXT PRIMARY KEY,
                          user_id     TEXT NOT NULL,
                          chat_id     TEXT NOT NULL,
                          goal        TEXT NOT NULL,
                          deliverable TEXT,
                          status      TEXT NOT NULL,
                          current_step INTEGER NOT NULL DEFAULT 0,
                          created_at  INTEGER NOT NULL,
                          updated_at  INTEGER NOT NULL,
                          final_text  TEXT,
                          error       TEXT
                        );
                        CREATE INDEX IF NOT EXISTS ix_deep_jobs_user ON deep_jobs(user_id, created_at DESC);
                        CREATE TABLE IF NOT EXISTS deep_steps (
                          job_id      TEXT NOT NULL,
                          idx         INTEGER NOT NULL,
                          title       TEXT NOT NULL,
                          status      TEXT NOT NULL,
                          attempts    INTEGER NOT NULL DEFAULT 0,
                          result_summary TEXT,
                          verify_note TEXT,
                          updated_at  INTEGER NOT NULL,
                          PRIMARY KEY (job_id, idx)
                        );
                        """)
            ));
            LOG.info("deep-work store ready at {}", dbPath);
        } catch (Exception e) {
            throw new RuntimeException("failed to open deep-work store at " + dbPath, e);
        }
    }

    private Connection open() throws Exception { return ds.getConnection(); }

    public void createJob(Job j, List<String> stepTitles) {
        String ins = """
            INSERT INTO deep_jobs (id, user_id, chat_id, goal, deliverable, status, current_step,
                                   created_at, updated_at, final_text, error)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, NULL)
            """;
        String insStep = """
            INSERT INTO deep_steps (job_id, idx, title, status, attempts, result_summary, verify_note, updated_at)
            VALUES (?, ?, ?, 'PENDING', 0, NULL, NULL, ?)
            """;
        try (Connection c = open()) {
            c.setAutoCommit(false);
            try {
                long now = System.currentTimeMillis();
                try (PreparedStatement ps = c.prepareStatement(ins)) {
                    ps.setString(1, j.id()); ps.setString(2, j.userId()); ps.setString(3, j.chatId());
                    ps.setString(4, j.goal()); ps.setString(5, j.deliverable());
                    ps.setString(6, j.status()); ps.setLong(7, now); ps.setLong(8, now);
                    ps.executeUpdate();
                }
                try (PreparedStatement ps = c.prepareStatement(insStep)) {
                    for (int i = 0; i < stepTitles.size(); i++) {
                        ps.setString(1, j.id()); ps.setInt(2, i);
                        ps.setString(3, stepTitles.get(i)); ps.setLong(4, now);
                        ps.addBatch();
                    }
                    ps.executeBatch();
                }
                c.commit();
            } catch (Exception e) {
                c.rollback(); throw e;
            } finally {
                c.setAutoCommit(true);
            }
        } catch (Exception e) {
            throw new RuntimeException("createJob failed: " + e.getMessage(), e);
        }
    }

    public void setJobStatus(String id, JobStatus status, String finalText, String error) {
        String sql = "UPDATE deep_jobs SET status=?, final_text=COALESCE(?, final_text), " +
                     "error=?, updated_at=? WHERE id=?";
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, status.name());
            ps.setString(2, finalText);
            ps.setString(3, error);
            ps.setLong(4, System.currentTimeMillis());
            ps.setString(5, id);
            ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("setJobStatus failed for {}: {}", id, e.toString());
        }
    }

    public void setCurrentStep(String id, int idx) {
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "UPDATE deep_jobs SET current_step=?, updated_at=? WHERE id=?")) {
            ps.setInt(1, idx); ps.setLong(2, System.currentTimeMillis()); ps.setString(3, id);
            ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("setCurrentStep failed for {}: {}", id, e.toString());
        }
    }

    public void updateStep(String jobId, int idx, StepStatus status, Integer attempts,
                           String resultSummary, String verifyNote) {
        String sql = "UPDATE deep_steps SET status=?, attempts=COALESCE(?, attempts), " +
                     "result_summary=COALESCE(?, result_summary), verify_note=COALESCE(?, verify_note), " +
                     "updated_at=? WHERE job_id=? AND idx=?";
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, status.name());
            if (attempts == null) ps.setNull(2, java.sql.Types.INTEGER); else ps.setInt(2, attempts);
            ps.setString(3, resultSummary);
            ps.setString(4, verifyNote);
            ps.setLong(5, System.currentTimeMillis());
            ps.setString(6, jobId);
            ps.setInt(7, idx);
            ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("updateStep failed for {}#{}: {}", jobId, idx, e.toString());
        }
    }

    public Optional<Job> get(String id) {
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "SELECT * FROM deep_jobs WHERE id=?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) return Optional.of(readJob(rs));
            }
        } catch (Exception e) {
            LOG.warn("get failed for {}: {}", id, e.toString());
        }
        return Optional.empty();
    }

    public List<Job> recent(String userId, int limit) {
        List<Job> out = new ArrayList<>();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "SELECT * FROM deep_jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?")) {
            ps.setString(1, userId);
            ps.setInt(2, Math.max(1, Math.min(limit, 100)));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(readJob(rs));
            }
        } catch (Exception e) {
            LOG.warn("recent failed: {}", e.toString());
        }
        return out;
    }

    /** Jobs that were mid-flight when the process died — resumed on boot. */
    public List<Job> resumable() {
        List<Job> out = new ArrayList<>();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "SELECT * FROM deep_jobs WHERE status IN ('PLANNING','RUNNING') ORDER BY created_at")) {
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(readJob(rs));
            }
        } catch (Exception e) {
            LOG.warn("resumable query failed: {}", e.toString());
        }
        return out;
    }

    public List<Step> steps(String jobId) {
        List<Step> out = new ArrayList<>();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "SELECT * FROM deep_steps WHERE job_id=? ORDER BY idx")) {
            ps.setString(1, jobId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(new Step(rs.getString("job_id"), rs.getInt("idx"),
                            rs.getString("title"), rs.getString("status"),
                            rs.getInt("attempts"), rs.getString("result_summary"),
                            rs.getString("verify_note"), rs.getLong("updated_at")));
                }
            }
        } catch (Exception e) {
            LOG.warn("steps failed for {}: {}", jobId, e.toString());
        }
        return out;
    }

    private static Job readJob(ResultSet rs) throws Exception {
        return new Job(rs.getString("id"), rs.getString("user_id"), rs.getString("chat_id"),
                rs.getString("goal"), rs.getString("deliverable"), rs.getString("status"),
                rs.getInt("current_step"), rs.getLong("created_at"), rs.getLong("updated_at"),
                rs.getString("final_text"), rs.getString("error"));
    }
}
