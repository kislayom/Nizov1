package ai.nizo.agent.cache;

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
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * SQLite-backed cache of completed stock-analysis reports.
 *
 * <p>Single-user for now (user_id is always "default"); the schema is per-user
 * already so adding multi-user later is a no-op. Each row is one finished
 * stock chat — keyed by chatId, indexed by (user_id, ticker, created_at).
 *
 * <p>Schema:
 * <pre>
 *   CREATE TABLE stock_reports (
 *     chat_id    TEXT PRIMARY KEY,
 *     user_id    TEXT NOT NULL,
 *     ticker     TEXT NOT NULL,
 *     created_at INTEGER NOT NULL,
 *     final_text TEXT NOT NULL,    -- master report markdown
 *     prompt     TEXT,             -- user kickoff message
 *     channel    TEXT,             -- "web" / "ios"
 *     iters      INTEGER,
 *     tools      INTEGER,
 *     elapsed_ms INTEGER,
 *     stop_reason TEXT
 *   );
 *   CREATE INDEX idx_stock_reports_user_created ON stock_reports(user_id, created_at DESC);
 *   CREATE INDEX idx_stock_reports_user_ticker  ON stock_reports(user_id, ticker, created_at DESC);
 * </pre>
 */
public final class StockReportStore {

    private static final Logger LOG = LoggerFactory.getLogger(StockReportStore.class);

    public static final String DEFAULT_USER = "default";

    public record Report(
            String chatId,
            String userId,
            String ticker,
            long createdAt,
            String finalText,
            String prompt,
            String channel,
            int iters,
            int tools,
            long elapsedMs,
            String stopReason
    ) {}

    private final SqliteDataSource ds;
    private final Path dbPath;

    public StockReportStore(Path dbPath) {
        this.dbPath = dbPath;
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            this.ds = new SqliteDataSource(dbPath);
            try (Connection c = open(); Statement st = c.createStatement()) {
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
            }
            new SchemaMigrator(ds, "stock_reports").migrate(MIGRATIONS);
            LOG.info("stock-reports store ready at {}", dbPath);
        } catch (Exception e) {
            throw new RuntimeException("failed to open stock-reports store at " + dbPath + ": " + e.getMessage(), e);
        }
    }

    /**
     * Migration ledger. The original {@code CREATE TABLE} included
     * {@code iters / tools / elapsed_ms / stop_reason} but those columns were added some
     * time after the first deploy — older databases never picked them up because
     * {@code CREATE TABLE IF NOT EXISTS} is silent on schema drift.
     *
     * <p>The split below repairs that: v1 = the ORIGINAL columns only, v2 = the four
     * additions. Fresh DBs run both back-to-back; in-the-wild DBs that already have v1's
     * columns just pick up v2.
     */
    private static final List<SchemaMigrator.Migration> MIGRATIONS = List.of(
            SchemaMigrator.of(1, "stock_reports baseline", """
                CREATE TABLE IF NOT EXISTS stock_reports (
                  chat_id     TEXT PRIMARY KEY,
                  user_id     TEXT NOT NULL,
                  ticker      TEXT NOT NULL,
                  created_at  INTEGER NOT NULL,
                  final_text  TEXT NOT NULL,
                  prompt      TEXT,
                  channel     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_stock_reports_user_created
                    ON stock_reports(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stock_reports_user_ticker
                    ON stock_reports(user_id, ticker, created_at DESC);
                """),
            // SQLite ignores ALTER TABLE ADD COLUMN if the column already exists? No — it
            // errors with "duplicate column name". We catch that per-statement so an
            // already-upgraded DB (where v1's old DDL had these columns from the start)
            // still ticks the version forward.
            new SchemaMigrator.Migration() {
                @Override public int version() { return 2; }
                @Override public String description() { return "add iters/tools/elapsed_ms/stop_reason"; }
                @Override public void apply(Connection c) throws java.sql.SQLException {
                    addColumnIfMissing(c, "stock_reports", "iters", "INTEGER");
                    addColumnIfMissing(c, "stock_reports", "tools", "INTEGER");
                    addColumnIfMissing(c, "stock_reports", "elapsed_ms", "INTEGER");
                    addColumnIfMissing(c, "stock_reports", "stop_reason", "TEXT");
                }
            }
    );

    /**
     * Idempotent ADD COLUMN. SQLite's {@code PRAGMA table_info} gives us the current shape;
     * we add only what's missing. The defensive check is needed because v1's original DDL
     * (the one we replaced) already had these columns inline — so a DB created against
     * the OLD code will be at v0 in {@code schema_version} but already have the columns.
     */
    private static void addColumnIfMissing(Connection c, String table, String column, String type)
            throws java.sql.SQLException {
        boolean present = false;
        try (Statement st = c.createStatement();
             ResultSet rs = st.executeQuery("PRAGMA table_info(" + table + ")")) {
            while (rs.next()) {
                if (column.equalsIgnoreCase(rs.getString("name"))) { present = true; break; }
            }
        }
        if (present) return;
        try (Statement st = c.createStatement()) {
            st.execute("ALTER TABLE " + table + " ADD COLUMN " + column + " " + type);
        }
    }

    private Connection open() throws Exception {
        return ds.getConnection();
    }

    /**
     * Save a report. Atomically replaces any prior row for the same (user_id, ticker)
     * so the library shows ONE entry per ticker (latest run). UPSERT on chat_id covers
     * the rare case of saving twice with the same chat_id (e.g. retry).
     *
     * <p>Why per-ticker dedup rather than full history: the user has been clear (May 2026)
     * that the library should show "just one last run" per ticker — multiple stale AAPL
     * cards is clutter. The full per-chat conversation history is still in sessions.db
     * if anyone needs it.
     */
    public void save(Report r) {
        if (r == null || r.chatId == null || r.chatId.isBlank()
                || r.finalText == null || r.finalText.isBlank()) return;
        String userId = (r.userId == null || r.userId.isBlank()) ? DEFAULT_USER : r.userId;
        String ticker = r.ticker == null ? "?" : r.ticker.toUpperCase();
        // Skip persisting "partial" / orchestrator-bypassed runs. A real deterministic
        // pipeline produces 100KB+ of master report; anything under 20KB is almost certainly
        // a shortcut where the outer LLM answered directly instead of calling the
        // skill_stock_analysis tool. Saving those would clobber a previously-saved good
        // report for the same ticker, and they're not useful to surface in the library.
        // Threshold is intentionally conservative — even tiny tickers (illiquid Indian
        // names, etc.) yield 30KB+ when the full pipeline runs.
        int MIN_REAL_REPORT_BYTES = 20_000;
        if (r.finalText.length() < MIN_REAL_REPORT_BYTES) {
            LOG.info("skip persisting partial report for chat={} ticker={} ({} chars < {} threshold)",
                    r.chatId, ticker, r.finalText.length(), MIN_REAL_REPORT_BYTES);
            return;
        }
        // Purge ALL older rows for this (user, ticker). The user wants a single library
        // card per ticker — multiple stale entries (e.g. 3 AAPL cards from successive
        // pipeline runs) is clutter. The 20KB threshold above already filters out the
        // partial / orchestrator-bypassed runs that originally motivated a size-comparison
        // guard here, so unconditional purge is safe: anything still reaching this DELETE
        // is a "real" pipeline output (60KB+ in practice, even for Indian stocks). If we
        // ever see a regression where a small-but-above-threshold dud overwrites a good
        // run, bump the threshold rather than re-introducing size-comparison here.
        String deleteSql = "DELETE FROM stock_reports WHERE user_id = ? AND ticker = ? AND chat_id <> ?";
        String insertSql = """
            INSERT INTO stock_reports
              (chat_id, user_id, ticker, created_at, final_text, prompt, channel, iters, tools, elapsed_ms, stop_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              ticker = excluded.ticker,
              created_at = excluded.created_at,
              final_text = excluded.final_text,
              prompt = excluded.prompt,
              channel = excluded.channel,
              iters = excluded.iters,
              tools = excluded.tools,
              elapsed_ms = excluded.elapsed_ms,
              stop_reason = excluded.stop_reason
            """;
        try (Connection c = open()) {
            c.setAutoCommit(false);
            int purged;
            try (PreparedStatement del = c.prepareStatement(deleteSql)) {
                del.setString(1, userId);
                del.setString(2, ticker);
                del.setString(3, r.chatId);
                purged = del.executeUpdate();
            }
            try (PreparedStatement ps = c.prepareStatement(insertSql)) {
                ps.setString(1, r.chatId);
                ps.setString(2, userId);
                ps.setString(3, ticker);
                ps.setLong  (4, r.createdAt > 0 ? r.createdAt : Instant.now().toEpochMilli());
                ps.setString(5, r.finalText);
                ps.setString(6, r.prompt);
                ps.setString(7, r.channel);
                ps.setInt   (8, r.iters);
                ps.setInt   (9, r.tools);
                ps.setLong  (10, r.elapsedMs);
                ps.setString(11, r.stopReason);
                ps.executeUpdate();
            }
            c.commit();
            if (purged > 0) {
                LOG.info("saved stock report chat={} ticker={} ({} chars), purged {} older runs for same ticker",
                        r.chatId, ticker, r.finalText.length(), purged);
            } else {
                LOG.info("saved stock report chat={} ticker={} ({} chars)", r.chatId, ticker, r.finalText.length());
            }
        } catch (Exception e) {
            LOG.warn("save failed for {}: {}", r.chatId, e.toString());
        }
    }

    /** List most-recent reports for a user. */
    public List<Report> recent(String userId, int limit) {
        String sql = """
            SELECT chat_id, user_id, ticker, created_at, final_text, prompt, channel, iters, tools, elapsed_ms, stop_reason
            FROM stock_reports
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """;
        return query(sql, userId, limit);
    }

    /** Get one by chatId. */
    public Optional<Report> get(String chatId) {
        String sql = """
            SELECT chat_id, user_id, ticker, created_at, final_text, prompt, channel, iters, tools, elapsed_ms, stop_reason
            FROM stock_reports WHERE chat_id = ?
            """;
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, chatId);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) return Optional.of(read(rs));
            }
        } catch (Exception e) {
            LOG.warn("get failed for {}: {}", chatId, e.toString());
        }
        return Optional.empty();
    }

    public boolean delete(String chatId) {
        try (Connection c = open();
             PreparedStatement ps = c.prepareStatement("DELETE FROM stock_reports WHERE chat_id = ?")) {
            ps.setString(1, chatId);
            return ps.executeUpdate() > 0;
        } catch (Exception e) {
            LOG.warn("delete failed for {}: {}", chatId, e.toString());
            return false;
        }
    }

    private List<Report> query(String sql, String userId, int limit) {
        List<Report> out = new ArrayList<>();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, (userId == null || userId.isBlank()) ? DEFAULT_USER : userId);
            ps.setInt(2, Math.max(1, Math.min(limit, 500)));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(read(rs));
            }
        } catch (Exception e) {
            LOG.warn("query failed: {}", e.toString());
        }
        return out;
    }

    private static Report read(ResultSet rs) throws Exception {
        return new Report(
                rs.getString("chat_id"),
                rs.getString("user_id"),
                rs.getString("ticker"),
                rs.getLong("created_at"),
                rs.getString("final_text"),
                rs.getString("prompt"),
                rs.getString("channel"),
                rs.getInt("iters"),
                rs.getInt("tools"),
                rs.getLong("elapsed_ms"),
                rs.getString("stop_reason")
        );
    }

    /** Heuristic: pull the ticker out of a stock-style chatId like "stock-amzn-1234" or "stock-amzn-validate-1234". */
    public static String tickerFromChatId(String chatId) {
        if (chatId == null) return "?";
        if (!chatId.startsWith("stock-")) return "?";
        String rest = chatId.substring("stock-".length()); // "amzn-1234" or "amzn-validate-1234"
        int dash = rest.indexOf('-');
        return (dash > 0) ? rest.substring(0, dash).toUpperCase() : rest.toUpperCase();
    }
}
