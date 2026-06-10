package ai.nizo.memory.store;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryTags;
import ai.nizo.memory.util.Fts;
import ai.nizo.memory.util.Tags;
import ai.nizo.memory.util.Vectors;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.time.Instant;
import java.util.*;

/**
 * Persistent memory store backed by SQLite.
 *
 * <p>Design notes for a 192 GB RAM host:
 * <ul>
 *   <li>SQLite page cache is sized aggressively via {@code PRAGMA cache_size}
 *       so hot memory items stay in RAM.</li>
 *   <li>Embeddings live in a BLOB column; {@link #all()} is only used by the
 *       in-memory vector index which pages lazily.</li>
 *   <li>WAL mode keeps readers non-blocking during consolidation.</li>
 *   <li>Every row carries a {@code user_id} column; all queries are scoped to a
 *       single userId to enforce tenant isolation.</li>
 * </ul>
 */
public final class SqliteMemoryStore implements AutoCloseable {

    private final Connection conn;

    public SqliteMemoryStore(Path dbPath) {
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
            try (Statement st = conn.createStatement()) {
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
                st.execute("PRAGMA cache_size=-524288");   // ~512 MB
                st.execute("PRAGMA mmap_size=17179869184"); // 16 GB mmap
                st.execute("""
                    CREATE TABLE IF NOT EXISTS memory_items (
                      id TEXT PRIMARY KEY,
                      user_id TEXT NOT NULL DEFAULT 'default',
                      session_id TEXT,
                      tier TEXT NOT NULL,
                      content TEXT NOT NULL,
                      embedding BLOB,
                      tags TEXT,
                      source TEXT,
                      confidence REAL NOT NULL DEFAULT 1.0,
                      created_at INTEGER NOT NULL,
                      last_accessed INTEGER NOT NULL,
                      access_count INTEGER NOT NULL DEFAULT 0,
                      tokens INTEGER NOT NULL DEFAULT 0
                    )
                    """);
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_tier ON memory_items(tier)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_created ON memory_items(created_at)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memory_items(user_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_tier ON memory_items(user_id, tier)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_session ON memory_items(user_id, session_id)");
                st.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(content, content='memory_items', content_rowid='rowid')
                    """);
                st.execute("""
                    CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_items BEGIN
                      INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
                    END
                    """);
                st.execute("""
                    CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_items BEGIN
                      INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    END
                    """);
                // Migration: add user_id column to existing databases that lack it.
                migrateIfNeeded(st);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Cannot open memory store: " + dbPath, e);
        }
    }

    /**
     * Idempotent column migrations for databases created by older schemas.
     * Adds {@code user_id} (pre-Phase-2) and {@code session_id} (pre-session-filter)
     * if missing. No-op on fresh databases where CREATE TABLE already declared them.
     */
    private void migrateIfNeeded(Statement st) {
        Set<String> existingColumns = new HashSet<>();
        try (ResultSet rs = st.executeQuery("PRAGMA table_info(memory_items)")) {
            while (rs.next()) existingColumns.add(rs.getString("name"));
        } catch (SQLException ignored) {
            return;  // Table doesn't exist yet — CREATE TABLE path will cover it.
        }
        if (!existingColumns.contains("user_id")) {
            try {
                st.execute("ALTER TABLE memory_items ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memory_items(user_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_tier ON memory_items(user_id, tier)");
            } catch (SQLException ignored) {}
        }
        if (!existingColumns.contains("session_id")) {
            try {
                st.execute("ALTER TABLE memory_items ADD COLUMN session_id TEXT");
                st.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_session ON memory_items(user_id, session_id)");
            } catch (SQLException ignored) {}
        }
    }

    public void upsert(MemoryItem item) {
        // Lift session_id from tags into the dedicated column so that
        // session-filtered recall hits the (user_id, session_id) index rather
        // than a LIKE-scan on the serialised tags blob. The tag is kept in the
        // tags payload as well (redundant but keeps the domain object honest).
        String sessionId = item.tags() == null ? null : item.tags().get(MemoryTags.SESSION_ID);
        String sql = """
            INSERT INTO memory_items(id, user_id, session_id, tier, content, embedding, tags, source,
                                     confidence, created_at, last_accessed, access_count, tokens)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              user_id=excluded.user_id,
              session_id=excluded.session_id,
              tier=excluded.tier,
              content=excluded.content,
              embedding=excluded.embedding,
              tags=excluded.tags,
              source=excluded.source,
              confidence=excluded.confidence,
              last_accessed=excluded.last_accessed,
              access_count=access_count+1,
              tokens=excluded.tokens
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, item.id());
            ps.setString(2, item.userId() == null ? "default" : item.userId());
            if (sessionId == null || sessionId.isBlank()) ps.setNull(3, Types.VARCHAR);
            else ps.setString(3, sessionId);
            ps.setString(4, item.tier().name());
            ps.setString(5, item.content());
            ps.setBytes(6, item.embedding() == null ? null : Vectors.toBytes(item.embedding()));
            ps.setString(7, Tags.encode(item.tags()));
            ps.setString(8, item.source());
            ps.setDouble(9, item.confidence());
            ps.setLong(10, item.createdAt().toEpochMilli());
            ps.setLong(11, item.lastAccessedAt().toEpochMilli());
            ps.setInt(12, item.accessCount());
            ps.setInt(13, item.tokens());
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    public Optional<MemoryItem> findById(String id) {
        try (PreparedStatement ps = conn.prepareStatement(
                "SELECT * FROM memory_items WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? Optional.of(map(rs)) : Optional.empty();
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /**
     * Full-text BM25 prefilter — scoped to userId.
     *
     * <p><b>Why we override BM25 length normalization</b>
     * (LongMemEval finding): the default FTS5 BM25 uses {@code b=0.75},
     * which strongly favours short documents. In conversational memory,
     * <em>user questions</em> are short (~15 words) while <em>assistant
     * responses</em> are long (200-1000 words). The default scoring
     * therefore returns the user's prompt and misses the assistant's actual
     * answer — exactly the failure mode behind 4 of our 9 LongMemEval recall
     * misses (single-session-assistant questions).
     *
     * <p>Setting {@code b=0.25} keeps mild length penalty (very long noisy
     * docs still get downweighted) but stops actively burying assistant
     * answers under user questions. This is calibration of an existing knob,
     * not a hack — every IR system has to pick a {@code b} for its corpus.
     */
    public List<MemoryItem> ftsSearch(String userId, String query, int limit) {
        return ftsSearch(userId, query, limit, null);
    }

    /**
     * Session-filtered FTS variant. When {@code sessionIds} is non-null the
     * search is restricted to memory items tagged with one of the given
     * session identifiers (hits the {@code idx_mem_user_session} index).
     * A {@code null} filter or an empty set keeps backwards-compatible
     * whole-user behaviour.
     *
     * <p>See {@link #ftsSearch(String, String, int)} for the BM25 parameter
     * rationale.
     */
    public List<MemoryItem> ftsSearch(String userId, String query, int limit,
                                       Set<String> sessionIds) {
        boolean hasFilter = sessionIds != null && !sessionIds.isEmpty();
        StringBuilder sql = new StringBuilder("""
            SELECT m.* FROM memory_fts f
            JOIN memory_items m ON m.rowid = f.rowid
            WHERE memory_fts MATCH ? AND m.user_id = ?
            """);
        if (hasFilter) {
            sql.append(" AND m.session_id IN (");
            for (int i = 0; i < sessionIds.size(); i++) sql.append(i == 0 ? "?" : ",?");
            sql.append(")");
        }
        sql.append(" ORDER BY bm25(memory_fts, 1.2, 0.25) LIMIT ?");
        try (PreparedStatement ps = conn.prepareStatement(sql.toString())) {
            int pi = 1;
            ps.setString(pi++, Fts.sanitiseMatch(query));
            ps.setString(pi++, userId == null ? "default" : userId);
            if (hasFilter) for (String s : sessionIds) ps.setString(pi++, s);
            ps.setInt(pi, limit);
            return collect(ps);
        } catch (SQLException e) {
            return List.of();
        }
    }

    /**
     * Return the set of memory-item ids that belong to any of the given
     * session IDs for this user. Used to pre-filter a vector-index topK so
     * cosine similarity is computed against candidate items only.
     */
    public Set<String> idsForSessions(String userId, Set<String> sessionIds) {
        if (sessionIds == null || sessionIds.isEmpty()) return Set.of();
        StringBuilder sql = new StringBuilder(
                "SELECT id FROM memory_items WHERE user_id = ? AND session_id IN (");
        for (int i = 0; i < sessionIds.size(); i++) sql.append(i == 0 ? "?" : ",?");
        sql.append(")");
        try (PreparedStatement ps = conn.prepareStatement(sql.toString())) {
            int pi = 1;
            ps.setString(pi++, userId == null ? "default" : userId);
            for (String s : sessionIds) ps.setString(pi++, s);
            Set<String> out = new LinkedHashSet<>();
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) out.add(rs.getString(1));
            }
            return out;
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Compact per-session summary for a user. One row per distinct {@code session_id}
     * that carries at least one stored item. The {@code preview} is the content
     * of the earliest item in the session, truncated to {@code previewChars} —
     * serves as the manifest entry fed to the {@link ai.nizo.memory.session.SessionPicker}
     * for LLM-driven session selection.
     *
     * <p>Sessions with a {@code null} / blank {@code session_id} are grouped under
     * the synthetic id {@code "__unsessioned__"} so existing data is not hidden
     * from callers that iterate the manifest.
     */
    public List<SessionInfo> sessionManifest(String userId, int previewChars) {
        // MIN/MAX created_at + COUNT give a cheap session summary; separate
        // subquery pulls the earliest item's content for the preview.
        String sql = """
            SELECT COALESCE(session_id, '__unsessioned__') AS sess,
                   MIN(created_at) AS started,
                   MAX(created_at) AS ended,
                   COUNT(*) AS n,
                   (SELECT m2.content FROM memory_items m2
                     WHERE m2.user_id = m.user_id
                       AND COALESCE(m2.session_id,'__unsessioned__') = COALESCE(m.session_id,'__unsessioned__')
                     ORDER BY m2.created_at ASC LIMIT 1) AS first_content
              FROM memory_items m
             WHERE m.user_id = ?
             GROUP BY sess
             ORDER BY started
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            List<SessionInfo> out = new ArrayList<>();
            int previewCap = Math.max(20, previewChars);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String preview = rs.getString("first_content");
                    if (preview == null) preview = "";
                    preview = preview.replace('\n', ' ');
                    if (preview.length() > previewCap) preview = preview.substring(0, previewCap) + "…";
                    out.add(new SessionInfo(
                            rs.getString("sess"),
                            Instant.ofEpochMilli(rs.getLong("started")),
                            Instant.ofEpochMilli(rs.getLong("ended")),
                            rs.getInt("n"),
                            preview));
                }
            }
            return out;
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Count of distinct sessions for a user (cheap gate for the picker). */
    public int distinctSessionCount(String userId) {
        String sql = """
            SELECT COUNT(DISTINCT COALESCE(session_id, '__unsessioned__')) AS n
              FROM memory_items WHERE user_id = ?
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getInt(1) : 0;
            }
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Compact per-session summary row. */
    public record SessionInfo(String sessionId, Instant startedAt, Instant endedAt,
                              int itemCount, String preview) {}

    public List<MemoryItem> recent(String userId, MemoryItem.Tier tier, int limit) {
        String sql = "SELECT * FROM memory_items WHERE user_id = ? AND tier = ? ORDER BY created_at DESC LIMIT ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            ps.setString(2, tier.name());
            ps.setInt(3, limit);
            return collect(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All items across all users — used only for vector index hydration at startup. */
    public List<MemoryItem> all() {
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT * FROM memory_items")) {
            List<MemoryItem> out = new ArrayList<>();
            while (rs.next()) out.add(map(rs));
            return out;
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Distinct userIds present in the store — used by the reflection worker
     *  to know which users have memory to distil. */
    public java.util.Set<String> distinctUserIds() {
        java.util.Set<String> out = new java.util.LinkedHashSet<>();
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT DISTINCT user_id FROM memory_items")) {
            while (rs.next()) out.add(rs.getString(1));
        } catch (SQLException e) { throw new IllegalStateException(e); }
        return out;
    }

    /** Episodes for {@code userId} older than {@code olderThanMillisAgo}ms,
     *  newest first, up to {@code limit}. Used by the reflection worker. */
    public List<MemoryItem> olderEpisodes(String userId, long olderThanMillisAgo, int limit) {
        long cutoff = System.currentTimeMillis() - olderThanMillisAgo;
        String sql = """
            SELECT * FROM memory_items
             WHERE user_id = ? AND tier = 'EPISODIC' AND created_at <= ?
             ORDER BY created_at DESC LIMIT ?
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            ps.setLong(2, cutoff);
            ps.setInt(3, limit);
            return collect(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /**
     * Scan for items whose serialised tags contain all required key=value pairs.
     * This is the retrieval path for tag-based queries like "give me all actions"
     * where the query text may not overlap with the item content. Scoped to userId.
     */
    public List<MemoryItem> findByTags(String userId, Map<String, String> requiredTags, int limit) {
        if (requiredTags == null || requiredTags.isEmpty()) return List.of();
        // Build SQL LIKE conditions for each tag — tags stored as "k1=v1;k2=v2"
        StringBuilder sql = new StringBuilder("SELECT * FROM memory_items WHERE user_id = ?");
        List<String> params = new ArrayList<>();
        params.add(userId == null ? "default" : userId);
        for (Map.Entry<String, String> e : requiredTags.entrySet()) {
            sql.append(" AND tags LIKE ?");
            params.add("%" + e.getKey() + "=" + e.getValue() + "%");
        }
        sql.append(" ORDER BY created_at DESC LIMIT ?");
        try (PreparedStatement ps = conn.prepareStatement(sql.toString())) {
            int i = 1;
            for (String p : params) ps.setString(i++, p);
            ps.setInt(i, limit);
            return collect(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    public void delete(String id) {
        try (PreparedStatement ps = conn.prepareStatement("DELETE FROM memory_items WHERE id = ?")) {
            ps.setString(1, id);
            ps.executeUpdate();
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All items for one user, newest first, capped at {@code limit}.
     *  Backs the customer-facing "show me everything you remember about me" API. */
    public List<MemoryItem> findByUserId(String userId, int limit) {
        String sql = "SELECT * FROM memory_items WHERE user_id = ? ORDER BY created_at DESC LIMIT ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            ps.setInt(2, limit);
            return collect(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Items whose content contains {@code substring} (case-insensitive),
     *  scoped to userId. Backs the customer-facing "forget about Mike" API. */
    public List<MemoryItem> findByContentLike(String userId, String substring, int limit) {
        if (substring == null || substring.isBlank()) return List.of();
        String sql = "SELECT * FROM memory_items WHERE user_id = ? AND lower(content) LIKE ? LIMIT ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            ps.setString(2, "%" + substring.toLowerCase() + "%");
            ps.setInt(3, limit);
            return collect(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Update only the tags column on an existing item — used by pin/unpin,
     *  reconfirm, sensitivity changes, etc. without rebuilding the whole row. */
    public boolean updateTags(String id, Map<String, String> newTags) {
        String sql = "UPDATE memory_items SET tags = ? WHERE id = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, Tags.encode(newTags == null ? Map.of() : newTags));
            ps.setString(2, id);
            return ps.executeUpdate() > 0;
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Delete all items for a given userId — GDPR forgetUser. */
    public int deleteAllForUser(String userId) {
        try (PreparedStatement ps = conn.prepareStatement("DELETE FROM memory_items WHERE user_id = ?")) {
            ps.setString(1, userId);
            return ps.executeUpdate();
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Per-tier counts scoped to a single user. */
    public Map<MemoryItem.Tier, Long> countsByTier(String userId) {
        Map<MemoryItem.Tier, Long> out = new EnumMap<>(MemoryItem.Tier.class);
        for (MemoryItem.Tier t : MemoryItem.Tier.values()) out.put(t, 0L);
        String sql = "SELECT tier, COUNT(*) c FROM memory_items WHERE user_id = ? GROUP BY tier";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId == null ? "default" : userId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.put(MemoryItem.Tier.valueOf(rs.getString("tier")), rs.getLong("c"));
                }
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
        return out;
    }

    /** Global counts across all users — for observability dashboards. */
    public Map<MemoryItem.Tier, Long> countsByTierGlobal() {
        Map<MemoryItem.Tier, Long> out = new EnumMap<>(MemoryItem.Tier.class);
        for (MemoryItem.Tier t : MemoryItem.Tier.values()) out.put(t, 0L);
        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery("SELECT tier, COUNT(*) c FROM memory_items GROUP BY tier")) {
            while (rs.next()) {
                out.put(MemoryItem.Tier.valueOf(rs.getString("tier")), rs.getLong("c"));
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
        return out;
    }

    /**
     * Demote the confidence of memory items matching a user and FTS query.
     * Used by the extraction pipeline to lower old contradicted facts so
     * newer corrections outrank them.
     *
     * @return number of items demoted
     */
    public int demoteConfidence(String userId, String query, double newConfidence) {
        // Find matching items via FTS, then update their confidence
        List<MemoryItem> matches = ftsSearch(userId, query, 10);
        int count = 0;
        String sql = "UPDATE memory_items SET confidence = ? WHERE id = ? AND confidence > ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            for (MemoryItem m : matches) {
                ps.setDouble(1, newConfidence);
                ps.setString(2, m.id());
                ps.setDouble(3, newConfidence);
                count += ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
        return count;
    }

    public void touch(String id) {
        String sql = "UPDATE memory_items SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, System.currentTimeMillis());
            ps.setString(2, id);
            ps.executeUpdate();
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    @Override
    public void close() {
        try { conn.close(); } catch (SQLException ignored) {}
    }

    // ---------------------- helpers ----------------------

    private static List<MemoryItem> collect(PreparedStatement ps) throws SQLException {
        try (ResultSet rs = ps.executeQuery()) {
            List<MemoryItem> out = new ArrayList<>();
            while (rs.next()) out.add(map(rs));
            return out;
        }
    }

    private static MemoryItem map(ResultSet rs) throws SQLException {
        byte[] emb = rs.getBytes("embedding");
        return new MemoryItem(
                rs.getString("id"),
                rs.getString("user_id"),
                MemoryItem.Tier.valueOf(rs.getString("tier")),
                rs.getString("content"),
                emb == null ? null : Vectors.fromBytes(emb),
                Tags.decode(rs.getString("tags")),
                rs.getString("source"),
                rs.getDouble("confidence"),
                Instant.ofEpochMilli(rs.getLong("created_at")),
                Instant.ofEpochMilli(rs.getLong("last_accessed")),
                rs.getInt("access_count"),
                rs.getInt("tokens")
        );
    }

}
