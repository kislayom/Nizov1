package ai.nizo.memory.graph;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.util.Fts;
import ai.nizo.memory.util.Json;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.time.Instant;
import java.util.*;

/**
 * Persistent knowledge-graph store backed by SQLite.
 *
 * <p>Stores {@link Node}s and {@link Edge}s in the same database as
 * {@code SqliteMemoryStore} (or a separate one &mdash; caller decides the path).
 * Follows the same JDBC patterns: WAL mode, aggressive cache, FTS5 for labels,
 * userId-scoped queries for tenant isolation.
 */
public final class SqliteGraphStore implements AutoCloseable {

    private final Connection conn;

    public SqliteGraphStore(Path dbPath) {
        try {
            if (dbPath.getParent() != null) Files.createDirectories(dbPath.getParent());
            conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath.toAbsolutePath());
            try (Statement st = conn.createStatement()) {
                // Same PRAGMAs as SqliteMemoryStore
                st.execute("PRAGMA journal_mode=WAL");
                st.execute("PRAGMA synchronous=NORMAL");
                st.execute("PRAGMA cache_size=-524288");    // ~512 MB
                st.execute("PRAGMA mmap_size=17179869184"); // 16 GB mmap

                // ---- knowledge_nodes ----
                st.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_nodes (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        label TEXT NOT NULL,
                        properties TEXT,
                        confidence REAL NOT NULL DEFAULT 0.8,
                        source TEXT NOT NULL DEFAULT 'extracted',
                        privacy_level TEXT NOT NULL DEFAULT 'public',
                        mention_count INTEGER NOT NULL DEFAULT 1,
                        first_seen_at INTEGER NOT NULL,
                        last_confirmed_at INTEGER NOT NULL
                    )
                    """);
                st.execute("CREATE INDEX IF NOT EXISTS idx_kn_user ON knowledge_nodes(user_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_kn_user_cat ON knowledge_nodes(user_id, category)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_kn_label ON knowledge_nodes(user_id, LOWER(label))");

                // FTS5 on labels
                st.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_nodes_fts
                    USING fts5(label, content='knowledge_nodes', content_rowid='rowid')
                    """);
                st.execute("""
                    CREATE TRIGGER IF NOT EXISTS kn_fts_ai AFTER INSERT ON knowledge_nodes BEGIN
                        INSERT INTO knowledge_nodes_fts(rowid, label) VALUES (new.rowid, new.label);
                    END
                    """);
                st.execute("""
                    CREATE TRIGGER IF NOT EXISTS kn_fts_ad AFTER DELETE ON knowledge_nodes BEGIN
                        INSERT INTO knowledge_nodes_fts(knowledge_nodes_fts, rowid, label)
                            VALUES('delete', old.rowid, old.label);
                    END
                    """);

                // ---- knowledge_edges ----
                st.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_edges (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        source_node_id TEXT NOT NULL,
                        target_node_id TEXT NOT NULL,
                        relationship TEXT NOT NULL,
                        properties TEXT,
                        valid_from INTEGER,
                        valid_to INTEGER,
                        confidence REAL NOT NULL DEFAULT 0.8,
                        source TEXT NOT NULL DEFAULT 'extracted',
                        invalidated_at INTEGER,
                        created_at INTEGER NOT NULL
                    )
                    """);
                st.execute("CREATE INDEX IF NOT EXISTS idx_ke_user ON knowledge_edges(user_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_edges(source_node_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_ke_target ON knowledge_edges(target_node_id)");
                st.execute("CREATE INDEX IF NOT EXISTS idx_ke_rel ON knowledge_edges(user_id, source_node_id, relationship)");
            }
        } catch (Exception e) {
            throw new IllegalStateException("Cannot open graph store: " + dbPath, e);
        }
    }

    // ===================== Node operations =====================

    /** Insert or replace a node. */
    public void upsertNode(Node node) {
        String sql = """
            INSERT OR REPLACE INTO knowledge_nodes
                (id, user_id, category, label, properties, confidence, source,
                 privacy_level, mention_count, first_seen_at, last_confirmed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, node.id());
            ps.setString(2, node.userId());
            ps.setString(3, node.category());
            ps.setString(4, node.label());
            ps.setString(5, node.properties() == null ? null : Json.stringify(node.properties()));
            ps.setDouble(6, node.confidence());
            ps.setString(7, node.source());
            ps.setString(8, node.privacyLevel());
            ps.setInt(9, node.mentionCount());
            ps.setLong(10, node.firstSeenAt().toEpochMilli());
            ps.setLong(11, node.lastConfirmedAt().toEpochMilli());
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    public Optional<Node> findNodeById(String id) {
        try (PreparedStatement ps = conn.prepareStatement(
                "SELECT * FROM knowledge_nodes WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? Optional.of(mapNode(rs)) : Optional.empty();
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Case-insensitive exact match on label within a category. */
    public Optional<Node> findByUserCategoryLabel(String userId, String category, String label) {
        String sql = "SELECT * FROM knowledge_nodes WHERE user_id = ? AND category = ? AND LOWER(label) = LOWER(?)";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, category);
            ps.setString(3, label);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? Optional.of(mapNode(rs)) : Optional.empty();
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Substring match on label (LIKE %label%). */
    public List<Node> findByUserAndLabelContaining(String userId, String label) {
        String sql = "SELECT * FROM knowledge_nodes WHERE user_id = ? AND label LIKE ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, "%" + label + "%");
            return collectNodes(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Full-text search on node labels, scoped to a user. */
    public List<Node> ftsSearchNodes(String userId, String query, int limit) {
        String sql = """
            SELECT n.* FROM knowledge_nodes_fts f
            JOIN knowledge_nodes n ON n.rowid = f.rowid
            WHERE knowledge_nodes_fts MATCH ? AND n.user_id = ?
            ORDER BY rank LIMIT ?
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, Fts.sanitiseMatch(query));
            ps.setString(2, userId);
            ps.setInt(3, limit);
            return collectNodes(ps);
        } catch (SQLException e) {
            // FTS syntax errors on weird queries -- fall back to empty
            return List.of();
        }
    }

    /** Find person nodes by name (case-insensitive). */
    public List<Node> findPersonsByName(String userId, String label) {
        String sql = """
            SELECT * FROM knowledge_nodes
            WHERE user_id = ? AND category = 'person' AND LOWER(label) = LOWER(?)
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, label);
            return collectNodes(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All nodes in a category, excluding deleted/redacted privacy levels. */
    public List<Node> findByCategory(String userId, String category) {
        String sql = """
            SELECT * FROM knowledge_nodes
            WHERE user_id = ? AND category = ?
              AND privacy_level NOT IN ('deleted', 'redacted')
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, category);
            return collectNodes(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All accessible nodes for a user (non-deleted, non-redacted). */
    public List<Node> findAllAccessible(String userId) {
        String sql = """
            SELECT * FROM knowledge_nodes
            WHERE user_id = ? AND privacy_level NOT IN ('deleted', 'redacted')
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            return collectNodes(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Hard delete a node by id. */
    public void deleteNode(String id) {
        try (PreparedStatement ps = conn.prepareStatement(
                "DELETE FROM knowledge_nodes WHERE id = ?")) {
            ps.setString(1, id);
            ps.executeUpdate();
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /**
     * Hard-delete every node and every edge belonging to {@code userId}.
     * Used by GDPR-grade {@code forgetUser}. Returns the number of rows
     * deleted (nodes + edges summed).
     *
     * <p>Runs both deletes in a single transaction so a partial purge can't
     * leave dangling edges referencing deleted nodes.
     */
    public int deleteAllForUser(String userId) {
        int nodes, edges;
        try {
            conn.setAutoCommit(false);
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM knowledge_edges WHERE user_id = ?")) {
                ps.setString(1, userId);
                edges = ps.executeUpdate();
            }
            try (PreparedStatement ps = conn.prepareStatement(
                    "DELETE FROM knowledge_nodes WHERE user_id = ?")) {
                ps.setString(1, userId);
                nodes = ps.executeUpdate();
            }
            conn.commit();
        } catch (SQLException e) {
            try { conn.rollback(); } catch (SQLException ignore) {}
            throw new IllegalStateException(e);
        } finally {
            try { conn.setAutoCommit(true); } catch (SQLException ignore) {}
        }
        return nodes + edges;
    }

    /** Count nodes grouped by category for a user. */
    public Map<String, Long> countByCategory(String userId) {
        String sql = "SELECT category, COUNT(*) c FROM knowledge_nodes WHERE user_id = ? GROUP BY category";
        Map<String, Long> out = new LinkedHashMap<>();
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.put(rs.getString("category"), rs.getLong("c"));
                }
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
        return out;
    }

    // ===================== Edge operations =====================

    /** Insert a new edge. */
    public void insertEdge(Edge edge) {
        String sql = """
            INSERT INTO knowledge_edges
                (id, user_id, source_node_id, target_node_id, relationship,
                 properties, valid_from, valid_to, confidence, source,
                 invalidated_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, edge.id());
            ps.setString(2, edge.userId());
            ps.setString(3, edge.sourceNodeId());
            ps.setString(4, edge.targetNodeId());
            ps.setString(5, edge.relationship());
            ps.setString(6, edge.properties() == null ? null : Json.stringify(edge.properties()));
            setNullableLong(ps, 7, edge.validFrom());
            setNullableLong(ps, 8, edge.validTo());
            ps.setDouble(9, edge.confidence());
            ps.setString(10, edge.source());
            setNullableLong(ps, 11, edge.invalidatedAt());
            ps.setLong(12, Instant.now().toEpochMilli());
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    public Optional<Edge> findEdgeById(String id) {
        try (PreparedStatement ps = conn.prepareStatement(
                "SELECT * FROM knowledge_edges WHERE id = ?")) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? Optional.of(mapEdge(rs)) : Optional.empty();
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Current (non-invalidated) edges touching a node in either direction. */
    public List<Edge> findCurrentEdgesForNode(String nodeId) {
        String sql = """
            SELECT * FROM knowledge_edges
            WHERE (source_node_id = ? OR target_node_id = ?)
              AND invalidated_at IS NULL
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, nodeId);
            ps.setString(2, nodeId);
            return collectEdges(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Current edges originating from a node. */
    public List<Edge> findCurrentEdgesFromNode(String nodeId) {
        String sql = """
            SELECT * FROM knowledge_edges
            WHERE source_node_id = ? AND invalidated_at IS NULL
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, nodeId);
            return collectEdges(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Find current edges from a source with the same relationship (conflict detection). */
    public List<Edge> findConflictingEdges(String userId, String sourceNodeId, String relationship) {
        String sql = """
            SELECT * FROM knowledge_edges
            WHERE user_id = ? AND source_node_id = ? AND relationship = ?
              AND invalidated_at IS NULL
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, sourceNodeId);
            ps.setString(3, relationship);
            return collectEdges(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All current edges between two nodes (bidirectional). */
    public List<Edge> findEdgesBetween(String userId, String nodeA, String nodeB) {
        String sql = """
            SELECT * FROM knowledge_edges
            WHERE user_id = ?
              AND ((source_node_id = ? AND target_node_id = ?)
                OR (source_node_id = ? AND target_node_id = ?))
              AND invalidated_at IS NULL
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            ps.setString(2, nodeA);
            ps.setString(3, nodeB);
            ps.setString(4, nodeB);
            ps.setString(5, nodeA);
            return collectEdges(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** All current (non-invalidated) edges for a user. */
    public List<Edge> findAllCurrentEdges(String userId) {
        String sql = "SELECT * FROM knowledge_edges WHERE user_id = ? AND invalidated_at IS NULL";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            return collectEdges(ps);
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    /** Update an edge (primarily for invalidation). Does not modify created_at. */
    public void updateEdge(Edge edge) {
        String sql = """
            UPDATE knowledge_edges SET
                user_id = ?, source_node_id = ?, target_node_id = ?, relationship = ?,
                properties = ?, valid_from = ?, valid_to = ?, confidence = ?,
                source = ?, invalidated_at = ?
            WHERE id = ?
            """;
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, edge.userId());
            ps.setString(2, edge.sourceNodeId());
            ps.setString(3, edge.targetNodeId());
            ps.setString(4, edge.relationship());
            ps.setString(5, edge.properties() == null ? null : Json.stringify(edge.properties()));
            setNullableLong(ps, 6, edge.validFrom());
            setNullableLong(ps, 7, edge.validTo());
            ps.setDouble(8, edge.confidence());
            ps.setString(9, edge.source());
            setNullableLong(ps, 10, edge.invalidatedAt());
            ps.setString(11, edge.id());
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Total node count for a user. */
    public long countNodes(String userId) {
        return countTable(userId, "knowledge_nodes");
    }

    /** Total edge count for a user. */
    public long countEdges(String userId) {
        return countTable(userId, "knowledge_edges");
    }

    /** Cascade-delete all edges referencing a node (either direction). */
    public void deleteEdgesForNode(String nodeId) {
        String sql = "DELETE FROM knowledge_edges WHERE source_node_id = ? OR target_node_id = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, nodeId);
            ps.setString(2, nodeId);
            ps.executeUpdate();
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    @Override
    public void close() {
        try { conn.close(); } catch (SQLException ignored) {}
    }

    // ===================== helpers =====================

    private long countTable(String userId, String table) {
        String sql = "SELECT COUNT(*) FROM " + table + " WHERE user_id = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, userId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getLong(1) : 0L;
            }
        } catch (SQLException e) { throw new IllegalStateException(e); }
    }

    private static void setNullableLong(PreparedStatement ps, int index, Instant instant) throws SQLException {
        if (instant == null) {
            ps.setNull(index, Types.INTEGER);
        } else {
            ps.setLong(index, instant.toEpochMilli());
        }
    }

    private static Instant nullableInstant(ResultSet rs, String column) throws SQLException {
        long val = rs.getLong(column);
        return rs.wasNull() ? null : Instant.ofEpochMilli(val);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseProperties(String json) {
        if (json == null || json.isEmpty()) return null;
        return Json.parseMap(json);
    }

    private static List<Node> collectNodes(PreparedStatement ps) throws SQLException {
        try (ResultSet rs = ps.executeQuery()) {
            List<Node> out = new ArrayList<>();
            while (rs.next()) out.add(mapNode(rs));
            return out;
        }
    }

    private static List<Edge> collectEdges(PreparedStatement ps) throws SQLException {
        try (ResultSet rs = ps.executeQuery()) {
            List<Edge> out = new ArrayList<>();
            while (rs.next()) out.add(mapEdge(rs));
            return out;
        }
    }

    private static Node mapNode(ResultSet rs) throws SQLException {
        return new Node(
                rs.getString("id"),
                rs.getString("user_id"),
                rs.getString("category"),
                rs.getString("label"),
                parseProperties(rs.getString("properties")),
                rs.getDouble("confidence"),
                rs.getString("source"),
                rs.getString("privacy_level"),
                rs.getInt("mention_count"),
                Instant.ofEpochMilli(rs.getLong("first_seen_at")),
                Instant.ofEpochMilli(rs.getLong("last_confirmed_at"))
        );
    }

    private static Edge mapEdge(ResultSet rs) throws SQLException {
        return new Edge(
                rs.getString("id"),
                rs.getString("user_id"),
                rs.getString("source_node_id"),
                rs.getString("target_node_id"),
                rs.getString("relationship"),
                parseProperties(rs.getString("properties")),
                nullableInstant(rs, "valid_from"),
                nullableInstant(rs, "valid_to"),
                rs.getDouble("confidence"),
                rs.getString("source"),
                nullableInstant(rs, "invalidated_at")
        );
    }
}
