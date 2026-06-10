package ai.nizo.agent.session;

import ai.nizo.agent.store.SchemaMigrator;
import ai.nizo.agent.store.SqliteDataSource;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.Role;
import ai.nizo.api.llm.ToolCall;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * SQLite-backed {@link SessionStore}. One row per persisted message; deserialised back to
 * {@link ChatMessage} on read. Schema is created on first open.
 *
 * <p>Why a dedicated table (not nizo-memory's {@code memory_items}): conversation history is
 * a tight, append-only ring per chat. nizo-memory's recall pipeline (BM25 + vector + KG) is
 * overkill for "last N turns of this chat" and not optimised for it. Memory recall is layered
 * on top via tools (memory_recall) when the agent needs cross-session retrieval.
 */
public final class SqliteSessionStore implements SessionStore {

    private static final Logger LOG = LoggerFactory.getLogger(SqliteSessionStore.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final SqliteDataSource ds;

    public SqliteSessionStore(Path dbFile) {
        try {
            if (dbFile.getParent() != null) Files.createDirectories(dbFile.getParent());
        } catch (Exception e) {
            throw new RuntimeException("could not prepare session db dir", e);
        }
        this.ds = new SqliteDataSource(dbFile);
        applyPragmas();
        new SchemaMigrator(ds, "sessions").migrate(MIGRATIONS);
        LOG.info("Session store: {}", dbFile.toAbsolutePath());
    }

    /**
     * Migration ledger. v1 captures the schema as it shipped before {@link SchemaMigrator}
     * existed — note the {@code IF NOT EXISTS} clauses, which keep v1 idempotent against
     * databases that already have these objects from the pre-migrator codepath.
     *
     * <p>Subsequent versions should be ADD-ONLY (e.g. {@code ALTER TABLE ... ADD COLUMN}).
     * Renames and drops require a copy-table dance under SQLite — write them carefully if
     * the time comes.
     */
    private static final List<SchemaMigrator.Migration> MIGRATIONS = List.of(
            SchemaMigrator.of(1, "session_messages baseline", """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id      TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT,
                    tool_call_id TEXT,
                    name         TEXT,
                    tool_calls   TEXT,
                    images       TEXT,
                    created_at   INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_session_chat_created
                    ON session_messages (chat_id, id);
                """),
            // Per-user namespacing (June 2026): nullable owner tag per row. Legacy rows
            // stay NULL and are attributed to the owner identity "web-user" at query
            // time — no backfill needed.
            SchemaMigrator.of(2, "add user_id owner tag", """
                ALTER TABLE session_messages ADD COLUMN user_id TEXT;
                """)
    );

    /** Owner identity legacy rows (user_id NULL) belong to — Kislay's web identity. */
    private static final String LEGACY_OWNER = "web-user";

    private void applyPragmas() {
        try (Connection c = ds.getConnection(); Statement s = c.createStatement()) {
            s.executeUpdate("PRAGMA journal_mode=WAL");
            s.executeUpdate("PRAGMA synchronous=NORMAL");
        } catch (SQLException e) {
            throw new RuntimeException("could not set session-store PRAGMAs", e);
        }
    }

    private Connection open() throws SQLException {
        return ds.getConnection();
    }

    @Override
    public List<ChatMessage> recent(String chatId, int limit) {
        if (chatId == null || chatId.isBlank()) return List.of();
        if (limit <= 0) return List.of();

        List<ChatMessage> out = new ArrayList<>(limit);
        String sql = """
            SELECT role, content, tool_call_id, name, tool_calls, images
            FROM session_messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
        """;
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, chatId);
            ps.setInt(2, limit);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    out.add(decode(rs));
                }
            }
        } catch (Exception e) {
            LOG.warn("session recent failed for {}: {}", chatId, e.toString());
            return List.of();
        }
        Collections.reverse(out); // oldest-first for prompt assembly
        return out;
    }

    @Override
    public synchronized void append(String chatId, ChatMessage message) {
        append(chatId, message, null);
    }

    @Override
    public synchronized void append(String chatId, ChatMessage message, String userId) {
        if (chatId == null || chatId.isBlank() || message == null) return;
        String sql = """
            INSERT INTO session_messages (chat_id, role, content, tool_call_id, name, tool_calls, images, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """;
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            ps.setString(1, chatId);
            ps.setString(2, message.role().name());
            if (message.content() == null) ps.setNull(3, java.sql.Types.VARCHAR); else ps.setString(3, message.content());
            if (message.toolCallId() == null) ps.setNull(4, java.sql.Types.VARCHAR); else ps.setString(4, message.toolCallId());
            if (message.name() == null) ps.setNull(5, java.sql.Types.VARCHAR); else ps.setString(5, message.name());
            ps.setString(6, encodeToolCalls(message.toolCalls()));
            ps.setString(7, encodeImages(message.images()));
            ps.setLong(8, System.currentTimeMillis());
            if (userId == null || userId.isBlank()) ps.setNull(9, java.sql.Types.VARCHAR); else ps.setString(9, userId);
            ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("session append failed for {}: {}", chatId, e.toString());
        }
    }

    @Override
    public synchronized void replaceHistory(String chatId, List<ChatMessage> messages) {
        if (chatId == null || chatId.isBlank()) return;
        String del = "DELETE FROM session_messages WHERE chat_id = ?";
        String ins = """
            INSERT INTO session_messages (chat_id, role, content, tool_call_id, name, tool_calls, images, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """;
        try (Connection c = open()) {
            c.setAutoCommit(false);
            try {
                // Preserve ownership across the rebuild — condense must not orphan the
                // chat from its user's listing. MAX() ignores NULLs, so a chat with any
                // tagged row keeps its owner; an all-NULL (legacy) chat stays NULL.
                String owner = null;
                try (PreparedStatement ps = c.prepareStatement(
                        "SELECT MAX(user_id) FROM session_messages WHERE chat_id = ?")) {
                    ps.setString(1, chatId);
                    try (ResultSet rs = ps.executeQuery()) {
                        if (rs.next()) owner = rs.getString(1);
                    }
                }
                try (PreparedStatement ps = c.prepareStatement(del)) {
                    ps.setString(1, chatId);
                    ps.executeUpdate();
                }
                if (messages != null && !messages.isEmpty()) {
                    long now = System.currentTimeMillis();
                    try (PreparedStatement ps = c.prepareStatement(ins)) {
                        for (int i = 0; i < messages.size(); i++) {
                            ChatMessage m = messages.get(i);
                            if (m == null) continue;
                            ps.setString(1, chatId);
                            ps.setString(2, m.role().name());
                            if (m.content() == null) ps.setNull(3, java.sql.Types.VARCHAR); else ps.setString(3, m.content());
                            if (m.toolCallId() == null) ps.setNull(4, java.sql.Types.VARCHAR); else ps.setString(4, m.toolCallId());
                            if (m.name() == null) ps.setNull(5, java.sql.Types.VARCHAR); else ps.setString(5, m.name());
                            ps.setString(6, encodeToolCalls(m.toolCalls()));
                            ps.setString(7, encodeImages(m.images()));
                            // Preserve insertion order across the rebuild — bump created_at by index.
                            ps.setLong(8, now + i);
                            if (owner == null) ps.setNull(9, java.sql.Types.VARCHAR); else ps.setString(9, owner);
                            ps.addBatch();
                        }
                        ps.executeBatch();
                    }
                }
                c.commit();
            } catch (SQLException e) {
                c.rollback();
                throw e;
            } finally {
                c.setAutoCommit(true);
            }
        } catch (Exception e) {
            LOG.warn("session replaceHistory failed for {}: {}", chatId, e.toString());
        }
    }

    @Override
    public void clear(String chatId) {
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "DELETE FROM session_messages WHERE chat_id = ?")) {
            ps.setString(1, chatId);
            ps.executeUpdate();
        } catch (Exception e) {
            LOG.warn("session clear failed for {}: {}", chatId, e.toString());
        }
    }

    @Override
    public long size(String chatId) {
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(
                "SELECT COUNT(*) FROM session_messages WHERE chat_id = ?")) {
            ps.setString(1, chatId);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getLong(1) : 0L;
            }
        } catch (Exception e) {
            return 0L;
        }
    }

    @Override
    public List<ChatSummary> listChats(int limit) {
        return listChats(limit, null);
    }

    @Override
    public List<ChatSummary> listChats(int limit, String userId) {
        // Aggregate per chat: count, last timestamp, last user msg, last assistant reply.
        // Ownership filter: a chat belongs to MAX(user_id) (NULL-safe — MAX ignores NULLs,
        // so any tagged row claims the chat); all-NULL legacy chats belong to LEGACY_OWNER.
        // userId == null → no filter (admin/all view, the pre-namespacing behavior).
        String ownerFilter = (userId == null || userId.isBlank())
                ? ""
                : "HAVING COALESCE(MAX(user_id), '" + LEGACY_OWNER + "') = ?";
        String sql = """
            WITH agg AS (
              SELECT chat_id,
                     COUNT(*) AS msg_count,
                     MAX(created_at) AS last_ts
              FROM session_messages
              GROUP BY chat_id
              %s
            ),""".formatted(ownerFilter) + """
            last_user AS (
              SELECT chat_id, content
              FROM session_messages s
              WHERE role = 'USER'
                AND created_at = (SELECT MAX(created_at) FROM session_messages WHERE chat_id = s.chat_id AND role = 'USER')
            ),
            last_ai AS (
              SELECT chat_id, content
              FROM session_messages s
              WHERE role = 'ASSISTANT'
                AND created_at = (SELECT MAX(created_at) FROM session_messages WHERE chat_id = s.chat_id AND role = 'ASSISTANT')
            )
            SELECT agg.chat_id, agg.msg_count, agg.last_ts,
                   COALESCE(lu.content, '') AS last_user,
                   COALESCE(la.content, '') AS last_ai
            FROM agg
            LEFT JOIN last_user lu ON lu.chat_id = agg.chat_id
            LEFT JOIN last_ai   la ON la.chat_id = agg.chat_id
            ORDER BY agg.last_ts DESC
            LIMIT ?
        """;
        List<ChatSummary> out = new ArrayList<>();
        boolean filtered = !ownerFilter.isEmpty();
        try (Connection c = open(); PreparedStatement ps = c.prepareStatement(sql)) {
            int p = 1;
            if (filtered) ps.setString(p++, userId);
            ps.setInt(p, Math.max(1, limit));
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String lu = rs.getString("last_user");
                    String la = rs.getString("last_ai");
                    out.add(new ChatSummary(
                            rs.getString("chat_id"),
                            rs.getLong("last_ts"),
                            rs.getLong("msg_count"),
                            lu == null ? "" : (lu.length() > 80 ? lu.substring(0, 80) : lu),
                            la == null ? "" : (la.length() > 80 ? la.substring(0, 80) : la)
                    ));
                }
            }
        } catch (Exception e) {
            LOG.warn("listChats failed: {}", e.toString());
        }
        return out;
    }

    // -------- (de)serialise --------

    private ChatMessage decode(ResultSet rs) throws Exception {
        Role role = Role.valueOf(rs.getString("role"));
        String content = rs.getString("content");
        String toolCallId = rs.getString("tool_call_id");
        String name = rs.getString("name");
        List<ToolCall> calls = decodeToolCalls(rs.getString("tool_calls"));
        List<String> images = decodeImages(rs.getString("images"));
        return new ChatMessage(role, content, images, calls, toolCallId, name);
    }

    private String encodeToolCalls(List<ToolCall> calls) {
        if (calls == null || calls.isEmpty()) return null;
        ArrayNode arr = MAPPER.createArrayNode();
        for (ToolCall c : calls) {
            ObjectNode n = arr.addObject();
            n.put("id", c.id());
            n.put("name", c.name());
            n.put("argumentsJson", c.argumentsJson());
        }
        try { return MAPPER.writeValueAsString(arr); }
        catch (Exception e) { return null; }
    }

    private List<ToolCall> decodeToolCalls(String json) {
        if (json == null || json.isBlank()) return List.of();
        try {
            JsonNode arr = MAPPER.readTree(json);
            List<ToolCall> out = new ArrayList<>();
            for (JsonNode n : arr) {
                out.add(new ToolCall(n.path("id").asText(),
                        n.path("name").asText(),
                        n.path("argumentsJson").asText()));
            }
            return out;
        } catch (Exception e) {
            return List.of();
        }
    }

    private String encodeImages(List<String> images) {
        if (images == null || images.isEmpty()) return null;
        try { return MAPPER.writeValueAsString(images); }
        catch (Exception e) { return null; }
    }

    private List<String> decodeImages(String json) {
        if (json == null || json.isBlank()) return List.of();
        try {
            JsonNode arr = MAPPER.readTree(json);
            List<String> out = new ArrayList<>();
            for (JsonNode n : arr) if (n.isTextual()) out.add(n.asText());
            return out;
        } catch (Exception e) {
            return List.of();
        }
    }
}
