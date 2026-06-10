package ai.nizo.agent.store;

import javax.sql.DataSource;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.logging.Logger;

/**
 * Tiny {@link DataSource} adapter over {@link DriverManager#getConnection(String)} for
 * a single SQLite file. No pooling — SQLite's WAL mode is happy with one connection
 * per call and the existing stores already open/close per query, so a pool would just
 * add ceremony.
 *
 * <p>Existence of this class is an explicit choice: {@link SchemaMigrator} takes a
 * {@code DataSource} to keep its public API generic (and unit-testable against in-memory
 * JDBC), but our production stores already know exactly which file they're talking to.
 */
public final class SqliteDataSource implements DataSource {

    private final String jdbcUrl;

    public SqliteDataSource(Path dbFile) {
        this("jdbc:sqlite:" + dbFile.toAbsolutePath());
    }

    public SqliteDataSource(String jdbcUrl) {
        this.jdbcUrl = jdbcUrl;
        try {
            // Ensure the driver is registered. Idempotent — a no-op if it's already loaded.
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("sqlite-jdbc driver missing on classpath", e);
        }
    }

    public String jdbcUrl() { return jdbcUrl; }

    @Override public Connection getConnection() throws SQLException {
        return DriverManager.getConnection(jdbcUrl);
    }

    @Override public Connection getConnection(String username, String password) throws SQLException {
        return DriverManager.getConnection(jdbcUrl, username, password);
    }

    // -------- DataSource boilerplate (we don't use any of these features) --------

    @Override public PrintWriter getLogWriter() { return null; }
    @Override public void setLogWriter(PrintWriter out) { }
    @Override public void setLoginTimeout(int seconds) { }
    @Override public int getLoginTimeout() { return 0; }
    @Override public Logger getParentLogger() throws SQLFeatureNotSupportedException {
        throw new SQLFeatureNotSupportedException();
    }

    @Override
    public <T> T unwrap(Class<T> iface) throws SQLException {
        if (iface.isAssignableFrom(getClass())) return iface.cast(this);
        throw new SQLException("not a wrapper for " + iface);
    }

    @Override
    public boolean isWrapperFor(Class<?> iface) {
        return iface.isAssignableFrom(getClass());
    }
}
