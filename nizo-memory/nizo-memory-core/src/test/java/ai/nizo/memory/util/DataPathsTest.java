package ai.nizo.memory.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class DataPathsTest {

    @Test
    void defaultDataDirIsUnderUserHome() {
        Path d = DataPaths.defaultDataDir();
        assertEquals(Path.of(System.getProperty("user.home"), ".nizo"), d);
    }

    @Test
    void defaultDatabaseFileSitsInsideDefaultDataDir() {
        Path f = DataPaths.defaultDatabaseFile();
        assertEquals(DataPaths.defaultDataDir().resolve("memory.db"), f);
    }

    @Test
    void resolvesUserDataToken() {
        Path p = DataPaths.resolve("${user.data}/memory.db");
        assertEquals(DataPaths.defaultDataDir().resolve("memory.db").toAbsolutePath(), p);
    }

    @Test
    void resolvesTildePrefix() {
        Path p = DataPaths.resolve("~/something/x.db");
        assertEquals(
                Path.of(System.getProperty("user.home"), "something", "x.db").toAbsolutePath(),
                p);
    }

    @Test
    void absolutePathPassesThrough(@TempDir Path tmp) {
        Path target = tmp.resolve("nested/inner/foo.db");
        Path resolved = DataPaths.resolve(target.toString());
        assertEquals(target.toAbsolutePath(), resolved);
        // parent dir should now exist
        assertTrue(Files.isDirectory(target.getParent()));
    }

    @Test
    void createsParentDirectoryOnDemand(@TempDir Path tmp) {
        Path target = tmp.resolve("a/b/c/db.sqlite");
        assertFalse(Files.exists(target.getParent()));
        DataPaths.resolve(target.toString());
        assertTrue(Files.isDirectory(target.getParent()));
    }

    @Test
    void blankOrNullPathRejected() {
        assertThrows(IllegalArgumentException.class, () -> DataPaths.resolve(null));
        assertThrows(IllegalArgumentException.class, () -> DataPaths.resolve(""));
        assertThrows(IllegalArgumentException.class, () -> DataPaths.resolve("   "));
    }
}
