package ai.nizo.tools.file;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Symlink-safe path resolution for the file tools.
 *
 * <p>The workspace boundary is enforced after fully resolving symlinks. A {@code normalize()}-only
 * check is unsafe because a symlink at {@code <workspace>/escape -> /etc/passwd} passes lexical
 * containment but reads from outside the workspace.
 *
 * <p>The trick: {@link Path#toRealPath} requires the path to exist. For writes we may target a
 * file that doesn't exist yet, so we resolve the closest existing ancestor (which might be the
 * workspace itself) and append the remaining components to its real path.
 */
final class WorkspacePaths {

    private WorkspacePaths() {}

    /**
     * Resolve {@code workspace} to its real (symlink-followed) absolute path. The workspace must
     * exist; if not, an {@link IOException} bubbles up — fail fast at construction.
     */
    static Path resolveWorkspaceReal(Path workspace) throws IOException {
        return workspace.toAbsolutePath().normalize().toRealPath();
    }

    /**
     * Resolve a workspace-relative path under {@code workspaceReal}, fully following symlinks
     * inside the workspace, and verify the result is contained in {@code workspaceReal}.
     *
     * <p>If the leaf doesn't exist yet (writes to new files), we compute the real path of the
     * closest existing ancestor, then re-attach the remaining components. This catches symlinks
     * that exist anywhere in the resolved chain — they get followed during {@code toRealPath} on
     * the existing prefix — while still permitting writes to brand-new paths.
     *
     * @throws SecurityException if the resolved target is outside {@code workspaceReal}
     * @throws IOException if a filesystem error happens during resolution
     */
    static Path resolveSafe(Path workspaceReal, String relPath) throws IOException {
        if (relPath == null) relPath = "";
        Path rel = Path.of(relPath);
        // Disallow rooted paths up front; resolve() would happily replace the workspace.
        if (rel.isAbsolute()) {
            throw new SecurityException("Access denied — absolute paths not allowed: " + relPath);
        }

        Path candidate = workspaceReal.resolve(rel).normalize();
        Path resolved;

        if (Files.exists(candidate)) {
            // Both leaf and all ancestors exist → toRealPath gives us the symlink-resolved truth.
            resolved = candidate.toRealPath();
        } else {
            resolved = resolveWithMissingTail(workspaceReal, candidate);
        }

        if (!resolved.startsWith(workspaceReal)) {
            throw new SecurityException(
                    "Access denied — path resolves outside workspace via symlink/traversal: "
                            + relPath);
        }
        return resolved;
    }

    /**
     * For a path whose leaf (or some suffix) doesn't exist, walk up until an existing ancestor is
     * found, resolve its real path (following any symlinks along the way), then re-attach the
     * missing components. The non-existing tail can't contain symlinks, so a lexical join is safe.
     */
    private static Path resolveWithMissingTail(Path workspaceReal, Path candidate) throws IOException {
        Path existing = candidate;
        while (existing != null && !Files.exists(existing)) {
            existing = existing.getParent();
        }
        if (existing == null) {
            throw new SecurityException(
                    "Access denied — path resolves outside workspace via symlink/traversal");
        }
        Path existingReal = existing.toRealPath();
        Path tail = existing.relativize(candidate);
        // tail.toString() may be empty if candidate equals existing; resolve() handles that.
        return existingReal.resolve(tail).normalize();
    }
}
