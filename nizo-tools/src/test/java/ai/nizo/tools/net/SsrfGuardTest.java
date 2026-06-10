package ai.nizo.tools.net;

import org.junit.jupiter.api.Test;

import java.net.URI;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Coverage for the SSRF guard. The defining requirement: an LLM-provided URL must NEVER
 * be able to coax us into hitting localhost, RFC1918, link-local (which covers AWS/GCP
 * metadata 169.254.169.254), or multicast — even via DNS rebinding-style hostnames.
 *
 * <p>Some of the asserts here resolve real-world DNS, so they only run reliably on an
 * online dev box. The "literal IP" cases work offline and form the spine of coverage.
 */
class SsrfGuardTest {

    @Test
    void blocksLoopbackV4() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://127.0.0.1/x")));
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://127.1.2.3/")));
    }

    @Test
    void blocksLoopbackV6() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://[::1]/")));
    }

    @Test
    void blocksLocalhostHostname() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://localhost/x")));
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://LOCALHOST/x")));
    }

    @Test
    void blocksAwsMetadataAddress() {
        // 169.254.169.254 is the canonical EC2/GCP metadata endpoint — the textbook SSRF target.
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://169.254.169.254/latest/meta-data/")));
    }

    @Test
    void blocksGcpMetadataHostname() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://metadata.google.internal/")));
    }

    @Test
    void blocksRfc1918() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://10.0.0.1/")));
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://192.168.1.1/")));
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://172.16.0.1/")));
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://172.31.255.255/")));
    }

    @Test
    void blocksLinkLocalV6() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://[fe80::1]/")));
    }

    @Test
    void blocksAnyLocal() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://0.0.0.0/")));
    }

    @Test
    void blocksMulticast() {
        assertThrows(SecurityException.class,
                () -> SsrfGuard.assertSafe(URI.create("http://224.0.0.1/")));
    }

    @Test
    void allowsPublicIpLiteral() {
        // 8.8.8.8 is Google DNS — a public, fixed IP that can be relied on offline.
        assertDoesNotThrow(() -> SsrfGuard.assertSafe(URI.create("https://8.8.8.8/")));
    }

    @Test
    void exceptionMentionsHost() {
        try {
            SsrfGuard.assertSafe(URI.create("http://192.168.4.200/"));
        } catch (SecurityException e) {
            // Caller (WebFetchTool) wraps this into the LLM-visible error; the host should
            // appear so the operator can debug from logs.
            String msg = String.valueOf(e.getMessage());
            org.junit.jupiter.api.Assertions.assertTrue(
                    msg.contains("192.168.4.200") || msg.toLowerCase().contains("private")
                            || msg.toLowerCase().contains("internal") || msg.toLowerCase().contains("blocked"),
                    "exception should name the host or category. Got: " + msg);
            return;
        }
        org.junit.jupiter.api.Assertions.fail("expected SecurityException for RFC1918 host");
    }
}
