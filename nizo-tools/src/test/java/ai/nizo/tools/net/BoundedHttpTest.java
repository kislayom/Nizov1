package ai.nizo.tools.net;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests the streaming byte-cap. The defining invariant is "we never hold more than maxBytes+1
 * in memory before we fail" — we test that by feeding a stream that's larger than the cap
 * and expecting an IOException.
 */
class BoundedHttpTest {

    @Test
    void readAll_underCap_returnsBody() throws IOException {
        String body = "hello, world";
        InputStream in = new ByteArrayInputStream(body.getBytes(StandardCharsets.UTF_8));
        assertEquals(body, BoundedHttp.readAll(in, 1024));
    }

    @Test
    void readAll_atCap_returnsBody() throws IOException {
        // Exactly at the cap is allowed (cap is exclusive in the > comparison)
        byte[] body = new byte[1000];
        java.util.Arrays.fill(body, (byte) 'a');
        InputStream in = new ByteArrayInputStream(body);
        String s = BoundedHttp.readAll(in, 1000);
        assertEquals(1000, s.length());
    }

    @Test
    void readAll_overCap_throws() {
        byte[] body = new byte[6 * 1024 * 1024]; // 6 MiB > 5 MiB cap
        InputStream in = new ByteArrayInputStream(body);
        IOException ex = assertThrows(IOException.class,
                () -> BoundedHttp.readAll(in, 5 * 1024 * 1024));
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("exceeded"),
                "error message should say 'exceeded'. Got: " + ex.getMessage());
    }

    @Test
    void readAll_largeButUnderCap_succeeds() throws IOException {
        // 4 MiB just under 5 MiB cap
        byte[] body = new byte[4 * 1024 * 1024];
        java.util.Arrays.fill(body, (byte) 'x');
        String s = BoundedHttp.readAll(new ByteArrayInputStream(body), 5 * 1024 * 1024);
        assertEquals(4 * 1024 * 1024, s.length());
    }

    @Test
    void readAll_nullStream_throws() {
        assertThrows(IOException.class, () -> BoundedHttp.readAll(null, 1024));
    }

    @Test
    void readAll_zeroOrNegativeMax_throws() {
        InputStream in = new ByteArrayInputStream(new byte[0]);
        assertThrows(IllegalArgumentException.class, () -> BoundedHttp.readAll(in, 0));
        assertThrows(IllegalArgumentException.class, () -> BoundedHttp.readAll(in, -1));
    }

    @Test
    void ofString_returnsHandler() {
        assertNotNull(BoundedHttp.ofString());
        assertNotNull(BoundedHttp.ofString(1024));
    }

    @Test
    void ofString_negativeCap_throws() {
        assertThrows(IllegalArgumentException.class, () -> BoundedHttp.ofString(0));
        assertThrows(IllegalArgumentException.class, () -> BoundedHttp.ofString(-1));
    }

    @Test
    void charsetFromContentType_picksDeclared() {
        assertEquals(Charset.forName("ISO-8859-1"),
                BoundedHttp.charsetFromContentType("text/html; charset=ISO-8859-1"));
        assertEquals(StandardCharsets.UTF_8,
                BoundedHttp.charsetFromContentType("text/html; charset=utf-8"));
        assertEquals(StandardCharsets.UTF_8,
                BoundedHttp.charsetFromContentType("text/html; charset=\"utf-8\""));
    }

    @Test
    void charsetFromContentType_missingDefaultsToUtf8() {
        assertEquals(StandardCharsets.UTF_8,
                BoundedHttp.charsetFromContentType("text/html"));
        assertEquals(StandardCharsets.UTF_8,
                BoundedHttp.charsetFromContentType(null));
    }

    @Test
    void charsetFromContentType_unknownCharsetFallsBackToUtf8() {
        assertEquals(StandardCharsets.UTF_8,
                BoundedHttp.charsetFromContentType("text/html; charset=this-charset-does-not-exist"));
    }
}
