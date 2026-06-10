package ai.nizo.tools.web;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the 404 short-circuit logic added to prevent wasting SmartProxy retries
 * on URLs that the LLM hallucinated and that genuinely don't exist.
 */
class WebFetchToolTest {

    @Test
    void realNotFound_yahooStyle404_isDetected() {
        // Real body returned by yahoo for /quote/AAPL/financials/ (verified May 2026)
        String body = "<html><meta charset='utf-8'>"
                + "<script>document.write('<p>Content is currently unavailable.</p>"
                + "<img src=\"//geo.yahoo.com/p?err=404\"/>');</script></html>";
        assertTrue(WebFetchTool.looksLikeRealNotFound(body),
                "Yahoo's 404 page should be detected as a real not-found");
    }

    @Test
    void realNotFound_pageNotFoundText_isDetected() {
        String body = "<html><body><h1>Page Not Found</h1>"
                + "<p>This page either doesn't exist, or it moved somewhere else.</p>"
                + "</body></html>";
        assertTrue(WebFetchTool.looksLikeRealNotFound(body),
                "Generic 'Page Not Found' should be detected");
    }

    @Test
    void realNotFound_stocktitanStyle_isDetected() {
        // Mirrors the user-reported stocktitan 404 page
        String body = "<html><body>Stock Titan<br>Error 404<br>Page Not Found<br>"
                + "This page either doesn't exist, or it moved somewhere else.</body></html>";
        assertTrue(WebFetchTool.looksLikeRealNotFound(body),
                "Stocktitan 404 page should be detected");
    }

    @Test
    void cloudflareInterstitial_isNOTRealNotFound() {
        String body = "<html><title>Just a moment...</title>"
                + "<body>Checking your browser before accessing... powered by Cloudflare</body></html>";
        assertFalse(WebFetchTool.looksLikeRealNotFound(body),
                "Cloudflare bot-block should NOT be flagged as 404 (let renderHtml handle it)");
    }

    @Test
    void datadomeCaptcha_isNOTRealNotFound() {
        String body = "<html><body>Please complete the captcha. captcha-delivery.com</body></html>";
        assertFalse(WebFetchTool.looksLikeRealNotFound(body),
                "DataDome captcha should NOT be flagged as 404");
    }

    @Test
    void akamaiBlock_isNOTRealNotFound() {
        String body = "<html><body>Access Denied. You don't have permission. Reference: Akamai</body></html>";
        assertFalse(WebFetchTool.looksLikeRealNotFound(body),
                "Akamai block should NOT be flagged as 404");
    }

    @Test
    void emptyOrNullBody_isNOTRealNotFound() {
        // We require positive evidence — empty body is ambiguous, let SmartProxy try.
        assertFalse(WebFetchTool.looksLikeRealNotFound(null));
        assertFalse(WebFetchTool.looksLikeRealNotFound(""));
    }

    @Test
    void hugePage_isNOTRealNotFound() {
        // A 50KB page that happens to mention "404" somewhere (e.g. in a docs link)
        // is not a 404. Real 404 pages are small.
        StringBuilder sb = new StringBuilder("<html>");
        for (int i = 0; i < 5000; i++) sb.append("<p>lorem ipsum 404 dolor</p>");
        sb.append("</html>");
        assertFalse(WebFetchTool.looksLikeRealNotFound(sb.toString()),
                "Huge body should not be flagged as 404 even if it contains '404'");
    }

    @Test
    void cleanContent_isNOTRealNotFound() {
        String body = "<html><body><h1>Apple Inc.</h1><p>Revenue: $391B in FY2024.</p></body></html>";
        assertFalse(WebFetchTool.looksLikeRealNotFound(body));
    }
}
