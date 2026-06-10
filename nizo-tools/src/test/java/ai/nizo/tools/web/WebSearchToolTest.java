package ai.nizo.tools.web;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for the markdown-in-query sanitizer added after seeing the LLM emit
 * `site:[reddit.com](http://reddit.com) AAPL` literally to web_search (May 2026 logs).
 */
class WebSearchToolTest {

    @Test
    void stripMarkdownLink_keepsLinkText() {
        // The actual user-reported case
        assertEquals("site:reddit.com AAPL stock 2026",
                WebSearchTool.sanitizeQuery("site:[reddit.com](http://reddit.com) AAPL stock 2026"));
    }

    @Test
    void stripMarkdownLink_macrotrendsCase() {
        assertEquals("macrotrends.net Apple AAPL income statement 10 years",
                WebSearchTool.sanitizeQuery(
                        "[macrotrends.net](http://macrotrends.net) Apple AAPL income statement 10 years"));
    }

    @Test
    void stripMarkdownLink_secGovCase() {
        assertEquals("site:sec.gov Apple 10-K 2024 filing",
                WebSearchTool.sanitizeQuery("site:[sec.gov](http://sec.gov) Apple 10-K 2024 filing"));
    }

    @Test
    void multipleMarkdownLinks_allStripped() {
        assertEquals("site:reddit.com AAPL site:stocktwits.com",
                WebSearchTool.sanitizeQuery(
                        "site:[reddit.com](http://reddit.com) AAPL site:[stocktwits.com](http://stocktwits.com)"));
    }

    @Test
    void cleanQuery_passesThrough() {
        assertEquals("AAPL income statement 2024 macrotrends",
                WebSearchTool.sanitizeQuery("AAPL income statement 2024 macrotrends"));
    }

    @Test
    void siteOperatorWithBareDomain_preserved() {
        assertEquals("site:macrotrends.net Apple income statement",
                WebSearchTool.sanitizeQuery("site:macrotrends.net Apple income statement"));
    }

    @Test
    void doubleSpacesCollapsed() {
        assertEquals("AAPL income statement",
                WebSearchTool.sanitizeQuery("AAPL    income   statement"));
    }

    @Test
    void leadingTrailingWhitespaceTrimmed() {
        assertEquals("AAPL income",
                WebSearchTool.sanitizeQuery("   AAPL income   "));
    }

    @Test
    void nullInput_returnsEmptyString() {
        assertEquals("", WebSearchTool.sanitizeQuery(null));
    }

    @Test
    void onlyMarkdown_keepsLinkTexts() {
        assertEquals("apple google",
                WebSearchTool.sanitizeQuery("[apple](https://apple.com) [google](https://google.com)"));
    }
}
