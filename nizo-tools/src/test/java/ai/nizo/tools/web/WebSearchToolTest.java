package ai.nizo.tools.web;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

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

    // ── Reliable keyless fallback parsers (added so search never returns "(no results)" for a
    //    factual query even when every bot-blockable web engine is walled off on the server IP) ──

    @Test
    void parseWikipedia_extractsTitlesUrlsAndStripsSnippetHtml() throws Exception {
        String json = """
            {"query":{"search":[
              {"title":"Noise Protocol Framework","snippet":"The <span class=\\"searchmatch\\">Noise</span> Protocol Framework is a framework for crypto protocols &amp; handshakes."},
              {"title":"WireGuard","snippet":"WireGuard uses the <span class=\\"searchmatch\\">Noise</span> framework."}
            ]}}""";
        var results = WebSearchTool.parseWikipedia(json, 5);
        assertEquals(2, results.size());
        String s = results.toString();
        // title + a correctly-built article URL (spaces -> underscores)
        assertTrue(s.contains("Noise Protocol Framework"), s);
        assertTrue(s.contains("https://en.wikipedia.org/wiki/Noise_Protocol_Framework"), s);
        // HTML tags + entities stripped from the snippet
        assertTrue(s.contains("framework for crypto protocols & handshakes"), s);
        assertFalse(s.contains("<span"), s);
        assertFalse(s.contains("&amp;"), s);
    }

    @Test
    void parseWikipedia_respectsLimit() throws Exception {
        String json = """
            {"query":{"search":[
              {"title":"A","snippet":"a"},{"title":"B","snippet":"b"},{"title":"C","snippet":"c"}
            ]}}""";
        assertEquals(2, WebSearchTool.parseWikipedia(json, 2).size());
    }

    @Test
    void parseWikipedia_emptyResultSet() throws Exception {
        assertEquals(0, WebSearchTool.parseWikipedia("{\"query\":{\"search\":[]}}", 5).size());
    }

    @Test
    void parseDuckDuckGoInstant_takesAbstractThenRelatedTopics() throws Exception {
        String json = """
            {"Heading":"Apple Inc.","AbstractText":"Apple Inc. is an American technology company.",
             "AbstractURL":"https://en.wikipedia.org/wiki/Apple_Inc.",
             "RelatedTopics":[
               {"Text":"IPhone - a smartphone by Apple","FirstURL":"https://duckduckgo.com/IPhone"},
               {"Name":"Products","Topics":[
                  {"Text":"MacBook - a laptop by Apple","FirstURL":"https://duckduckgo.com/MacBook"}
               ]}
             ]}""";
        var results = WebSearchTool.parseDuckDuckGoInstant(json, 5);
        assertEquals(3, results.size());
        String s = results.toString();
        assertTrue(s.contains("Apple Inc."), s);
        assertTrue(s.contains("https://en.wikipedia.org/wiki/Apple_Inc."), s);
        // related topic title is the clause before " - "; nested Topics group is flattened in
        assertTrue(s.contains("IPhone"), s);
        assertTrue(s.contains("MacBook"), s);
    }

    @Test
    void parseDuckDuckGoInstant_skipsTopicsMissingTextOrUrl() throws Exception {
        String json = """
            {"AbstractText":"","AbstractURL":"",
             "RelatedTopics":[
               {"Text":"has text but no url"},
               {"FirstURL":"https://duckduckgo.com/x"},
               {"Text":"Good - real one","FirstURL":"https://duckduckgo.com/good"}
             ]}""";
        var results = WebSearchTool.parseDuckDuckGoInstant(json, 5);
        assertEquals(1, results.size());
        assertTrue(results.toString().contains("Good"), results.toString());
    }
}
