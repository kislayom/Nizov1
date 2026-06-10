package ai.nizo.tools.finance;

import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Plumbing tests for the four new structured-data finance tools. We don't hit Yahoo from
 * unit tests — that's flaky and slow. We DO verify:
 *
 * <ul>
 *   <li>Tool name + description are non-empty (the LLM sees these and shouldn't see blanks)</li>
 *   <li>parametersJsonSchema is valid JSON (a malformed schema breaks the OpenAI tool-call format)</li>
 *   <li>Empty / blank ticker returns a clean error (not a NPE or stack trace)</li>
 *   <li>Each tool's execute path doesn't throw on degenerate input</li>
 * </ul>
 */
class FinanceToolsBasicTest {

    private static final ObjectMapper M = new ObjectMapper();
    // Shared client for the four tools; never actually exercised in these tests because we
    // only hit error paths. Each tool defends against the ticker check before calling into
    // the network.
    private final YahooQuoteSummary yahoo = new YahooQuoteSummary();

    @Test
    void fundamentals_nameAndSchema() throws Exception {
        StockFundamentalsTool t = new StockFundamentalsTool(yahoo);
        assertEquals("stock_fundamentals", t.name());
        assertFalse(t.description().isBlank());
        assertNotNull(M.readTree(t.parametersJsonSchema()));
    }

    @Test
    void analystRatings_nameAndSchema() throws Exception {
        AnalystRatingsTool t = new AnalystRatingsTool(yahoo);
        assertEquals("stock_analyst_ratings", t.name());
        assertFalse(t.description().isBlank());
        assertNotNull(M.readTree(t.parametersJsonSchema()));
    }

    @Test
    void insiderActivity_nameAndSchema() throws Exception {
        InsiderActivityTool t = new InsiderActivityTool(yahoo);
        assertEquals("stock_insider_activity", t.name());
        assertFalse(t.description().isBlank());
        assertNotNull(M.readTree(t.parametersJsonSchema()));
    }

    @Test
    void earningsHistory_nameAndSchema() throws Exception {
        EarningsHistoryTool t = new EarningsHistoryTool(yahoo);
        assertEquals("stock_earnings_history", t.name());
        assertFalse(t.description().isBlank());
        assertNotNull(M.readTree(t.parametersJsonSchema()));
    }

    @Test
    void fundamentals_blankTicker_returnsError() {
        StockFundamentalsTool t = new StockFundamentalsTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute("{\"ticker\": \"\"}"));
        assertFalse(r.ok(), "expected error for blank ticker");
        assertTrue(r.content().toLowerCase().contains("ticker"));
    }

    @Test
    void analystRatings_blankTicker_returnsError() {
        AnalystRatingsTool t = new AnalystRatingsTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute("{\"ticker\": \"\"}"));
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("ticker"));
    }

    @Test
    void insiderActivity_blankTicker_returnsError() {
        InsiderActivityTool t = new InsiderActivityTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute("{\"ticker\": \"\"}"));
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("ticker"));
    }

    @Test
    void earningsHistory_blankTicker_returnsError() {
        EarningsHistoryTool t = new EarningsHistoryTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute("{\"ticker\": \"\"}"));
        assertFalse(r.ok());
        assertTrue(r.content().toLowerCase().contains("ticker"));
    }

    @Test
    void fundamentals_emptyArgs_returnsError() {
        StockFundamentalsTool t = new StockFundamentalsTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute(""));
        assertFalse(r.ok());
    }

    @Test
    void analystRatings_emptyArgs_returnsError() {
        AnalystRatingsTool t = new AnalystRatingsTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute(""));
        assertFalse(r.ok());
    }

    @Test
    void insiderActivity_emptyArgs_returnsError() {
        InsiderActivityTool t = new InsiderActivityTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute(""));
        assertFalse(r.ok());
    }

    @Test
    void earningsHistory_emptyArgs_returnsError() {
        EarningsHistoryTool t = new EarningsHistoryTool(yahoo);
        ToolResult r = assertDoesNotThrow(() -> t.execute(""));
        assertFalse(r.ok());
    }

    @Test
    void yahooQuoteSummary_rejectsBlankTicker() {
        YahooQuoteSummary q = new YahooQuoteSummary();
        Exception ex = assertDoesNotThrow(() -> {
            try { q.fetch(""); return null; } catch (Exception e) { return e; }
        });
        assertNotNull(ex);
        assertTrue(ex instanceof IllegalArgumentException);
    }

    @Test
    void yahooQuoteSummary_rejectsNoModules() {
        YahooQuoteSummary q = new YahooQuoteSummary();
        Exception ex = assertDoesNotThrow(() -> {
            try { q.fetch("AAPL"); return null; } catch (Exception e) { return e; }
        });
        assertNotNull(ex);
        assertTrue(ex instanceof IllegalArgumentException);
    }
}
