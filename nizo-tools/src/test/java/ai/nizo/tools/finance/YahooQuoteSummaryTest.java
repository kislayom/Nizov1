package ai.nizo.tools.finance;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the Yahoo {raw, fmt, longFmt} unwrap helpers. Yahoo wraps numeric values
 * inconsistently — some endpoints return raw numbers, others return wrapped objects, and
 * occasionally a missing field is absent vs explicitly null. The helpers must paper over all
 * three shapes without throwing.
 */
class YahooQuoteSummaryTest {

    private static final ObjectMapper M = new ObjectMapper();

    @Test
    void rawDouble_unwrapsWrappedValue() throws Exception {
        JsonNode n = M.readTree("""
            { "totalRevenue": { "raw": 391035000000, "fmt": "391.04B", "longFmt": "391,035,000,000" } }
            """);
        assertEquals(391035000000d, YahooQuoteSummary.rawDouble(n.path("totalRevenue")), 0.001);
    }

    @Test
    void rawDouble_returnsNanOnMissing() throws Exception {
        JsonNode n = M.readTree("{}");
        assertTrue(Double.isNaN(YahooQuoteSummary.rawDouble(n.path("totalRevenue"))));
    }

    @Test
    void rawDouble_returnsNanOnExplicitNull() throws Exception {
        JsonNode n = M.readTree("""
            { "totalRevenue": null }
            """);
        assertTrue(Double.isNaN(YahooQuoteSummary.rawDouble(n.path("totalRevenue"))));
    }

    @Test
    void rawDouble_handlesBareNumber() throws Exception {
        // Some Yahoo modules (recommendationTrend) emit bare numeric counts, not {raw,fmt}.
        JsonNode n = M.readTree("{ \"strongBuy\": 12 }");
        assertEquals(12.0, YahooQuoteSummary.rawDouble(n.path("strongBuy")), 0.001);
    }

    @Test
    void rawLong_unwrapsWrappedValue() throws Exception {
        JsonNode n = M.readTree("""
            { "marketCap": { "raw": 3500000000000, "fmt": "3.50T" } }
            """);
        assertEquals(3500000000000L, YahooQuoteSummary.rawLong(n.path("marketCap")));
    }

    @Test
    void rawLong_returnsZeroOnMissing() throws Exception {
        JsonNode n = M.readTree("{}");
        assertEquals(0L, YahooQuoteSummary.rawLong(n.path("marketCap")));
    }

    @Test
    void rawLong_handlesBareNumber() throws Exception {
        JsonNode n = M.readTree("{ \"shares\": 1000000 }");
        assertEquals(1000000L, YahooQuoteSummary.rawLong(n.path("shares")));
    }

    @Test
    void fmtString_returnsFmtField() throws Exception {
        JsonNode n = M.readTree("""
            { "marketCap": { "raw": 3500000000000, "fmt": "3.50T" } }
            """);
        assertEquals("3.50T", YahooQuoteSummary.fmtString(n.path("marketCap")));
    }

    @Test
    void fmtString_returnsEmptyOnMissing() throws Exception {
        JsonNode n = M.readTree("{}");
        assertEquals("", YahooQuoteSummary.fmtString(n.path("marketCap")));
    }

    @Test
    void fmtString_falsbackForBareString() throws Exception {
        JsonNode n = M.readTree("{ \"period\": \"-1m\" }");
        assertEquals("-1m", YahooQuoteSummary.fmtString(n.path("period")));
    }
}
