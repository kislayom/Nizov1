package ai.nizo.tools.finance.buffett;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BuffettAnalysisEngine} — locks in the scoring rules so the engine
 * stays deterministic across edits. Each test exercises a slice of the score chain with
 * hand-crafted inputs and asserts the expected verdict / score band.
 */
class BuffettAnalysisEngineTest {

    private static final ObjectMapper M = new ObjectMapper();

    /**
     * Build a quote node {@code {currentPrice, marketCap}}.
     */
    private static ObjectNode quote(double price, double marketCap) {
        ObjectNode n = M.createObjectNode();
        n.put("currentPrice", price);
        n.put("marketCap", marketCap);
        return n;
    }

    /**
     * Flat metrics object — set fields by key. The engine reads bare numbers via {@code num()}
     * which falls through {@code {raw, fmt}} unwrapping, so test inputs can be primitives.
     */
    private static ObjectNode metrics() { return M.createObjectNode(); }

    /** Helper: append a single annual income/cashflow row. */
    private static ArrayNode addRow(ArrayNode arr, ObjectNode row) { arr.add(row); return arr; }

    private static ObjectNode incomeRow(double revenue, double netIncome, double eps, long shares) {
        ObjectNode r = M.createObjectNode();
        r.put("revenue", revenue);
        r.put("netIncome", netIncome);
        r.put("eps", eps);
        r.put("dilutedEPS", eps);
        r.put("sharesOutstanding", shares);
        return r;
    }

    private static ObjectNode cashFlowRow(double fcf, double capex, double dividends, double repurchase) {
        ObjectNode r = M.createObjectNode();
        r.put("freeCashFlow", fcf);
        r.put("capitalExpenditures", capex);
        r.put("dividendsPaid", -Math.abs(dividends));
        r.put("repurchaseOfStock", -Math.abs(repurchase));
        r.put("operatingCashFlow", fcf + Math.abs(capex));
        return r;
    }

    private static ObjectNode profile(String sector, String industry, String desc) {
        ObjectNode p = M.createObjectNode();
        p.put("sector", sector);
        p.put("industry", industry);
        p.put("longBusinessSummary", desc);
        return p;
    }

    // ======================================================================
    // Moat scoring
    // ======================================================================

    @Test
    void wideMoatStock_scoresHigh() {
        // High ROE + margins + low debt → moat 9/10
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.30);
        m.put("grossMargins", 0.50);
        m.put("operatingMargins", 0.30);
        m.put("returnOnInvestedCapital", 0.25);
        m.put("debtToEquity", 0.2);
        m.put("trailingPE", 25);
        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(100, 1_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(100e9, 20e9, 5.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(18e9, -5e9, 2e9, 8e9)),
                M.createArrayNode(),
                profile("Technology", "Software", "Big software co"),
                Double.NaN);
        assertTrue(result.moatScore >= 9, "expected wide moat; got " + result.moatScore);
        assertTrue(result.greenFlags.stream().anyMatch(f -> f.contains("Wide economic moat")));
    }

    @Test
    void weakBusinessStock_scoresLowMoat() {
        // Mediocre ROE + skinny margins + high debt → moat ≤ 5
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.05);
        m.put("grossMargins", 0.18);
        m.put("operatingMargins", 0.05);
        m.put("returnOnInvestedCapital", 0.04);
        m.put("debtToEquity", 2.0);
        m.put("trailingPE", 60);
        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(50, 5_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(10e9, 0.5e9, 1.0, 500_000_000L)),
                M.createArrayNode().add(cashFlowRow(0.3e9, -1e9, 0, 0)),
                M.createArrayNode(),
                profile("Industrial", "Manufacturing", "Capital-heavy biz"),
                Double.NaN);
        assertTrue(result.moatScore <= 5, "expected weak moat; got " + result.moatScore);
    }

    // ======================================================================
    // Verdict logic — score + MoS thresholds
    // ======================================================================

    @Test
    void strongBuy_requiresHighScoreAndDeepMoS() {
        // High-quality, deeply undervalued
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.30);
        m.put("grossMargins", 0.55);
        m.put("operatingMargins", 0.30);
        m.put("returnOnInvestedCapital", 0.25);
        m.put("debtToEquity", 0.1);
        m.put("currentRatio", 2.5);
        m.put("interestCoverageRatio", 30);
        m.put("trailingPE", 15);
        m.put("revenueGrowth", 0.10);
        m.put("earningsGrowth", 0.18);
        m.put("eps", 10.0);
        m.put("bookValuePerShare", 50.0);

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(80.0, 80_000_000_000.0),    // priced 80 vs intrinsic ~150 → big MoS
                m,
                M.createArrayNode().add(incomeRow(50e9, 12e9, 10.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(11e9, -2e9, 3e9, 5e9)),
                M.createArrayNode(),
                profile("Consumer", "Beverages", "Iconic brand"),
                Double.NaN);

        assertTrue(result.buffettScore >= 70, "score should be at least 70; got " + result.buffettScore);
        assertTrue(result.marginOfSafety > 0.10, "MoS should be positive; got " + result.marginOfSafety);
        assertTrue(result.verdict.equals("Strong Buy") || result.verdict.equals("Buy"),
                "expected Strong Buy or Buy; got " + result.verdict);
    }

    @Test
    void overvaluedHighQuality_returnsWatchOrPass() {
        // Same wide-moat profile but priced ABOVE intrinsic
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.30);
        m.put("grossMargins", 0.55);
        m.put("operatingMargins", 0.30);
        m.put("returnOnInvestedCapital", 0.25);
        m.put("debtToEquity", 0.1);
        m.put("currentRatio", 2.5);
        m.put("interestCoverageRatio", 30);
        m.put("trailingPE", 60);
        m.put("revenueGrowth", 0.05);
        m.put("eps", 5.0);
        m.put("bookValuePerShare", 20.0);

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(500.0, 500_000_000_000.0),     // wildly overpriced
                m,
                M.createArrayNode().add(incomeRow(50e9, 15e9, 5.0, 3_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(13e9, -2e9, 5e9, 4e9)),
                M.createArrayNode(),
                profile("Consumer", "Tech", "Premium product"),
                Double.NaN);

        assertTrue(result.marginOfSafety < 0, "MoS should be negative; got " + result.marginOfSafety);
        assertTrue(result.verdict.equals("Watch") || result.verdict.equals("Pass"),
                "expected Watch or Pass; got " + result.verdict);
    }

    // ======================================================================
    // Capital allocation grading
    // ======================================================================

    @Test
    void aggressiveBuybacks_gradeAplus() {
        // 90% of FCF returned to shareholders → A+
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.20);
        m.put("revenueGrowth", 0.04);   // mature
        m.put("eps", 5);
        m.put("bookValuePerShare", 20);

        ArrayNode cf = M.createArrayNode();
        cf.add(cashFlowRow(10e9, -1e9, 3e9, 6e9));   // $9B returned / $10B FCF = 90%

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(50, 50_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(50e9, 5e9, 5.0, 1_000_000_000L)),
                cf, M.createArrayNode(),
                profile("Consumer", "Retail", "Mature consumer co"),
                Double.NaN);
        assertEquals("A+", result.capitalAllocationGrade);
    }

    @Test
    void highGrowthCo_gradedOnReinvestment_notReturns() {
        // Revenue growth > 20% → grade A even with no buybacks
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.18);
        m.put("revenueGrowth", 0.25);    // hyper-growth
        m.put("eps", 2);
        m.put("bookValuePerShare", 10);

        ArrayNode cf = M.createArrayNode();
        cf.add(cashFlowRow(2e9, -1e9, 0, 0));  // 0% return ratio

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(100, 100_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(20e9, 2e9, 2.0, 1_000_000_000L)),
                cf, M.createArrayNode(),
                profile("Technology", "SaaS", "Hyper-growth SaaS"),
                Double.NaN);
        assertEquals("A", result.capitalAllocationGrade);
        assertTrue(result.capitalAllocationCommentary.toLowerCase().contains("growth"));
    }

    // ======================================================================
    // Margin of Safety assessment bands
    // ======================================================================

    @Test
    void mosAssessment_bands() {
        // Use a deterministic intrinsic via fixed inputs; vary the price to land in each band.
        // Build a stable, easily-priced setup: EPS 10, BVPS 50, no growth → fair PE 12
        // → growth-PE intrinsic = 12 × 10 = $120
        // → graham = sqrt(22.5 × 10 × 50) = sqrt(11250) ≈ $106
        // → 10-cap = OE/share × 10  (OE depends on netIncome - 0.7×capex)
        // The weighted average ends up around ~$110; tweak per band.

        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.20);
        m.put("eps", 10.0);
        m.put("bookValuePerShare", 50.0);
        m.put("revenueGrowth", 0.04);

        // Intrinsic ~110. Set price to 60 (45% MoS) → "Excellent"
        BuffettMetrics excellent = BuffettAnalysisEngine.analyze(
                quote(60.0, 60_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(50e9, 10e9, 10.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(8e9, -2e9, 3e9, 3e9)),
                M.createArrayNode(),
                profile("Consumer", "Retail", "Solid co"),
                Double.NaN);
        assertEquals("Excellent", excellent.marginOfSafetyAssessment,
                "MoS=" + excellent.marginOfSafety + ", intrinsic=" + excellent.averageIntrinsicValue);

        // price 130 → MoS negative → "Overvalued" or "Significantly Overvalued"
        BuffettMetrics over = BuffettAnalysisEngine.analyze(
                quote(130.0, 130_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(50e9, 10e9, 10.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(8e9, -2e9, 3e9, 3e9)),
                M.createArrayNode(),
                profile("Consumer", "Retail", "Solid co"),
                Double.NaN);
        assertTrue(over.marginOfSafety < 0);
        assertTrue(over.marginOfSafetyAssessment.startsWith("Overvalued") ||
                   over.marginOfSafetyAssessment.startsWith("Significantly"));
    }

    // ======================================================================
    // Munger checklist
    // ======================================================================

    @Test
    void mungerChecklist_perfectScore() {
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.30);
        m.put("grossMargins", 0.55);
        m.put("operatingMargins", 0.30);
        m.put("returnOnInvestedCapital", 0.25);
        m.put("debtToEquity", 0.1);
        m.put("currentRatio", 2.5);
        m.put("interestCoverageRatio", 30);
        m.put("trailingPE", 15);
        m.put("revenueGrowth", 0.04);    // mature → grade A+ achievable
        m.put("eps", 10);
        m.put("bookValuePerShare", 50);

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(50, 50_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(50e9, 10e9, 10.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(10e9, -1e9, 3e9, 6e9)),
                M.createArrayNode(),
                profile("Consumer", "Beverages", "Iconic global brand with deep moat"),
                Double.NaN);

        assertEquals(5, result.mungerChecklistScore(), "all 5 gates should pass");
    }

    // ======================================================================
    // CAGR computation
    // ======================================================================

    @Test
    void cagr_acrossThreeYears() {
        // 10 → 13.31 over 3 years = 10% CAGR. Note arr is newest-first.
        ArrayNode arr = M.createArrayNode();
        arr.add(M.createObjectNode().put("revenue", 13.31));
        arr.add(M.createObjectNode().put("revenue", 12.10));
        arr.add(M.createObjectNode().put("revenue", 11.00));
        arr.add(M.createObjectNode().put("revenue", 10.00));
        Double cagr = BuffettAnalysisEngine.cagr(arr, "revenue");
        assertNotNull(cagr);
        assertEquals(0.10, cagr, 0.01, "expected ~10% CAGR; got " + cagr);
    }

    @Test
    void cagr_returnsNullOnInsufficientData() {
        ArrayNode arr = M.createArrayNode();
        arr.add(M.createObjectNode().put("revenue", 100));
        arr.add(M.createObjectNode().put("revenue", 90));
        assertNull(BuffettAnalysisEngine.cagr(arr, "revenue"), "needs ≥3 rows");
    }

    // ======================================================================
    // Output JSON shape
    // ======================================================================

    @Test
    void toJson_hasAllExpectedFields() {
        ObjectNode m = metrics();
        m.put("returnOnEquity", 0.20);
        m.put("eps", 5);
        m.put("bookValuePerShare", 20);
        m.put("revenueGrowth", 0.10);

        BuffettMetrics result = BuffettAnalysisEngine.analyze(
                quote(50, 50_000_000_000.0), m,
                M.createArrayNode().add(incomeRow(20e9, 5e9, 5.0, 1_000_000_000L)),
                M.createArrayNode().add(cashFlowRow(4e9, -1e9, 1e9, 1e9)),
                M.createArrayNode(),
                profile("Consumer", "Retail", "OK co"),
                Double.NaN);

        JsonNode json = result.toJson("AAPL", "USD");
        assertEquals("AAPL", json.path("ticker").asText());
        assertEquals("USD", json.path("currency").asText());
        assertTrue(json.has("score"));
        assertTrue(json.has("verdict"));
        assertTrue(json.has("moat"));
        assertTrue(json.has("intrinsicValue"));
        assertTrue(json.has("marginOfSafety"));
        assertTrue(json.has("valuationMethods"));
        assertTrue(json.has("capitalAllocationGrade"));
        assertTrue(json.has("mungerChecklist"));
        assertTrue(json.has("redFlags"));
        assertTrue(json.has("greenFlags"));
        // Score breakdown subfields
        JsonNode score = json.path("score");
        assertTrue(score.has("total"));
        assertTrue(score.has("businessQuality"));
        assertTrue(score.has("management"));
        assertTrue(score.has("financialStrength"));
        assertTrue(score.has("valuation"));
        assertTrue(score.has("growth"));
    }
}
