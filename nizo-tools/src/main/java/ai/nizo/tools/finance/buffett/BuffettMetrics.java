package ai.nizo.tools.finance.buffett;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Buffett-Munger investment scorecard for a public stock — the deterministic, no-LLM
 * output that {@link BuffettAnalysisEngine} produces. Ported from Kimaya's BuffettMetrics.
 *
 * <h2>Why a flat mutable record</h2>
 * The engine builds this in many phases (moat → valuation → flags → score → verdict), each
 * of which depends on prior fields. Pure-immutable records would force a single huge
 * constructor with 30+ parameters or builders that obscure the compute order. This is the
 * "internal compute object" — convert to clean JSON at the end via {@link #toJson(String)}.
 *
 * <p>All numeric fields are doubles for JSON friendliness. Financial precision matters at
 * the input stage (FMP returns big-int dollars + decimal percentages); by the time we
 * compute scores and ratios, the LLM/UI consumer doesn't need {@code BigDecimal} fidelity.
 */
public final class BuffettMetrics {

    // ── Owner economics ────────────────────────────────────────────────────
    public double ownerEarnings;            // dollars; Net Income + Depreciation − Maintenance CapEx
    public double ownerEarningsPerShare;
    public double ownerEarningsYield;       // OE / Market Cap, decimal (0.05 = 5%)
    public double freeCashFlow;
    public double freeCashFlowPerShare;
    public double freeCashFlowYield;        // FCF/Share / Price

    // ── Moat (0-10) ────────────────────────────────────────────────────────
    public int    moatScore;                // 0-10; 8+ = wide
    public String moatType = "UNKNOWN";     // BRAND / NETWORK_EFFECT / SWITCHING_COSTS / BRAND_INTANGIBLES / MIXED
    public String moatTrend = "STABLE";     // WIDENING / STABLE / NARROWING
    public String moatEvidence = "";

    // ── Valuation ──────────────────────────────────────────────────────────
    public double dcfIntrinsicValue;        // 0 if unavailable
    public double tenCapValue;
    public double grahamNumberValue;
    public double growthAdjustedPeValue;
    public double averageIntrinsicValue;    // weighted across active methods
    public double currentPrice;
    public double marginOfSafety;           // (intrinsic - price) / intrinsic
    public String marginOfSafetyAssessment = "Unknown";   // Excellent/Adequate/Minimal/Overvalued/Significantly Overvalued
    public final List<ValuationMethod> valuationMethodsUsed = new ArrayList<>();

    // ── Capital allocation ─────────────────────────────────────────────────
    public String capitalAllocationGrade = "?";       // A+ A B C D
    public String capitalAllocationCommentary = "";

    // ── Long-term consistency (10y) ────────────────────────────────────────
    public Double tenYearRevenueCAGR;       // null if insufficient history
    public Double tenYearEpsCAGR;
    public Double tenYearFcfCAGR;

    // ── Component scores (each 0-20) ──────────────────────────────────────
    public int businessQualityScore;
    public int managementQualityScore;
    public int financialStrengthScore;
    public int valuationScore;
    public int growthProspectsScore;

    // ── Aggregate score ───────────────────────────────────────────────────
    public int    buffettScore;             // 0-100 sum of components
    public String verdict = "Pass";         // Strong Buy / Buy / Watch / Pass
    public double buyPrice;                 // intrinsic × 0.80
    public double strongBuyPrice;           // intrinsic × 0.65
    public String positionSizing = "Avoid";

    // ── Flags + Munger checklist ──────────────────────────────────────────
    public final List<String> redFlags = new ArrayList<>();
    public final List<String> greenFlags = new ArrayList<>();
    public boolean understandBusiness;
    public boolean durableMoat;
    public boolean honestManagement;
    public boolean fairPrice;
    public boolean canHoldForever;

    public int mungerChecklistScore() {
        int n = 0;
        if (understandBusiness) n++;
        if (durableMoat) n++;
        if (honestManagement) n++;
        if (fairPrice) n++;
        if (canHoldForever) n++;
        return n;
    }

    /** A single valuation method's contribution to the weighted intrinsic value. */
    public record ValuationMethod(
            String method,         // "DCF", "Growth-Adjusted PE", "10-Cap", "Graham Number"
            double value,          // intrinsic value from this method (per share)
            double weight,         // weight % (0-100)
            String formula,        // human-readable, e.g. "Fair PE 22.5 × EPS 6.40"
            boolean excluded,      // true if engine excluded this method (e.g. growth co)
            String excludeReason   // null if !excluded
    ) {}

    /**
     * Serialize to a JSON shape consumed by the {@code chart-buffett} front-end fence
     * renderer. The shape is stable; consumer fields:
     * <ul>
     *   <li>{@code ticker, asOf, source, currentPrice, currency}</li>
     *   <li>{@code score: { total, businessQuality, management, financialStrength, valuation, growth }}</li>
     *   <li>{@code verdict, positionSizing, buyPrice, strongBuyPrice}</li>
     *   <li>{@code moat: { score, type, trend, evidence }}</li>
     *   <li>{@code marginOfSafety, marginOfSafetyAssessment, intrinsicValue}</li>
     *   <li>{@code valuationMethods: [{method, value, weight, formula, excluded, excludeReason}]}</li>
     *   <li>{@code capitalAllocationGrade, capitalAllocationCommentary}</li>
     *   <li>{@code ownerEarnings, ownerEarningsPerShare, ownerEarningsYield, freeCashFlow*, fcfYield}</li>
     *   <li>{@code consistency: { tenYearRevenueCAGR, tenYearEpsCAGR, tenYearFcfCAGR }}</li>
     *   <li>{@code mungerChecklist: { understandBusiness, durableMoat, ..., score }}</li>
     *   <li>{@code redFlags: [..], greenFlags: [..]}</li>
     * </ul>
     */
    public ObjectNode toJson(String ticker, String currency) {
        ObjectMapper M = new ObjectMapper();
        ObjectNode out = M.createObjectNode();
        out.put("ticker", ticker == null ? "" : ticker);
        out.put("asOf", java.time.LocalDate.now(java.time.ZoneOffset.UTC).toString());
        out.put("source", "buffett-engine");
        out.put("currency", currency == null ? "USD" : currency);
        out.put("currentPrice", round2(currentPrice));

        ObjectNode score = out.putObject("score");
        score.put("total", buffettScore);
        score.put("businessQuality", businessQualityScore);
        score.put("management", managementQualityScore);
        score.put("financialStrength", financialStrengthScore);
        score.put("valuation", valuationScore);
        score.put("growth", growthProspectsScore);

        out.put("verdict", verdict);
        out.put("positionSizing", positionSizing);
        out.put("buyPrice", round2(buyPrice));
        out.put("strongBuyPrice", round2(strongBuyPrice));

        ObjectNode moat = out.putObject("moat");
        moat.put("score", moatScore);
        moat.put("type", moatType);
        moat.put("trend", moatTrend);
        moat.put("evidence", moatEvidence);

        out.put("intrinsicValue", round2(averageIntrinsicValue));
        out.put("marginOfSafety", round4(marginOfSafety));
        out.put("marginOfSafetyAssessment", marginOfSafetyAssessment);

        ArrayNode methods = out.putArray("valuationMethods");
        for (ValuationMethod m : valuationMethodsUsed) {
            ObjectNode n = methods.addObject();
            n.put("method", m.method());
            n.put("value", round2(m.value()));
            n.put("weight", round2(m.weight()));
            n.put("formula", m.formula());
            n.put("excluded", m.excluded());
            if (m.excludeReason() != null) n.put("excludeReason", m.excludeReason());
        }

        out.put("capitalAllocationGrade", capitalAllocationGrade);
        out.put("capitalAllocationCommentary", capitalAllocationCommentary);

        out.put("ownerEarnings", round2(ownerEarnings));
        out.put("ownerEarningsPerShare", round2(ownerEarningsPerShare));
        out.put("ownerEarningsYield", round4(ownerEarningsYield));
        out.put("freeCashFlow", round2(freeCashFlow));
        out.put("freeCashFlowPerShare", round2(freeCashFlowPerShare));
        out.put("freeCashFlowYield", round4(freeCashFlowYield));

        ObjectNode consistency = out.putObject("consistency");
        if (tenYearRevenueCAGR != null) consistency.put("tenYearRevenueCAGR", round4(tenYearRevenueCAGR));
        if (tenYearEpsCAGR != null)     consistency.put("tenYearEpsCAGR", round4(tenYearEpsCAGR));
        if (tenYearFcfCAGR != null)     consistency.put("tenYearFcfCAGR", round4(tenYearFcfCAGR));

        ObjectNode munger = out.putObject("mungerChecklist");
        munger.put("understandBusiness", understandBusiness);
        munger.put("durableMoat", durableMoat);
        munger.put("honestManagement", honestManagement);
        munger.put("fairPrice", fairPrice);
        munger.put("canHoldForever", canHoldForever);
        munger.put("score", mungerChecklistScore());

        ArrayNode rf = out.putArray("redFlags");
        for (String f : redFlags) rf.add(f);
        ArrayNode gf = out.putArray("greenFlags");
        for (String f : greenFlags) gf.add(f);

        return out;
    }

    private static double round2(double v) { return Double.isNaN(v) ? 0 : Math.round(v * 100.0) / 100.0; }
    private static double round4(double v) { return Double.isNaN(v) ? 0 : Math.round(v * 10000.0) / 10000.0; }
}
