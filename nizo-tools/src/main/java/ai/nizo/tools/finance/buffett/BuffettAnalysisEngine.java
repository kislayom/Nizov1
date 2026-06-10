package ai.nizo.tools.finance.buffett;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Locale;

/**
 * Deterministic, no-LLM Buffett-Munger investment scorecard. Ported from Kimaya's
 * {@code BuffettAnalysisEngine} with the Spring/Lombok dependencies stripped.
 *
 * <h2>What it computes</h2>
 * Given a ticker's quote + key metrics + financial statements + company profile, produces:
 * <ul>
 *   <li><b>Moat score</b> 0-10 plus type (BRAND / NETWORK / SWITCHING / etc.) and trend</li>
 *   <li><b>Owner earnings</b> & FCF yield (the metrics Buffett actually uses)</li>
 *   <li><b>Intrinsic value</b> via weighted multi-method consensus (DCF + Growth-PE + 10-Cap + Graham)</li>
 *   <li><b>Margin of safety</b> with assessment band</li>
 *   <li><b>Capital allocation grade</b> A+/A/B/C/D — buybacks/dividends vs reinvestment quality</li>
 *   <li><b>Five component scores</b> (each 0-20): business quality / management / financial strength / valuation / growth</li>
 *   <li><b>Buffett score</b> 0-100 = sum of components</li>
 *   <li><b>Verdict</b>: Strong Buy / Buy / Watch / Pass + position sizing</li>
 *   <li><b>Buy / Strong Buy price</b> (intrinsic × 0.80 / 0.65)</li>
 *   <li><b>Munger checklist</b>: 5 boolean gates (understand / durable moat / honest mgmt / fair price / hold forever)</li>
 *   <li><b>Red & green flags</b>: explicit issue + strength lists</li>
 * </ul>
 *
 * <h2>Inputs as JsonNode</h2>
 * Rather than re-implement Kimaya's StockQuote/KeyMetrics/Profile/FinancialStatements POJOs,
 * this engine consumes the same field paths Nizo's existing tools already produce — Yahoo
 * {@code quoteSummary}-shaped JSON for fundamentals + a few extras we feed in. Keeps the
 * port thin.
 */
public final class BuffettAnalysisEngine {

    // ── Hard-coded thresholds (Kimaya defaults) ──────────────────────────
    private static final double MIN_ROE = 0.15;            // 15% — Buffett floor for "high quality"
    private static final double GROWTH_THRESHOLD = 0.08;   // 8% revenue growth = growth co
    private static final double VERY_HIGH_GROWTH = 0.20;   // 20% = A-grade reinvestment
    private static final double HIGH_GROWTH_THRESHOLD = 0.15;

    private static final double FAIR_PE_MIN = 12.0;
    private static final double FAIR_PE_MAX = 30.0;
    private static final double GROWTH_PE_MULTIPLIER = 150.0;

    private static final double CAPITAL_RETURN_A_PLUS = 0.80;
    private static final double CAPITAL_RETURN_A      = 0.60;
    private static final double CAPITAL_RETURN_B      = 0.40;
    private static final double CAPITAL_RETURN_C      = 0.20;

    private static final double CAPEX_MAINTENANCE_RATIO = 0.70;  // 70% of total CapEx is maintenance

    private BuffettAnalysisEngine() { /* no instances */ }

    /**
     * Run the full Buffett scorecard given pre-fetched data.
     *
     * @param quote        quote module (price, marketCap, currency)  — Yahoo {@code price} or FMP profile shape
     * @param metrics      key-metrics + ratios — fields like roe, debtToEquity, peRatio, ...
     * @param income       array of annual income statements newest-first (revenue, netIncome, eps, sharesOutstanding)
     * @param cashFlow     array of annual cash-flow statements newest-first (FCF, dividends, repurchase, capex)
     * @param balanceSheet array of annual balance sheets newest-first (totalEquity, totalDebt, cash) — used for some metrics
     * @param profile      profile/assetProfile (sector, industry, description)
     * @param dcfPerShare  optional external DCF intrinsic value per share, NaN if not provided
     * @return populated {@link BuffettMetrics}
     */
    public static BuffettMetrics analyze(JsonNode quote, JsonNode metrics, JsonNode income,
                                         JsonNode cashFlow, JsonNode balanceSheet, JsonNode profile,
                                         double dcfPerShare) {
        BuffettMetrics m = new BuffettMetrics();

        // ── Pull primitives we'll reference often ────────────────────────────
        double price        = num(quote, "currentPrice", num(quote, "regularMarketPrice", 0));
        double marketCap    = num(quote, "marketCap", 0);
        double roe          = num(metrics, "returnOnEquity", num(metrics, "roe", 0));
        double roic         = num(metrics, "returnOnInvestedCapital", num(metrics, "roic", 0));
        double grossMargin  = num(metrics, "grossMargins", num(metrics, "grossMargin", 0));
        double opMargin     = num(metrics, "operatingMargins", num(metrics, "operatingMargin", 0));
        double netMargin    = num(metrics, "profitMargins", num(metrics, "netMargin", num(metrics, "netProfitMargin", 0)));
        double pe           = num(metrics, "trailingPE", num(metrics, "peRatio", 0));
        double debtEquity   = num(metrics, "debtToEquity", 0);
        // FMP returns debtToEquity as fraction × 100 sometimes (e.g. "79.548"); detect.
        if (debtEquity > 5) debtEquity /= 100.0;
        double currentRatio = num(metrics, "currentRatio", 0);
        double interestCov  = num(metrics, "interestCoverageRatio", num(metrics, "interestCoverage", 0));
        double eps          = num(metrics, "earningsPerShare", num(metrics, "eps", 0));
        double bvps         = num(metrics, "bookValuePerShare", 0);
        double divYield     = num(metrics, "dividendYield", 0);
        double revGrowth    = num(metrics, "revenueGrowth", 0);
        double earnGrowth   = num(metrics, "earningsGrowth", 0);

        m.currentPrice = price;

        // ── 1. Owner earnings & FCF yields ───────────────────────────────────
        double netIncomeLatest = arrField(income, 0, "netIncome");
        long sharesOutstanding = (long) arrField(income, 0, "sharesOutstanding");
        if (sharesOutstanding <= 0 && marketCap > 0 && price > 0) {
            sharesOutstanding = (long) (marketCap / price);
        }
        double capexLatest = Math.abs(arrField(cashFlow, 0, "capitalExpenditures"));
        double maintenanceCapex = capexLatest * CAPEX_MAINTENANCE_RATIO;
        m.ownerEarnings = netIncomeLatest - maintenanceCapex;   // depreciation usually offset by capex; simplify
        m.ownerEarningsPerShare = sharesOutstanding > 0 ? m.ownerEarnings / sharesOutstanding : 0;
        m.ownerEarningsYield = marketCap > 0 ? m.ownerEarnings / marketCap : 0;
        m.freeCashFlow = arrField(cashFlow, 0, "freeCashFlow");
        if (m.freeCashFlow == 0) m.freeCashFlow = arrField(cashFlow, 0, "operatingCashFlow") - capexLatest;
        m.freeCashFlowPerShare = sharesOutstanding > 0 ? m.freeCashFlow / sharesOutstanding : 0;
        m.freeCashFlowYield = price > 0 ? m.freeCashFlowPerShare / price : 0;

        // ── 2. Moat ──────────────────────────────────────────────────────────
        m.moatScore = scoreMoat(roe, grossMargin, opMargin, roic, debtEquity, pe);
        m.moatType = classifyMoatType(profile, grossMargin, roe);
        m.moatTrend = classifyMoatTrend(revGrowth);
        m.moatEvidence = buildMoatEvidence(m.moatScore, m.moatType, roe, grossMargin, opMargin);

        // ── 3. Long-term consistency CAGRs ───────────────────────────────────
        m.tenYearRevenueCAGR = cagr(income, "revenue");
        m.tenYearEpsCAGR     = cagr(income, "eps");
        if (m.tenYearEpsCAGR == null) m.tenYearEpsCAGR = cagr(income, "dilutedEPS");
        m.tenYearFcfCAGR     = cagr(cashFlow, "freeCashFlow");

        // ── 4. Intrinsic value via weighted multi-method consensus ──────────
        boolean isGrowthCompany = revGrowth > GROWTH_THRESHOLD;
        computeValuation(m, eps, bvps, dcfPerShare, revGrowth, isGrowthCompany);

        // ── 5. Margin of safety ─────────────────────────────────────────────
        if (m.averageIntrinsicValue > 0) {
            m.marginOfSafety = (m.averageIntrinsicValue - price) / m.averageIntrinsicValue;
        }
        m.marginOfSafetyAssessment =
                m.marginOfSafety >= 0.35 ? "Excellent" :
                m.marginOfSafety >= 0.20 ? "Adequate" :
                m.marginOfSafety >= 0.00 ? "Minimal" :
                m.marginOfSafety >= -0.20 ? "Overvalued" : "Significantly Overvalued";

        // ── 6. Capital allocation grade ─────────────────────────────────────
        gradeCapitalAllocation(m, cashFlow, revGrowth, isGrowthCompany);

        // ── 7. Component scores 0-20 each ───────────────────────────────────
        m.businessQualityScore   = scoreBusinessQuality(m.moatScore, roe);
        m.managementQualityScore = scoreManagement(m.capitalAllocationGrade, roic, revGrowth);
        m.financialStrengthScore = scoreFinancialStrength(currentRatio, debtEquity, interestCov);
        m.valuationScore         = scoreValuation(m.marginOfSafety);
        m.growthProspectsScore   = scoreGrowth(revGrowth, earnGrowth);
        m.buffettScore = m.businessQualityScore + m.managementQualityScore + m.financialStrengthScore
                + m.valuationScore + m.growthProspectsScore;

        // ── 8. Verdict + buy prices + position sizing ───────────────────────
        if (m.averageIntrinsicValue > 0) {
            m.buyPrice = m.averageIntrinsicValue * 0.80;
            m.strongBuyPrice = m.averageIntrinsicValue * 0.65;
        }
        m.verdict = pickVerdict(m.buffettScore, m.marginOfSafety, m.moatScore);
        m.positionSizing = pickPositionSizing(m.buffettScore, m.moatScore);

        // ── 9. Red / Green flags ────────────────────────────────────────────
        addFlags(m, debtEquity, currentRatio, pe, roe, netMargin);

        // ── 10. Munger checklist ────────────────────────────────────────────
        m.understandBusiness = profile != null && !text(profile, "longBusinessSummary",
                text(profile, "description", "")).isBlank();
        m.durableMoat = m.moatScore >= 7;
        m.honestManagement = m.capitalAllocationGrade.startsWith("A") || m.capitalAllocationGrade.equals("B");
        m.fairPrice = m.marginOfSafety > 0;
        m.canHoldForever = m.moatScore >= 6 && m.redFlags.size() <= 2;

        return m;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Computation helpers
    // ───────────────────────────────────────────────────────────────────────

    private static int scoreMoat(double roe, double grossMargin, double opMargin,
                                 double roic, double debtEquity, double pe) {
        int s = 5;
        if (roe > 0.15)         s += 2;
        if (grossMargin > 0.40) s += 1;
        if (opMargin > 0.20)    s += 1;
        if (roic > 0.15)        s += 1;
        if (debtEquity < 0.5)   s += 1;
        if (pe > 50)            s -= 1;
        return Math.max(0, Math.min(10, s));
    }

    private static String classifyMoatType(JsonNode profile, double grossMargin, double roe) {
        String sector = text(profile, "sector", "").toLowerCase(Locale.ROOT);
        String industry = text(profile, "industry", "").toLowerCase(Locale.ROOT);
        if (grossMargin > 0.60) return "BRAND_INTANGIBLES";
        if (sector.contains("technology") && roe > 0.25) return "NETWORK_EFFECT";
        if (industry.contains("software") || industry.contains("services")) return "SWITCHING_COSTS";
        if (sector.contains("consumer")) return "BRAND";
        return "MIXED";
    }

    private static String classifyMoatTrend(double revGrowth) {
        if (revGrowth > 0.10) return "WIDENING";
        if (revGrowth < 0)    return "NARROWING";
        return "STABLE";
    }

    private static String buildMoatEvidence(int moatScore, String moatType,
                                            double roe, double grossMargin, double opMargin) {
        StringBuilder sb = new StringBuilder();
        sb.append("Moat ").append(moatScore).append("/10 (").append(moatType).append(")");
        if (roe > 0)         sb.append(" · ROE ").append(pct1(roe));
        if (grossMargin > 0) sb.append(" · gross ").append(pct1(grossMargin));
        if (opMargin > 0)    sb.append(" · operating ").append(pct1(opMargin));
        return sb.toString();
    }

    private static void computeValuation(BuffettMetrics m, double eps, double bvps,
                                         double dcfPerShare, double revGrowth, boolean isGrowth) {
        double weightedSum = 0;
        double weightTotal = 0;

        // Method 1: DCF (50% weight if available)
        m.dcfIntrinsicValue = (Double.isNaN(dcfPerShare) || dcfPerShare <= 0) ? 0 : dcfPerShare;
        if (m.dcfIntrinsicValue > 0) {
            double w = 50;
            weightedSum += m.dcfIntrinsicValue * w;
            weightTotal += w;
            m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                    "DCF", m.dcfIntrinsicValue, w, "External DCF model", false, null));
        }

        // Method 2: Growth-Adjusted P/E (always)
        if (eps > 0) {
            double fairPe = Math.max(FAIR_PE_MIN, Math.min(FAIR_PE_MAX, revGrowth * GROWTH_PE_MULTIPLIER));
            if (fairPe < FAIR_PE_MIN) fairPe = FAIR_PE_MIN;
            m.growthAdjustedPeValue = fairPe * eps;
            double w = 30;
            weightedSum += m.growthAdjustedPeValue * w;
            weightTotal += w;
            m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                    "Growth-Adjusted PE", m.growthAdjustedPeValue, w,
                    String.format(Locale.ROOT, "Fair PE %.1f × EPS %.2f", fairPe, eps),
                    false, null));
        }

        // Method 3: 10-Cap (10% weight, only for mature)
        if (m.ownerEarningsPerShare > 0) {
            m.tenCapValue = m.ownerEarningsPerShare * 10;
            if (!isGrowth) {
                double w = 10;
                weightedSum += m.tenCapValue * w;
                weightTotal += w;
                m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                        "10-Cap", m.tenCapValue, w,
                        String.format(Locale.ROOT, "Owner Earnings/Share %.2f × 10", m.ownerEarningsPerShare),
                        false, null));
            } else {
                m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                        "10-Cap", m.tenCapValue, 0,
                        String.format(Locale.ROOT, "Owner Earnings/Share %.2f × 10", m.ownerEarningsPerShare),
                        true, "Excluded — high-growth company (revenue growth > 8%)"));
            }
        }

        // Method 4: Graham Number (10% weight, only for value)
        if (eps > 0 && bvps > 0) {
            m.grahamNumberValue = Math.sqrt(22.5 * eps * bvps);
            if (!isGrowth) {
                double w = 10;
                weightedSum += m.grahamNumberValue * w;
                weightTotal += w;
                m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                        "Graham Number", m.grahamNumberValue, w,
                        String.format(Locale.ROOT, "√(22.5 × EPS %.2f × BVPS %.2f)", eps, bvps),
                        false, null));
            } else {
                m.valuationMethodsUsed.add(new BuffettMetrics.ValuationMethod(
                        "Graham Number", m.grahamNumberValue, 0,
                        String.format(Locale.ROOT, "√(22.5 × EPS %.2f × BVPS %.2f)", eps, bvps),
                        true, "Excluded — high-growth company (revenue growth > 8%)"));
            }
        }

        m.averageIntrinsicValue = weightTotal > 0 ? Math.round((weightedSum / weightTotal) * 100.0) / 100.0 : 0;
    }

    private static void gradeCapitalAllocation(BuffettMetrics m, JsonNode cashFlow, double revGrowth, boolean isGrowth) {
        if (revGrowth > VERY_HIGH_GROWTH) {
            m.capitalAllocationGrade = "A";
            m.capitalAllocationCommentary = String.format(Locale.ROOT,
                    "High-growth (%s rev) — earnings appropriately reinvested in expansion.",
                    pct1(revGrowth));
            return;
        }
        if (isGrowth && revGrowth > HIGH_GROWTH_THRESHOLD) {
            m.capitalAllocationGrade = "B";
            m.capitalAllocationCommentary = String.format(Locale.ROOT,
                    "Growth phase (%s rev) — reinvesting cash; early to judge return profile.",
                    pct1(revGrowth));
            return;
        }
        // Mature path
        double fcf = m.freeCashFlow > 0 ? m.freeCashFlow : 1;     // avoid /0
        double dividends = Math.abs(arrField(cashFlow, 0, "dividendsPaid"));
        double repurchase = Math.abs(arrField(cashFlow, 0, "repurchaseOfStock"));
        if (repurchase == 0) repurchase = Math.abs(arrField(cashFlow, 0, "shareRepurchases"));
        double returnRatio = (dividends + repurchase) / fcf;
        if (returnRatio >= CAPITAL_RETURN_A_PLUS) m.capitalAllocationGrade = "A+";
        else if (returnRatio >= CAPITAL_RETURN_A) m.capitalAllocationGrade = "A";
        else if (returnRatio >= CAPITAL_RETURN_B) m.capitalAllocationGrade = "B";
        else if (returnRatio >= CAPITAL_RETURN_C) m.capitalAllocationGrade = "C";
        else m.capitalAllocationGrade = "D";
        m.capitalAllocationCommentary = String.format(Locale.ROOT,
                "Returned %s of FCF to shareholders via buybacks + dividends.", pct1(returnRatio));
    }

    private static int scoreBusinessQuality(int moatScore, double roe) {
        int s = moatScore * 2;        // 0-20
        if (roe > 0.15) s = Math.min(20, s + 2);
        return s;
    }

    private static int scoreManagement(String capitalGrade, double roic, double revGrowth) {
        int s = switch (capitalGrade) {
            case "A+" -> 18;
            case "A"  -> 15;
            case "B"  -> 12;
            case "C"  -> 8;
            default   -> 5;
        };
        if (roic > 0.15) s = Math.min(20, s + 3);
        if (revGrowth > 0.15) s = Math.max(s, 12);    // growth-co floor
        return Math.min(20, s);
    }

    private static int scoreFinancialStrength(double currentRatio, double debtEquity, double interestCoverage) {
        int s = 10;
        if (currentRatio > 2.0) s += 4;
        else if (currentRatio > 1.5) s += 2;
        else if (currentRatio < 1.0) s -= 4;

        if (debtEquity < 0.3) s += 4;
        else if (debtEquity < 0.5) s += 2;
        else if (debtEquity > 1.5) s -= 4;

        if (interestCoverage > 5) s += 2;
        return Math.max(0, Math.min(20, s));
    }

    private static int scoreValuation(double mos) {
        if (mos >= 0.35) return 20;
        if (mos >= 0.25) return 16;
        if (mos >= 0.15) return 12;
        if (mos >= 0.00) return 8;
        if (mos >= -0.15) return 5;
        return 2;
    }

    private static int scoreGrowth(double revGrowth, double earnGrowth) {
        int s = 10;
        if (revGrowth > 0.15) s += 4;
        else if (revGrowth > 0.05) s += 2;
        else if (revGrowth < 0) s -= 3;

        if (earnGrowth > 0.15) s += 4;
        else if (earnGrowth > 0.05) s += 2;
        else if (earnGrowth < 0) s -= 3;
        return Math.max(0, Math.min(20, s));
    }

    private static String pickVerdict(int buffettScore, double mos, int moatScore) {
        if (buffettScore >= 80 && mos > 0.25) return "Strong Buy";
        if (buffettScore >= 70 && mos > 0)    return "Buy";
        if (buffettScore >= 60 && moatScore >= 6) return "Watch";
        return "Pass";
    }

    private static String pickPositionSizing(int buffettScore, int moatScore) {
        if (buffettScore >= 80 && moatScore >= 8) return "Up to 5% of portfolio";
        if (buffettScore >= 70) return "2-3% of portfolio";
        if (buffettScore >= 60) return "1-2% of portfolio";
        return "Avoid";
    }

    private static void addFlags(BuffettMetrics m, double debtEquity, double currentRatio,
                                 double pe, double roe, double netMargin) {
        if (debtEquity > 1.5)   m.redFlags.add("High leverage (D/E " + dec2(debtEquity) + ")");
        if (currentRatio < 1.0) m.redFlags.add("Liquidity tight (current ratio " + dec2(currentRatio) + ")");
        if (pe > 50)            m.redFlags.add("Premium valuation (P/E " + dec2(pe) + ")");
        if (roe > 0 && roe < 0.10) m.redFlags.add("Mediocre returns (ROE " + pct1(roe) + ")");

        if (roe > 0.20)         m.greenFlags.add("Excellent returns (ROE " + pct1(roe) + ")");
        if (netMargin > 0.15)   m.greenFlags.add("Strong net margins (" + pct1(netMargin) + ")");
        if (debtEquity < 0.3 && debtEquity > 0) m.greenFlags.add("Conservative leverage (D/E " + dec2(debtEquity) + ")");
        if (m.moatScore >= 8)   m.greenFlags.add("Wide economic moat");
    }

    // ───────────────────────────────────────────────────────────────────────
    // Field-extraction helpers — handle both Yahoo-shaped {raw, fmt} and bare numbers
    // ───────────────────────────────────────────────────────────────────────

    /** Extract a numeric field. Handles Yahoo's {raw, fmt} wrapper, bare numbers, and missing. */
    static double num(JsonNode parent, String field, double fallback) {
        if (parent == null || parent.isMissingNode() || parent.isNull()) return fallback;
        JsonNode n = parent.path(field);
        if (n.isMissingNode() || n.isNull()) return fallback;
        if (n.isObject()) {
            JsonNode raw = n.path("raw");
            if (!raw.isMissingNode() && !raw.isNull()) return raw.asDouble(fallback);
            return fallback;
        }
        return n.asDouble(fallback);
    }

    /** Get an array element's field — used for income/cashFlow rows that are lists. */
    static double arrField(JsonNode arr, int idx, String field) {
        if (arr == null || !arr.isArray() || idx >= arr.size()) return 0;
        return num(arr.get(idx), field, 0);
    }

    static String text(JsonNode parent, String field, String fallback) {
        if (parent == null || parent.isMissingNode() || parent.isNull()) return fallback;
        JsonNode n = parent.path(field);
        if (n.isMissingNode() || n.isNull()) return fallback;
        return n.asText(fallback);
    }

    /** Compute compound annual growth rate over an income/cashflow array. Returns null if &lt;3 rows. */
    static Double cagr(JsonNode arr, String field) {
        if (arr == null || !arr.isArray() || arr.size() < 3) return null;
        // arr is newest-first per Kimaya/our convention; oldest is last index.
        int last = arr.size() - 1;
        double newest = num(arr.get(0), field, 0);
        double oldest = num(arr.get(last), field, 0);
        if (newest <= 0 || oldest <= 0) return null;
        int years = last;     // arr.get(0) → arr.get(last) is `last` years apart
        double ratio = newest / oldest;
        return Math.pow(ratio, 1.0 / years) - 1.0;
    }

    private static String pct1(double v) {
        return String.format(Locale.ROOT, "%.1f%%", v * 100.0);
    }
    private static String dec2(double v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }
}
