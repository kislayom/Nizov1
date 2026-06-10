package ai.nizo.skills;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link SubAgentSkillTool#injectChartFences(String, Map, boolean)}.
 *
 * <p>This is the workhorse of the chart-fence-injection optimisation: the LLM emits a tiny
 * placeholder like {@code [CHART:chart-financials]} instead of retyping ~50KB of JSON, and the
 * post-process pass expands it into a real fenced block from the cached tool result. These
 * tests lock in the contract.
 */
class InjectChartFencesTest {

    @Test
    void emptyCache_returnsContentUnchanged() {
        String content = "Some report\n[CHART:chart-financials]\nMore.";
        assertEquals(content, SubAgentSkillTool.injectChartFences(content, new HashMap<>(), false));
    }

    @Test
    void nullCache_returnsContentUnchanged() {
        String content = "Some report\n[CHART:chart-financials]\nMore.";
        assertEquals(content, SubAgentSkillTool.injectChartFences(content, null, false));
    }

    @Test
    void nullContent_returnsEmptyString() {
        Map<String, String> cache = Map.of("chart-financials", "{}");
        // null + non-empty cache → just append the unused chart
        String out = SubAgentSkillTool.injectChartFences(null, cache, true);
        assertTrue(out.contains("```chart-financials"));
        assertTrue(out.contains("{}"));
    }

    @Test
    void singlePlaceholder_replacedWithFencedBlock() {
        Map<String, String> cache = Map.of(
                "chart-financials",
                "{\"ticker\":\"AAPL\",\"revenue\":391000000000}");
        String content = "## Income\n[CHART:chart-financials]\nApple revenue grew.";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-financials\n"
                + "{\"ticker\":\"AAPL\",\"revenue\":391000000000}\n```"));
        // Original placeholder line gone
        assertFalse(out.contains("[CHART:chart-financials]"));
        // Surrounding text preserved
        assertTrue(out.startsWith("## Income\n"));
        assertTrue(out.contains("Apple revenue grew."));
    }

    @Test
    void multiplePlaceholders_allReplaced() {
        Map<String, String> cache = new LinkedHashMap<>();
        cache.put("chart-financials", "{\"f\":1}");
        cache.put("chart-analyst",    "{\"a\":2}");
        cache.put("chart-earnings",   "{\"e\":3}");

        String content = "Intro\n[CHART:chart-financials]\nMid\n[CHART:chart-analyst]\nEnd\n[CHART:chart-earnings]\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-financials\n{\"f\":1}\n```"));
        assertTrue(out.contains("```chart-analyst\n{\"a\":2}\n```"));
        assertTrue(out.contains("```chart-earnings\n{\"e\":3}\n```"));
        assertFalse(out.contains("[CHART:"));
    }

    @Test
    void unknownPlaceholder_replacedWithFriendlyNotice() {
        Map<String, String> cache = Map.of("chart-financials", "{\"f\":1}");
        String content = "[CHART:chart-financials]\n[CHART:chart-nonexistent]\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-financials\n{\"f\":1}\n```"));
        // Unknown placeholder is replaced with an italic "chart unavailable" notice
        // (May 2026 change — leaving the raw [CHART:X] in production looked like a
        // templating bug. The friendly notice is much clearer to end users.)
        assertFalse(out.contains("[CHART:chart-nonexistent]"),
                "raw placeholder should be replaced, not surfaced to user");
        assertTrue(out.contains("nonexistent chart unavailable"),
                "should contain friendly 'unavailable' notice");
    }

    @Test
    void appendUnused_addsCachedChartsThatLLMDidntReference() {
        Map<String, String> cache = new LinkedHashMap<>();
        cache.put("chart-financials", "{\"f\":1}");
        cache.put("chart-analyst",    "{\"a\":2}");

        // LLM only referenced the financials placeholder
        String content = "Some text\n[CHART:chart-financials]\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, true);

        assertTrue(out.contains("```chart-financials\n{\"f\":1}\n```"));
        assertTrue(out.contains("```chart-analyst\n{\"a\":2}\n```"),
                "unused chart should be appended at the end");
        assertTrue(out.contains("_Data widgets:_"),
                "appended block should be labelled");
    }

    @Test
    void appendUnused_skippedWhenLLMReferencedAllCharts() {
        Map<String, String> cache = new LinkedHashMap<>();
        cache.put("chart-financials", "{\"f\":1}");

        String content = "Body\n[CHART:chart-financials]\nMore\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, true);

        // Should appear exactly once (in the placeholder location), not duplicated at the end
        int firstIdx = out.indexOf("```chart-financials");
        int secondIdx = out.indexOf("```chart-financials", firstIdx + 1);
        assertTrue(firstIdx >= 0, "should contain the fence once");
        assertEquals(-1, secondIdx, "should NOT contain the fence twice");
        assertFalse(out.contains("_Data widgets:_"));
    }

    @Test
    void blankCacheValueTreatedAsMissing_emitsFriendlyNotice() {
        Map<String, String> cache = new HashMap<>();
        cache.put("chart-financials", "");      // blank
        cache.put("chart-analyst",    "  ");    // whitespace
        String content = "[CHART:chart-financials]\n[CHART:chart-analyst]\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, true);

        // Both raw [CHART:X] placeholders are replaced with the friendly "unavailable" notice
        // (cache values are blank → treated same as "unknown placeholder" in the renderer).
        assertFalse(out.contains("[CHART:chart-financials]"),
                "blank-cache placeholder should not surface raw");
        assertFalse(out.contains("[CHART:chart-analyst]"),
                "blank-cache placeholder should not surface raw");
        assertTrue(out.contains("financials chart unavailable"),
                "should contain friendly 'unavailable' notice for financials");
        assertTrue(out.contains("analyst chart unavailable"),
                "should contain friendly 'unavailable' notice for analyst");
        // No appended canonical fences for blank values (appendUnused skips blank cache entries)
        assertFalse(out.contains("```chart-financials\n"));
        assertFalse(out.contains("```chart-analyst\n"));
    }

    @Test
    void jsonWithDollarSigns_doesNotBreakRegexReplacement() {
        // Matcher.appendReplacement uses $-substitution — quoteReplacement must be applied
        // to keep $1, $0 literals in the JSON safe.
        Map<String, String> cache = Map.of(
                "chart-financials",
                "{\"price\":\"$276.83\",\"footnote\":\"see $1 of the 10-K\"}");
        String content = "[CHART:chart-financials]";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("$276.83"));
        assertTrue(out.contains("see $1 of the 10-K"));
    }

    @Test
    void leadingWhitespaceOnPlaceholderLineStillMatches() {
        Map<String, String> cache = Map.of("chart-tech", "{\"signal\":\"BUY\"}");
        // Indented placeholder — should still match (regex uses ^[ \t]*)
        String content = "Section\n    [CHART:chart-tech]\nMore.\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-tech\n{\"signal\":\"BUY\"}\n```"));
        assertFalse(out.contains("[CHART:chart-tech]"));
    }

    @Test
    void inlinePlaceholder_NOT_matched_onlyWholeLines() {
        // We only match whole-line placeholders so a literal "[CHART:x]" in prose isn't munged.
        Map<String, String> cache = Map.of("chart-financials", "{\"f\":1}");
        String content = "Use the [CHART:chart-financials] syntax for things.\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        // Inline mention preserved
        assertTrue(out.contains("[CHART:chart-financials] syntax"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pass-2 rescue tests (malformed LLM-emitted fence attempts)
    //
    // These cover the May 2026 chrome-inspect bug where Qwen3.6 ignored SKILL.md and emitted
    // `chart-earnings {JSON}` with single backticks instead of either the [CHART:type]
    // placeholder OR a triple-backtick fence. The page rendered the JSON as inline text and
    // the underscores in "yahoo_quoteSummary_v10" got parsed as <em> emphasis.
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    void rescue_singleBacktickWrap_replacedWithCanonicalFence() {
        Map<String, String> cache = Map.of(
                "chart-earnings",
                "{\"ticker\":\"AAPL\",\"summary\":{\"beat\":4}}");
        String content = "Consensus: 32 of 49 analysts rate Buy.\n\n"
                + "`chart-earnings {\"ticker\":\"AAPL\",\"source\":\"yahoo_LLM_made_this_up_v10\"}`\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-earnings\n"
                + "{\"ticker\":\"AAPL\",\"summary\":{\"beat\":4}}\n```"),
                "should rewrite to canonical fence with CACHED JSON, not the LLM's mangled JSON");
        // The LLM's bad JSON should be GONE — replaced with cached authoritative.
        assertFalse(out.contains("yahoo_LLM_made_this_up_v10"),
                "the LLM's transcribed JSON should be discarded in favor of cached");
        // Original prose preserved
        assertTrue(out.contains("Consensus: 32 of 49 analysts rate Buy."));
    }

    @Test
    void rescue_noBackticks_replacedWithCanonicalFence() {
        Map<String, String> cache = Map.of("chart-financials", "{\"f\":1}");
        String content = "Income statement:\n\n"
                + "chart-financials {\"ticker\":\"AAPL\"}\n\n"
                + "Apple grew revenue.";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("```chart-financials\n{\"f\":1}\n```"));
        assertTrue(out.contains("Apple grew revenue."));
    }

    @Test
    void rescue_nestedJsonBraces_balancedScan() {
        Map<String, String> cache = Map.of(
                "chart-analyst",
                "{\"ticker\":\"AAPL\",\"summary\":{\"good\":true}}");
        String content = "`chart-analyst {\"a\":{\"b\":{\"c\":1}},\"d\":[{\"e\":2}]}`\n\nNext paragraph.";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        // Cached JSON used (LLM's nested object discarded)
        assertTrue(out.contains("```chart-analyst\n{\"ticker\":\"AAPL\",\"summary\":{\"good\":true}}\n```"));
        // The "Next paragraph." after the malformed block should still be present
        assertTrue(out.contains("Next paragraph."));
        // The LLM's mangled inner braces should be gone (no leftover "}}` or unterminated text)
        assertFalse(out.contains("\"b\":{\"c\":1}"), "LLM's bad inner braces should be replaced");
    }

    @Test
    void rescue_canonicalFence_passesThroughUntouched() {
        Map<String, String> cache = Map.of("chart-financials", "{\"cached\":1}");
        // A proper triple-backtick fence WITH the correct cached JSON already.
        String content = "```chart-financials\n{\"cached\":1}\n```\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        // Should NOT double-wrap — content unchanged (or trivially equivalent).
        // Count occurrences of the fence opening — should be exactly 1.
        int count = 0;
        int idx = 0;
        while ((idx = out.indexOf("```chart-financials", idx)) >= 0) {
            count++;
            idx += "```chart-financials".length();
        }
        assertEquals(1, count, "canonical fence should not be duplicated");
    }

    @Test
    void rescue_unbalancedBraces_leavesContentAlone() {
        Map<String, String> cache = Map.of("chart-financials", "{\"f\":1}");
        // Unclosed JSON — bail rather than chew unrelated text
        String content = "`chart-financials {\"open\":\"never closed\"\n\nNext section.";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        // Original text should still be there because we couldn't safely identify the end.
        assertTrue(out.contains("Next section."));
    }

    @Test
    void rescue_unknownChartType_leavesContentAlone() {
        Map<String, String> cache = Map.of("chart-financials", "{\"f\":1}");
        // chart-bogus isn't in the cache — leave it alone (visible bug > silent munge).
        String content = "`chart-bogus {\"x\":1}`";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        assertTrue(out.contains("chart-bogus"));
        assertTrue(out.contains("{\"x\":1}"));
    }

    @Test
    void rescue_realWorldBugReproduction() {
        // The exact pattern from the May 2026 chrome-inspect screenshot.
        Map<String, String> cache = Map.of(
                "chart-earnings",
                "{\"ticker\":\"AAPL\",\"source\":\"fmp\",\"summary\":{\"beatCount\":4}}");
        String content =
                "Consensus: 32 of 49 analysts rate Buy/Strong Buy. Mean target $316.67.\n\n"
                + "`chart-earnings {\"ticker\":\"AAPL\",\"source\":\"yahoo_quoteSummary_v10\","
                + "\"asOf\":\"2026-05-08\",\"history\":[{\"date\":\"2026-03-31\",\"period\":\"-1q\","
                + "\"epsEstimate\":1.94275,\"epsActual\":2.01,\"beat\":true}]}`\n";
        String out = SubAgentSkillTool.injectChartFences(content, cache, false);

        // The LLM's bad text gone, cached fence in
        assertFalse(out.contains("yahoo_quoteSummary_v10"));
        assertTrue(out.contains("```chart-earnings\n"
                + "{\"ticker\":\"AAPL\",\"source\":\"fmp\",\"summary\":{\"beatCount\":4}}\n```"));
        assertTrue(out.contains("Consensus: 32 of 49"));
    }

    @Test
    void toolToChartMap_coversAllStructuredFinanceTools() {
        // Lock in: every chart-emitting tool in the system is in the injection map.
        // If we add a new structured-data tool, this test reminds the author to register it.
        assertEquals("chart-financials",  SubAgentSkillTool.TOOL_TO_CHART.get("stock_fundamentals"));
        assertEquals("chart-analyst",     SubAgentSkillTool.TOOL_TO_CHART.get("stock_analyst_ratings"));
        assertEquals("chart-insider",     SubAgentSkillTool.TOOL_TO_CHART.get("stock_insider_activity"));
        assertEquals("chart-earnings",    SubAgentSkillTool.TOOL_TO_CHART.get("stock_earnings_history"));
        assertEquals("chart-interactive", SubAgentSkillTool.TOOL_TO_CHART.get("historical_price"));
        assertEquals("chart-tech",        SubAgentSkillTool.TOOL_TO_CHART.get("technical_indicators"));
    }
}
