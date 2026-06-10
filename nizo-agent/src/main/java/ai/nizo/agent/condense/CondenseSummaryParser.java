package ai.nizo.agent.condense;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses the model's condense response. The contract from the prompt is:
 * <pre>
 *   &lt;analysis&gt;...thought process...&lt;/analysis&gt;
 *   &lt;summary&gt;1. Primary Request and Intent: ... 9. Optional Next Step: ...&lt;/summary&gt;
 * </pre>
 *
 * <p>We are forgiving:
 * <ul>
 *   <li>If the model emits {@code <summary>} but no closing tag (rare but possible at max-token cutoff),
 *       we take everything after the open tag.</li>
 *   <li>If the model omits the {@code <summary>} tags entirely (Qwen sometimes drops them when
 *       the response is short), we strip the analysis block and use the remainder.</li>
 *   <li>The analysis block is always discarded — it's chain-of-thought, not for the next agent.</li>
 * </ul>
 */
public final class CondenseSummaryParser {

    private static final Pattern SUMMARY_OPEN_CLOSE = Pattern.compile(
            "<summary>(.*?)</summary>", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    private static final Pattern SUMMARY_OPEN_ONLY  = Pattern.compile(
            "<summary>(.*)$", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    private static final Pattern ANALYSIS_BLOCK     = Pattern.compile(
            "<analysis>.*?</analysis>", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);

    private CondenseSummaryParser() {}

    /** @return the raw {@code <summary>} body, or {@code null} if no usable content. */
    public static String extractRawSummary(String modelResponse) {
        if (modelResponse == null || modelResponse.isBlank()) return null;

        Matcher m = SUMMARY_OPEN_CLOSE.matcher(modelResponse);
        if (m.find()) return m.group(1).trim();

        Matcher m2 = SUMMARY_OPEN_ONLY.matcher(modelResponse);
        if (m2.find()) return m2.group(1).trim();

        // No tags at all — strip analysis and return the rest.
        String stripped = ANALYSIS_BLOCK.matcher(modelResponse).replaceAll("").trim();
        return stripped.isEmpty() ? null : stripped;
    }

    /**
     * Wrap the raw summary in the fixed system-context preamble. The next conversation turn
     * sees this as the very first message of the new history.
     */
    public static String formatForReinjection(String rawSummary) {
        String body = rawSummary == null ? "" : rawSummary.trim();
        return """
                This session is being continued from a previous conversation that ran out of context. \
                The summary below covers the earlier portion of the conversation.

                Summary:
                """ + body;
    }
}
