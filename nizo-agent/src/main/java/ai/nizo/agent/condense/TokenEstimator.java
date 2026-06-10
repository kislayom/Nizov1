package ai.nizo.agent.condense;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.llm.ToolDef;

import java.util.List;

/**
 * Heuristic token counter. We don't ship Qwen's BPE tokenizer in-process (it's HF-Python and
 * adds ~30 MB to the uber-jar for marginal value here), so we approximate with a character
 * ratio calibrated against Qwen3 traces:
 * <ul>
 *   <li>~3.5 chars / token for English prose</li>
 *   <li>~3.0 chars / token for code and JSON (denser)</li>
 * </ul>
 * We use 3.5 as a slight over-estimate so the condense trigger fires <em>earlier</em> than the
 * true context limit — this is the conservative direction (better to condense a turn early than
 * to OOM the prompt).
 *
 * <p>For images we count an explicit Qwen3-VL approximation: the projector emits roughly
 * {@code (height/14) * (width/14)} visual tokens, capped at the resize bounds. We treat each
 * data URI as 768 visual tokens (typical 1024×1024 image after our 2048-cap normalize).
 */
public final class TokenEstimator {

    /** Average chars per token for our calibration. Conservative (over-estimates). */
    public static final double CHARS_PER_TOKEN = 3.5;

    /** Per-image token cost. A safe over-estimate for ≤1024×1024 images on Qwen3-VL. */
    public static final int VISUAL_TOKENS_PER_IMAGE = 768;

    /** Per-message overhead — chat templates add role markers and separators. */
    public static final int PER_MESSAGE_OVERHEAD = 4;

    public static int estimate(String s) {
        if (s == null || s.isEmpty()) return 0;
        return (int) Math.ceil(s.length() / CHARS_PER_TOKEN);
    }

    public static int estimate(ChatMessage m) {
        if (m == null) return 0;
        int n = PER_MESSAGE_OVERHEAD;
        n += estimate(m.content());
        if (m.hasImages()) n += m.images().size() * VISUAL_TOKENS_PER_IMAGE;
        for (ToolCall tc : m.toolCalls()) {
            n += estimate(tc.name()) + estimate(tc.argumentsJson()) + 4;
        }
        if (m.toolCallId() != null) n += estimate(m.toolCallId());
        if (m.name() != null) n += estimate(m.name());
        return n;
    }

    public static int estimateMessages(List<ChatMessage> ms) {
        if (ms == null || ms.isEmpty()) return 0;
        int n = 0;
        for (ChatMessage m : ms) n += estimate(m);
        return n;
    }

    public static int estimateTools(List<ToolDef> tools) {
        if (tools == null || tools.isEmpty()) return 0;
        int n = 0;
        for (ToolDef t : tools) {
            n += estimate(t.name()) + estimate(t.description()) + estimate(t.parametersJsonSchema()) + 8;
        }
        return n;
    }

    private TokenEstimator() {}
}
