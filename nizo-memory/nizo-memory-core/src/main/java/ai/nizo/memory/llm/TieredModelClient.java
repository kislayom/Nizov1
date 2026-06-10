package ai.nizo.memory.llm;

import ai.nizo.memory.api.Modality;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelCapability;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.api.model.ModelResponse;
import ai.nizo.memory.util.Tokens;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Set;

/**
 * Two-tier {@link ModelClient} that routes each request to a small (cheap,
 * fast) backend below an input-token threshold and a big (capable, slower)
 * backend above it. Mirrors Mastra's {@code ModelByInputTokens} pattern.
 *
 * <p>Why this matters for nizo: the {@code SessionPicker}, extractor, LLM
 * gate, and reflection worker each call into a {@code ModelClient}. With a
 * single configured model they all pay the big-model price even when the
 * input is trivial (a 200-token manifest, a 100-token YES/NO prompt). Routing
 * lets routine ops use a fast model and reserves the big one for genuinely
 * hard reasoning, cutting cost ~5–10× on a typical workload.
 *
 * <p>The threshold is measured on the <em>concatenated message bodies</em>
 * via {@link Tokens#count}, which is the same approximation our recall
 * pipeline uses for budgeting — keeps the math consistent across the system.
 *
 * <h3>Capability surface</h3>
 * The decorator reports a synthesised {@link ModelCapability}: it inherits
 * the small model's latency hint and locality, but advertises whichever
 * model name routed the most recent call (initially "tiered:&lt;small&gt;|&lt;big&gt;").
 * Costs are conservatively reported as the big model's prices so callers
 * tracking spend never under-estimate.
 *
 * <h3>Bypass</h3>
 * Set {@link #ROUTING_BYPASS_PROPERTY} system property to {@code "small"} or
 * {@code "big"} to force every call through one tier — useful for benchmarks
 * where you want apples-to-apples comparisons without retuning the threshold.
 */
public final class TieredModelClient implements ModelClient {

    /** System property to force every call to the named tier. */
    public static final String ROUTING_BYPASS_PROPERTY = "nizo.tiered-model.force";

    private static final Logger log = LoggerFactory.getLogger(TieredModelClient.class);

    private final ModelClient small;
    private final ModelClient big;
    private final int thresholdTokens;

    /**
     * @param small             cheap/fast backend (e.g. qwen2.5:7b, gemini-flash)
     * @param big               capable/expensive backend (e.g. qwen3.6:35b, gpt-4o)
     * @param thresholdTokens   route to {@code big} when message-body tokens
     *                          exceed this; else {@code small}
     */
    public TieredModelClient(ModelClient small, ModelClient big, int thresholdTokens) {
        if (small == null) throw new IllegalArgumentException("small ModelClient required");
        if (big == null) throw new IllegalArgumentException("big ModelClient required");
        if (thresholdTokens < 1) throw new IllegalArgumentException("thresholdTokens must be ≥ 1");
        this.small = small;
        this.big = big;
        this.thresholdTokens = thresholdTokens;
    }

    @Override
    public ModelCapability capability() {
        ModelCapability sc = small.capability();
        ModelCapability bc = big.capability();
        // Synthesise: union of input/output modalities, max context, conservative
        // (big-model) costs, small-model latency hint, isLocal only if BOTH local.
        Set<Modality> inputs = new HashSet<>(sc.inputs());
        inputs.addAll(bc.inputs());
        Set<Modality> outputs = new HashSet<>(sc.outputs());
        outputs.addAll(bc.outputs());
        return new ModelCapability(
                "tiered:" + sc.id() + "|" + bc.id(),
                "tiered",
                inputs,
                outputs,
                Math.max(sc.maxContextTokens(), bc.maxContextTokens()),
                sc.supportsTools() || bc.supportsTools(),
                sc.isLocal() && bc.isLocal(),
                bc.usdPerMInput(),    // conservative — assume big model price
                bc.usdPerMOutput(),
                sc.latencyHintMs());
    }

    @Override
    public ModelResponse complete(ModelRequest request) {
        ModelClient routed = pick(request);
        return routed.complete(request);
    }

    /** Visible for tests. Decides which backend handles {@code request}. */
    ModelClient pick(ModelRequest request) {
        // System-property override wins — useful for benchmark sweeps.
        String forced = System.getProperty(ROUTING_BYPASS_PROPERTY);
        if ("small".equalsIgnoreCase(forced)) return small;
        if ("big".equalsIgnoreCase(forced)) return big;

        int tokens = inputTokens(request);
        boolean useBig = tokens > thresholdTokens;
        if (log.isDebugEnabled()) {
            log.debug("TieredModelClient routed to {} (input ≈ {} tokens, threshold {})",
                    useBig ? "big" : "small", tokens, thresholdTokens);
        }
        return useBig ? big : small;
    }

    /**
     * Approximate input size as the sum of message-body token counts. We
     * deliberately ignore tools/options metadata — they're constant overhead
     * per call and the threshold should reflect the variable user payload.
     */
    static int inputTokens(ModelRequest request) {
        if (request == null || request.messages() == null) return 0;
        int n = 0;
        for (Message m : request.messages()) {
            String text = m == null ? null : m.text();
            if (text != null) n += Tokens.count(text);
        }
        return n;
    }
}
