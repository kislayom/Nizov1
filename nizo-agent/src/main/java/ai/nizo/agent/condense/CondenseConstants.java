package ai.nizo.agent.condense;

/**
 * Knobs for the condense subsystem. Names follow the spec verbatim — don't rename without
 * updating both the engine and the tests that reference them.
 *
 * <p>The default {@link #DEFAULT_EFFECTIVE_CONTEXT} matches our llama.cpp serve config
 * ({@code --ctx-size 262144}). Override at runtime via {@code NIZO_CTX_SIZE}.
 */
public final class CondenseConstants {

    /** Headroom (in tokens) we keep below the model's context window before triggering condense. */
    public static final int CONDENSE_BUFFER_TOKENS = 13_000;

    /** Circuit breaker — stop attempting condenses after this many consecutive failures. */
    public static final int CONDENSE_MAX_FAILURES = 3;

    /** Total tokens budgeted for re-injecting recently-read files. */
    public static final int REINJECT_FILE_BUDGET = 50_000;
    /** Per-file cap on re-injection. */
    public static final int REINJECT_FILE_PER_FILE = 5_000;
    /** Hard cap on the number of files re-injected. */
    public static final int REINJECT_FILE_MAX_COUNT = 5;

    /** Total tokens budgeted for re-injecting active skills. */
    public static final int REINJECT_SKILL_BUDGET = 25_000;
    /** Per-skill cap on re-injection. */
    public static final int REINJECT_SKILL_PER_SKILL = 5_000;

    /** Default effective context window — overridden by {@code NIZO_CTX_SIZE}. */
    public static final int DEFAULT_EFFECTIVE_CONTEXT = 262_144;

    /**
     * Effective context window honoring {@code NIZO_CTX_SIZE}. We resolve once per call (cheap)
     * to avoid stale reads if the env changes between runs.
     */
    public static int effectiveContextWindow() {
        String v = System.getenv("NIZO_CTX_SIZE");
        if (v == null || v.isBlank()) return DEFAULT_EFFECTIVE_CONTEXT;
        try { return Math.max(8_192, Integer.parseInt(v.trim())); }
        catch (NumberFormatException e) { return DEFAULT_EFFECTIVE_CONTEXT; }
    }

    /** Token threshold above which the auto-condense should fire. */
    public static int autoCondenseThreshold() {
        return effectiveContextWindow() - CONDENSE_BUFFER_TOKENS;
    }

    private CondenseConstants() {}
}
