package ai.nizo.api.condense;

/**
 * User-extensible hook around a condense operation.
 *
 * <p>Implementations are invoked twice per condense: {@link #preCondense} can inject extra
 * summarization instructions (returned strings are appended into the condense prompt's
 * "There may be additional summarization instructions" footer), and {@link #postCondense}
 * runs after the new history has been written back. Throw exceptions from either method
 * are swallowed so a misbehaving hook never breaks condensing.
 */
public interface CondenseHook {

    /**
     * @return text to inject into the condense prompt as additional instructions, or {@code null}
     *         to skip. Multiple hooks' return values are concatenated.
     */
    default String preCondense(CondenseRequest request) { return null; }

    default void postCondense(CondenseResult result) {}
}
