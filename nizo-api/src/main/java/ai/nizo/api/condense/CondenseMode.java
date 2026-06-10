package ai.nizo.api.condense;

/**
 * What slice of the conversation a {@code condense} request operates over.
 *
 * <ul>
 *   <li>{@link #FULL} — summarize the whole conversation; result replaces all prior history.</li>
 *   <li>{@link #PARTIAL_FROM} — keep messages BEFORE the pivot index, summarize everything AFTER it.
 *       Useful when the head is well-formed context (e.g. an initial spec) and the tail has gone
 *       off on a tangent we want compacted.</li>
 *   <li>{@link #PARTIAL_UP_TO} — summarize messages BEFORE the pivot index, keep messages AFTER it.
 *       Useful when current work is in the recent tail and we want to compress earlier exploration.</li>
 * </ul>
 *
 * <p>For partial modes the {@code pivot index} is exclusive on the kept side
 * (i.e. {@code keepRange = [0, pivot)} for {@link #PARTIAL_FROM} and
 * {@code keepRange = [pivot, n)} for {@link #PARTIAL_UP_TO}).
 */
public enum CondenseMode {
    FULL,
    PARTIAL_FROM,
    PARTIAL_UP_TO
}
