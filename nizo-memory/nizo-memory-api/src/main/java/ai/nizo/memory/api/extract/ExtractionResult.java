package ai.nizo.memory.api.extract;

import java.util.Map;
import java.util.Set;

/**
 * Output of a single extraction pass over one user message.
 *
 * <p>{@link #count} is the total number of discrete extractions found;
 * {@link #types} enumerates which categories were detected; and
 * {@link #raw} carries the structured extraction payload for downstream
 * graph and memory writers.
 *
 * @param count number of discrete extractions found
 * @param types categories detected in this pass
 * @param raw   structured extraction payload keyed by extraction id or label
 */
public record ExtractionResult(
        int count,
        Set<ExtractionCategory> types,
        Map<String, Object> raw
) {

    /** Returns an empty result indicating nothing was extracted. */
    public static ExtractionResult empty() {
        return new ExtractionResult(0, Set.of(), Map.of());
    }

    /** Returns {@code true} if at least one extraction was found. */
    public boolean hasExtractions() {
        return count > 0;
    }
}
