package ai.nizo.api.condense;

import java.time.Instant;
import java.util.List;

/**
 * Outcome of a condense run.
 *
 * @param ok                 success / failure flag
 * @param mode               echo of the requested mode
 * @param trigger            echo of the trigger
 * @param messagesBefore     how many messages the chat had before condensing
 * @param messagesAfter      how many messages it has after (kept + 1 summary + re-injected)
 * @param tokensBefore       estimated token count before condensing
 * @param tokensAfter        estimated token count after condensing (summary + re-injected)
 * @param rawSummary         the raw {@code <summary>} block extracted from the model
 * @param formattedSummary   the wrapped summary as it was inserted back into history
 *                           (with the "this session is being continued..." preamble)
 * @param reinjectedFiles    paths re-attached after condensing
 * @param reinjectedSkills   skill names re-attached after condensing
 * @param durationMs         wall time of the whole condense operation
 * @param at                 when it finished
 * @param error              null if ok; otherwise short message
 */
public record CondenseResult(
        boolean ok,
        CondenseMode mode,
        CondenseRequest.Trigger trigger,
        int messagesBefore,
        int messagesAfter,
        int tokensBefore,
        int tokensAfter,
        String rawSummary,
        String formattedSummary,
        List<String> reinjectedFiles,
        List<String> reinjectedSkills,
        long durationMs,
        Instant at,
        String error
) {
    public CondenseResult {
        reinjectedFiles  = reinjectedFiles  == null ? List.of() : List.copyOf(reinjectedFiles);
        reinjectedSkills = reinjectedSkills == null ? List.of() : List.copyOf(reinjectedSkills);
    }

    public static CondenseResult failure(CondenseRequest req, String error, long durationMs) {
        return new CondenseResult(
                false, req.mode(), req.trigger(),
                0, 0, 0, 0,
                null, null, List.of(), List.of(),
                durationMs, Instant.now(), error
        );
    }
}
