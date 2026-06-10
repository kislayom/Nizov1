package ai.nizo.agent.condense;

import ai.nizo.api.condense.CondenseMode;

import java.util.List;
import java.util.Objects;

/**
 * Builds the verbatim {@code condense} prompt sent to the forked worker. The wording is fixed
 * by spec — DO NOT paraphrase. Section 8 ("Current Work") is rewritten to "Work Completed"
 * for {@link CondenseMode#PARTIAL_UP_TO}, since in {@code up_to} mode the messages we KEEP are
 * the recent/current work and the messages we SUMMARIZE are the historical phase.
 *
 * <p>The prompt is wrapped in {@code <analysis>} tags by the model itself (per the
 * "Output format" instruction inside the prompt), not by us.
 *
 * <p>Pre-condense hooks may contribute additional summarization instructions which we splice
 * into the "There may be additional summarization instructions provided in the included
 * context" sentence at the bottom of the prompt.
 */
public final class CondensePromptBuilder {

    private CondensePromptBuilder() {}

    /**
     * @param mode             determines section 8 wording (current work vs. work completed)
     * @param hookInstructions optional list of extra instructions from pre-condense hooks; nulls/blanks ignored
     */
    public static String build(CondenseMode mode, List<String> hookInstructions) {
        Objects.requireNonNull(mode, "mode");
        boolean upTo = mode == CondenseMode.PARTIAL_UP_TO;

        String section8 = upTo
                ? """
                  8. Work Completed: Describe in detail precisely what work has been completed in the \
                  portion of the conversation being summarized. Include file names and code snippets where applicable. \
                  Note that the conversation continues with messages kept after this summary — your job here is \
                  to capture the historical phase, not the current task.
                  """.replace('\n', ' ').trim()
                : """
                  8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, \
                  paying special attention to the most recent messages from both user and assistant. Include file names and \
                  code snippets where applicable.
                  """.replace('\n', ' ').trim();

        StringBuilder hookBlock = new StringBuilder();
        if (hookInstructions != null) {
            int idx = 0;
            for (String h : hookInstructions) {
                if (h == null || h.isBlank()) continue;
                if (idx == 0) hookBlock.append("\n\nAdditional summarization instructions:\n");
                hookBlock.append("- ").append(h.trim()).append('\n');
                idx++;
            }
        }

        return """
RESPOND WITH TEXT ONLY. DO NOT CALL ANY TOOLS. Tool calls will be REJECTED.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions. This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:
- Chronologically analyze each message and section of the conversation
- For each section thoroughly identify:
  - The user's explicit requests and intents
  - Your approach to addressing the user's requests
  - Key decisions, technical concepts and code patterns
  - Specific details like: file names, full code snippets, function signatures, file edits
  - Errors that you ran into and how you fixed them
- Pay special attention to specific user feedback that you received, especially if the user told you to do something differently
- Double check for technical accuracy and completeness

Your summary MUST include these sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on
%SECTION_8%
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Output format:
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>
<summary>
1. Primary Request and Intent: [...]
2. Key Technical Concepts: [...]
...
9. Optional Next Step: [...]
</summary>

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary.%HOOKS%

RESPOND WITH TEXT ONLY. DO NOT CALL ANY TOOLS.
"""
                .replace("%SECTION_8%", section8)
                .replace("%HOOKS%", hookBlock.toString());
    }
}
