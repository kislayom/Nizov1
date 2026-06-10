package ai.nizo.memory.session;

import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.store.SqliteMemoryStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * {@link SessionPicker} backed by a small LLM. Given a user query and a
 * manifest of sessions (each with a short preview of its first turn), asks
 * the model to return the 1-indexed positions of the sessions most likely to
 * contain the answer. The returned positions are then mapped back to the
 * manifest's session IDs.
 *
 * <p>The prompt uses positions rather than raw session IDs so that opaque
 * identifiers (UUIDs, timestamps) don't eat prompt tokens or confuse the
 * parser. Output must be a comma-separated list of integers or the literal
 * token {@code NONE}; anything else is treated as an abstention.
 *
 * <p>Typical latency on a 3B-active-param MoE: ~300 ms for a 54-session
 * manifest (~5 KB prompt) at temperature 0. That's the all-in cost of the
 * pre-filter on every recall that crosses the session threshold.
 */
public final class LlmSessionPicker implements SessionPicker {

    private static final Logger log = LoggerFactory.getLogger(LlmSessionPicker.class);

    /** One-shot picker prompt. Positions are 1-indexed; NONE abstains. */
    private static final String PROMPT = """
            You are filtering conversation sessions to find the ones likely to contain
            the answer to a user's question.

            USER QUESTION:
            %s

            SESSIONS (numbered 1..N, each is a chronological snapshot):
            %s

            Return the numbers of the 1 to %d sessions most likely to contain the answer,
            separated by commas (e.g. "3, 7, 12"). If none look relevant, reply NONE.
            Output ONLY the numbers or NONE — no commentary, no prose.
            """;

    private static final Pattern NUMBER = Pattern.compile("\\b(\\d{1,3})\\b");

    private final ModelClient llm;
    private final int previewCharsPerSession;

    public LlmSessionPicker(ModelClient llm) {
        this(llm, 120);
    }

    /**
     * @param llm                     model client used for the pick call
     * @param previewCharsPerSession  hard cap on each session preview in the
     *                                prompt (runtime manifests may already be
     *                                shorter); keeps prompt size bounded for
     *                                large haystacks
     */
    public LlmSessionPicker(ModelClient llm, int previewCharsPerSession) {
        this.llm = llm;
        this.previewCharsPerSession = Math.max(40, previewCharsPerSession);
    }

    @Override
    public Set<String> pick(String query,
                             List<SqliteMemoryStore.SessionInfo> manifest,
                             int topN) {
        if (query == null || query.isBlank() || manifest == null || manifest.isEmpty()) {
            return Set.of();
        }
        if (manifest.size() == 1) {
            // Only one session — picker is a no-op; let the caller decide.
            return Set.of(manifest.get(0).sessionId());
        }
        int limit = Math.max(1, Math.min(topN, manifest.size()));

        StringBuilder sessionsBlock = new StringBuilder();
        for (int i = 0; i < manifest.size(); i++) {
            var s = manifest.get(i);
            String preview = s.preview() == null ? "" : s.preview();
            if (preview.length() > previewCharsPerSession) {
                preview = preview.substring(0, previewCharsPerSession) + "…";
            }
            sessionsBlock.append(String.format("[%d] (%d turns) \"%s\"%n",
                    i + 1, s.itemCount(), preview.replace('"', '\'')));
        }

        String prompt = PROMPT.formatted(query.strip(), sessionsBlock.toString(), limit);
        String response;
        try {
            response = llm.complete(ModelRequest.of(List.of(Message.user(prompt)))).text();
        } catch (RuntimeException e) {
            log.debug("Session picker LLM call failed; abstaining: {}", e.toString());
            return Set.of();
        }
        if (response == null) return Set.of();
        String trimmed = response.strip();
        if (trimmed.isEmpty() || trimmed.equalsIgnoreCase("NONE")) return Set.of();

        // Extract integer positions from the response; tolerate chattier models
        // that wrap the answer in prose (e.g. "I'd pick 3 and 7.") — we just
        // collect digits. Positions are 1-indexed and must map to a manifest row.
        Set<String> picks = new LinkedHashSet<>();
        Matcher m = NUMBER.matcher(trimmed);
        while (m.find() && picks.size() < limit) {
            int pos;
            try {
                pos = Integer.parseInt(m.group(1));
            } catch (NumberFormatException nfe) {
                continue;
            }
            if (pos >= 1 && pos <= manifest.size()) {
                picks.add(manifest.get(pos - 1).sessionId());
            }
        }
        return picks;
    }
}
