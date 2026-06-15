package ai.nizo.agent.schedule;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import ai.nizo.scheduler.NaturalTimeParser;
import ai.nizo.scheduler.ParsedSchedule;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;

/**
 * Chat-facing scheduling: set / list / cancel reminders and recurring jobs. When a schedule fires,
 * its prompt runs through the agent and the result is delivered back to the chat that set it.
 */
public final class ScheduleTool implements Tool {

    private static final ObjectMapper M = new ObjectMapper();
    private static final DateTimeFormatter LOCAL = DateTimeFormatter.ofPattern("EEE d MMM HH:mm", Locale.ENGLISH);

    private final ScheduleStore store;
    private final ZoneId zone;

    public ScheduleTool(ScheduleStore store, ZoneId zone) {
        this.store = store;
        this.zone = zone;
    }

    @Override public String name() { return "schedule"; }

    @Override
    public String description() {
        return "Schedule a prompt to run later — once or recurring. Use whenever the user says "
                + "'remind me…', 'every morning…', 'in 2 hours…', 'each weekday at 8', etc. When it "
                + "fires, the prompt runs through you and the result is delivered to THIS chat. "
                + "Actions: set {when, prompt}; list; cancel {id}. 'when' accepts natural language "
                + "('tomorrow at 9am', 'every weekday at 8:30', 'in 30 minutes') or a 5-field cron "
                + "('0 9 * * 1-5'). Put the actual instruction to run later in 'prompt'.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "action": { "type": "string", "enum": ["set","list","cancel"], "description": "set a schedule, list active ones, or cancel by id." },
                "when":   { "type": "string", "description": "For set: when to run — natural language or a 5-field cron." },
                "prompt": { "type": "string", "description": "For set: the instruction to run when it fires (e.g. 'give me a market briefing for AAPL')." },
                "id":     { "type": "string", "description": "For cancel: the schedule id." }
              },
              "required": ["action"]
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        JsonNode a;
        try { a = M.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson); }
        catch (Exception e) { return ToolResult.error("bad arguments JSON: " + e.getMessage()); }

        String action = a.path("action").asText("").toLowerCase(Locale.ROOT).trim();
        return switch (action) {
            case "set" -> set(a);
            case "list" -> list();
            case "cancel" -> cancel(a.path("id").asText(""));
            default -> ToolResult.error("action must be set | list | cancel");
        };
    }

    private ToolResult set(JsonNode a) {
        String when = a.path("when").asText("").trim();
        String prompt = a.path("prompt").asText("").trim();
        if (when.isEmpty()) return ToolResult.error("set requires 'when'");
        if (prompt.isEmpty()) return ToolResult.error("set requires 'prompt' (the instruction to run later)");

        long now = System.currentTimeMillis();
        Optional<ParsedSchedule> parsed = NaturalTimeParser.parse(when, zone, now);
        if (parsed.isEmpty()) {
            return ToolResult.error("Couldn't understand the time \"" + when + "\". Try 'tomorrow at 9am', "
                    + "'every weekday at 8', 'in 30 minutes', or a 5-field cron like '0 9 * * 1-5'.");
        }
        ParsedSchedule p = parsed.get();
        if (p.firstFireMs() <= 0) return ToolResult.error("that schedule never fires — check the time/cron.");

        String id = "sch-" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        ScheduledTask task = new ScheduledTask(id, p.kind(), p.spec(), prompt,
                UserContext.currentChat(), UserContext.current(), p.firstFireMs(), 0, true, now);
        store.add(task);
        return ToolResult.ok("Scheduled (" + p.human() + ") — first run " + local(p.firstFireMs())
                + ". Prompt: \"" + prompt + "\". id=" + id + " (cancel with schedule/cancel).");
    }

    private ToolResult list() {
        List<ScheduledTask> tasks = store.listForUser(UserContext.current());
        if (tasks.isEmpty()) return ToolResult.ok("No active schedules.");
        StringBuilder sb = new StringBuilder("Active schedules:\n");
        for (ScheduledTask t : tasks) {
            sb.append("• ").append(t.id()).append("  [").append(t.kind()).append(' ').append(t.spec()).append("]  next ")
              .append(local(t.nextFireMs())).append("\n    \"").append(trim(t.prompt(), 100)).append("\"\n");
        }
        return ToolResult.ok(sb.toString());
    }

    private ToolResult cancel(String id) {
        if (id == null || id.isBlank()) return ToolResult.error("cancel requires an id (see schedule/list)");
        boolean ok = store.delete(id.trim());
        return ok ? ToolResult.ok("Cancelled schedule " + id) : ToolResult.error("no schedule with id " + id);
    }

    private String local(long ms) {
        return Instant.ofEpochMilli(ms).atZone(zone).format(LOCAL);
    }

    private static String trim(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n) + "…";
    }
}
