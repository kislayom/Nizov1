package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.time.Month;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * India event calendar — Phase 4 of the "India Picks" build.
 *
 * <p>Returns a forward-looking calendar of macro / political / monetary events
 * that historically move Indian equity. The {@code india_top_picks} orchestrator
 * uses this to bias picks AWAY from sectors with imminent event risk (e.g. tilt
 * down financials if RBI meeting is within 7 days; tilt down capex if election
 * is within 30 days; tilt up FMCG ahead of Union Budget if government likely to
 * announce rural / consumption-focused outlays).
 *
 * <p>Sources:
 * <ul>
 *   <li><b>RBI MPC meetings</b> — bi-monthly, the next 12 months are deterministic.
 *       Hardcoded with annual review.</li>
 *   <li><b>Union Budget</b> — Feb 1 each year (vote-on-account in election years).</li>
 *   <li><b>General elections</b> — last held April 2024; next 2029.</li>
 *   <li><b>Major state elections</b> — varies. We track the next few using a static
 *       schedule (refreshed annually).</li>
 *   <li><b>Earnings windows</b> — quarterly, Indian companies report on a
 *       Mar/Jun/Sep/Dec quarter-end with results 6-8 weeks later. The window we mark
 *       is the most active 6-week period after each quarter.</li>
 * </ul>
 *
 * <p>Output: a list of events sorted by date, plus a {@code nextRiskWindow} object
 * highlighting the single most-imminent event affecting picks.
 *
 * <p>No external network calls — entirely curated. Update the static schedule
 * annually or when new RBI / election dates are announced.
 */
public final class IndiaEventCalendarTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaEventCalendarTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final DateTimeFormatter ISO = DateTimeFormatter.ISO_LOCAL_DATE;

    /** Calendar entries — date, type, label, sector impact tags, severity (1-3). */
    private static final List<Event> SCHEDULE = List.of(
            // RBI MPC meetings — bi-monthly, 6/year. Forward-looking 12 months.
            new Event(LocalDate.of(2026, 6, 6),  "rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            new Event(LocalDate.of(2026, 8, 8),  "rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            new Event(LocalDate.of(2026, 10, 10),"rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            new Event(LocalDate.of(2026, 12, 5), "rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            new Event(LocalDate.of(2027, 2, 6),  "rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            new Event(LocalDate.of(2027, 4, 7),  "rbi-mpc", "RBI MPC policy meeting",
                    new String[]{"bank", "psu bank", "realty"}, 2),
            // Union Budget — Feb 1 annually.
            new Event(LocalDate.of(2027, 2, 1),  "budget", "Union Budget 2027-28",
                    new String[]{"infrastructure", "auto", "fmcg", "defence", "psu"}, 3),
            // Election cycle — next general election expected 2029. State elections in between.
            new Event(LocalDate.of(2026, 10, 15),"state-election", "Bihar Assembly election (expected)",
                    new String[]{"fmcg", "rural", "agri"}, 1),
            new Event(LocalDate.of(2026, 11, 30),"state-election", "Bengal Assembly election (expected)",
                    new String[]{"fmcg", "rural"}, 1),
            new Event(LocalDate.of(2027, 3, 15), "state-election", "UP Assembly election (expected)",
                    new String[]{"fmcg", "rural", "auto"}, 2),
            new Event(LocalDate.of(2029, 4, 1),  "general-election", "Lok Sabha 2029 (expected)",
                    new String[]{"defence", "infrastructure", "fmcg", "all"}, 3),
            // Earnings reporting windows — major quarterly clusters (mid-quarter +6w).
            new Event(LocalDate.of(2026, 7, 15), "earnings-window", "Q1FY27 earnings (Jul-Aug)",
                    new String[]{"all"}, 2),
            new Event(LocalDate.of(2026, 10, 25),"earnings-window", "Q2FY27 earnings (Oct-Nov)",
                    new String[]{"all"}, 2),
            new Event(LocalDate.of(2027, 2, 5),  "earnings-window", "Q3FY27 earnings (Feb)",
                    new String[]{"all"}, 2),
            new Event(LocalDate.of(2027, 5, 5),  "earnings-window", "Q4FY27 earnings (May)",
                    new String[]{"all"}, 2)
    );

    @Override public String name() { return "india_event_calendar"; }

    @Override
    public String description() {
        return "Forward-looking calendar of India macro / political events (RBI MPC meetings, "
                + "Union Budget, general + state elections, quarterly earnings windows). Each event "
                + "carries sector-impact tags + severity (1-3). india_top_picks consumes this to "
                + "apply event-risk adjustments to picks (e.g. RBI in 7 days → trim banks).";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "lookaheadDays": { "type": "integer", "description": "Only return events within N days (default 180)." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            int lookaheadDays = Math.max(7, Math.min(1825, args.path("lookaheadDays").asInt(180)));
            LocalDate now = LocalDate.now();
            LocalDate horizon = now.plusDays(lookaheadDays);

            // Sort + filter
            List<Event> upcoming = new ArrayList<>(SCHEDULE);
            upcoming.removeIf(e -> e.date.isBefore(now) || e.date.isAfter(horizon));
            upcoming.sort(Comparator.comparing(e -> e.date));

            ObjectNode out = MAPPER.createObjectNode();
            out.put("asOf", now.toString());
            out.put("lookaheadDays", lookaheadDays);
            ArrayNode arr = out.putArray("events");
            for (Event e : upcoming) {
                ObjectNode o = MAPPER.createObjectNode();
                o.put("date", ISO.format(e.date));
                o.put("daysAway", (int) ChronoUnit.DAYS.between(now, e.date));
                o.put("type", e.type);
                o.put("label", e.label);
                ArrayNode sectors = o.putArray("sectorImpact");
                for (String s : e.sectorImpact) sectors.add(s);
                o.put("severity", e.severity);
                arr.add(o);
            }
            // Next risk window — the single closest non-routine event.
            Event next = upcoming.stream().filter(e -> e.severity >= 2).findFirst().orElse(null);
            if (next != null) {
                ObjectNode nrw = out.putObject("nextRiskWindow");
                nrw.put("date", ISO.format(next.date));
                nrw.put("daysAway", (int) ChronoUnit.DAYS.between(now, next.date));
                nrw.put("label", next.label);
                nrw.put("type", next.type);
                ArrayNode si = nrw.putArray("sectorImpact");
                for (String s : next.sectorImpact) si.add(s);
                // Generate a sector-tilt map: tilt DOWN sectors affected by imminent event (within 14d)
                ObjectNode tilts = out.putObject("eventTiltByKeyword");
                if (ChronoUnit.DAYS.between(now, next.date) <= 14) {
                    for (String kw : next.sectorImpact) {
                        // -3 for severity 2, -5 for severity 3, scaled by closeness
                        double bias = -1.5 * next.severity;
                        tilts.put(kw, bias);
                    }
                }
            }

            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("india_event_calendar failed: {}", e.toString());
            return ToolResult.error("india_event_calendar failed: " + e.getMessage());
        }
    }

    private record Event(LocalDate date, String type, String label, String[] sectorImpact, int severity) {}
}
