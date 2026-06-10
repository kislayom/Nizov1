package ai.nizo.tools.system;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Returns the current wall-clock time. Surprisingly load-bearing — agents otherwise hallucinate
 * dates from training data ("today is January 2026").
 */
public final class CurrentTimeTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Override public String name() { return "current_time"; }

    @Override
    public String description() {
        return "Get the current date and time. Use this when the user asks about today, "
                + "now, the current date, or anything time-relative — never guess from training data. "
                + "Optional 'timezone' parameter is an IANA zone like 'Asia/Singapore'; defaults to system zone.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "timezone": {
                  "type": "string",
                  "description": "IANA timezone, e.g. 'UTC', 'Asia/Singapore', 'America/New_York'."
                }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        String tz = null;
        if (argumentsJson != null && !argumentsJson.isBlank()) {
            JsonNode args = MAPPER.readTree(argumentsJson);
            JsonNode tzNode = args.get("timezone");
            if (tzNode != null && tzNode.isTextual()) tz = tzNode.asText();
        }

        ZoneId zone = (tz == null || tz.isBlank()) ? ZoneId.systemDefault() : ZoneId.of(tz);
        ZonedDateTime now = ZonedDateTime.now(zone);
        String iso = now.format(DateTimeFormatter.ISO_OFFSET_DATE_TIME);
        String human = now.format(DateTimeFormatter.ofPattern("EEEE, d MMMM yyyy 'at' HH:mm zzz"));
        return ToolResult.ok(human + "  (ISO: " + iso + ")");
    }
}
