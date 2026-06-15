package ai.nizo.scheduler;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses everyday reminder phrasing into a {@link ParsedSchedule} (ONCE instant or recurring cron).
 *
 * <p>Deliberately covers the common cases people actually say — "in 20 minutes", "tomorrow at 9am",
 * "every morning", "every day at 7:30pm", "every weekday at 9", "every monday at 18:00", "every 2
 * hours" — plus an explicit 5-field cron passthrough. Anything it can't confidently parse returns
 * {@link Optional#empty()}, so the caller can ask the model to supply a cron expression instead of
 * silently guessing a time. All computation is relative to an injected {@code now} + {@code zone}
 * so it's deterministic and unit-testable.
 */
public final class NaturalTimeParser {

    private NaturalTimeParser() {}

    private static final Pattern CRON_5 = Pattern.compile("^[\\d*/,\\-]+(\\s+[\\d*/,\\-]+){4}$");
    private static final Pattern IN_REL = Pattern.compile(
            "\\bin\\s+(\\d+)\\s*(min(?:ute)?s?|hours?|hrs?|days?|weeks?)\\b");
    private static final Pattern EVERY_N = Pattern.compile(
            "\\bevery\\s+(\\d+)\\s*(min(?:ute)?s?|hours?|hrs?)\\b");
    private static final Pattern TIME = Pattern.compile(
            "(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?", Pattern.CASE_INSENSITIVE);
    private static final DateTimeFormatter ISO = DateTimeFormatter.ISO_INSTANT;

    private static final String[] DOW_NAMES = {"sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"};
    private static final String[] DOW_ABBR = {"sun", "mon", "tue", "wed", "thu", "fri", "sat"};

    public static Optional<ParsedSchedule> parse(String text, ZoneId zone, long nowMs) {
        if (text == null || text.isBlank()) return Optional.empty();
        String t = text.trim().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
        ZonedDateTime now = Instant.ofEpochMilli(nowMs).atZone(zone);

        // 1) Explicit 5-field cron passthrough.
        if (CRON_5.matcher(t).matches() && CronSupport.isValid(t)) {
            return Optional.of(cron(t, "cron " + t, zone, nowMs));
        }

        // 2) Relative one-shot: "in 20 minutes", "in 2 hours", "in 3 days".
        Matcher rel = IN_REL.matcher(t);
        if (rel.find()) {
            long n = Long.parseLong(rel.group(1));
            String unit = rel.group(2);
            ZonedDateTime fire = unit.startsWith("min") ? now.plusMinutes(n)
                    : unit.startsWith("h") ? now.plusHours(n)
                    : unit.startsWith("d") ? now.plusDays(n)
                    : now.plusWeeks(n);
            return Optional.of(once(fire, "in " + n + " " + unit));
        }

        // 3) Recurring "every N minutes/hours".
        Matcher en = EVERY_N.matcher(t);
        if (en.find()) {
            int n = Integer.parseInt(en.group(1));
            String unit = en.group(2);
            String expr = unit.startsWith("min") ? "*/" + n + " * * * *" : "0 */" + n + " * * *";
            return Optional.of(cron(expr, "every " + n + " " + unit, zone, nowMs));
        }
        if (t.matches(".*\\bevery\\s+hour\\b.*")) return Optional.of(cron("0 * * * *", "every hour", zone, nowMs));
        if (t.matches(".*\\bevery\\s+minute\\b.*")) return Optional.of(cron("* * * * *", "every minute", zone, nowMs));

        // 4) Named recurring times of day.
        if (t.contains("every morning")) return Optional.of(cron("0 8 * * *", "every day at 08:00", zone, nowMs));
        if (t.contains("every afternoon")) return Optional.of(cron("0 14 * * *", "every day at 14:00", zone, nowMs));
        if (t.contains("every evening")) return Optional.of(cron("0 18 * * *", "every day at 18:00", zone, nowMs));
        if (t.contains("every night")) return Optional.of(cron("0 21 * * *", "every day at 21:00", zone, nowMs));

        Optional<LocalTime> time = parseTime(t);

        // 5) Recurring weekday-specific: "every monday at 9am", "every weekday at 7:30".
        if (t.contains("every")) {
            LocalTime tm = time.orElse(LocalTime.of(9, 0));
            if (t.contains("weekday")) {
                return Optional.of(cron(tm.getMinute() + " " + tm.getHour() + " * * 1-5",
                        "every weekday at " + hm(tm), zone, nowMs));
            }
            for (int d = 0; d < 7; d++) {
                if (t.contains(DOW_NAMES[d]) || t.matches(".*\\b" + DOW_ABBR[d] + "\\b.*")) {
                    return Optional.of(cron(tm.getMinute() + " " + tm.getHour() + " * * " + d,
                            "every " + DOW_NAMES[d] + " at " + hm(tm), zone, nowMs));
                }
            }
            if (t.contains("every day") || t.contains("daily")) {
                return Optional.of(cron(tm.getMinute() + " " + tm.getHour() + " * * *",
                        "every day at " + hm(tm), zone, nowMs));
            }
        }

        // 6) One-shot day references.
        if (time.isPresent()) {
            LocalTime tm = time.get();
            LocalDate day = now.toLocalDate();
            boolean explicitDay = false;
            if (t.contains("tomorrow")) { day = day.plusDays(1); explicitDay = true; }
            else {
                // weekday name → next occurrence of that weekday
                for (int d = 0; d < 7; d++) {
                    if (t.contains(DOW_NAMES[d]) || t.matches(".*\\b" + DOW_ABBR[d] + "\\b.*")) {
                        int target = (d == 0) ? 7 : d;           // ISO: Mon=1..Sun=7
                        int cur = now.toLocalDate().getDayOfWeek().getValue();
                        int add = (target - cur + 7) % 7;
                        if (add == 0) add = 7;                   // "next" that weekday
                        day = now.toLocalDate().plusDays(add);
                        explicitDay = true;
                        break;
                    }
                }
            }
            ZonedDateTime fire = ZonedDateTime.of(LocalDateTime.of(day, tm), zone);
            if (!explicitDay && !fire.isAfter(now)) fire = fire.plusDays(1);   // bare "at 9am" already past → tomorrow
            return Optional.of(once(fire, dayHuman(t) + " at " + hm(tm)));
        }

        return Optional.empty();
    }

    /** Parse a clock time out of free text: 9am, 9:30pm, 21:00, noon, midnight, tonight. */
    static Optional<LocalTime> parseTime(String t) {
        if (t.contains("noon")) return Optional.of(LocalTime.of(12, 0));
        if (t.contains("midnight")) return Optional.of(LocalTime.MIDNIGHT);
        if (t.contains("tonight")) return Optional.of(LocalTime.of(21, 0));
        Matcher m = TIME.matcher(t);
        // Require an am/pm or a colon to avoid matching stray numbers ("in 3 days").
        while (m.find()) {
            String ap = m.group(3);
            boolean hasColon = m.group(2) != null;
            if (ap == null && !hasColon) continue;
            int h = Integer.parseInt(m.group(1));
            int min = hasColon ? Integer.parseInt(m.group(2)) : 0;
            if (h > 23 || min > 59) continue;
            if (ap != null) {
                ap = ap.toLowerCase(Locale.ROOT);
                if (ap.equals("pm") && h < 12) h += 12;
                if (ap.equals("am") && h == 12) h = 0;
            }
            if (h > 23) continue;
            return Optional.of(LocalTime.of(h, min));
        }
        return Optional.empty();
    }

    private static ParsedSchedule once(ZonedDateTime fire, String human) {
        return new ParsedSchedule(ScheduleKind.ONCE, ISO.format(fire.toInstant()),
                fire.toInstant().toEpochMilli(), human);
    }

    private static ParsedSchedule cron(String expr, String human, ZoneId zone, long nowMs) {
        long next = CronSupport.nextFireMs(expr, nowMs, zone);
        return new ParsedSchedule(ScheduleKind.CRON, expr, next, human);
    }

    private static String hm(LocalTime t) {
        return String.format("%02d:%02d", t.getHour(), t.getMinute());
    }

    private static String dayHuman(String t) {
        if (t.contains("tomorrow")) return "tomorrow";
        for (int d = 0; d < 7; d++) if (t.contains(DOW_NAMES[d])) return DOW_NAMES[d];
        return "today";
    }
}
