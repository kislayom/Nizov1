package ai.nizo.scheduler;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.ZoneId;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Deterministic tests for {@link NaturalTimeParser} — fixed "now" + UTC so firings are exact.
 * NOW = Tuesday 2026-06-16 10:00:00Z.
 */
class NaturalTimeParserTest {

    private static final ZoneId UTC = ZoneId.of("UTC");
    private static final long NOW = Instant.parse("2026-06-16T10:00:00Z").toEpochMilli();

    private static ParsedSchedule parse(String s) {
        Optional<ParsedSchedule> p = NaturalTimeParser.parse(s, UTC, NOW);
        assertTrue(p.isPresent(), "expected to parse: " + s);
        return p.get();
    }

    private static long at(String iso) { return Instant.parse(iso).toEpochMilli(); }

    @Test
    void relativeMinutesAndHours() {
        ParsedSchedule m = parse("remind me in 20 minutes to stretch");
        assertEquals(ScheduleKind.ONCE, m.kind());
        assertEquals(NOW + 20 * 60_000L, m.firstFireMs());

        assertEquals(NOW + 2 * 3600_000L, parse("in 2 hours").firstFireMs());
        assertEquals(NOW + 3 * 86_400_000L, parse("in 3 days").firstFireMs());
    }

    @Test
    void tomorrowAtTime() {
        ParsedSchedule p = parse("tomorrow at 9am");
        assertEquals(ScheduleKind.ONCE, p.kind());
        assertEquals(at("2026-06-17T09:00:00Z"), p.firstFireMs());
    }

    @Test
    void bareTimeAlreadyPastRollsToTomorrow() {
        // 9am is before NOW (10:00) → fire tomorrow.
        assertEquals(at("2026-06-17T09:00:00Z"), parse("at 9am").firstFireMs());
        // 3pm is after NOW → today.
        assertEquals(at("2026-06-16T15:00:00Z"), parse("at 3pm").firstFireMs());
    }

    @Test
    void dailyRecurring() {
        ParsedSchedule p = parse("every day at 7:30pm");
        assertEquals(ScheduleKind.CRON, p.kind());
        assertEquals("30 19 * * *", p.spec());
        assertTrue(p.firstFireMs() > NOW);
    }

    @Test
    void namedTimesOfDay() {
        assertEquals("0 8 * * *", parse("every morning send me a briefing").spec());
        assertEquals("0 18 * * *", parse("every evening").spec());
        assertEquals("0 21 * * *", parse("every night").spec());
    }

    @Test
    void weekdaySpecificCron() {
        assertEquals("0 9 * * 1", parse("every monday at 9am").spec());
        assertEquals("30 7 * * 1-5", parse("every weekday at 7:30").spec());
        assertEquals("0 18 * * 5", parse("every friday at 6pm").spec());
    }

    @Test
    void everyNUnits() {
        assertEquals("*/15 * * * *", parse("every 15 minutes").spec());
        assertEquals("0 */2 * * *", parse("every 2 hours").spec());
        assertEquals("0 * * * *", parse("every hour").spec());
    }

    @Test
    void explicitCronPassthrough() {
        ParsedSchedule p = parse("0 6 * * 1-5");
        assertEquals(ScheduleKind.CRON, p.kind());
        assertEquals("0 6 * * 1-5", p.spec());
    }

    @Test
    void nextWeekdayOneShot() {
        // NOW is Tuesday; "friday at noon" → this coming Friday 2026-06-19 12:00Z.
        ParsedSchedule p = parse("friday at noon");
        assertEquals(ScheduleKind.ONCE, p.kind());
        assertEquals(at("2026-06-19T12:00:00Z"), p.firstFireMs());
    }

    @Test
    void unparseableReturnsEmpty() {
        assertTrue(NaturalTimeParser.parse("just do the thing", UTC, NOW).isEmpty());
        assertTrue(NaturalTimeParser.parse("", UTC, NOW).isEmpty());
    }
}
