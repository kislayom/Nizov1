package ai.nizo.scheduler;

/**
 * Result of {@link NaturalTimeParser}: a normalized schedule the store can persist.
 *
 * @param kind        ONCE or CRON
 * @param spec        for ONCE — an ISO-8601 instant string; for CRON — a 5-field UNIX cron expression
 * @param firstFireMs epoch-millis of the next firing (already computed relative to "now")
 * @param human       a short human-readable description (e.g. "every day at 08:00")
 */
public record ParsedSchedule(ScheduleKind kind, String spec, long firstFireMs, String human) {}
