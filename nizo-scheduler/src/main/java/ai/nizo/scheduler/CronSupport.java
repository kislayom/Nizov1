package ai.nizo.scheduler;

import com.cronutils.model.Cron;
import com.cronutils.model.CronType;
import com.cronutils.model.definition.CronDefinitionBuilder;
import com.cronutils.model.time.ExecutionTime;
import com.cronutils.parser.CronParser;

import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;

/**
 * Thin wrapper over cron-utils for 5-field UNIX cron expressions ({@code min hour dom month dow}).
 * Used to validate cron strings and compute the next firing relative to a given instant.
 */
public final class CronSupport {

    private static final CronParser UNIX =
            new CronParser(CronDefinitionBuilder.instanceDefinitionFor(CronType.UNIX));

    private CronSupport() {}

    /** True if {@code expr} is a parseable, valid 5-field UNIX cron. */
    public static boolean isValid(String expr) {
        if (expr == null || expr.isBlank()) return false;
        try {
            UNIX.parse(expr.trim()).validate();
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Next firing of {@code cronExpr} strictly after {@code fromMs}, in epoch-millis, or -1 if the
     * expression never fires again.
     */
    public static long nextFireMs(String cronExpr, long fromMs, ZoneId zone) {
        Cron cron = UNIX.parse(cronExpr.trim());
        ZonedDateTime from = Instant.ofEpochMilli(fromMs).atZone(zone);
        return ExecutionTime.forCron(cron).nextExecution(from)
                .map(z -> z.toInstant().toEpochMilli())
                .orElse(-1L);
    }
}
