package ai.nizo.scheduler;

/** A schedule is either a single future firing ({@code ONCE}) or a recurring cron ({@code CRON}). */
public enum ScheduleKind {
    ONCE,
    CRON
}
