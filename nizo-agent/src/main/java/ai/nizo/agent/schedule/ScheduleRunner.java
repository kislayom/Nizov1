package ai.nizo.agent.schedule;

/**
 * What to do when a scheduled task fires: run its prompt and deliver the result to its chat.
 * Kept as a callback so {@link SchedulerEngine} stays decoupled from the agent loop (and unit-testable).
 */
@FunctionalInterface
public interface ScheduleRunner {
    void run(ScheduledTask task);
}
