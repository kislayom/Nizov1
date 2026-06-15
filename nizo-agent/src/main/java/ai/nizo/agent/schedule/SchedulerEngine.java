package ai.nizo.agent.schedule;

import ai.nizo.scheduler.CronSupport;
import ai.nizo.scheduler.ScheduleKind;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.ZoneId;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Ticks on a fixed interval, fires due {@link ScheduledTask}s, and reschedules them.
 *
 * <p>Key correctness property: a task's next firing (CRON) is computed and persisted, or the task
 * is disabled (ONCE), BEFORE the run is dispatched — so a slow run can't be re-picked and
 * double-fired by the next tick. Runs are dispatched to an injected {@link Executor} (a
 * virtual-thread pool in prod) so a long agent run never blocks the tick loop.
 */
public final class SchedulerEngine {

    private static final Logger LOG = LoggerFactory.getLogger(SchedulerEngine.class);

    private final ScheduleStore store;
    private final ScheduleRunner runner;
    private final ZoneId zone;
    private final Executor worker;
    private ScheduledExecutorService ticker;

    public SchedulerEngine(ScheduleStore store, ScheduleRunner runner, ZoneId zone, Executor worker) {
        this.store = store;
        this.runner = runner;
        this.zone = zone;
        this.worker = worker;
    }

    /** Start ticking every 30s (first tick after 10s). */
    public void start() {
        ticker = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "nizo-scheduler"); t.setDaemon(true); return t;
        });
        ticker.scheduleAtFixedRate(() -> {
            try { runDue(System.currentTimeMillis()); }
            catch (Exception e) { LOG.warn("scheduler tick failed: {}", e.toString()); }
        }, 10, 30, TimeUnit.SECONDS);
        LOG.info("scheduler engine ENABLED (tick every 30s, zone {})", zone);
    }

    public void stop() { if (ticker != null) ticker.shutdownNow(); }

    /**
     * Fire everything due as of {@code nowMs}. Reschedules CRON to its next firing and disables
     * ONCE, persisting that BEFORE dispatching the run. Returns the number fired. Package-private
     * so tests can drive it deterministically.
     */
    int runDue(long nowMs) {
        List<ScheduledTask> due = store.due(nowMs);
        for (ScheduledTask t : due) {
            try {
                if (t.kind() == ScheduleKind.CRON) {
                    long next = CronSupport.nextFireMs(t.spec(), nowMs, zone);
                    if (next > 0) {
                        store.markFired(t.id(), nowMs, next);
                    } else {                 // cron that never fires again — retire it
                        store.markFired(t.id(), nowMs, t.nextFireMs());
                        store.setEnabled(t.id(), false);
                    }
                } else {                     // ONCE — fire and retire
                    store.markFired(t.id(), nowMs, t.nextFireMs());
                    store.setEnabled(t.id(), false);
                }
                LOG.info("scheduler firing {} ({}: {}) for chat={}", t.id(), t.kind(), t.spec(), t.chatId());
                worker.execute(() -> {
                    try { runner.run(t); }
                    catch (Exception e) { LOG.warn("scheduled task {} run failed: {}", t.id(), e.toString()); }
                });
            } catch (Exception e) {
                LOG.warn("scheduler could not process task {}: {}", t.id(), e.toString());
            }
        }
        return due.size();
    }
}
