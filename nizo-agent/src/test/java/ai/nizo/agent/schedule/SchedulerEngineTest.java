package ai.nizo.agent.schedule;

import ai.nizo.scheduler.ScheduleKind;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Offline tests for the store + tick engine using a temp SQLite DB and a synchronous worker. */
class SchedulerEngineTest {

    @Test
    void firesDue_retiresOnce_reschedulesCron_noDoubleFire(@TempDir Path tmp) {
        ScheduleStore store = new ScheduleStore(tmp.resolve("sched.db"));
        long now = System.currentTimeMillis();
        store.add(new ScheduledTask("once-1", ScheduleKind.ONCE, "x", "do A", "c1", "u1", now - 1000, 0, true, now));
        store.add(new ScheduledTask("cron-1", ScheduleKind.CRON, "* * * * *", "do B", "c1", "u1", now - 1000, 0, true, now));
        store.add(new ScheduledTask("future-1", ScheduleKind.ONCE, "x", "do C", "c1", "u1", now + 3_600_000, 0, true, now));

        List<String> fired = new ArrayList<>();
        SchedulerEngine eng = new SchedulerEngine(store, t -> fired.add(t.id()), ZoneId.of("UTC"), Runnable::run);

        assertEquals(2, eng.runDue(now), "once-1 and cron-1 are due");
        assertTrue(fired.contains("once-1") && fired.contains("cron-1"));
        assertFalse(fired.contains("future-1"));

        assertFalse(store.get("once-1").orElseThrow().enabled(), "ONCE retired after firing");
        ScheduledTask cron = store.get("cron-1").orElseThrow();
        assertTrue(cron.enabled(), "CRON stays enabled");
        assertTrue(cron.nextFireMs() > now, "CRON rescheduled to the future");
        assertEquals(now, cron.lastFireMs());

        // Immediate re-tick: nothing fires (once disabled, cron now in the future) — no double-fire.
        fired.clear();
        assertEquals(0, eng.runDue(now));
        assertTrue(fired.isEmpty());

        // The future task fires once its time arrives.
        fired.clear();
        eng.runDue(now + 3_600_000 + 1);
        assertTrue(fired.contains("future-1"));
    }

    @Test
    void listAndCancel(@TempDir Path tmp) {
        ScheduleStore store = new ScheduleStore(tmp.resolve("s.db"));
        long now = System.currentTimeMillis();
        store.add(new ScheduledTask("x1", ScheduleKind.ONCE, "x", "p", "c", "u", now + 1000, 0, true, now));
        store.add(new ScheduledTask("x2", ScheduleKind.CRON, "0 9 * * *", "p2", "c", "u", now + 1000, 0, true, now));
        assertEquals(2, store.listForUser("u").size());
        assertTrue(store.delete("x1"));
        assertEquals(1, store.listForUser("u").size());
        assertFalse(store.delete("does-not-exist"));
    }
}
