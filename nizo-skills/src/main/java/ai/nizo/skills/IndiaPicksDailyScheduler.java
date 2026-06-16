package ai.nizo.skills;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Lightweight daily refresh for {@code india_top_picks} — Phase 6.3.
 *
 * <p>Fires the picks pipeline once a day so the library always has a fresh ranking
 * waiting when the user opens the India Picks tab. Default fire time: 09:00 IST
 * (before NSE open at 09:15), with an 8h post-fire skip to avoid re-runs if the
 * process restarts mid-day.
 *
 * <p>Persistence: last-run timestamp written to {@code ~/.nizo/india-picks-last-run}.
 * On startup we read this and only schedule a near-immediate run if &gt;24h have
 * passed since the previous one; otherwise the next fire is at the next 09:00 IST.
 *
 * <p>This is a stop-gap until the full {@code nizo-scheduler} module is built —
 * once that exists, migrate the cron expression there for unified scheduling +
 * Telegram delivery.
 *
 * <p>Env var {@code NIZO_INDIA_PICKS_AUTO_REFRESH=0} disables auto-refresh entirely.
 */
public final class IndiaPicksDailyScheduler {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaPicksDailyScheduler.class);
    private static final ZoneId IST = ZoneId.of("Asia/Kolkata");
    private static final int FIRE_HOUR_IST = 9;    // 09:00 IST — before NSE open
    private static final long MIN_INTERVAL_HOURS = 8L;
    private static final String UNIVERSE = "NIFTY 500";
    private static final int TOP_N = 10;
    private static final int CANDIDATE_N = 60;

    private final Supplier<ToolRegistry> registry;
    private final Path stateFile;
    private final ScheduledExecutorService exec;

    public IndiaPicksDailyScheduler(Supplier<ToolRegistry> registry) {
        this.registry = registry;
        String home = System.getProperty("user.home", ".");
        this.stateFile = Paths.get(home, ".nizo", "india-picks-last-run");
        this.exec = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "india-picks-daily-scheduler");
            t.setDaemon(true);
            return t;
        });
    }

    /** Begin scheduling. Idempotent — safe to call once at boot. */
    public void start() {
        if ("0".equals(System.getenv("NIZO_INDIA_PICKS_AUTO_REFRESH"))) {
            LOG.info("india-picks daily scheduler DISABLED via NIZO_INDIA_PICKS_AUTO_REFRESH=0");
            return;
        }
        long delayMs = computeInitialDelayMs();
        long periodMs = Duration.ofHours(24).toMillis();
        exec.scheduleAtFixedRate(this::runOnce, delayMs, periodMs, TimeUnit.MILLISECONDS);
        LOG.info("india-picks daily scheduler armed — first fire in {} min, then every 24h",
                delayMs / 60_000L);
    }

    public void shutdown() {
        try { exec.shutdown(); } catch (Exception e) { /* best-effort */ }
    }

    /**
     * Initial delay: if we've never run before (or last run >24h ago) AND the next 09:00 IST
     * is more than {@link #MIN_INTERVAL_HOURS} away, fire in 60s. Otherwise wait for next 09:00.
     */
    private long computeInitialDelayMs() {
        long lastRun = readLastRun();
        long now = System.currentTimeMillis();
        long sinceLast = now - lastRun;
        ZonedDateTime nowIst = ZonedDateTime.now(IST);
        ZonedDateTime nextFire = nowIst.withHour(FIRE_HOUR_IST).withMinute(0).withSecond(0).withNano(0);
        if (!nextFire.isAfter(nowIst)) nextFire = nextFire.plusDays(1);
        long delayToNext = Duration.between(nowIst, nextFire).toMillis();

        // If never run, or last run was >24h ago, kick off in 60s (warms cache, populates library).
        if (lastRun <= 0 || sinceLast > Duration.ofHours(24).toMillis()) {
            return Math.min(60_000L, delayToNext);
        }
        return delayToNext;
    }

    /** Fire one picks run. Bounded to ~30 min — long enough for NIFTY 500 deep pass. */
    private void runOnce() {
        try {
            ToolRegistry reg = registry.get();
            if (reg == null) { LOG.warn("india-picks daily: registry not ready, skipping"); return; }
            Tool picksT = reg.byName("india_top_picks").orElse(null);
            if (picksT == null) { LOG.warn("india-picks daily: tool not registered, skipping"); return; }

            // concurrency=3 matches the llama-server's --parallel slot count. Was 6: that fanned out
            // 6 LLM-bound scorers against 3 slots, so the daily run thrashed each scorer's KV cache
            // (measured June 2026: a NIFTY-500 pass took ~19 min) AND starved any interactive stock
            // report that overlapped it. Capping to the real slot budget roughly halves the run and
            // leaves the interactive analyst gate room to make progress when the two collide.
            String args = "{\"universe\":\"" + UNIVERSE + "\",\"topN\":" + TOP_N
                        + ",\"candidateN\":" + CANDIDATE_N + ",\"concurrency\":3}";
            LOG.info("india-picks daily: starting scheduled run (universe={}, topN={}, candidateN={})",
                    UNIVERSE, TOP_N, CANDIDATE_N);
            long t0 = System.currentTimeMillis();
            ToolResult r = picksT.execute(args);
            long elapsed = System.currentTimeMillis() - t0;
            if (r != null && r.ok()) {
                int chars = r.content() != null ? r.content().length() : 0;
                LOG.info("india-picks daily: DONE in {} ms, {} chars output", elapsed, chars);
                writeLastRun(System.currentTimeMillis());
            } else {
                LOG.warn("india-picks daily: FAILED in {} ms: {}",
                        elapsed, r == null ? "null result" : r.content());
            }
        } catch (Throwable t) {
            // Catch Throwable so the scheduler thread survives any pathological failure.
            LOG.warn("india-picks daily: uncaught exception, scheduler stays armed: {}", t.toString());
        }
    }

    private long readLastRun() {
        try {
            if (!Files.isRegularFile(stateFile)) return 0L;
            String s = Files.readString(stateFile).trim();
            return Long.parseLong(s);
        } catch (Exception e) { return 0L; }
    }

    private void writeLastRun(long ts) {
        try {
            Files.createDirectories(stateFile.getParent());
            Files.writeString(stateFile, Long.toString(ts),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (Exception e) {
            LOG.debug("india-picks daily: state write failed: {}", e.toString());
        }
    }
}
