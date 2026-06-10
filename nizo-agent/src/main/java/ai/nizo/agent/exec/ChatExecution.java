package ai.nizo.agent.exec;

import ai.nizo.agent.loop.AgentLoop;
import ai.nizo.api.agent.AgentEvent;
import ai.nizo.api.chat.IncomingMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Per-chat executor that owns a single AgentLoop worker thread, a message inbox, a ring buffer of
 * recent events, and a set of live event subscribers.
 *
 * <p><b>Why this exists</b>: prior model coupled chat execution to the SSE HTTP connection. Close
 * the browser tab → SSE dies → AgentLoop kept running but had nowhere to send events → the
 * in-flight work was effectively orphaned. Reopen the tab and you saw stale state.
 *
 * <p><b>New model</b>: the chat is an autonomous worker. SSE connections are subscribers that
 * come and go. Subscribers can replay missed events from the ring buffer ({@link #eventsSince}).
 * Stopping a turn is a thread interrupt away. Multiple subscribers (different tabs / devices)
 * see the same stream.
 *
 * <p><b>Threading</b>: a single permanent worker thread per chat. {@link #enqueue} is non-
 * blocking (just adds to the inbox). The worker pulls one message at a time, runs
 * {@link AgentLoop#runStreaming}, and the events flow through {@link #emit} to all subscribers
 * and the ring.
 */
public final class ChatExecution implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(ChatExecution.class);
    private static final int RING_CAP_DEFAULT = 5_000;

    /**
     * Hard cap on pending messages per chat. Prevents an unbounded queue blowing up the JVM if
     * a client (buggy retry loop, attacker, or a fast-typing user) spams enqueue while a long
     * LLM call is in flight. Override via {@code NIZO_CHAT_INBOX_CAP}.
     */
    private static final int INBOX_CAP_DEFAULT = readIntEnv("NIZO_CHAT_INBOX_CAP", 64);

    private static int readIntEnv(String key, int defaultValue) {
        String v = System.getenv(key);
        if (v == null || v.isBlank()) return defaultValue;
        try { return Math.max(1, Integer.parseInt(v.trim())); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    public enum Status { IDLE, PROCESSING, CLOSED }

    /** Result of {@link #enqueue}. {@link #ACCEPTED} on success, {@link #QUEUE_FULL} when over cap. */
    public enum EnqueueResult { ACCEPTED, QUEUE_FULL, CLOSED }

    private final String chatId;
    private final AgentLoop agent;
    private final ExecutorService workers;
    private final int ringCap;

    private final LinkedBlockingDeque<IncomingMessage> inbox = new LinkedBlockingDeque<>();
    private final int inboxCap = INBOX_CAP_DEFAULT;
    private final Deque<ChatEvent> ring = new ArrayDeque<>();
    private final Set<EventSubscriber> subscribers = ConcurrentHashMap.newKeySet();

    private final AtomicLong seqGen = new AtomicLong(0);
    private final AtomicReference<Status> status = new AtomicReference<>(Status.IDLE);
    private final AtomicReference<Thread> workerThread = new AtomicReference<>();
    private volatile boolean closed = false;
    private volatile long createdAt = System.currentTimeMillis();
    private volatile long lastActivityAt = System.currentTimeMillis();

    ChatExecution(String chatId, AgentLoop agent, ExecutorService workers) {
        this(chatId, agent, workers, RING_CAP_DEFAULT);
    }

    ChatExecution(String chatId, AgentLoop agent, ExecutorService workers, int ringCap) {
        this.chatId = chatId;
        this.agent = agent;
        this.workers = workers;
        this.ringCap = Math.max(64, ringCap);
        // Spin up the permanent worker — it blocks on inbox.take() until messages arrive.
        workers.submit(this::workerLoop);
    }

    public String chatId()       { return chatId; }
    public Status status()       { return status.get(); }
    public int queueDepth()      { return inbox.size(); }
    public long latestSeq()      { return seqGen.get(); }
    public long createdAt()      { return createdAt; }
    public long lastActivityAt() { return lastActivityAt; }
    public boolean isClosed()    { return closed; }

    /**
     * Add a user message to the inbox. Returns immediately — does not block on the LLM.
     *
     * @return {@link EnqueueResult#ACCEPTED} on success; {@link EnqueueResult#QUEUE_FULL} if the
     *         inbox is at capacity (caller should map to HTTP 429 or similar);
     *         {@link EnqueueResult#CLOSED} if the chat has been closed.
     */
    public synchronized EnqueueResult enqueue(IncomingMessage msg) {
        if (closed) return EnqueueResult.CLOSED;
        if (inbox.size() >= inboxCap) {
            LOG.warn("chat {} inbox full ({}/{}) — rejecting enqueue", chatId, inbox.size(), inboxCap);
            return EnqueueResult.QUEUE_FULL;
        }
        inbox.offer(msg);
        lastActivityAt = System.currentTimeMillis();
        return EnqueueResult.ACCEPTED;
    }

    /** Snapshot of pending input messages (for the UI's queue display). */
    public synchronized List<IncomingMessage> pending() {
        return new ArrayList<>(inbox);
    }

    /**
     * Subscribe to live events. Returns a token that, when called, removes the subscriber.
     * Typically used as a try-with-style finally cleanup after a SSE handler returns.
     */
    public Subscription subscribe(EventSubscriber sub) {
        subscribers.add(sub);
        return () -> subscribers.remove(sub);
    }

    /** Events with {@code seq > sinceSeq}, oldest first. Used for catch-up replay on reconnect. */
    public synchronized List<ChatEvent> eventsSince(long sinceSeq) {
        if (sinceSeq < 0) sinceSeq = -1;
        List<ChatEvent> out = new ArrayList<>();
        for (ChatEvent e : ring) {
            if (e.seq() > sinceSeq) out.add(e);
        }
        return out;
    }

    /**
     * Stop the currently-processing turn (if any). Returns true if there was a turn to stop.
     * The interrupt propagates as InterruptedException through the agent loop's blocking
     * waits on the LLM — AgentLoop catches it and emits a "stopped by user" event.
     *
     * <p>Pending queued messages are kept (the user only intends to abort the current turn).
     */
    public boolean stopCurrent() {
        if (status.get() != Status.PROCESSING) return false;
        Thread t = workerThread.get();
        if (t == null) return false;
        // Tell subscribers BEFORE interrupting — gives them a clean signal even if cleanup is messy.
        emit(new AgentEvent.Warning(0, "[stopped by user]"));
        t.interrupt();
        return true;
    }

    /** Stop current turn AND drop any queued messages. */
    public boolean stopAndClearQueue() {
        boolean stopped = stopCurrent();
        int dropped = inbox.size();
        inbox.clear();
        if (dropped > 0) {
            emit(new AgentEvent.Warning(0, "[dropped " + dropped + " queued message(s)]"));
        }
        return stopped || dropped > 0;
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        status.set(Status.CLOSED);
        Thread t = workerThread.get();
        if (t != null) t.interrupt();
        for (EventSubscriber sub : subscribers) {
            try { sub.onEnd(); } catch (Exception ignored) {}
        }
        subscribers.clear();
        ring.clear();
        inbox.clear();
    }

    // ─────────────────────────── internals ───────────────────────────

    private void workerLoop() {
        Thread.currentThread().setName("chat-worker-" + chatId);
        workerThread.set(Thread.currentThread());
        try {
            while (!closed) {
                IncomingMessage msg;
                try {
                    msg = inbox.take(); // blocks
                } catch (InterruptedException ie) {
                    if (closed) break;
                    // Spurious interrupt (e.g. from stopCurrent after queue drained) — clear and continue.
                    Thread.interrupted();
                    continue;
                }
                status.set(Status.PROCESSING);
                lastActivityAt = System.currentTimeMillis();
                try {
                    agent.runStreaming(msg, this::emit);
                } catch (Throwable t) {
                    LOG.warn("agent runStreaming threw for chat {}: {}", chatId, t.toString(), t);
                    emit(new AgentEvent.Warning(0, "[turn error: " + t.getClass().getSimpleName() + "]"));
                } finally {
                    // Clear any leftover interrupt flag from stopCurrent — next take() must block normally.
                    Thread.interrupted();
                    status.set(Status.IDLE);
                    lastActivityAt = System.currentTimeMillis();
                }
            }
        } finally {
            workerThread.set(null);
        }
    }

    /**
     * Single emit point. Stamps the event with a sequence number, appends to ring (evicting
     * oldest if full), then fans out to subscribers. Subscriber exceptions are swallowed —
     * a misbehaving consumer never blocks the producer.
     *
     * <p>TokenChunk and ThinkingChunk are deliberately EXCLUDED from the ring buffer.
     * A single stock-analysis run emits 75-100K content tokens (one event each), which
     * would evict every durable ToolCallStart / ToolCallResult / FinalReply event from a
     * 5,000-slot ring before a mid-pipeline reload could replay them. Token chunks are
     * transient by nature — live subscribers still receive them via fan-out below, but
     * reconnecting clients can pull the persisted assistant message from sessions.db
     * instead. Keeping the ring small + durable means tool tiles always backfill after
     * a refresh, even hours into a long run.
     */
    private void emit(AgentEvent event) {
        ChatEvent ce = new ChatEvent(seqGen.incrementAndGet(), event, System.currentTimeMillis());
        boolean buffer = !(event instanceof AgentEvent.TokenChunk)
                && !(event instanceof AgentEvent.ThinkingChunk);
        if (buffer) {
            synchronized (this) {
                ring.addLast(ce);
                while (ring.size() > ringCap) ring.pollFirst();
            }
        }
        lastActivityAt = ce.timestampMs();
        for (EventSubscriber sub : subscribers) {
            try { sub.onEvent(ce); } catch (Exception e) {
                LOG.debug("subscriber threw, ignored: {}", e.toString());
            }
        }
    }

    /** Returned by {@link #subscribe} — call to detach. Safe to call multiple times. */
    @FunctionalInterface
    public interface Subscription extends AutoCloseable {
        void unsubscribe();
        @Override default void close() { unsubscribe(); }
    }
}
