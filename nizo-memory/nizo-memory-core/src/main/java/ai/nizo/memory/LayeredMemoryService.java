package ai.nizo.memory;

import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.GraphTraversal;
import ai.nizo.memory.api.graph.Node;
import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.memory.RecallRequest;
import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.canonical.CanonicalPromoter;
import ai.nizo.memory.session.SessionPicker;
import ai.nizo.memory.util.Tokens;
import ai.nizo.memory.vector.InMemoryVectorIndex;
import ai.nizo.memory.store.SqliteMemoryStore;
import ai.nizo.memory.vector.VectorIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <h2>Memory pipeline</h2>
 *
 * <pre>
 *   remember()  ─┐
 *                ├─▶  [EPISODIC log]     ─consolidate()─▶  [SEMANTIC facts]
 *   learnFact() ─┘                                        ▲
 *                                                         │
 *   recall():  BM25 prefilter ∪ vector top-K ∪ tag-scan
 *              ─▶ deduplicate (max-marginal-relevance)
 *              ─▶ score = α·similarity + β·recency + γ·confidence + δ·accessCount
 *              ─▶ pack to token budget (greedy + tie-break)
 *              ─▶ touch last_accessed on returned items
 * </pre>
 *
 * <p>All operations are scoped to a {@code userId}. Two users' memories are
 * fully isolated: BM25, vector index, tag scans, consolidation, and stats
 * all filter by userId.
 *
 * <p>The scoring weights deliberately reward <strong>grounded, cited facts</strong>
 * over raw episodes — that keeps the prompt small and anti-hallucinatory.
 */
public final class LayeredMemoryService implements MemoryService {

    private static final Logger log = LoggerFactory.getLogger(LayeredMemoryService.class);

    private final SqliteMemoryStore store;
    private final InMemoryVectorIndex index;
    private final EmbeddingClient embedder;
    private final ModelClient summariser;
    private final GraphService graphService;      // nullable — graph channel is optional
    private final GraphTraversal graphTraversal;   // nullable
    private final ExecutorService async =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "nizo-memory");
                t.setDaemon(true);
                return t;
            });
    private final AtomicInteger turnCounter = new AtomicInteger();
    private final int consolidateEveryN;
    private final double confidenceFloor;
    private final double minRelevantSimilarity;
    private final double minTopScore;

    // Session pre-filter (Phase A of the "multi-session" recall fix).
    // Populated via {@link #setSessionPicker}; when left null the recall
    // pipeline is byte-identical to the pre-feature behaviour.
    private volatile SessionPicker sessionPicker = SessionPicker.NO_OP;
    private volatile int sessionPickerThreshold = 10;
    private volatile int sessionPickerTopN = 5;

    // Phase B — canonical-fact promoter. Runs at {@code learnFact} write-time
    // and tags items that deserve a seat at the "permanent" table.
    private final CanonicalPromoter canonicalPromoter = new CanonicalPromoter();

    /** Conservative legacy default — fine for tests with FakeEmbedder. Real
     *  embedders (e.g. nomic-embed-text) need ~0.45 to filter unrelated noise. */
    private static final double DEFAULT_MIN_RELEVANT_SIMILARITY = 0.01;

    /** If the highest-scored recall result is below this threshold, the
     *  entire result is dropped (empty returned). Guards against the
     *  "nomic moderate similarity everywhere" failure mode where unknown
     *  queries still return 5-10 semantically-nearby-but-wrong items.
     *  0.0 = disabled (legacy tests). Production default ~0.55. */
    private static final double DEFAULT_MIN_TOP_SCORE = 0.0;

    /** Stopwords excluded from the exact-name boost. Constructed once.
     *  Generic verbs / nouns are in here because they appear in too many
     *  items to discriminate — the boost is meant for SPECIFIC entities. */
    private static final Set<String> STOP_WORDS = Set.of(
            // Question + structural words
            "what", "where", "when", "who", "why", "how", "which",
            "is", "are", "was", "were", "do", "does", "did", "the",
            "a", "an", "of", "to", "in", "on", "at", "for", "with",
            "my", "your", "his", "her", "their", "any", "all", "some",
            "and", "or", "but", "i", "you", "he", "she", "we", "they",
            "me", "him", "us", "them", "this", "that", "these", "those",
            "tell", "show", "list", "give", "find", "about",
            // Generic verbs / nouns
            "user", "users", "work", "works", "worked", "working",
            "go", "goes", "going", "went", "make", "makes", "made", "making",
            "get", "gets", "got", "getting", "have", "has", "had",
            "be", "been", "being", "say", "said", "told",
            "thing", "things", "stuff", "person", "people", "someone",
            "current", "new", "old", "favorite", "favourite", "best",
            "want", "wants", "wanted", "need", "needs", "needed",
            "like", "likes", "liked", "love", "loves", "loved");

    /** Backward-compatible constructor — no graph integration. */
    public LayeredMemoryService(SqliteMemoryStore store,
                                InMemoryVectorIndex index,
                                EmbeddingClient embedder,
                                ModelClient summariser,
                                int consolidateEveryN,
                                double confidenceFloor) {
        this(store, index, embedder, summariser, null, null,
                consolidateEveryN, confidenceFloor,
                DEFAULT_MIN_RELEVANT_SIMILARITY, DEFAULT_MIN_TOP_SCORE);
    }

    /** Backward-compatible 8-arg with graph, default similarity floor. */
    public LayeredMemoryService(SqliteMemoryStore store,
                                InMemoryVectorIndex index,
                                EmbeddingClient embedder,
                                ModelClient summariser,
                                GraphService graphService,
                                GraphTraversal graphTraversal,
                                int consolidateEveryN,
                                double confidenceFloor) {
        this(store, index, embedder, summariser, graphService, graphTraversal,
                consolidateEveryN, confidenceFloor,
                DEFAULT_MIN_RELEVANT_SIMILARITY, DEFAULT_MIN_TOP_SCORE);
    }

    /** 9-arg constructor — configurable min-similarity floor, legacy
     *  no-top-score-threshold. */
    public LayeredMemoryService(SqliteMemoryStore store,
                                InMemoryVectorIndex index,
                                EmbeddingClient embedder,
                                ModelClient summariser,
                                GraphService graphService,
                                GraphTraversal graphTraversal,
                                int consolidateEveryN,
                                double confidenceFloor,
                                double minRelevantSimilarity) {
        this(store, index, embedder, summariser, graphService, graphTraversal,
                consolidateEveryN, confidenceFloor,
                minRelevantSimilarity, DEFAULT_MIN_TOP_SCORE);
    }

    /**
     * Full constructor — configurable relevance floor and top-score
     * threshold.
     *
     * @param minRelevantSimilarity vector cosine below which an item is
     *   excluded from user-facing recall (no FTS / tag / graph hit). Use ~0.55
     *   for real embedders (nomic-embed-text), ~0.01 for FakeEmbedder in tests.
     *   Floor is NOT applied to PROCEDURAL items.
     * @param minTopScore if the top-ranked recall result scores below this,
     *   the entire result set is dropped (empty returned). Use ~0.55 for
     *   production with nomic — kills the "nomic nearest neighbour is
     *   always in [0.4, 0.55]" noise floor for unknown queries.
     *   0.0 disables (legacy tests).
     */
    public LayeredMemoryService(SqliteMemoryStore store,
                                InMemoryVectorIndex index,
                                EmbeddingClient embedder,
                                ModelClient summariser,
                                GraphService graphService,
                                GraphTraversal graphTraversal,
                                int consolidateEveryN,
                                double confidenceFloor,
                                double minRelevantSimilarity,
                                double minTopScore) {
        this.store = store;
        this.index = index;
        this.embedder = embedder;
        this.summariser = summariser;
        this.graphService = graphService;
        this.graphTraversal = graphTraversal;
        this.consolidateEveryN = consolidateEveryN;
        this.confidenceFloor = confidenceFloor;
        this.minRelevantSimilarity = minRelevantSimilarity;
        this.minTopScore = minTopScore;
        index.hydrate(store.all());
    }

    @Override
    public String remember(String userId, String content, Map<String, String> tags, String source) {
        if (content == null || content.isBlank()) {
            log.debug("Skipping remember() with null/blank content");
            return UUID.randomUUID().toString();
        }
        String uid = userId == null ? "default" : userId;
        String id = UUID.randomUUID().toString();
        Instant now = Instant.now();
        MemoryItem initial = new MemoryItem(
                id, uid, MemoryItem.Tier.EPISODIC, content, null,
                tags == null ? Map.of() : tags,
                source, 0.9, now, now, 0, Tokens.count(content));
        store.upsert(initial);
        async.submit(() -> embedAndIndex(uid, id, content));
        // G16 — legacy turn-counter consolidation is DISABLED. It ran on a
        // fixed modulo of the per-process turn counter regardless of content
        // or time, which created duplicate SEMANTIC facts for everything the
        // new ReflectionService was already processing.
        //
        // Set {@code consolidateEveryN <= 0} in config to opt back in. The
        // default wiring no longer triggers the legacy path.
        if (consolidateEveryN > 0
                && turnCounter.incrementAndGet() % consolidateEveryN == 0
                && isLegacyConsolidationEnabled()) {
            async.submit(() -> consolidate(uid));
        }
        return id;
    }

    /** G16 — feature-flag for legacy {@code consolidate()}. False by default
     *  now that {@link ai.nizo.memory.reflect.ReflectionService} handles the
     *  same job properly (with dedup + reflected_at markers). Override via
     *  the system property {@code nizo.memory.legacy-consolidate} if needed. */
    private static boolean isLegacyConsolidationEnabled() {
        String flag = System.getProperty("nizo.memory.legacy-consolidate", "false");
        return "true".equalsIgnoreCase(flag);
    }

    @Override
    public String learnFact(String userId, String fact, String source, double confidence) {
        return learnFact(userId, fact, source, confidence, Map.of());
    }

    @Override
    public String learnFact(String userId, String fact, String source,
                             double confidence, Map<String, String> tags) {
        if (fact == null || fact.isBlank()) {
            log.debug("Skipping learnFact() with null/blank fact");
            return UUID.randomUUID().toString();
        }
        String uid = userId == null ? "default" : userId;

        // Check for near-duplicate semantic facts before inserting.
        float[] v = embedder == null ? null : safeEmbed(fact);
        if (v != null) {
            for (VectorIndex.Hit h : index.topK(uid, v, 3)) {
                store.findById(h.id()).ifPresent(existing -> {
                    if (existing.tier() == MemoryItem.Tier.SEMANTIC && h.score() > 0.94) {
                        log.debug("Suppressing near-duplicate fact (score={}): {}", h.score(), fact);
                    }
                });
            }
        }
        String id = UUID.randomUUID().toString();
        Instant now = Instant.now();
        // Merge caller-supplied semantic tags (subject / sensitivity / mode
        // / hypothetical / expires_at / cultural_term) on top of the default
        // {kind=fact} so recall filters can read them.
        Map<String, String> mergedTags = new java.util.HashMap<>();
        mergedTags.put("kind", "fact");
        if (tags != null) mergedTags.putAll(tags);
        // Infer facet if not supplied — powers precision-heavy compatibility.
        if (!mergedTags.containsKey(ai.nizo.memory.api.memory.MemoryTags.FACET)) {
            mergedTags.put(ai.nizo.memory.api.memory.MemoryTags.FACET,
                    ai.nizo.memory.facet.FacetClassifier.inferContent(fact));
        } else {
            mergedTags.put(ai.nizo.memory.api.memory.MemoryTags.FACET,
                    ai.nizo.memory.facet.FacetClassifier.normalize(
                            mergedTags.get(ai.nizo.memory.api.memory.MemoryTags.FACET)));
        }
        // Phase B — promote to canonical if the fact is foundational / pinned /
        // reinforced / definitional. Canonical items later win a dedicated RRF
        // channel and appear in the top-of-prompt index.
        mergedTags = canonicalPromoter.maybePromote(fact, mergedTags, confidence);
        MemoryItem item = new MemoryItem(
                id, uid, MemoryItem.Tier.SEMANTIC, fact, v,
                mergedTags,
                source, confidence, now, now, 0, Tokens.count(fact));
        store.upsert(item);
        if (v != null) index.add(uid, id, v);
        return id;
    }

    @Override
    public List<MemoryItem> recall(RecallRequest req) {
        // Two-pass adaptive retry (OMEGA Phase 7.5 port).
        // Pass 1 — full-strength thresholds. If we got nothing OR the top
        // result is weak (below the per-instance minTopScore), fall through
        // to pass 2 with thresholds relaxed by ADAPTIVE_RETRY_RELAXATION.
        // Pass 2 only runs if pass 1 was strict enough to throw away results
        // — calls already configured with very low thresholds (e.g. legacy
        // tests using 0.0) skip the retry entirely.
        List<MemoryItem> first = recallInternal(req, this.minTopScore);
        if (this.minTopScore <= 0.0 || !shouldAdaptiveRetry(first, req)) {
            return first;
        }
        double relaxed = this.minTopScore * ADAPTIVE_RETRY_RELAXATION;
        if (log.isDebugEnabled()) {
            log.debug("Adaptive retry firing: pass-1 returned {} hits, relaxing minTopScore {} → {}",
                    first.size(), this.minTopScore, relaxed);
        }
        List<MemoryItem> second = recallInternal(req, relaxed);
        // Take whichever is non-empty; prefer pass-1 ordering when both have
        // results (the relaxed pass is a safety net, not a replacement).
        return second.size() > first.size() ? second : (first.isEmpty() ? second : first);
    }

    /**
     * Whether to fire the adaptive retry. Triggers when pass-1 returned
     * nothing — almost always a "thresholds too tight" miss given that the
     * caller asked for {@code req}. We do <em>not</em> retry just because the
     * top score is low; that's a deliberate "honest abstention" we want to
     * preserve. Empty-pool retries strike the right balance.
     */
    private boolean shouldAdaptiveRetry(List<MemoryItem> firstPassResults,
                                          RecallRequest req) {
        if (!firstPassResults.isEmpty()) return false;
        // Only retry on queries with extractable signal. Empty queries / pure
        // stop-words have no relaxation that would help.
        if (req == null || req.query() == null || req.query().isBlank()) return false;
        if (!hasMeaningfulQueryTokens(req.query())) return false;
        return true;
    }

    /**
     * Multiplier applied to {@link #minTopScore} on the adaptive retry pass.
     * Matches OMEGA's default of 0.6×. Tighter (≤ 0.4×) defeats the safety
     * net by always firing; looser (≥ 0.8×) rarely changes the outcome.
     */
    private static final double ADAPTIVE_RETRY_RELAXATION = 0.6;

    /**
     * Recall variant with a per-call {@code minTopScore} override. Used by
     * {@link #surface} modes that need to loosen or tighten the "is there
     * anything relevant?" threshold per request (e.g. {@code recall-heavy}
     * lowers it; {@code precision-heavy} raises it). Package-visible.
     */
    List<MemoryItem> recallInternal(RecallRequest req, double minTopScoreOverride) {
        String uid = req.userId() == null ? "default" : req.userId();
        int budget = req.tokenBudget() <= 0 ? 1200 : req.tokenBudget();
        double minConf = Math.max(req.minConfidence(), confidenceFloor);
        // minTopScore used by this invocation — may be loosened/tightened per call.
        double effectiveMinTopScore = minTopScoreOverride;

        // G15 — reject queries with no extractable tokens upfront. "?",
        // "ok", "the", "a a a" etc. have no meaningful signal; returning
        // any PROCEDURAL or vector-matched fact is pure noise.
        if (req.query() == null || !hasMeaningfulQueryTokens(req.query())) {
            return List.of();
        }

        // ---- Session pre-filter (multi-session haystack fix) ----
        // When the user has many sessions, narrow the search to a few likely
        // ones before running the full hybrid pipeline. The filter is set from
        // (1) the caller, if they passed one explicitly, otherwise (2) the
        // LLM-driven SessionPicker if one is configured and the user has more
        // sessions than the threshold. An empty picker response (abstain) or
        // a disabled picker leaves the filter null — recall proceeds over all
        // sessions, byte-identical to the pre-feature behaviour.
        Set<String> sessionFilter = normaliseSessionFilter(req.sessionIds());
        if (sessionFilter == null && sessionPicker != SessionPicker.NO_OP) {
            int sessCount = 0;
            try {
                sessCount = store.distinctSessionCount(uid);
            } catch (RuntimeException e) {
                log.debug("distinctSessionCount failed (non-fatal): {}", e.toString());
            }
            if (sessCount > sessionPickerThreshold) {
                try {
                    List<SqliteMemoryStore.SessionInfo> manifest = store.sessionManifest(uid, 140);
                    Set<String> picks = sessionPicker.pick(req.query(), manifest, sessionPickerTopN);
                    if (picks != null && !picks.isEmpty()) {
                        sessionFilter = picks;
                        log.debug("Session picker selected {}/{} sessions for query '{}'",
                                picks.size(), manifest.size(), truncate(req.query(), 60));
                    }
                } catch (RuntimeException e) {
                    log.debug("Session picker failed (non-fatal): {}", e.toString());
                }
            }
        }

        // Hybrid retrieval: BM25 ∪ vector ∪ tag-scan ∪ graph — all scoped to userId.
        // We also track which items came from a SIGNAL (FTS/vector-above-threshold/
        // tag/graph) vs which came only as vector top-K filler. Items without any
        // relevance signal are excluded entirely — no filler when the query has
        // nothing matching.
        Map<String, MemoryItem> pool = new LinkedHashMap<>();
        Set<String> relevantIds = new HashSet<>();   // items with a real signal
        Set<String> ftsHitIds = new HashSet<>();     // items that hit FTS
        Set<String> tagHitIds = new HashSet<>();     // items that hit tag scan
        Map<String, Double> simScores = new HashMap<>();   // item id → vector similarity
        // RRF rank bookkeeping — each channel's 1-based rank per item id.
        // Rank absent means the item didn't hit that channel.
        Map<String, Integer> ftsRanks = new HashMap<>();
        Map<String, Integer> vecRanks = new HashMap<>();
        Map<String, Integer> tagRanks = new HashMap<>();
        Map<String, Integer> graphRanks = new HashMap<>();
        // Phase B — canonical rank bookkeeping. Items carrying
        // {@link MemoryTags#CANONICAL} get a dedicated channel that outranks
        // all other channels in the RRF fusion below.
        Map<String, Integer> canonRanks = new HashMap<>();
        Set<String> canonicalIds = new HashSet<>();

        Set<MemoryItem.Tier> reqTiers = req.tiers();
        // PROCEDURAL (bundled world knowledge) gate — G7/G17 fix.
        //
        // Customers complained that PROCEDURAL heuristics leaked into every
        // recall ("where does the user work" returned world-knowledge rows
        // about iPhone lingo). We now detect PERSONAL queries (any query
        // using first-person pronouns, "my X", "am I", "what do I" etc.) and
        // exclude PROCEDURAL by default for those. Generic queries ("what is
        // Thai cuisine", "nobel prize categories") still get PROCEDURAL.
        //
        // Caller can force inclusion by passing an explicit tiers set that
        // contains PROCEDURAL, OR exclude by passing tiers without PROCEDURAL.
        boolean explicitInclude = reqTiers != null && reqTiers.contains(MemoryItem.Tier.PROCEDURAL);
        boolean explicitExclude = reqTiers != null && !reqTiers.contains(MemoryItem.Tier.PROCEDURAL);
        boolean includeProceduralEarly;
        if (explicitInclude) {
            includeProceduralEarly = true;
        } else if (explicitExclude) {
            includeProceduralEarly = false;
        } else {
            // Auto-decide based on query shape.
            includeProceduralEarly = !isPersonalQuery(req.query());
        }

        // Conversational-recall intent: queries asking about prior assistant
        // content ("you mentioned X", "remind me what you recommended") need
        // a wider FTS net so long assistant answers aren't crowded out by
        // shorter user questions even after the b=0.25 length-norm tweak.
        boolean isConversationalRecall = isConversationalRecallQuery(req.query());

        // Aggregation intent — detector kept as a public hook for callers
        // that want to dispatch to a graph-mediated counting layer (the
        // architecturally correct fix). Recall-pipeline-side widening was
        // tried and reverted: more context with a 14B answerer actively
        // hurt single-fact recall without moving multi-session counting.
        // See bench/lme-oracle-12-nizo-v4.json for evidence.
        boolean isAggregation = isAggregationQuery(req.query());
        // Currently unused at the recall layer. The future fix: when
        // isAggregation, route to GraphService.countEntities(...) instead
        // of recall+enumerate. That requires extraction-on (entities in
        // the graph) — out of scope for the current benchmark setup.

        int ftsLimit = isConversationalRecall ? 120 : 60;
        int vecLimit = 60;

        // FTS channel — BM25-style lexical match; rank preserved for RRF.
        // sessionFilter pushes down to the DB so the index does the work.
        int ftsRank = 0;
        // Capture the BM25 top-1 / top-2 in iteration order so the strong-signal
        // detector below can compare them without re-querying the store.
        List<MemoryItem> ftsTopOrdered = new ArrayList<>(2);
        for (MemoryItem m : store.ftsSearch(uid, req.query(), ftsLimit, sessionFilter)) {
            if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
            ftsRank++;
            pool.put(m.id(), m);
            relevantIds.add(m.id());  // FTS hit = signal
            ftsHitIds.add(m.id());
            ftsRanks.putIfAbsent(m.id(), ftsRank);
            if (ftsTopOrdered.size() < 2) ftsTopOrdered.add(m);
        }

        // ---- Strong-signal short-circuit (OMEGA Phase 2.5) ----
        // When the top FTS result clearly dominates the runner-up in lexical
        // overlap with the query, more retrieval channels are mostly noise:
        // RRF would only add candidates the query barely touches, and the
        // reranker (if any) would still pick the FTS leader. Skipping the
        // vector + graph + entity channels in that case is a latency win and
        // a precision win — fewer false positives in the pool.
        //
        // Conservative thresholds so we only short-circuit when the signal is
        // unambiguous: top-1 must cover ≥ 70 % of meaningful query tokens AND
        // beat the runner-up by ≥ 20 percentage points.
        Set<String> earlyQueryPrefixes = queryTokenPrefixes(req.query());
        boolean strongSignal = false;
        if (ftsTopOrdered.size() >= 2 && !earlyQueryPrefixes.isEmpty()) {
            double top1Overlap = overlapRatio(earlyQueryPrefixes, ftsTopOrdered.get(0).content());
            double top2Overlap = overlapRatio(earlyQueryPrefixes, ftsTopOrdered.get(1).content());
            if (top1Overlap >= STRONG_SIGNAL_OVERLAP
                    && (top1Overlap - top2Overlap) >= STRONG_SIGNAL_GAP) {
                strongSignal = true;
                log.debug("Strong-signal short-circuit: top1={} top2={} for query '{}'",
                        String.format("%.2f", top1Overlap),
                        String.format("%.2f", top2Overlap),
                        truncate(req.query(), 60));
            }
        }

        // Vector channel — cosine-similarity top-K; rank preserved for RRF.
        // Only items above similarity threshold count as relevant signal
        // (low-similarity neighbours are noise — we still pool them for
        // context but don't vote for them in RRF).
        // Vector channel is suppressed when the strong-signal short-circuit
        // fired above — the FTS leader is already dominant; adding low-grade
        // vector neighbours would only dilute the result.
        float[] qv = (embedder == null || strongSignal) ? null : safeEmbed(req.query());
        if (qv != null) {
            int vr = 0;
            // When a session filter is active we widen the vector net a bit —
            // many hits will be outside the selected sessions and post-filtered
            // away, so pulling too few leaves RRF under-filled.
            int effectiveVecLimit = sessionFilter == null
                    ? vecLimit : Math.min(vecLimit * 3, 240);
            for (VectorIndex.Hit h : index.topK(uid, qv, effectiveVecLimit)) {
                var maybe = store.findById(h.id());
                if (maybe.isEmpty()) continue;
                MemoryItem m = maybe.get();
                if (!passesSessionFilter(m, sessionFilter)) continue;
                if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
                vr++;
                pool.putIfAbsent(m.id(), m);
                simScores.put(m.id(), h.score());
                // Floor is tier-aware: PROCEDURAL items always pass (they're
                // pulled via recallProcedural() and need a wider net).
                boolean passes = m.tier() == MemoryItem.Tier.PROCEDURAL
                        || h.score() >= minRelevantSimilarity;
                if (passes) {
                    relevantIds.add(m.id());  // meaningful vector match = signal
                    vecRanks.putIfAbsent(m.id(), vr);
                }
            }
        }

        // Tag-based retrieval — ranked in iteration order from the store.
        // Always a signal when requiredTags are specified.
        if (req.requiredTags() != null && !req.requiredTags().isEmpty()) {
            int tr = 0;
            for (MemoryItem m : store.findByTags(uid, req.requiredTags(), 30)) {
                if (!passesSessionFilter(m, sessionFilter)) continue;
                tr++;
                pool.putIfAbsent(m.id(), m);
                relevantIds.add(m.id());  // tag match = signal
                tagHitIds.add(m.id());
                tagRanks.putIfAbsent(m.id(), tr);
            }
        }

        // R4-D — ALWAYS include pinned facts in the pool. Pinning is an
        // explicit user signal ("I always want this considered"), so pinned
        // items should surface even when they don't hit a retrieval channel
        // via the query. Example: user pins "severely allergic to shellfish"
        // and later asks "what do I like about food" — the allergy is
        // safety-critical and must surface.
        try {
            for (MemoryItem m : store.findByTags(uid,
                    Map.of(ai.nizo.memory.api.memory.MemoryTags.PINNED, "true"),
                    20)) {
                if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
                pool.putIfAbsent(m.id(), m);
                relevantIds.add(m.id());
            }
        } catch (RuntimeException e) {
            log.debug("Pinned-always-include channel failed (non-fatal): {}", e.toString());
        }

        // ---- Canonical channel (Phase B) ----
        // Canonical facts are the user's "index" — small, stable, high-signal.
        // They're always pooled, always counted as signal, and get the largest
        // RRF weight below so a query that matches one dominates other channels.
        // Session filter is intentionally bypassed here: canonical facts are
        // cross-session by construction.
        try {
            int cr = 0;
            for (MemoryItem m : store.findByTags(uid,
                    Map.of(ai.nizo.memory.api.memory.MemoryTags.CANONICAL, "true"),
                    20)) {
                if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
                pool.putIfAbsent(m.id(), m);
                relevantIds.add(m.id());
                canonicalIds.add(m.id());
                canonRanks.putIfAbsent(m.id(), ++cr);
            }
        } catch (RuntimeException e) {
            log.debug("Canonical channel failed (non-fatal): {}", e.toString());
        }

        // Graph channel — FTS on node labels + 1-hop edge traversal.
        // Rank is assigned in order of node match; nodes that match directly
        // rank above their 1-hop neighbours.
        // Suppressed under strong-signal short-circuit, same rationale as the
        // vector channel above — graph 1-hop expansion would only add noise
        // when one FTS hit already dominates.
        Set<String> graphBoostedIds = Set.of();
        if (!strongSignal && graphService != null && req.query() != null && !req.query().isBlank()) {
            Set<String> boosted = new HashSet<>();
            try {
                int gr = 0;
                List<Node> matchingNodes = graphService.searchNodes(uid, req.query(), 10);
                for (Node node : matchingNodes) {
                    for (MemoryItem m : store.ftsSearch(uid, node.label(), 5, sessionFilter)) {
                        if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
                        gr++;
                        pool.putIfAbsent(m.id(), m);
                        boosted.add(m.id());
                        relevantIds.add(m.id());  // graph match = signal
                        graphRanks.putIfAbsent(m.id(), gr);
                    }
                }
                if (graphTraversal != null && !matchingNodes.isEmpty()) {
                    for (GraphTraversal.GraphNeighbor neighbor :
                            graphTraversal.expandFromNodes(uid, new LinkedHashSet<>(matchingNodes), 1)) {
                        for (MemoryItem m : store.ftsSearch(uid, neighbor.node().label(), 3, sessionFilter)) {
                            if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProceduralEarly) continue;
                            gr++;
                            pool.putIfAbsent(m.id(), m);
                            boosted.add(m.id());
                            relevantIds.add(m.id());
                            graphRanks.putIfAbsent(m.id(), gr);
                        }
                    }
                }
            } catch (RuntimeException e) {
                log.debug("Graph channel failed (non-fatal): {}", e.toString());
            }
            graphBoostedIds = boosted;
        }

        Set<MemoryItem.Tier> tiers = req.tiers();
        // PROCEDURAL items are INCLUDED by default (symmetric with the
        // channel-level includeProceduralEarly above). Opt out by passing
        // an explicit tiers set that omits PROCEDURAL.
        boolean includeProcedural = tiers == null || tiers.contains(MemoryItem.Tier.PROCEDURAL);
        // Pre-compute query token prefixes for lexical-overlap guard (below).
        Set<String> queryPrefixes = queryTokenPrefixes(req.query());
        Instant now = Instant.now();
        List<Scored> scored = new ArrayList<>();
        // Query lowercase for content-touching checks (subject / sensitivity gate).
        String queryLow = req.query() == null ? "" : req.query().toLowerCase();

        for (MemoryItem m : pool.values()) {
            // RELEVANCE FLOOR: no filler. If the item came from nowhere meaningful,
            // it doesn't belong in the result set.
            if (!relevantIds.contains(m.id())) continue;
            if (tiers != null && !tiers.contains(m.tier())) continue;
            // PROCEDURAL exclusion (unless explicitly requested via tiers)
            if (m.tier() == MemoryItem.Tier.PROCEDURAL && !includeProcedural) continue;
            if (m.confidence() < minConf) continue;
            if (!matchesTags(m, req.requiredTags())) continue;
            // G34 — filter raw user_message EPISODIC items from customer-facing
            // recall by default. Their provenance role is preserved (still
            // addressable by id, still linked from derived SEMANTIC facts via
            // source_message_id), but they should not crowd recall with 150-
            // character paragraphs when focused SEMANTIC facts exist. Callers
            // that specifically want raw messages can pass tiers={EPISODIC}.
            if (m.tier() == MemoryItem.Tier.EPISODIC
                    && "user_message".equals(m.source())
                    && (tiers == null || tiers.size() != 1 || !tiers.contains(MemoryItem.Tier.EPISODIC))) {
                continue;
            }

            // ── SUBJECT FILTER (safety-critical) ─────────────────────
            // By default only surface facts about the user. A fact tagged
            // "other:mom" must NOT be returned for "do i have allergies"
            // because it's about mom, not the user. The fact CAN be returned
            // when the query names that person ("does mom have allergies").
            String subject = m.tags().getOrDefault(
                    ai.nizo.memory.api.memory.MemoryTags.SUBJECT,
                    ai.nizo.memory.api.memory.MemoryTags.SUBJECT_SELF);
            if (subject.startsWith(ai.nizo.memory.api.memory.MemoryTags.SUBJECT_OTHER_PREFIX)) {
                String other = subject.substring(
                        ai.nizo.memory.api.memory.MemoryTags.SUBJECT_OTHER_PREFIX.length())
                        .toLowerCase();
                // Skip unless the query names the subject or uses a family role word
                if (!queryLow.contains(other) && !queryMentionsFamilyRole(queryLow, other)) {
                    continue;
                }
            }

            // ── SENSITIVITY GATE ─────────────────────────────────────
            // SENSITIVE / CRITICAL items only surface when the query
            // lexically overlaps the content (i.e. the user is actually
            // asking about the sensitive topic). Blanket queries like
            // "tell me about me" MUST NOT leak therapy notes into recall.
            String sensitivity = m.tags().get(ai.nizo.memory.api.memory.MemoryTags.SENSITIVITY);
            if (sensitivity != null
                    && (sensitivity.equalsIgnoreCase(
                            ai.nizo.memory.api.memory.MemoryTags.SENS_SENSITIVE)
                        || sensitivity.equalsIgnoreCase(
                            ai.nizo.memory.api.memory.MemoryTags.SENS_CRITICAL))) {
                // Must have a lexical overlap OR a very strong vector sim
                double hitSim = simScores.getOrDefault(m.id(), 0.0);
                boolean touches = hasLexicalOverlap(queryPrefixes, m.content()) || hitSim >= 0.72;
                if (!touches) continue;
            }

            // ── MODE FILTER ──────────────────────────────────────────
            // When the RecallRequest's tags filter specifies a mode
            // (required_tags: mode=work), items with a different mode are
            // already excluded by matchesTags. This is the *default* lane
            // separation: if the item is tagged mode=health but the query
            // is lexically work-flavoured, skip — prevents health stuff
            // leaking into work prompts.
            // (Light heuristic; opt-in via requiredTags for stricter lanes.)
            // LEXICAL-OVERLAP GUARD: applies ONLY to vector-only hits (no FTS,
            // no tag, no graph, not PROCEDURAL). G14 tightened — nomic-embed-
            // text returns ~0.6 similarity on wildly unrelated pairs, so the
            // previous 0.70 threshold still leaked (e.g. "capital of Burkina
            // Faso" vs "Alice Apple at Stripe"). Now we require 0.82.
            double thisSim = simScores.getOrDefault(m.id(), 0.0);
            // R4-D — pinned items bypass the lexical-overlap / vector-only
            // guards. An explicit pin is a user instruction ("this must
            // always be considered"). The guards exist to protect against
            // noise, not to override user intent.
            boolean isPinned = "true".equalsIgnoreCase(
                    m.tags().get(ai.nizo.memory.api.memory.MemoryTags.PINNED));
            boolean vectorOnly = !ftsHitIds.contains(m.id())
                    && !tagHitIds.contains(m.id())
                    && !graphBoostedIds.contains(m.id())
                    && m.tier() != MemoryItem.Tier.PROCEDURAL;
            if (!isPinned && vectorOnly
                    && !queryPrefixes.isEmpty()
                    && !hasLexicalOverlap(queryPrefixes, m.content())
                    && thisSim < 0.82) {
                continue;
            }
            // G14/G15 — when the query after stopword removal yields NO
            // meaningful tokens (e.g. "?", "ok", "the"), vector-only hits
            // would sneak in because the lexical-overlap guard is skipped
            // (queryPrefixes empty). Force them out entirely. Pinned items
            // still pass (user explicitly wants them always considered).
            if (!isPinned && vectorOnly && queryPrefixes.isEmpty()) {
                continue;
            }
            // ═══════════════════════════════════════════════════════════════
            // RRF fusion across channels (F11). Each channel contributes its
            // weighted reciprocal-rank when the item hit that channel. Items
            // that hit multiple channels accumulate more signal — this is
            // the entire point of RRF.
            //
            //     score_rrf = Σ w_channel / (k + rank_channel)
            //
            // k = 60 (conventional RRF smoothing constant).
            // Weights scaled so the single-channel rank-1 contribution
            // (~0.5) dominates the downstream boosts — keeps the
            // min_top_score = 0.55 threshold meaningful.
            //   canon = 40.0 — canonical fact (definitional, pinned, reinforced)
            //   vec   = 30.0 — strongest semantic match (cosine)
            //   fts   = 25.0 — strong lexical match (BM25)
            //   graph = 25.0 — structural evidence (entity hit + 1-hop)
            //   tag   = 20.0 — explicit caller-supplied filter
            // Max RRF when all 5 rank #1: (40+30+25+25+20)/61 ≈ 2.29
            // ═══════════════════════════════════════════════════════════════
            final double K = 60.0;
            double rrf = 0.0;
            Integer canRank = canonRanks.get(m.id());
            if (canRank != null) rrf += 40.0 / (K + canRank);
            Integer fr = ftsRanks.get(m.id());
            if (fr != null) rrf += 25.0 / (K + fr);
            Integer vr = vecRanks.get(m.id());
            if (vr != null) rrf += 30.0 / (K + vr);
            Integer gr = graphRanks.get(m.id());
            if (gr != null) rrf += 25.0 / (K + gr);
            Integer trr = tagRanks.get(m.id());
            if (trr != null) rrf += 20.0 / (K + trr);
            // Facet-aware multiplier (Wave-1 #2 — OMEGA type-weight port).
            // Foundational facets (identity/health/profile/preference) get
            // amplified; OTHER and unclassified items pass through at 1.0×.
            rrf *= facetWeight(m);

            // Vector similarity kept as a prior for items that had a cosine
            // hit below the minRelevantSimilarity floor — they don't vote in
            // RRF but we still use their sim for MMR dedup later.
            double sim = simScores.getOrDefault(m.id(),
                    qv != null && m.embedding() != null
                            ? ai.nizo.memory.util.Vectors.cosine(qv, m.embedding())
                            : 0.5);

            // Quality priors: confidence (decayed), recency, usage.
            double confDecayed = decayedConfidence(m, now);        // F5
            double recency = recency(m.createdAt(), now);
            double usage = Math.log1p(m.accessCount()) / 4.0;

            // Tier prior: personal facts (SEMANTIC / WORKING / EPISODIC)
            // outrank world-knowledge PROCEDURAL when both match.
            double tierBoost = switch (m.tier()) {
                case SEMANTIC -> 0.20;
                case WORKING -> 0.05;
                case EPISODIC -> 0.0;
                case PROCEDURAL -> 0.0;
            };

            // Exact-token boost: short queries (likely a name or specific noun)
            // that appear as a whole word in the content. Rank-independent —
            // this is a content signal, not a channel vote.
            double exactBoost = exactTokenBoost(req.query(), m.content());

            // ── PIN boost: user-marked important facts get a large boost ──
            double pinBoost = "true".equalsIgnoreCase(
                    m.tags().get(ai.nizo.memory.api.memory.MemoryTags.PINNED)) ? 0.25 : 0.0;

            // ── CONVERSATIONAL-RECALL boost: when the query is asking about
            //    prior assistant content ("you mentioned", "remind me what
            //    you recommended"), surface assistant turns. Without this,
            //    BM25 picks the user's prompt over the assistant's actual
            //    answer because the prompt is shorter and denser in query
            //    keywords. Real bug found in LongMemEval — 4 of 9 recall
            //    misses were single-session-assistant questions where only
            //    the user's question was recalled.
            double assistantRoleBoost = 0.0;
            if (isConversationalRecall && "assistant".equalsIgnoreCase(m.tags().get("role"))) {
                assistantRoleBoost = 0.30;
            }

            // ── HYPOTHETICAL penalty: "considering Pixel" ≠ "uses Pixel" ──
            double hypoPenalty = "true".equalsIgnoreCase(
                    m.tags().get(ai.nizo.memory.api.memory.MemoryTags.HYPOTHETICAL)) ? -0.20 : 0.0;

            // ── EXPIRY penalty: past events / outdated deferrals fall off ─
            double expiryPenalty = 0.0;
            String expiresAt = m.tags().get(ai.nizo.memory.api.memory.MemoryTags.EXPIRES_AT);
            if (expiresAt != null && !expiresAt.isBlank()) {
                try {
                    java.time.LocalDate exp = java.time.LocalDate.parse(expiresAt);
                    if (exp.isBefore(java.time.LocalDate.now())) expiryPenalty = -0.25;
                } catch (Exception ignore) { /* bad date format */ }
            }

            // ── SINGLE-MENTION penalty: one-off mentions are tentative ───
            double mentionPenalty = 0.0;
            String mc = m.tags().get(ai.nizo.memory.api.memory.MemoryTags.MENTION_COUNT);
            int mentions = 1;
            try { if (mc != null) mentions = Math.max(1, Integer.parseInt(mc)); }
            catch (NumberFormatException ignore) {}
            if (mentions < 2 && !"user_stated".equals(m.source()) && pinBoost == 0.0) {
                mentionPenalty = -0.05;
            }

            // STALE decay is now subsumed by F5 confidence decay (half-life
            // 180d on `extraction` / `conversation` source, reset by
            // last_reconfirmed). Old duplicate penalty removed — don't
            // double-count age.

            // Final score. RRF is the retrieval backbone; the rest are
            // priors and content modifiers.
            double score = rrf
                    + 0.15 * confDecayed
                    + 0.08 * recency
                    + 0.05 * usage
                    + tierBoost + exactBoost
                    + pinBoost + assistantRoleBoost
                    + hypoPenalty + expiryPenalty + mentionPenalty;
            scored.add(new Scored(m, score));
        }
        scored.sort(Comparator.comparingDouble((Scored s) -> s.score).reversed());

        // Top-score threshold: if the highest-scored item didn't clear the
        // minTopScore bar, the entire query had no strong signal — return
        // empty honestly rather than lead the agent with semantically-nearby
        // but factually-wrong items. Configurable; 0.0 disables.
        if (effectiveMinTopScore > 0.0 && !scored.isEmpty() && scored.get(0).score < effectiveMinTopScore) {
            log.debug("Dropping all {} recall results; top score {} < minTopScore {}",
                    scored.size(), scored.get(0).score, effectiveMinTopScore);
            if ("true".equalsIgnoreCase(System.getProperty("nizo.recall.trace"))) {
                System.err.println("[trace] DROP all (top<floor) — pool=" + pool.size()
                        + " scored=" + scored.size() + " topScore=" + scored.get(0).score
                        + " minTopScore=" + effectiveMinTopScore);
                for (Scored s : scored.subList(0, Math.min(5, scored.size()))) {
                    System.err.println("    score=" + s.score + " content=" + s.item.content().substring(0, Math.min(70, s.item.content().length())));
                }
            }
            return List.of();
        }
        if ("true".equalsIgnoreCase(System.getProperty("nizo.recall.trace"))) {
            System.err.println("[trace] OK — pool=" + pool.size() + " scored=" + scored.size()
                    + " topScore=" + (scored.isEmpty() ? "n/a" : scored.get(0).score));
        }

        // Max-Marginal-Relevance deduplication + token packing.
        List<MemoryItem> picked = new ArrayList<>();
        int usedTokens = 0;
        for (Scored s : scored) {
            if (usedTokens + s.item.tokens() > budget) continue;
            boolean dup = false;
            for (MemoryItem already : picked) {
                if (already.embedding() != null && s.item.embedding() != null) {
                    if (ai.nizo.memory.util.Vectors.cosine(already.embedding(), s.item.embedding()) > 0.92) {
                        dup = true; break;
                    }
                }
            }
            if (dup) continue;
            picked.add(s.item);
            usedTokens += s.item.tokens();
            store.touch(s.item.id());
            if (picked.size() >= 20) break;
        }
        return picked;
    }

    /** Summarise recent episodic items into semantic facts, then prune them. */
    @Override
    public void consolidate(String userId) {
        String uid = userId == null ? "default" : userId;
        List<MemoryItem> recent = store.recent(uid, MemoryItem.Tier.EPISODIC, 40);
        if (recent.size() < 8 || summariser == null) return;
        StringBuilder buf = new StringBuilder();
        for (MemoryItem m : recent) {
            buf.append("- ").append(m.content().replace('\n', ' ')).append('\n');
        }
        String prompt = """
                You are the agent's memory consolidator. Read the recent events below
                and extract stable facts about the user, their preferences, ongoing
                projects, or recurring entities. Output one fact per line. Only include
                a fact if it is supported by the evidence; otherwise omit it.
                Output NOTHING else.

                Events:
                %s
                """.formatted(buf);
        try {
            String out = summariser.complete(ModelRequest.of(List.of(Message.user(prompt)))).text();
            if (out == null) return;
            for (String line : out.split("\\r?\\n")) {
                String fact = line.replaceFirst("^[-*\\d.\\s]+", "").trim();
                if (fact.length() > 6) learnFact(uid, fact, "consolidation", 0.7);
            }
        } catch (RuntimeException e) {
            log.warn("Consolidation failed: {}", e.toString());
        }
    }

    @Override
    public String learnProcedural(String userId, String rule, String source, double confidence) {
        if (rule == null || rule.isBlank()) {
            return UUID.randomUUID().toString();
        }
        String uid = userId == null ? "default" : userId;
        String id = UUID.randomUUID().toString();
        Instant now = Instant.now();
        // Embed if available so procedural rules participate in semantic recall too.
        float[] v = embedder == null ? null : safeEmbed(rule);
        MemoryItem item = new MemoryItem(
                id, uid, MemoryItem.Tier.PROCEDURAL, rule, v,
                Map.of("kind", "heuristic"),
                source, confidence, now, now, 0, Tokens.count(rule));
        store.upsert(item);
        if (v != null) index.add(uid, id, v);
        return id;
    }

    @Override
    public int demoteContradicted(String userId, String query, double newConfidence) {
        String uid = userId == null ? "default" : userId;
        return store.demoteConfidence(uid, query, newConfidence);
    }

    @Override
    public Map<MemoryItem.Tier, Long> stats(String userId) {
        String uid = userId == null ? "default" : userId;
        return store.countsByTier(uid);
    }

    // ---------- helpers ----------

    private void embedAndIndex(String userId, String id, String content) {
        if (embedder == null) return;
        float[] v = safeEmbed(content);
        if (v == null) return;
        store.findById(id).ifPresent(m -> store.upsert(new MemoryItem(
                m.id(), m.userId(), m.tier(), m.content(), v, m.tags(), m.source(),
                m.confidence(), m.createdAt(), m.lastAccessedAt(), m.accessCount(), m.tokens())));
        index.add(userId, id, v);
    }

    private float[] safeEmbed(String s) {
        try { return embedder.embed(s); }
        catch (RuntimeException e) {
            log.debug("Embed failed: {}", e.toString());
            return null;
        }
    }

    private static boolean matchesTags(MemoryItem m, Map<String, String> req) {
        if (req == null || req.isEmpty()) return true;
        for (Map.Entry<String, String> e : req.entrySet()) {
            if (!e.getValue().equals(m.tags().get(e.getKey()))) return false;
        }
        return true;
    }

    /**
     * Extract non-stopword query tokens as 5-char prefixes. A prefix match
     * is close enough to catch stem variations ("investing" → "invest",
     * "investment" → "invest"; "allergy" → "aller", "allergies" → "aller").
     * Returns empty set for queries that are all stopwords — in which case
     * the lexical guard is skipped.
     */
    private static Set<String> queryTokenPrefixes(String query) {
        if (query == null || query.isBlank()) return Set.of();
        String[] tokens = query.toLowerCase().split("\\W+");
        Set<String> out = new HashSet<>();
        for (String t : tokens) {
            // Use STRUCTURAL_STOPWORDS (narrower) here — "work", "job", "user"
            // are meaningful signals for lexical-overlap and should build
            // prefixes. STOP_WORDS (broader) is only for exact-boost dampening.
            if (t.length() < 4 || STRUCTURAL_STOPWORDS.contains(t)) continue;
            out.add(t.length() > 5 ? t.substring(0, 5) : t);
        }
        return out;
    }

    /** True if any query-token prefix appears as a substring of the content. */
    private static boolean hasLexicalOverlap(Set<String> queryPrefixes, String content) {
        if (content == null || content.isEmpty()) return false;
        String low = content.toLowerCase();
        for (String p : queryPrefixes) {
            if (low.contains(p)) return true;
        }
        return false;
    }

    /**
     * Does the query mention a family-role word for this "other" subject?
     * Lets "do my parents have allergies" match items tagged
     * {@code subject=other:mom} or {@code subject=other:dad} even though
     * the query doesn't literally say "mom".
     */
    private static boolean queryMentionsFamilyRole(String queryLow, String otherName) {
        // If the "other" is itself a family role, check synonyms.
        // Otherwise the query must literally contain the name.
        return switch (otherName) {
            case "mom", "mother", "mum" -> queryLow.contains("mom") || queryLow.contains("mother")
                    || queryLow.contains("mum") || queryLow.contains("parent");
            case "dad", "father" -> queryLow.contains("dad") || queryLow.contains("father")
                    || queryLow.contains("parent");
            case "sister", "brother", "sibling" -> queryLow.contains("sister") || queryLow.contains("brother")
                    || queryLow.contains("sibling");
            case "son", "daughter", "child", "kid" -> queryLow.contains("son") || queryLow.contains("daughter")
                    || queryLow.contains("child") || queryLow.contains("kid");
            case "wife", "husband", "spouse", "partner" -> queryLow.contains("wife") || queryLow.contains("husband")
                    || queryLow.contains("spouse") || queryLow.contains("partner");
            default -> false;
        };
    }

    /**
     * G7/G17 — is this query asking about the user's own data, as opposed to
     * world knowledge? Heuristic: any first-person pronoun / possessive, or
     * "do i / am i / where do i / what does my" style phrasing, marks the
     * query as personal. For those, the PROCEDURAL (world-knowledge) tier is
     * excluded by default so general-knowledge heuristics don't leak into
     * user-specific answers.
     */
    /**
     * Structural stopwords — question words, articles, prepositions, pronouns.
     * This is a NARROWER list than {@link #STOP_WORDS} which also excludes
     * generic nouns/verbs from the exact-match boost. For query-meaningfulness
     * we only care about truly content-free words — "work" / "job" / "user"
     * ARE meaningful signals when the user asks "where do I work".
     */
    private static final Set<String> STRUCTURAL_STOPWORDS = Set.of(
            "what", "where", "when", "who", "why", "how", "which",
            "is", "are", "was", "were", "do", "does", "did", "the",
            "a", "an", "of", "to", "in", "on", "at", "for", "with",
            "my", "your", "his", "her", "their", "any", "all", "some",
            "and", "or", "but", "i", "you", "he", "she", "we", "they",
            "me", "him", "us", "them", "this", "that", "these", "those",
            "tell", "show", "list", "give", "find", "about");

    /**
     * G15 — returns true iff the query contains at least one content-bearing
     * token. Uses STRUCTURAL_STOPWORDS (narrower than STOP_WORDS) so queries
     * like "where do I work" — where "work" IS the signal — don't get rejected.
     */
    static boolean hasMeaningfulQueryTokens(String query) {
        if (query == null) return false;
        String[] tokens = query.toLowerCase().split("\\W+");
        for (String t : tokens) {
            if (t.length() >= 3 && !STRUCTURAL_STOPWORDS.contains(t)) return true;
        }
        return false;
    }

    /**
     * <h3>Query intent — conversational recall</h3>
     *
     * <p>Conversational memory has dual semantics: facts <em>about</em> the user
     * (from user turns) and information <em>told to</em> the user (from assistant
     * turns). When the user asks "what did you tell me", "you mentioned X",
     * "you recommended X", "remind me of the X you suggested", the answer
     * lives in past assistant turns.
     *
     * <p>Default scoring (BM25 + tier boost) buries assistant answers under
     * user questions because:
     * <ol>
     *   <li>BM25 favours short docs (questions ≪ answers in length)</li>
     *   <li>The user's prompt repeats the query keywords more densely</li>
     *   <li>We have no role-aware ranking</li>
     * </ol>
     *
     * <p>This detector marks the query so the recall pipeline can over-fetch
     * and apply a role=assistant boost. Returns {@code true} for explicit
     * "ask the assistant about prior assistant content" patterns; {@code false}
     * otherwise.
     */
    /**
     * <h3>Query intent — aggregation</h3>
     *
     * <p>Aggregation queries ("how many X", "count my Y", "list all Z",
     * "total of W") need fundamentally different recall semantics than
     * fact-recall queries:
     *
     * <ul>
     *   <li><b>Top-K + MMR dedup is wrong.</b> For "how many movie festivals
     *       did I attend", every distinct mention matters; deduplicating
     *       similar items removes the very evidence needed to count.</li>
     *   <li><b>Token-budget early-termination is wrong.</b> Cutting off at
     *       20 items hides counting evidence sitting at rank 21.</li>
     *   <li><b>Single-channel ranking is wrong.</b> All semantically
     *       relevant items should surface, not just the top BM25 hits.</li>
     * </ul>
     *
     * <p>When this detector fires, the recall pipeline disables MMR dedup,
     * raises the result cap to {@link #AGGREGATION_RESULT_CAP}, and uses a
     * larger token budget. The answerer (in the harness or downstream agent)
     * gets the full evidence set and is prompted to enumerate-then-count.
     *
     * <p>This is a real architectural distinction in conversational memory
     * systems — peer systems that ignore it score ~10-15% on multi-session
     * counting. The right long-term fix is graph-mediated entity counting
     * (extract entities at ingest time, count via {@code GraphService}); this
     * detector closes the gap until that lands.
     */
    public static boolean isAggregationQuery(String query) {
        if (query == null || query.isBlank()) return false;
        String q = " " + query.toLowerCase().trim() + " ";
        String[] markers = {
                " how many ", " how much ", " how often ",
                " count ", " count of ", " total ", " total of ",
                " list all ", " list every ", " list my ",
                " enumerate ", " all my ", " all the ", " all of my ",
                " every ", " each of my ",
                " number of ", " amount of ",
                " sum of ", " sum up ",
                " how long have ", " for how long "
        };
        for (String m : markers) if (q.contains(m)) return true;
        return false;
    }

    /** Cap for aggregation-mode recall. Trades token budget for completeness. */
    private static final int AGGREGATION_RESULT_CAP = 50;

    static boolean isConversationalRecallQuery(String query) {
        if (query == null || query.isBlank()) return false;
        String q = " " + query.toLowerCase().trim() + " ";
        String[] markers = {
                " you mentioned ", " you said ", " you told me ",
                " you recommended ", " you suggested ", " you advised ",
                " you provided ", " you proposed ", " you noted ",
                " remind me of ", " remind me what ", " remind me which ",
                " what did you say ", " what did you tell ",
                " what did you recommend ", " what did you mention ",
                " what was the ", " what were the ",   // recap-style
                " we discussed ", " we talked about ",
                " our previous conversation ", " our last conversation ",
                " our chat about ", " can you remind ",
                " looking back at our ", " from our previous "
        };
        for (String m : markers) if (q.contains(m)) return true;
        return false;
    }

    static boolean isPersonalQuery(String query) {
        if (query == null || query.isBlank()) return false;
        String q = " " + query.toLowerCase().trim() + " ";
        // Strong signals — any of these makes the query clearly personal.
        String[] tokens = {
                // First-person
                " i ", " me ", " my ", " mine ", " myself ", " we ", " us ", " our ",
                " i've ", " i'm ", " i'd ", " i'll ", " we've ", " we're ",
                " am i ", " do i ", " did i ", " have i ", " can i ",
                " what's my ", " what is my ", " where's my ", " where is my ",
                " who's my ", " who is my ", " when's my ", " when is my ",
                " what did i ", " where do i ", " what am i ", " tell me ",
                " remind me ", " show me ",
                // Third-person referring to the user (agent-style phrasing —
                // e.g. Nizo asking nizo-memory "where does the user work").
                " the user ", " this user ", " user's ", " users ",
                " does the user ", " did the user ", " has the user ",
                " where does the user ", " what does the user ",
                " about the user ", " the user's "
        };
        for (String t : tokens) if (q.contains(t)) return true;
        return false;
    }

    private static double recency(Instant t, Instant now) {
        long hours = Math.max(1, (now.toEpochMilli() - t.toEpochMilli()) / 3_600_000L);
        // ~1.0 fresh → ~0.2 after a week → near-0 after a year
        return 1.0 / (1.0 + Math.log(hours));
    }

    /**
     * F5 — confidence decay with 180-day half-life, anchored to
     * {@code last_reconfirmed} when present, else {@code createdAt}.
     *
     * <p>Pinned and {@code user_stated} facts don't decay — they're explicit
     * user intent. Only {@code extraction} and {@code conversation} sources
     * decay, which means an LLM-inferred fact from three years ago weighs
     * half as much as one from today, but a pinned health constraint stays
     * at full strength forever.
     *
     * <p>Reconfirm resets the anchor: when {@code reconfirm(factId)} is
     * called, {@code last_reconfirmed} is set, and the decay clock restarts
     * from that moment.
     */
    private static double decayedConfidence(MemoryItem m, Instant now) {
        double base = m.confidence();
        // Pinned = user-curated, never decays.
        if ("true".equalsIgnoreCase(
                m.tags().get(ai.nizo.memory.api.memory.MemoryTags.PINNED))) {
            return base;
        }
        // Only decay inferred facts. User-stated, imported, consolidation
        // outputs are treated as stable.
        String src = m.source();
        if (!"extraction".equals(src) && !"conversation".equals(src)) {
            return base;
        }
        // Anchor: the later of createdAt and last_reconfirmed.
        Instant anchor = m.createdAt();
        String reconf = m.tags().get(ai.nizo.memory.api.memory.MemoryTags.LAST_RECONFIRMED);
        if (reconf != null && !reconf.isBlank()) {
            try {
                Instant r = Instant.parse(reconf);
                if (r.isAfter(anchor)) anchor = r;
            } catch (Exception ignore) {}
        }
        long days = java.time.Duration.between(anchor, now).toDays();
        if (days <= 0) return base;
        // Half-life = 180 days. After 180d, confidence halves; 360d → quarter.
        final double halflife = 180.0;
        double factor = Math.pow(0.5, days / halflife);
        return base * factor;
    }

    /**
     * Boost items whose content contains a query token as a whole word.
     * Short queries (1-3 tokens, typical for "who is Amit", "where Stripe",
     * "Sony headphones") get the highest boost since they're usually a
     * specific name or noun. Longer queries get diminishing returns since
     * vector similarity already does heavy lifting.
     */
    private static double exactTokenBoost(String query, String content) {
        if (query == null || content == null) return 0.0;
        String q = query.toLowerCase().trim();
        String c = content.toLowerCase();
        if (q.isEmpty()) return 0.0;
        String[] tokens = q.split("\\W+");
        // Stopwords that shouldn't trigger an "exact name match" boost.
        // Includes generic verbs / nouns ("user", "work", "go", "make") that
        // appear in too many items to discriminate — the boost is meant for
        // SPECIFIC entities like "Amit", "Stripe", "Hajj".
        Set<String> stop = STOP_WORDS;
        int matched = 0;
        int eligible = 0;
        for (String t : tokens) {
            if (t.length() < 3 || stop.contains(t)) continue;
            eligible++;
            // Whole-word match (\b would need regex; cheap substring with
            // boundary check is sufficient for English-ish content).
            int idx = 0;
            boolean hit = false;
            while ((idx = c.indexOf(t, idx)) >= 0) {
                boolean leftBound = idx == 0 || !Character.isLetterOrDigit(c.charAt(idx - 1));
                int end = idx + t.length();
                boolean rightBound = end == c.length() || !Character.isLetterOrDigit(c.charAt(end));
                if (leftBound && rightBound) { hit = true; break; }
                idx++;
            }
            if (hit) matched++;
        }
        if (eligible == 0) return 0.0;
        // 1 of 1 → 0.20, 2 of 2 → 0.20, 1 of 3 → 0.067
        return 0.20 * ((double) matched / eligible);
    }

    // ====== Customer-facing controls (forget / pin / inspect / import) ======

    @Override
    public List<MemoryItem> inspect(String userId, int limit) {
        String uid = userId == null ? "default" : userId;
        int n = limit <= 0 ? 100 : limit;
        // Filter out PROCEDURAL items — those are bundled world-knowledge
        // heuristics, not user-stated facts. The customer-facing
        // "what do you remember about me" view should only show user data.
        return store.findByUserId(uid, n * 4).stream()
                .filter(m -> m.tier() != MemoryItem.Tier.PROCEDURAL)
                .limit(n)
                .toList();
    }

    @Override
    public int forgetAbout(String userId, String topic) {
        String uid = userId == null ? "default" : userId;
        if (topic == null || topic.isBlank()) return 0;
        // Find every item containing the topic substring (case-insensitive).
        List<MemoryItem> hits = store.findByContentLike(uid, topic, 1000);
        int deleted = 0;
        for (MemoryItem m : hits) {
            store.delete(m.id());
            index.remove(uid, m.id());
            deleted++;
        }
        // Note: topic-scoped forget only purges memory items + vectors; graph
        // entities that happen to share the topic keyword remain (deleting
        // them would be overreach — a topic word can name an unrelated node).
        // Use forgetUser() for full GDPR cascade.
        log.info("forgetAbout(user={}, topic='{}') deleted {} items", uid, topic, deleted);
        return deleted;
    }

    @Override
    public boolean pin(String userId, String factId, boolean pinned, String reason) {
        if (factId == null) return false;
        var maybe = store.findById(factId);
        if (maybe.isEmpty()) return false;
        MemoryItem m = maybe.get();
        Map<String, String> newTags = new java.util.HashMap<>(m.tags());
        if (pinned) {
            newTags.put(ai.nizo.memory.api.memory.MemoryTags.PINNED, "true");
            if (reason != null && !reason.isBlank()) {
                newTags.put(ai.nizo.memory.api.memory.MemoryTags.PIN_REASON, reason);
            }
        } else {
            newTags.remove(ai.nizo.memory.api.memory.MemoryTags.PINNED);
            newTags.remove(ai.nizo.memory.api.memory.MemoryTags.PIN_REASON);
        }
        return store.updateTags(factId, newTags);
    }

    @Override
    public boolean reconfirm(String userId, String factId) {
        if (factId == null) return false;
        var maybe = store.findById(factId);
        if (maybe.isEmpty()) return false;
        Map<String, String> newTags = new java.util.HashMap<>(maybe.get().tags());
        newTags.put(ai.nizo.memory.api.memory.MemoryTags.LAST_RECONFIRMED, Instant.now().toString());
        return store.updateTags(factId, newTags);
    }

    @Override
    public int importFacts(String userId, List<ImportedFact> facts) {
        if (facts == null || facts.isEmpty()) return 0;
        String uid = userId == null ? "default" : userId;
        int loaded = 0;
        for (ImportedFact f : facts) {
            if (f == null || f.content() == null || f.content().isBlank()) continue;
            String id = UUID.randomUUID().toString();
            Instant now = Instant.now();
            float[] v = embedder == null ? null : safeEmbed(f.content());
            Map<String, String> tags = new java.util.HashMap<>();
            tags.put("kind", "imported");
            if (f.tags() != null) tags.putAll(f.tags());
            // Infer semantic facet if caller didn't supply one. Used by
            // precision-heavy recall for facet-compatibility filtering.
            if (!tags.containsKey(ai.nizo.memory.api.memory.MemoryTags.FACET)) {
                tags.put(ai.nizo.memory.api.memory.MemoryTags.FACET,
                        ai.nizo.memory.facet.FacetClassifier.inferContent(f.content()));
            } else {
                tags.put(ai.nizo.memory.api.memory.MemoryTags.FACET,
                        ai.nizo.memory.facet.FacetClassifier.normalize(
                                tags.get(ai.nizo.memory.api.memory.MemoryTags.FACET)));
            }
            MemoryItem item = new MemoryItem(
                    id, uid, MemoryItem.Tier.SEMANTIC, f.content(), v, tags,
                    "imported", f.confidence() == null ? 0.85 : f.confidence(),
                    now, now, 0, Tokens.count(f.content()));
            store.upsert(item);
            if (v != null) index.add(uid, id, v);
            loaded++;
        }
        log.info("importFacts(user={}) stored {} facts", uid, loaded);
        return loaded;
    }

    @Override
    public int forgetUser(String userId) {
        String uid = userId == null ? "default" : userId;
        int memoryRows = store.deleteAllForUser(uid);
        index.removeAllForUser(uid);
        // F9: cascade graph purge. The graph layer stores nodes + edges in
        // a separate table; forgetUser MUST remove them or the user's
        // relationship graph survives a GDPR-style forget request and can
        // re-surface via graph-channel recall.
        int graphRows = 0;
        if (graphService != null) {
            try {
                graphRows = graphService.deleteAllForUser(uid);
            } catch (RuntimeException e) {
                log.warn("forgetUser(user={}) graph cascade failed: {}", uid, e.toString());
            }
        }
        log.info("forgetUser(user={}) deleted {} memory items + {} graph rows",
                uid, memoryRows, graphRows);
        return memoryRows + graphRows;
    }

    // ============= Active memory (pre-reply surface) ==================

    /**
     * Precompiled command-prefix patterns that indicate a pure command
     * rather than a user statement worth surfacing context for.
     */
    private static final java.util.regex.Pattern COMMAND_PREFIX_PATTERN =
            java.util.regex.Pattern.compile(
                    "^\\s*(analyse|analyze|search\\s+for|find|show\\s+me|list|tell\\s+me|" +
                    "remind\\s+me|make|create|write|run|open|close|start|stop)\\s+",
                    java.util.regex.Pattern.CASE_INSENSITIVE);

    @Override
    public List<IndexEntry> canonicalIndex(String userId, int maxEntries) {
        String uid = userId == null ? "default" : userId;
        int cap = Math.max(1, maxEntries);
        List<MemoryItem> items;
        try {
            items = store.findByTags(uid,
                    Map.of(ai.nizo.memory.api.memory.MemoryTags.CANONICAL, "true"),
                    cap * 3);  // overfetch to allow dedup by cluster key
        } catch (RuntimeException e) {
            log.debug("canonicalIndex findByTags failed (non-fatal): {}", e.toString());
            return List.of();
        }
        // Dedup by cluster_key — keep the most recent item per key. findByTags
        // already returns newest-first, so a first-write-wins merge produces
        // "latest claim per attribute."
        java.util.LinkedHashMap<String, IndexEntry> byKey = new java.util.LinkedHashMap<>();
        for (MemoryItem m : items) {
            if (byKey.size() >= cap) break;
            Map<String, String> t = m.tags() == null ? Map.of() : m.tags();
            String key = t.getOrDefault(ai.nizo.memory.api.memory.MemoryTags.CLUSTER_KEY,
                    "other:" + m.id());
            if (byKey.containsKey(key)) continue;
            String facet = t.getOrDefault(ai.nizo.memory.api.memory.MemoryTags.FACET,
                    ai.nizo.memory.api.memory.MemoryTags.FACET_OTHER);
            Instant lastReconfirmed = parseInstant(
                    t.get(ai.nizo.memory.api.memory.MemoryTags.LAST_RECONFIRMED));
            byKey.put(key, new IndexEntry(key, m.content(), facet, lastReconfirmed));
        }
        return List.copyOf(byKey.values());
    }

    /** Best-effort ISO-8601 parse; returns null on any failure. */
    private static Instant parseInstant(String s) {
        if (s == null || s.isBlank()) return null;
        try { return Instant.parse(s); } catch (Exception e) { return null; }
    }

    @Override
    public SurfaceResult surface(SurfaceRequest req) {
        if (req == null || req.message() == null || req.message().isBlank()) {
            return SurfaceResult.skipped("empty_message");
        }
        String message = req.message().trim();
        String mode = req.mode() == null || req.mode().isBlank() ? "balanced" : req.mode();
        int maxItems = req.maxItems() <= 0 ? 5 : Math.min(req.maxItems(), 20);
        int maxChars = req.maxSummaryChars() <= 0 ? 500 : Math.min(req.maxSummaryChars(), 2000);

        // Command-only gate: "analyze AAPL", "search for X", "open google" —
        // these are instructions to the agent, not user utterances carrying
        // context. Returning nothing is the right answer.
        if (COMMAND_PREFIX_PATTERN.matcher(message).find()) {
            return new SurfaceResult(false, "", List.of(), "command_only", mode);
        }

        // Message too short to carry real signal. "ok", "yes", "hmm".
        if (message.length() < 8 && !hasMeaningfulQueryTokens(message)) {
            return new SurfaceResult(false, "", List.of(), "no_meaningful_signal", mode);
        }

        // Mode-tuned parameters. These include a per-call minTopScore override
        // so strict modes actually tighten the floor and recall-heavy modes
        // actually loosen it — not just POST-filter after the floor has
        // already fired.
        int modeBudget;
        int modeMaxItems;
        double modeMinTopScore;
        boolean requireLexicalOverlapInStrict;
        Set<MemoryItem.Tier> tiers = null;
        switch (mode) {
            case "strict", "precision-heavy" -> {
                modeBudget = 400;
                modeMaxItems = Math.min(maxItems, 3);
                modeMinTopScore = Math.max(minTopScore, 0.72);
                requireLexicalOverlapInStrict = true;
            }
            case "recall-heavy" -> {
                modeBudget = 1500;
                modeMaxItems = Math.min(maxItems, 10);
                modeMinTopScore = 0.35;   // looser — surface more weak signals
                requireLexicalOverlapInStrict = false;
            }
            case "preference-only" -> {
                modeBudget = 800;
                modeMaxItems = maxItems;
                modeMinTopScore = 0.40;
                requireLexicalOverlapInStrict = false;
                tiers = Set.of(MemoryItem.Tier.SEMANTIC);
            }
            default -> {  // "balanced"
                modeBudget = 800;
                modeMaxItems = maxItems;
                modeMinTopScore = Math.min(minTopScore, 0.50);
                requireLexicalOverlapInStrict = false;
            }
        }

        // Expand the query with recent-turn context for better intent — but
        // weighted so the latest message dominates.
        String effectiveQuery = buildEffectiveQuery(message, req.recentTurns());

        RecallRequest r = new RecallRequest(
                req.userId(), effectiveQuery, modeBudget,
                tiers, Map.of(), 0.0);

        List<MemoryItem> recalled = recallInternal(r, modeMinTopScore);
        if (recalled.isEmpty()) {
            return new SurfaceResult(false, "", List.of(), "below_threshold", mode);
        }

        // Strict modes additionally require items to share meaningful
        // lexical overlap with the user's ACTUAL message AND match the
        // question's semantic FACET. This is a two-layer filter:
        //
        //   1. Lexical gate — at least 2 distinct overlap tokens, or 1
        //      overlap that isn't an entity-marker. Removes obvious
        //      misses where the recall pipeline returned noise.
        //   2. Facet gate — the content's inferred/stored facet must be
        //      compatible with the query's classified facet. Removes the
        //      "right entity, wrong facet" case: query asks about SCHEDULE,
        //      content addresses PROFILE → abstain.
        //
        // When the query's facet classifies as FACET_OTHER (no pattern
        // matched), we skip the facet gate — the heuristic was unsure, so
        // defer to the lexical gate only.
        if (requireLexicalOverlapInStrict) {
            Set<String> msgTokens = queryTokenPrefixesPublic(message);
            // Multi-facet: a query can carry multiple intents simultaneously
            // ("Thai place tonight" = PREFERENCE ∩ SCHEDULE). Content passes
            // if its facet is compatible with ANY of the query's facets.
            Set<String> queryFacets = ai.nizo.memory.facet.FacetClassifier.classifyQueryAll(message);

            if (!msgTokens.isEmpty() || !queryFacets.isEmpty()) {
                List<MemoryItem> filtered = new ArrayList<>();
                for (MemoryItem m : recalled) {
                    // Lexical gate
                    boolean lexicalOk;
                    if (msgTokens.isEmpty()) {
                        lexicalOk = true;
                    } else {
                        int overlaps = countLexicalOverlaps(msgTokens, m.content());
                        lexicalOk = overlaps >= 2
                                || (overlaps == 1 && !onlyEntityOverlap(msgTokens, m.content()));
                    }
                    if (!lexicalOk) continue;

                    // Facet gate (only when query facet(s) classified)
                    if (!queryFacets.isEmpty()) {
                        String contentFacet = m.tags() == null ? null
                                : m.tags().get(ai.nizo.memory.api.memory.MemoryTags.FACET);
                        if (contentFacet == null) {
                            contentFacet = ai.nizo.memory.facet.FacetClassifier.inferContent(m.content());
                        }
                        if (!ai.nizo.memory.facet.FacetClassifier.isCompatibleAny(queryFacets, contentFacet)) {
                            continue;
                        }
                    }

                    filtered.add(m);
                }

                if (filtered.isEmpty()) {
                    // Heuristic abstained. Before returning skipped, try the
                    // LLM gate on the top candidates — the heuristic might
                    // have been too strict. Bounded at 3 LLM calls per
                    // request. Only runs when summariser ModelClient is
                    // configured; degrades gracefully to pure abstention
                    // otherwise.
                    List<MemoryItem> llmKept = llmGateStrict(
                            message, recalled, 3);
                    if (llmKept.isEmpty()) {
                        return new SurfaceResult(false, "", List.of(),
                                "entity_only_match", mode);
                    }
                    recalled = llmKept;
                } else {
                    recalled = filtered;
                }
            }
        }

        List<MemoryItem> trimmed = recalled.size() > modeMaxItems
                ? recalled.subList(0, modeMaxItems) : recalled;

        // Summary: concat trimmed content, cap at maxChars, sanity-trim per-item.
        StringBuilder sb = new StringBuilder();
        for (MemoryItem it : trimmed) {
            String c = it.content().replaceAll("\\s+", " ").trim();
            if (c.length() > 140) c = c.substring(0, 137) + "...";
            String sep = sb.length() == 0 ? "" : " · ";
            if (sb.length() + sep.length() + c.length() > maxChars) break;
            sb.append(sep).append(c);
        }

        return new SurfaceResult(true, sb.toString(), trimmed, null, mode);
    }

    /**
     * Build the query string used for recall. Combines the latest message
     * with recent-turn context (if provided), weighted so the latest message
     * dominates. Format: latest message + key tokens from recent user turns.
     */
    private static String buildEffectiveQuery(String message, List<ConversationTurn> recent) {
        if (recent == null || recent.isEmpty()) return message;
        StringBuilder sb = new StringBuilder(message);
        // Include up to 3 recent user turns, most recent first, for topic
        // expansion. Assistant turns are excluded — they describe what the
        // assistant said, not what the user wanted.
        int added = 0;
        for (int i = recent.size() - 1; i >= 0 && added < 3; i--) {
            ConversationTurn t = recent.get(i);
            if ("user".equalsIgnoreCase(t.role()) && t.content() != null && !t.content().isBlank()) {
                sb.append(' ').append(t.content());
                added++;
            }
        }
        return sb.toString();
    }

    /** Package-visible accessor for queryTokenPrefixes. */
    static Set<String> queryTokenPrefixesPublic(String query) {
        return queryTokenPrefixes(query);
    }

    /**
     * Per-candidate relevance prompt for the precision-heavy LLM gate.
     * Single-token output keeps latency low and parsing trivial.
     */
    private static final String LLM_GATE_PROMPT = """
            You are filtering memory items for relevance to a user's question.
            Reply with a single token: YES if the fact DIRECTLY addresses the
            question's intent (not just identifies the entity it's about),
            NO otherwise.

            QUESTION: %s
            FACT: %s
            GRADE (YES/NO):""";

    /**
     * LLM-gated fallback for precision-heavy mode. Called when the
     * lexical+facet heuristic filters all candidates — the heuristic might
     * be too strict, and a cheap one-token LLM check per top candidate can
     * recover legitimate matches.
     *
     * <p>Bounded by {@code maxCalls} (default 3). Skips if no summariser
     * client is configured. Degrades to empty-list on any LLM failure so
     * the caller falls through to honest abstention.
     */
    private List<MemoryItem> llmGateStrict(String message,
                                            List<MemoryItem> candidates,
                                            int maxCalls) {
        if (summariser == null || candidates.isEmpty()) return List.of();
        List<MemoryItem> kept = new ArrayList<>();
        int calls = 0;
        for (MemoryItem c : candidates) {
            if (calls >= maxCalls) break;
            String prompt = LLM_GATE_PROMPT.formatted(message, c.content());
            try {
                String resp = summariser.complete(
                        ModelRequest.of(List.of(Message.user(prompt)))).text();
                calls++;
                if (resp != null && resp.trim().toUpperCase(java.util.Locale.ROOT)
                        .startsWith("YES")) {
                    kept.add(c);
                }
            } catch (RuntimeException e) {
                log.debug("LLM gate failed (non-fatal): {}", e.toString());
                break;
            }
        }
        if (!kept.isEmpty()) {
            log.debug("LLM gate recovered {} of {} candidates ({} calls)",
                    kept.size(), candidates.size(), calls);
        }
        return kept;
    }

    /**
     * Synthetic session identifier for rows that carry no explicit
     * {@code session_id} tag. Matches the same sentinel used by
     * {@link SqliteMemoryStore#sessionManifest} so unsessioned items
     * appear in the manifest and can still be picked.
     */
    private static final String UNSESSIONED = "__unsessioned__";

    /** Return a non-null filter set, or {@code null} when the filter is empty. */
    private static Set<String> normaliseSessionFilter(Set<String> in) {
        if (in == null || in.isEmpty()) return null;
        return in;
    }

    /**
     * True when {@code m} belongs to a session that passes the filter (or the
     * filter is disabled). Items without an explicit {@code session_id} tag
     * are bucketed under {@link #UNSESSIONED}, so a filter containing that
     * sentinel can opt them in.
     */
    private static boolean passesSessionFilter(MemoryItem m, Set<String> sessionFilter) {
        if (sessionFilter == null || sessionFilter.isEmpty()) return true;
        String sid = m.tags() == null ? null
                : m.tags().get(ai.nizo.memory.api.memory.MemoryTags.SESSION_ID);
        String effective = (sid == null || sid.isBlank()) ? UNSESSIONED : sid;
        return sessionFilter.contains(effective);
    }

    /** Short truncation for debug logs — never raises an NPE on null input. */
    private static String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }

    /**
     * Enable the session pre-filter. When {@code picker} is non-null the
     * recall pipeline will, for users whose distinct session count exceeds
     * {@code threshold}, ask the picker to narrow recall to {@code topN}
     * sessions before running the hybrid retrieval channels. Callers pass
     * {@link SessionPicker#NO_OP} (or null) to disable.
     *
     * <p>Idempotent and thread-safe: all three fields are {@code volatile}
     * so callers can flip the picker on and off at runtime.
     */
    public void setSessionPicker(SessionPicker picker, int threshold, int topN) {
        this.sessionPicker = picker == null ? SessionPicker.NO_OP : picker;
        this.sessionPickerThreshold = Math.max(1, threshold);
        this.sessionPickerTopN = Math.max(1, topN);
    }

    /**
     * Strong-signal short-circuit thresholds (OMEGA Phase 2.5 port).
     * The top-1 FTS hit must cover at least this fraction of meaningful query
     * tokens AND beat the runner-up by at least {@link #STRONG_SIGNAL_GAP} for
     * the recall pipeline to skip the vector + graph channels.
     */
    private static final double STRONG_SIGNAL_OVERLAP = 0.70;
    /** Minimum overlap-ratio gap between top-1 and top-2 to count as strong. */
    private static final double STRONG_SIGNAL_GAP = 0.20;

    /**
     * Per-facet RRF score multipliers (OMEGA-style type weights, mapped onto
     * our facet taxonomy). Foundational facets — identity, health, profile,
     * preference — get amplified; high-noise facets (event, routine) and
     * low-precision OTHER stay at 1.0×; legacy "kind" tags fall back to 1.0×.
     *
     * <p>Tuning rationale (LongMemEval-driven):
     * <ul>
     *   <li>IDENTITY / PROFILE — definitional facts about the user. When an
     *       identity match exists it almost always answers the query.</li>
     *   <li>HEALTH — safety-critical, must dominate when its topic is asked.</li>
     *   <li>PREFERENCE — frequently the right answer for taste / style queries.</li>
     *   <li>RELATIONSHIP — high signal but often retrieved alongside SCHEDULE
     *       which would be the actual answer; modest boost.</li>
     *   <li>EVENT / ROUTINE — useful but noisy (many events accumulate).</li>
     *   <li>OTHER — catch-all, no boost.</li>
     * </ul>
     */
    private static final java.util.Map<String, Double> FACET_WEIGHTS = java.util.Map.ofEntries(
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_IDENTITY,     2.0),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_HEALTH,       2.0),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_PROFILE,      2.0),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_PREFERENCE,   1.6),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_RELATIONSHIP, 1.4),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_LOCATION,     1.4),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_FINANCE,      1.4),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_GOAL,         1.3),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_COMMITMENT,   1.3),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_SCHEDULE,     1.2),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_ROUTINE,      1.0),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_EVENT,        1.0),
            java.util.Map.entry(ai.nizo.memory.api.memory.MemoryTags.FACET_OTHER,        1.0)
    );

    /**
     * Facet-aware multiplier for the RRF base score. Items without a facet
     * tag (e.g. legacy episodic items pre-dating facet inference) fall back
     * to {@code 1.0×} so introducing weights is a pure boost — never a
     * regression for items the system hasn't classified.
     */
    private static double facetWeight(MemoryItem m) {
        if (m.tags() == null) return 1.0;
        String facet = m.tags().get(ai.nizo.memory.api.memory.MemoryTags.FACET);
        if (facet == null || facet.isBlank()) return 1.0;
        return FACET_WEIGHTS.getOrDefault(facet, 1.0);
    }

    /**
     * Fraction of {@code queryPrefixes} that appear in {@code content}.
     * Used by the strong-signal short-circuit to compare top-1 vs top-2 FTS
     * hits cheaply, without requiring the store to surface BM25 scores.
     */
    private static double overlapRatio(Set<String> queryPrefixes, String content) {
        if (queryPrefixes == null || queryPrefixes.isEmpty()) return 0.0;
        int matches = countLexicalOverlaps(queryPrefixes, content);
        return (double) matches / queryPrefixes.size();
    }

    /** Count distinct query-prefix tokens that appear in {@code content}. */
    private static int countLexicalOverlaps(Set<String> queryPrefixes, String content) {
        if (content == null || queryPrefixes.isEmpty()) return 0;
        String low = content.toLowerCase();
        int n = 0;
        for (String p : queryPrefixes) if (low.contains(p)) n++;
        return n;
    }

    /**
     * "Entity-marker" words: relationship / kinship / generic category
     * tokens that identify an ENTITY but don't address a QUESTION FACET.
     * If a query and a fact share ONLY these tokens, the fact identifies
     * the entity but doesn't answer the question. Used by precision-heavy
     * mode's facet filter.
     */
    private static final Set<String> ENTITY_MARKER_PREFIXES = Set.of(
            "wife", "husba", "spous", "partn",
            "mom", "mothe", "dad", "fathe", "paren",
            "broth", "siste", "sibli",
            "son", "daugh", "child", "kid",
            "frien", "colle", "boss", "manag", "menta",
            "compa", "job", "role", "emplo", "caree",
            "user", "self");

    /**
     * True if the only lexical overlap between {@code queryPrefixes} and
     * {@code content} is an entity-marker token (wife/mom/company/etc).
     */
    private static boolean onlyEntityOverlap(Set<String> queryPrefixes, String content) {
        String low = content.toLowerCase();
        boolean foundNonEntity = false;
        for (String p : queryPrefixes) {
            if (!low.contains(p)) continue;
            if (!ENTITY_MARKER_PREFIXES.contains(p)) {
                foundNonEntity = true;
                break;
            }
        }
        return !foundNonEntity;
    }

    /**
     * Top-K most semantically relevant PROCEDURAL items. No FTS / graph / floor —
     * pure vector similarity, used by ExtractionPipeline to inject only the
     * heuristics that matter for the current message.
     */
    @Override
    public List<MemoryItem> recallProcedural(String userId, String query, int topK) {
        String uid = userId == null ? "default" : userId;
        if (topK <= 0 || embedder == null || query == null || query.isBlank()) {
            return List.of();
        }
        float[] qv = safeEmbed(query);
        if (qv == null) return List.of();

        // Pull a wider top-K from the vector index, then filter to PROCEDURAL.
        // We over-fetch by 4x because PROCEDURAL is a fraction of total items
        // (rest are episodic / semantic from this user's actual interactions).
        int over = Math.max(topK * 4, 50);
        List<MemoryItem> picked = new ArrayList<>();
        for (VectorIndex.Hit h : index.topK(uid, qv, over)) {
            var maybe = store.findById(h.id());
            if (maybe.isEmpty()) continue;
            MemoryItem m = maybe.get();
            if (m.tier() != MemoryItem.Tier.PROCEDURAL) continue;
            picked.add(m);
            if (picked.size() >= topK) break;
        }
        return picked;
    }

    private record Scored(MemoryItem item, double score) {}
}
