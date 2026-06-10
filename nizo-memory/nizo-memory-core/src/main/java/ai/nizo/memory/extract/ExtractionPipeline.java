package ai.nizo.memory.extract;

import ai.nizo.memory.api.extract.ExtractionCategory;
import ai.nizo.memory.api.extract.ExtractionResult;
import ai.nizo.memory.api.extract.ExtractionService;
import ai.nizo.memory.api.memory.MemoryService;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.api.model.ModelResponse;
import ai.nizo.memory.util.Json;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * LLM-backed extraction pipeline implementing {@link ExtractionService}.
 *
 * <p>Sends user messages through a structured extraction prompt, parses the
 * JSON response, then fans out results to the knowledge graph (via
 * {@link GraphFactRouter}) and to the semantic/episodic memory tier (via
 * {@link MemoryService}).
 *
 * <p>De-Springified port from Kimaya's {@code FactExtractionService} (~1077 LOC),
 * significantly simplified. No Spring, no JPA, no Lombok -- plain Java 21.
 */
public final class ExtractionPipeline implements ExtractionService {

    private static final Logger LOG = Logger.getLogger(ExtractionPipeline.class.getName());

    private static final String SOURCE = "extraction";

    // ── Command-only detection ──────────────────────────────────────────

    private static final int MIN_EXTRACTABLE_LENGTH = 15;

    private static final List<Pattern> COMMAND_PATTERNS = List.of(
            Pattern.compile("^analyze\\s", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^search\\s", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^find\\s", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^remind me\\b", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^make\\s", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^show me\\b", Pattern.CASE_INSENSITIVE),
            Pattern.compile("^(what can you|how do you|can you|do you)\\b", Pattern.CASE_INSENSITIVE)
    );

    // ── Category key mapping (JSON key -> enum) ─────────────────────────

    private static final Map<String, ExtractionCategory> CATEGORY_KEYS = Map.ofEntries(
            Map.entry("PROFILE", ExtractionCategory.PROFILE),
            Map.entry("RELATIONSHIP", ExtractionCategory.RELATIONSHIP),
            Map.entry("PREFERENCE", ExtractionCategory.PREFERENCE),
            Map.entry("EVENT", ExtractionCategory.EVENT),
            Map.entry("GOAL", ExtractionCategory.GOAL),
            Map.entry("FOLLOW_UP", ExtractionCategory.FOLLOW_UP),
            Map.entry("IMPLICIT_COMMITMENT", ExtractionCategory.IMPLICIT_COMMITMENT),
            Map.entry("INVESTMENT_INTEREST", ExtractionCategory.INVESTMENT_INTEREST),
            Map.entry("DEFERRAL", ExtractionCategory.DEFERRAL),
            Map.entry("RESOLUTION", ExtractionCategory.RESOLUTION)
    );

    // ── Extraction prompt ───────────────────────────────────────────────
    // Structured like Claude Code's system prompt — sections, bullets, explicit
    // guardrails, anti-pattern examples, output contract. Keeps the LLM honest.

    private static final String EXTRACTION_PROMPT = """
            You are a personal knowledge extractor for a long-term memory system.
            You read a single user message and extract structured facts about
            their life, work, relationships, preferences, and intentions.

            # Output contract
            - Output ONE valid JSON object and nothing else. No prose, no markdown fences.
            - Include ONLY top-level keys for categories that apply. Empty input → {}.
            - Every string field must be a non-empty, fully-formed phrase. Never output bare keywords like "prefer" or "work" as values.
            - Never invent details. If a field is not in the message, omit it.

            # Extraction discipline (ported from Mastra's Observer agent — the
            # pattern that drove their 94.87% on LongMemEval)
            ## Assertions vs questions
            - "I have two kids" → assertion (extract as fact: kids=2 or RELATIONSHIP entries).
            - "How many kids do I have?" → question (NEVER extract a fact from a question).
            - "I'm looking forward to <X>" → statement of intent (extract as future EVENT or GOAL with the date if mentioned).
            - "Can you recommend <X>?" → question (do not extract).
            - User assertions are AUTHORITATIVE. If a user previously stated something and later asks a question on the same topic, the assertion is still the answer — the question doesn't invalidate it.

            ## State changes
            When the user updates or supersedes a previous fact, frame it explicitly:
            - "I switched from Vim to Cursor" → PREFERENCE with note "switched from Vim".
            - "I moved from Sydney to Melbourne" → PROFILE location updated; mention the previous city if relevant.
            - "I started a new job at Acme" → PROFILE company=Acme; the prior employer (if any) is now historic.
            - GOOD: "User uses Cursor for code (switched from Vim)."
            - BAD: "User uses Cursor."   ← drops the supersession signal

            ## Temporal anchoring
            Every dated fact carries TWO times:
            1. WHEN STATED (the message timestamp — implicit, no need to repeat).
            2. WHAT IS REFERENCED — only when there's a relative time word.
            Convert relative time references to absolute dates whenever possible:
            - "yesterday", "last week", "this weekend", "tomorrow", "next month" → produce an ISO date in the relevant date field.
            - "recently", "soon", "lately", "a while ago" → too vague, omit a date rather than guess.

            ## Multi-event splitting
            If a single sentence contains MULTIPLE events, output a SEPARATE entry per event.
            EACH split entry MUST carry its own date when applicable.
            Example: "I'm visiting parents this weekend and going to the dentist tomorrow"
            → TWO EVENT entries: one with date=this-weekend, one with date=tomorrow.

            ## Preserve unusual phrasing
            When the user uses a non-standard or domain-specific term, quote it.
            - "User exercised."           ← BAD (information lost)
            - "User did a 'movement session' (their term for exercise)."   ← GOOD

            # Common knowledge (apply these guardrails)
            - "iPhone user", "Android user", "Pixel user" → phone PREFERENCE, NOT occupation/PROFILE.
            - "Linux user", "macOS user", "Windows user", "Vim user" → tool/OS PREFERENCE, NOT occupation/PROFILE.
            - "AAPL", "TSLA", "HDFC", "NIFTY50ETF" etc. → stock tickers → INVESTMENT_INTEREST, NOT EVENT.
            - "EpiPen", "inhaler", "insulin" → indicates a serious medical condition → PREFERENCE with domain=health AND high importance.
            - "vegetarian", "vegan", "kosher", "halal", "lactose intolerant", "gluten free", peanut/nut allergy → dietary/health PREFERENCE.
            - ₹ = Indian Rupees, $ = USD by default, € = Euro, £ = GBP.
            - "WFH" = working from home, "OOO" = out of office, "EOD" = end of day, "ASAP" = urgent.
            - "my wife/husband/partner/spouse <Name>" → RELATIONSHIP person_name=<Name>, type=spouse/partner.
            - "my mom/dad/mother/father <Name>" → RELATIONSHIP person_name=<Name>, type=parent.
            - "my son/daughter <Name>" → RELATIONSHIP person_name=<Name>, type=child.
            - Job titles look like: "engineer", "manager", "director", "designer", "founder", "analyst", "consultant", "doctor", "lawyer". They do NOT look like "iPhone user".

            # Categories

            ## PROFILE (single object — biographical facts about the speaker)
            Fields: name, nickname, location_city, location_country, occupation,
            company, industry, birthday (YYYY-MM-DD), timezone.
            - occupation MUST be a real job title. Examples: "principal engineer", "doctor", "product manager".
            - Anti-examples (NEVER use as occupation): "iPhone user", "morning person", "Vim user", "vegetarian", "married".
            - Only emit PROFILE when the user is talking about THEMSELVES. "My wife is a surgeon" → RELATIONSHIP, not PROFILE.

            ## RELATIONSHIP (array — people in the user's life)
            Fields: person_name (actual name; fall back to role only if no name given),
            relationship_type (spouse/partner/child/parent/sibling/family/friend/
            colleague/coworker/manager/mentor/client/acquaintance/knows/co-founder),
            context (free-form: occupation, location, how they know each other).
            - "my wife Sarah" → person_name="Sarah", relationship_type="spouse".
            - "my friend" (no name) → person_name="friend", relationship_type="friend".
            - context must be a full sentence-like phrase, e.g. "surgeon at City Hospital".
            - **person_name MUST be an actual human name. NEVER use:**
              - a festival ("Pongal", "Diwali", "Eid", "Christmas")
              - a place ("Tirupati", "Mecca", "Bangalore")
              - a possessive reference ("Arjun's wife" is NOT a name — if the message is
                "Arjun's wife is a teacher", the subject is the USER herself, not a third person)
              - generic group words ("participants", "friends", "family", "attendees")
            - Skip the whole RELATIONSHIP entry if you cannot identify a valid person name.
            - **relationship_type must be a real relationship.** NEVER use "visit",
              "participant", "activity", or verbs. If a sentence describes an activity
              involving multiple people, it's an EVENT, not a RELATIONSHIP.

            ## PREFERENCE (array — what the user likes / believes / does habitually)
            Fields: subject (the topic), assertion (the full preference statement —
            never just "prefer" or "like"), domain (work/personal/technical/
            lifestyle/health/social/finance).
            - "I love mango ice cream" → {subject: "ice cream flavor", assertion: "loves mango ice cream", domain: "lifestyle"}.
            - "I have a peanut allergy" → {subject: "allergy", assertion: "severe peanut allergy, carries EpiPen if mentioned", domain: "health"}.
            - "I use Vim" → {subject: "code editor", assertion: "uses Vim", domain: "technical"}.
            - assertion MUST be a complete phrase. Never bare verbs like "prefer", "specifically", "use".

            ## EVENT (array — things that happened or will happen in the user's life)
            Fields: summary, event_type (meeting/conversation/decision/experience/
            task/milestone/problem/achievement/deadline), date (YYYY-MM-DD if known),
            participants (list of names), emotional_valence (positive/negative/
            neutral/mixed).
            - SKIP commands: "analyze AAPL", "search for X", "remind me to Y" — these are NOT events.

            ## GOAL (array — LONG-TERM aspirations, not short-term tasks)
            Fields: title, description, category (health/career/learning/finance/
            personal/relationship/creative), target_date (YYYY-MM-DD), priority
            (high/medium/low).
            - Triggers: "I want to", "my goal is", "aiming to", "working toward",
              "trying to", "plan to eventually", "dream of", "aspire to", "hoping to",
              "by next year", "in five years", "long-term plan", "save up for",
              "build toward", "lose X kg", "learn X", "get fit", "retire by".
            - GOAL has a HORIZON (weeks, months, years). A task due tomorrow is
              FOLLOW_UP, not GOAL. "Learn Spanish" is GOAL; "finish Spanish lesson
              tonight" is FOLLOW_UP.
            - Examples:
              * "I want to run a marathon next year" → {title: "Run a marathon", category: "health", target_date: "~next year", priority: "high"}
              * "Saving up to buy a house in Bangalore by 2028" → {title: "Buy a house in Bangalore", category: "finance", target_date: "2028"}
              * "Learning French, aiming for B2 in 12 months" → {title: "Reach B2 French", category: "learning"}

            ## FOLLOW_UP (array — SHORT-TERM action items the user owes themselves)
            Fields: description, follow_up_days (default 1).
            - Triggers: "need to", "have to", "should", "must", "gotta", "to-do",
              "reminder", "don't forget", "chase up", "send X by", "call X tomorrow",
              "email by EOD", "book", "schedule", "pick up", "pay the bill".
            - FOLLOW_UP has a SHORT horizon (hours to ~2 weeks) and a specific
              discrete action. Longer / vaguer → GOAL. An intent without a task →
              IMPLICIT_COMMITMENT.
            - Examples:
              * "Need to email the accountant by Friday" → {description: "Email the accountant", follow_up_days: 2}
              * "Reminder: pick up dry cleaning tomorrow" → {description: "Pick up dry cleaning", follow_up_days: 1}
              * "Gotta book tickets for the Chennai trip" → {description: "Book tickets for Chennai trip", follow_up_days: 3}

            ## IMPLICIT_COMMITMENT (array — intent / waiting / pending decision — NOT a scheduled task)
            Fields: description, commitment_type (will_do/waiting_for/need_to/
            planning_to/pending_decision), related_person, estimated_timeframe
            (days: 1/3/5/7/14/30).
            - Timing hints: "soon"=3, "this week"=5, "next week"=7, "this month"=14, "next month"=30.
            - Triggers: "I'll", "I will", "gonna", "planning to", "thinking of doing",
              "we're working on", "waiting for X to reply", "Ravi still hasn't gotten back",
              "yet to decide", "pending on", "blocked on", "hoping to hear from".
            - Use IMPLICIT_COMMITMENT when there's INTENT or WAITING without an
              explicit deadline action. Use FOLLOW_UP when there's a concrete task.
              Use DEFERRAL when the user explicitly postponed a decision.
            - Examples:
              * "I'll circle back with Priya next week about the offer" → {description: "Circle back with Priya on the offer", commitment_type: "will_do", related_person: "Priya", estimated_timeframe: 7}
              * "Still waiting for HR to send the revised contract" → {description: "Waiting for HR to send revised contract", commitment_type: "waiting_for", related_person: "HR", estimated_timeframe: 5}
              * "Planning to start yoga classes this month" → {description: "Start yoga classes", commitment_type: "planning_to", estimated_timeframe: 14}

            ## INVESTMENT_INTEREST (single object or array — investing preferences and holdings)
            Fields: summary (ONE sentence describing the user's investment approach —
            REQUIRED if the message mentions any investment), tickers (list of actual
            stock symbols — 1-5 uppercase letters ONLY, like "AAPL", "HDFCBANK", "TSLA"),
            sectors (list of industry sectors), vehicles (free-form: SIP, mutual funds,
            ETF, 401k, NPS, PPF, EPF, stocks, bonds, crypto, real estate, etc.), style
            (value/growth/dividend/momentum/index/passive), risk_appetite
            (conservative/moderate/aggressive).
            - **summary is REQUIRED** — capture the narrative so an agent can reason
              about retirement/tax/savings questions later.
            - **tickers is uppercase-letters ONLY.** NEVER include "YC alums",
              "Mexican angels", "Y Combinator", country codes ("SG", "US"),
              or names of investors.
            - "SIP of ₹80K in Nifty 50 and Next 50 via Zerodha + PPF + EPF + NPS" →
              {summary: "SIP ₹80K/month in Nifty 50 + Next 50 index funds via Zerodha; also PPF, EPF, NPS Tier-1", sectors: [], vehicles: "SIP, index funds, PPF, EPF, NPS", style: "index"}.
            - "watching HDFC Bank" → {summary: "watching HDFC Bank for a potential buy", tickers: ["HDFCBANK"]}.
            - SKIP commands: "analyze AAPL" is a request, not a preference.

            ## DEFERRAL (array — decisions the user is putting off)
            Fields: decision, context (the reason — the BLOCKER), days_until_followup (default 3).
            - Triggers: "I'll think about it", "let me get back to you", "haven't decided",
              "not sure yet", "will wait", "too expensive", "will pass for now",
              "holding off", "parked for now", "revisit later", "back burner",
              "not right now", "maybe later", "on the fence".
            - context MUST capture the blocker (price, time, uncertainty, etc.).
            - Examples:
              * "Sony WH-1000XM5 at ₹28,000 is too much. Will wait for sale." → {decision: "Buying Sony WH-1000XM5 headphones", context: "₹28,000 is too expensive, waiting for sale", days_until_followup: 60}
              * "House decision parked until my bonus comes through in March" → {decision: "Buying a house", context: "Waiting for March bonus for down payment", days_until_followup: 60}
              * "Still on the fence about joining the boot camp" → {decision: "Join boot camp", context: "Undecided on commitment", days_until_followup: 7}

            ## RESOLUTION (array — decisions the user has made / is making)
            Fields: decision, choice (what they decided).
            - Triggers: "I decided to", "I'm going with", "I chose", "I switched to",
              "I'll go ahead with", "settled on", "going ahead", "pulling the trigger",
              "locked in", "finalised", "signed up for", "bought", "picked X over Y".
            - When the user resolves an old DEFERRAL, emit RESOLUTION (NOT DEFERRAL).
            - Examples:
              * "Went ahead and booked the Bali trip" → {decision: "Bali trip", choice: "Booked it"}
              * "Decided to switch from Aviva to LIC for term insurance" → {decision: "Term insurance provider", choice: "Switched from Aviva to LIC"}
              * "Signed up for the AWS certification course" → {decision: "AWS certification", choice: "Signed up for the course"}

            # Worked examples

            Input: "I'm Kislay, I work at Stripe as a Staff Engineer."
            Output: {"PROFILE": {"name": "Kislay", "occupation": "Staff Engineer", "company": "Stripe"}}

            Input: "I prefer dark mode in my apps. iPhone user."
            Output: {"PREFERENCE": [{"subject": "UI theme", "assertion": "prefers dark mode in all applications", "domain": "technical"}, {"subject": "phone", "assertion": "iPhone user, prefers iOS over Android", "domain": "technical"}]}
            (Note: "iPhone user" is a phone preference — NOT occupation.)

            Input: "I love mango ice cream from Naturals."
            Output: {"PREFERENCE": [{"subject": "ice cream flavor", "assertion": "loves mango ice cream from Naturals brand", "domain": "lifestyle"}]}

            Input: "I have a peanut allergy. Always carry an EpiPen."
            Output: {"PREFERENCE": [{"subject": "allergy", "assertion": "severe peanut allergy, always carries EpiPen", "domain": "health"}]}

            Input: "Sony WH-1000XM5 is too expensive at ₹28,000. Will wait for a sale."
            Output: {"DEFERRAL": [{"decision": "Buying Sony WH-1000XM5 headphones", "context": "Price ₹28,000 is too expensive, waiting for a sale", "days_until_followup": 60}]}

            Input: "I got the promotion! Now Staff Engineer instead of Principal."
            Output: {"PROFILE": {"occupation": "Staff Engineer"}, "EVENT": [{"summary": "Got promoted from Principal to Staff Engineer", "event_type": "achievement", "emotional_valence": "positive"}], "RESOLUTION": [{"decision": "Career progression", "choice": "Promoted to Staff Engineer"}]}

            Input: "Need to email the accountant by Friday about the GST return."
            Output: {"FOLLOW_UP": [{"description": "Email accountant about GST return", "follow_up_days": 2}]}

            Input: "I want to run a half-marathon in 2027. Already started interval training."
            Output: {"GOAL": [{"title": "Run a half-marathon", "category": "health", "target_date": "2027", "priority": "high"}], "EVENT": [{"summary": "Started interval training for half-marathon", "event_type": "milestone", "emotional_valence": "positive"}]}

            Input: "Still waiting for HR to send the revised offer letter."
            Output: {"IMPLICIT_COMMITMENT": [{"description": "Waiting for revised offer letter from HR", "commitment_type": "waiting_for", "related_person": "HR", "estimated_timeframe": 5}]}

            Input: "House-buying decision is parked until my March bonus comes through."
            Output: {"DEFERRAL": [{"decision": "Buying a house", "context": "Waiting for March bonus for down payment", "days_until_followup": 60}]}

            Input: "Signed up for the AWS Solutions Architect course. Will start next week."
            Output: {"RESOLUTION": [{"decision": "AWS Solutions Architect certification", "choice": "Signed up for the course"}], "IMPLICIT_COMMITMENT": [{"description": "Start AWS course", "commitment_type": "planning_to", "estimated_timeframe": 7}]}

            Input: "analyze AAPL"
            Output: {}
            (Pure command — no extractable life facts.)

            # Disambiguation (when categories overlap)
            Ask, in order:
            1. Does the user say they DECIDED something? → RESOLUTION.
            2. Does the user say they're PUTTING IT OFF / UNDECIDED? → DEFERRAL.
            3. Is there a DISCRETE TASK with a near-term deadline? → FOLLOW_UP.
            4. Is there INTENT or WAITING without a task/deadline? → IMPLICIT_COMMITMENT.
            5. Is it a LONG-TERM aspiration (weeks, months, years)? → GOAL.
            6. Is it a stable HABIT / LIKE / BELIEF? → PREFERENCE.
            7. Did something HAPPEN (past or scheduled)? → EVENT.
            Pick EVERY category that applies — a single message often spans 2-3.

            # Rules
            - Read the message carefully before deciding categories.
            - Never put preferences/hobbies/health-conditions into PROFILE.occupation.
            - assertion / context / summary fields must always be complete phrases — never single words.
            - A single message can produce MULTIPLE categories simultaneously. Don't force one answer.
            - Output the JSON object directly — no preamble, no explanation, no markdown fences.
            """;

    // ── Heuristic injection budget ──────────────────────────────────────

    /** Top-K relevant PROCEDURAL heuristics retrieved per message. Trades
     *  prompt cost for grounding. 12 fits ~600 tokens with our compact entries. */
    private static final int TOP_K_RELEVANT_HEURISTICS = 12;

    /** Always-on guards loaded once from the bundled YAML. These short rules
     *  (extraction-guards category) prevent common LLM misclassifications and
     *  must apply to every message, not just semantically related ones. */
    private static final List<String> ALWAYS_ON_GUARDS = loadAlwaysOnGuards();

    /** Placeholder words an LLM emits when it can't fill a PROFILE field.
     *  Treated as garbage and dropped before storage. */
    private static final Set<String> PROFILE_PLACEHOLDERS = Set.of(
            "unspecified", "unknown", "n/a", "na", "none", "tbd", "tba",
            "current", "current company", "current employer", "current job",
            "current role", "current position", "working", "working at",
            "employed", "employee", "person", "user", "member",
            "not specified", "not provided", "not mentioned", "not stated",
            // Generic / possessive references LLMs substitute for a real name
            "your company", "your employer", "your firm", "your startup",
            "our company", "our employer", "our firm", "our startup", "our team",
            "my company", "my employer", "my firm", "my startup", "my team",
            "the company", "the employer", "the firm", "the startup", "the team",
            "a company", "a firm", "a startup",
            "small company", "small firm", "small startup",
            // Role names masquerading as companies
            "company", "employer", "firm", "startup", "organization", "org",
            "team", "business", "enterprise");

    /** Words that are NOT valid person names for RELATIONSHIP. Covers
     *  festivals, places, generic roles, and common LLM hallucinations. */
    private static final Set<String> NOT_A_PERSON_NAME = Set.of(
            // Indian festivals
            "diwali", "deepavali", "holi", "navratri", "dussehra",
            "ganesh chaturthi", "janmashtami", "raksha bandhan", "karva chauth",
            "onam", "pongal", "makar sankranti", "ram navami", "maha shivratri",
            // Islamic / Jewish / Christian festivals
            "eid", "eid al-fitr", "eid al-adha", "ramadan", "muharram",
            "christmas", "easter", "good friday", "hanukkah", "passover",
            "rosh hashanah", "yom kippur",
            // Other festivals
            "lunar new year", "chinese new year", "songkran", "vesak",
            "thanksgiving", "halloween", "valentine's day", "new year",
            // Common places that get mistaken for names
            "tirupati", "mecca", "jerusalem", "varanasi", "haridwar", "vrindavan",
            "bangalore", "mumbai", "delhi", "chennai", "kolkata", "pune",
            "singapore", "new york", "london", "paris", "tokyo",
            // Generic roles that aren't names
            "participants", "participant", "attendees", "attendee",
            "visitors", "visitor", "guest", "guests", "colleagues",
            "friends", "family", "parents", "kids", "children",
            "everyone", "someone", "anyone", "nobody", "people",
            // Cultural family-relation terms — NOT person names
            // (Tamil / Hindi / Bengali / Spanish / Italian / German / etc.)
            "amma", "appa", "thatha", "paati",
            "ma", "pa", "papa", "mama", "mamma", "mum", "mummy", "daddy",
            "naani", "naana", "dadi", "dada", "nani", "nana",
            "maasi", "mausi", "chacha", "chachi", "bua", "phupi",
            "abuela", "abuelo", "tia", "tio", "madre", "padre",
            "oma", "opa", "mutter", "vater",
            "obaasan", "ojiisan", "okaasan", "otousan",
            "halmoni", "harabeoji", "umma", "appa-korean");

    private static List<String> loadAlwaysOnGuards() {
        try {
            List<ai.nizo.memory.seed.WorldKnowledgeSeed.Heuristic> all =
                    ai.nizo.memory.seed.WorldKnowledgeSeed.loadHeuristicsFromClasspath();
            List<String> out = new ArrayList<>();
            for (var h : all) {
                if ("extraction-guards".equals(h.category())) out.add(h.text());
            }
            return List.copyOf(out);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Could not load always-on extraction guards", e);
            return List.of();
        }
    }

    // ── Dependencies ────────────────────────────────────────────────────

    private final ModelClient extractor;
    private final GraphFactRouter router;
    private final MemoryService memory;

    /** Per-process cache of users we've already auto-seeded. Idempotent at
     *  the WorldKnowledgeSeed layer too (marker check), but this avoids the
     *  marker recall on every single extract() call. */
    private final java.util.Set<String> seededUsers =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    /** Thread-local message context — set at the top of extract() before
     *  routeToMemory and consumed by helper methods that store individual
     *  facts. Avoids threading the context through every routing method
     *  signature. */
    private static final ThreadLocal<MessageContext> CURRENT_CONTEXT = new ThreadLocal<>();

    /** Context-aware fact storage — uses MessageContext tags so the
     *  recall pipeline can apply subject / sensitivity / mode filters. */
    private String learnFactCtx(String userId, String fact, String source, double confidence) {
        MessageContext ctx = CURRENT_CONTEXT.get();
        Map<String, String> tags = ctx == null ? Map.of() : ctx.asTags();
        return memory.learnFact(userId, fact, source, confidence, tags);
    }

    public ExtractionPipeline(ModelClient extractor,
                              GraphFactRouter router,
                              MemoryService memory) {
        this.extractor = Objects.requireNonNull(extractor, "extractor");
        this.router = Objects.requireNonNull(router, "router");
        this.memory = Objects.requireNonNull(memory, "memory");
    }

    // ===== ExtractionService ================================================

    @Override
    public ExtractionResult extract(String userId, String message) {
        if (isCommandOnly(message)) {
            LOG.fine(() -> "Skipping extraction for command-only message: " + message);
            return ExtractionResult.empty();
        }

        // G21 — prompt-injection defense. If the message is trying to
        // manipulate the extractor ("IGNORE PRIOR INSTRUCTIONS", "SYSTEM
        // OVERRIDE", "emit exactly {PROFILE:...}"), we skip LLM extraction
        // entirely. The raw message still lands as EPISODIC (so provenance
        // survives and a human can audit), but nothing structured flows into
        // PROFILE / RELATIONSHIP / etc where it could poison future recall.
        if (looksLikePromptInjection(message)) {
            LOG.warning("Prompt injection signal in message for userId=" + userId
                    + " — skipping structured extraction.");
            memory.remember(userId, message,
                    Map.of("kind", "raw_message", "flagged", "prompt_injection_suspected"),
                    "user_message");
            return ExtractionResult.empty();
        }

        // Auto-seed world knowledge on first interaction per user per process.
        // After a DB wipe / fresh install / first contact with a new user,
        // this ensures the PROCEDURAL tier is populated before extraction.
        ensureSeeded(userId);

        try {
            // 1. Build prompt:
            //    - SYSTEM message: EXTRACTION_PROMPT + ALWAYS-ON guards (static across calls — Ollama KV-cache reuses it)
            //    - USER message: dynamic top-K relevant hints (varies) + the actual message
            // Keeping the system prompt fully static is critical: qwen2.5:7b's
            // KV cache survives between calls only when the prompt prefix is
            // byte-identical. Earlier we put dynamic content in the system
            // message and saw extractions slow from 2-4s to 15-17s.
            String systemPrompt = EXTRACTION_PROMPT;
            String learned = collectLearnedHeuristics(userId, message);
            if (!learned.isEmpty()) {
                systemPrompt = EXTRACTION_PROMPT + "\n\n# Applicable heuristics\n" + learned;
            }
            // NOTE: dynamic top-K heuristics are intentionally NOT injected.
            // We tested putting them in either system or user message — both
            // produced 15-17s latency (vs 2-4s with static-only) AND degraded
            // extraction quality (most messages returned `none`). The 326-item
            // bundle was the wrong abstraction for runtime injection. The
            // always-on extraction-guards above are sufficient for category
            // classification; domain knowledge belongs in EXTRACTION_PROMPT
            // examples, not retrieved per call. {@link #buildDynamicHeuristics}
            // remains available for future experiments.
            ModelRequest request = ModelRequest.of(List.of(
                    Message.system(systemPrompt),
                    Message.user(message)
            ));
            ModelResponse response = extractor.complete(request);
            String raw = response.text();

            // 2. Parse JSON response
            Map<String, Object> extracted = parseResponse(raw);
            if (extracted.isEmpty()) {
                LOG.fine(() -> "No extractions found in response for userId=" + userId);
                // Even on parse failure, run the investment fallback — we
                // refuse to silently lose financial context just because
                // the LLM emitted malformed JSON.
                try {
                    maybeStoreInvestmentFallback(userId, message, Map.of());
                } catch (Exception e) {
                    LOG.log(Level.WARNING, "Investment fallback (on empty parse) failed", e);
                }
                return ExtractionResult.empty();
            }

            // 3. Build result metadata
            Set<ExtractionCategory> types = resolveCategories(extracted);
            int count = countExtractions(extracted);

            // 4. Route to graph
            try {
                router.routeToGraph(userId, extracted);
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Graph routing failed for userId=" + userId, e);
            }

            // 5. Demote contradicted facts FIRST (before storing new ones)
            try {
                demoteContradicted(userId, extracted);
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Contradiction demotion failed for userId=" + userId, e);
            }

            // 6. Store raw message as EPISODIC for provenance BEFORE any fact
            //    extraction lands. Every SEMANTIC fact we derive below will be
            //    tagged with this episode id so "why do you think X about me?"
            //    can be answered by fetching the source message by id.
            String episodeId = null;
            String excerpt = null;
            try {
                episodeId = memory.remember(
                        userId, message,
                        Map.of("kind", "raw_message", "from", "extraction"),
                        "user_message");
                excerpt = truncateExcerpt(message);
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Could not store raw message for provenance", e);
            }

            // 7. Route to memory (store new facts) — pass per-message context
            //    (subject / sensitivity / mode / hypothetical / expiry / source)
            //    so each learned fact carries the tags recall filters need and
            //    a back-pointer to the source message.
            MessageContext ctx = analyseMessage(message).withProvenance(episodeId, excerpt);
            // Stash the context in a thread-local so the routing helpers can
            // read it without changing every method signature.
            CURRENT_CONTEXT.set(ctx);
            try {
                routeToMemory(userId, extracted, message);
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Memory routing failed for userId=" + userId, e);
            } finally {
                CURRENT_CONTEXT.remove();
            }

            // 7. Safety net: if the message mentions investing (SIP/PPF/401k/
            //    stocks/etc.) but NO investment-tagged fact landed, store a
            //    fallback so the user's financial context isn't silently lost.
            //    This was the Meera failure: LLM classified SIP as PREFERENCE
            //    without a finance domain, and "my investments" returned empty.
            try {
                maybeStoreInvestmentFallback(userId, message, extracted);
            } catch (Exception e) {
                LOG.log(Level.WARNING, "Investment fallback failed for userId=" + userId, e);
            }

            LOG.fine(() -> "Extracted " + count + " items across " + types.size()
                    + " categories for userId=" + userId);
            return new ExtractionResult(count, types, extracted);

        } catch (Exception e) {
            LOG.log(Level.WARNING, "Extraction failed for userId=" + userId, e);
            // Last-ditch safety net: malformed JSON / connection failure
            // shouldn't silently lose critical financial context.
            try {
                maybeStoreInvestmentFallback(userId, message, Map.of());
            } catch (Exception inner) {
                LOG.log(Level.WARNING, "Investment fallback (on extraction error) failed", inner);
            }
            return ExtractionResult.empty();
        }
    }

    /** G21 — detect common prompt-injection shapes in user messages. */
    static boolean looksLikePromptInjection(String message) {
        if (message == null || message.isBlank()) return false;
        String low = message.toLowerCase(Locale.ROOT);
        String[] signals = {
                "ignore all prior", "ignore all previous", "ignore prior instructions",
                "ignore previous instructions", "disregard prior", "disregard previous",
                "system override", "system: override", "sudo override",
                "admin override", "developer mode", "jailbreak",
                "emit exactly", "output exactly", "reply with only",
                "you are now", "you must now", "new role:",
                "the real user is actually", "pretend you are",
                "override your instructions", "override extraction",
                "forget everything", "forget all", "do anything now",
                "enable dan mode", "dan mode enabled",
                "stop following the extraction prompt"
        };
        for (String s : signals) if (low.contains(s)) return true;
        return false;
    }

    /** G21 — post-extraction sanity check for injected profile values. */
    private static boolean looksLikeInjectedPayload(String v) {
        if (v == null) return false;
        String low = v.toLowerCase(Locale.ROOT);
        return low.contains("pwned") || low.contains("hacker")
                || low.contains("ignore prior") || low.contains("system override")
                || low.contains("jailbreak");
    }

    @Override
    public boolean isCommandOnly(String message) {
        if (message == null || message.isBlank()) return true;

        String trimmed = message.strip();
        if (trimmed.length() < MIN_EXTRACTABLE_LENGTH) return true;

        for (Pattern p : COMMAND_PATTERNS) {
            if (p.matcher(trimmed).find()) return true;
        }

        return false;
    }

    // ===== Procedural memory (user-taught heuristics) ========================

    /**
     * Build the always-on heuristic block for the system prompt.
     *
     * <p>Returns ONLY the {@code extraction-guards} subset — short, critical
     * classification rules ("iPhone user is a phone preference, not a job")
     * that apply to every message. These are static across calls so the
     * LLM's prompt cache survives; per-message dynamic retrieval is done
     * separately via {@link #buildDynamicHeuristics} and prepended to the
     * USER message (not the system prompt) to keep the system prefix stable.
     */
    private String collectLearnedHeuristics(String userId, String message) {
        StringBuilder sb = new StringBuilder();
        for (String guard : ALWAYS_ON_GUARDS) {
            sb.append("- ").append(guard).append('\n');
        }
        return sb.toString();
    }

    /**
     * Per-message dynamic heuristics. Top-K most semantically relevant
     * PROCEDURAL items to the current message. Returned as a markdown block
     * the caller prepends to the user message — keeping the system prompt
     * static (so qwen2.5:7b's KV cache survives across calls).
     */
    private String buildDynamicHeuristics(String userId, String message) {
        try {
            var hits = memory.recallProcedural(userId, message, TOP_K_RELEVANT_HEURISTICS);
            if (hits.isEmpty()) return "";
            // Filter out items already in always-on (avoid duplication).
            java.util.Set<String> alwaysOn = new java.util.HashSet<>(ALWAYS_ON_GUARDS);
            StringBuilder sb = new StringBuilder("# Hints for this message:\n");
            int added = 0;
            for (var item : hits) {
                if (alwaysOn.contains(item.content())) continue;
                sb.append("- ").append(item.content()).append('\n');
                added++;
            }
            return added > 0 ? sb.append('\n').toString() : "";
        } catch (Exception e) {
            LOG.fine(() -> "buildDynamicHeuristics failed (non-fatal): " + e);
            return "";
        }
    }

    /** Truncate a message into a compact excerpt suitable for tagging.
     *  Collapses whitespace, caps at 140 chars with an ellipsis. */
    private static String truncateExcerpt(String s) {
        if (s == null) return null;
        String compact = s.replaceAll("\\s+", " ").trim();
        if (compact.isEmpty()) return null;
        if (compact.length() <= 140) return compact;
        return compact.substring(0, 137) + "...";
    }

    /** Auto-seed world knowledge for a user on first encounter per process. */
    private void ensureSeeded(String userId) {
        if (userId == null || userId.isBlank()) return;
        if (!seededUsers.add(userId)) return;
        try {
            ai.nizo.memory.seed.WorldKnowledgeSeed.seedIfNeeded(memory, userId);
        } catch (Exception e) {
            LOG.fine(() -> "Auto-seed failed for " + userId + " (non-fatal): " + e);
        }
    }

    /** Substring patterns we reject anywhere inside a PROFILE value —
     *  e.g. "Company name not provided", "Name withheld for privacy". */
    private static final List<String> PLACEHOLDER_CONTAINS = List.of(
            "not provided", "not specified", "not mentioned", "not stated",
            "not disclosed", "not given", "not available",
            "name withheld", "name not", "name unknown",
            "to be determined", "to be decided", "to be announced",
            "placeholder", "redacted", "anonymous", "undisclosed");

    /** True if the value looks like an LLM placeholder — exact match against
     *  {@link #PROFILE_PLACEHOLDERS} OR a substring match against
     *  {@link #PLACEHOLDER_CONTAINS} (catches "Company name not provided",
     *  "Name withheld", etc.). */
    private static boolean isPlaceholder(String value) {
        if (value == null) return true;
        String v = value.toLowerCase().trim();
        if (v.isEmpty() || v.length() < 2) return true;
        if (PROFILE_PLACEHOLDERS.contains(v)) return true;
        for (String pat : PLACEHOLDER_CONTAINS) {
            if (v.contains(pat)) return true;
        }
        return false;
    }

    /** True if the value is a pronoun the LLM mistook for a name. */
    private static boolean isPronounish(String value) {
        if (value == null) return false;
        String v = value.toLowerCase().trim();
        return Set.of("you", "i", "me", "my", "we", "us", "they",
                       "he", "she", "him", "her", "their", "them",
                       "the user", "the person", "user").contains(v);
    }

    /**
     * True if the value is a valid actual person name for a RELATIONSHIP
     * entry. Rejects:
     *  - Possessive references ("Arjun's wife", "Ira's school")
     *  - Known festivals / places / generic roles ("Pongal participants",
     *    "Tirupati", "participants")
     *  - Generic role words ("friend", "colleague") when no name is given
     *  - Placeholders + pronouns
     */
    private static boolean isValidPersonName(String name) {
        if (name == null) return false;
        String v = name.trim();
        if (v.isEmpty() || v.length() < 2) return false;
        if (isPlaceholder(v) || isPronounish(v)) return false;
        // Possessive references like "Arjun's wife" are references, not names
        if (v.contains("'s ") || v.contains("'s ")) return false;
        String low = v.toLowerCase();
        if (NOT_A_PERSON_NAME.contains(low)) return false;
        // Check each word isn't a festival/place
        for (String word : low.split("\\s+")) {
            if (NOT_A_PERSON_NAME.contains(word)) return false;
        }
        // Valid-looking names start with a letter (including non-ASCII)
        if (!Character.isLetter(v.charAt(0))) return false;
        return true;
    }

    /** True if the relationship_type is a real relationship (not the LLM
     *  confusing an action/activity for a relationship). */
    private static boolean isValidRelationshipType(String type) {
        if (type == null) return false;
        String t = type.toLowerCase().trim();
        if (t.isEmpty()) return false;
        return Set.of(
                "spouse", "wife", "husband", "partner",
                "child", "son", "daughter",
                "parent", "mother", "father", "mom", "dad",
                "sibling", "brother", "sister",
                "family", "relative",
                "mentor", "mentee", "manager", "report", "colleague", "coworker",
                "friend", "best friend", "acquaintance", "knows",
                "client", "customer", "vendor", "advisor",
                "teacher", "student", "classmate",
                "boss", "subordinate", "peer",
                "co-founder", "cofounder", "business partner",
                "neighbor", "neighbour", "roommate", "flatmate",
                "in-law", "father-in-law", "mother-in-law",
                "brother-in-law", "sister-in-law", "son-in-law", "daughter-in-law"
        ).contains(t);
    }

    /**
     * Safety net. When a message contains clear investment vocabulary
     * (SIP / PPF / NPS / 401k / stocks / mutual funds / crypto / specific
     * tickers), we ALWAYS store a retrieval-friendly fact tagged as an
     * investment approach. This is additive to whatever the LLM routed
     * (INVESTMENT_INTEREST or PREFERENCE(finance)) and guarantees that
     * natural-language queries like "my investments" / "how am i saving"
     * match — the LLM's field-level classification is unreliable and
     * sometimes omits the financial domain entirely.
     *
     * <p>Cheap duplicate protection: skips if the exact message already
     * landed as an investment-approach fact this session.
     */
    private void maybeStoreInvestmentFallback(String userId, String message,
                                               Map<String, Object> extracted) {
        if (message == null || !looksLikeInvestment(message)) return;

        // Store the raw message as an investment-context fact so retrieval
        // queries like "my investments" / "retirement savings" always match.
        // Separate from any INVESTMENT_INTEREST the LLM emitted — the LLM
        // frequently omits key context fields (summary / vehicles / etc.),
        // so having the raw statement is our floor.
        learnFactCtx(userId,
                "User investment approach: " + message.trim(),
                "extraction_fallback", 0.85);
    }

    // ====================================================================
    // Customer-problem guardrails — derived once per message, attached as
    // tags to every fact extracted from that message.
    // ====================================================================

    /**
     * Per-message context that decorates every fact extracted from this
     * message. Computed once via {@link #analyseMessage} and threaded into
     * every {@code learnFact} call so the recall pipeline can apply
     * subject / sensitivity / mode / hypothetical filters.
     */
    private record MessageContext(
            String subject,        // "self" | "other:mom" | etc
            String sensitivity,    // PUBLIC / PERSONAL / SENSITIVE / CRITICAL
            String mode,           // work / personal / health / finance / social
            boolean hypothetical,  // "considering / might / maybe" → true
            String expiresAt,      // ISO date if message names a future date
            String sourceMessageId,// id of the raw EPISODIC message this context came from
            String sourceExcerpt   // short quoted excerpt (<=140 chars) of the raw message
    ) {
        MessageContext withProvenance(String msgId, String excerpt) {
            return new MessageContext(subject, sensitivity, mode, hypothetical,
                    expiresAt, msgId, excerpt);
        }

        Map<String, String> asTags() {
            Map<String, String> t = new java.util.LinkedHashMap<>();
            if (subject != null && !subject.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.SUBJECT, subject);
            if (sensitivity != null && !sensitivity.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.SENSITIVITY, sensitivity);
            if (mode != null && !mode.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.MODE, mode);
            if (hypothetical)
                t.put(ai.nizo.memory.api.memory.MemoryTags.HYPOTHETICAL, "true");
            if (expiresAt != null && !expiresAt.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.EXPIRES_AT, expiresAt);
            if (sourceMessageId != null && !sourceMessageId.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.SOURCE_MESSAGE_ID, sourceMessageId);
            if (sourceExcerpt != null && !sourceExcerpt.isBlank())
                t.put(ai.nizo.memory.api.memory.MemoryTags.SOURCE_EXCERPT, sourceExcerpt);
            return t;
        }
    }

    /**
     * Detect message-level context. This runs once per message and is
     * intentionally simple regex / keyword based (not another LLM call) —
     * the goal is reliable, fast guardrails that don't depend on the
     * extraction LLM correctly tagging things.
     */
    static MessageContext analyseMessage(String message) {
        if (message == null) return new MessageContext(
                "self", "PERSONAL", "personal", false, null, null, null);
        String low = message.toLowerCase(Locale.ROOT);

        // ── Subject attribution ───────────────────────────────────────
        // "my mom's X / my dad's X / my sister has Y / my brother is Z"
        // → the fact is about that family member, NOT about the user.
        // Critical for safety: peanut allergies, medications, etc.
        //
        // R8-2 — but if the message ALSO contains first-person statements
        // ("I am Kim, my wife Priya is..."), keep subject=self. Mixed
        // messages need PROFILE routing for the user's own facts; the
        // RELATIONSHIP routing handles the wife/child/etc. separately.
        String subject = "self";
        java.util.regex.Matcher relMatcher = THIRD_PARTY_SUBJECT.matcher(low);
        if (relMatcher.find() && !hasFirstPersonStatement(low)) {
            subject = "other:" + relMatcher.group(1);
        }

        // ── Hypothetical / hedged ─────────────────────────────────────
        boolean hypo = HEDGE_PATTERN.matcher(low).find();

        // ── Mode (work / personal / health / finance / social) ────────
        String mode = "personal";
        if (containsAny(low, MODE_KEYWORDS_WORK))     mode = "work";
        else if (containsAny(low, MODE_KEYWORDS_HEALTH))  mode = "health";
        else if (containsAny(low, MODE_KEYWORDS_FINANCE)) mode = "finance";
        else if (containsAny(low, MODE_KEYWORDS_SOCIAL))  mode = "social";

        // ── Sensitivity ───────────────────────────────────────────────
        // CRITICAL = safety (allergies, meds, conditions; subject doesn't matter)
        // SENSITIVE = mental health, finance details, relationship status
        String sensitivity = "PERSONAL";
        if (containsAny(low, CRITICAL_KEYWORDS)) sensitivity = "CRITICAL";
        else if (containsAny(low, SENSITIVE_KEYWORDS)) sensitivity = "SENSITIVE";

        // ── Expiry hint ───────────────────────────────────────────────
        // Detect "on Friday / next Monday / on May 15 / by 2025-12-15"
        // and emit an ISO expiry. Past-date events get expiry=today so
        // recall demotes them.
        String expiresAt = guessExpiry(low);

        return new MessageContext(subject, sensitivity, mode, hypo, expiresAt, null, null);
    }

    /** R8-2 — true if the message contains a first-person statement about
     *  the user themselves ("i am ...", "i'm ...", "i work ...", "i live ...",
     *  "i have ..."). Used to suppress third-party subject attribution in
     *  mixed-subject messages. */
    static boolean hasFirstPersonStatement(String low) {
        String[] markers = {
                "i am ", "i'm ", "i've ", "i'll ", "i'd ",
                "i work", "i live", "i have", "i had ", "i do ",
                "i use ", "i love ", "i like ", "i prefer ",
                "i want ", "i need ", "i think ", "i feel ",
                "me and my ", "my name ", "my job ", "my role "
        };
        for (String m : markers) if (low.contains(m)) return true;
        return false;
    }

    private static final java.util.regex.Pattern THIRD_PARTY_SUBJECT =
            java.util.regex.Pattern.compile(
                    "\\bmy (mom|mother|mum|dad|father|sister|brother|son|daughter|"
                    + "wife|husband|spouse|partner|friend|colleague|boss|manager|"
                    + "child|kid|parent|sibling|cousin|aunt|uncle|niece|nephew|"
                    + "grandma|grandpa|grandmother|grandfather|in-law)\\b");

    private static final java.util.regex.Pattern HEDGE_PATTERN =
            java.util.regex.Pattern.compile(
                    "\\b(considering|thinking about|might|maybe|perhaps|possibly|"
                    + "i'd like to|i would like to|i want to try|"
                    + "looking into|exploring|tempted to|on the fence|"
                    + "not sure if|wondering if|hoping to|planning to maybe)\\b");

    private static final List<String> MODE_KEYWORDS_WORK = List.of(
            "meeting", "deploy", "sprint", "standup", "ticket", "ship ",
            "promotion", "promoted", "manager", "boss", "colleague",
            "office", "wfh", "ooo", "deadline", "kpi", "okr", "salary",
            "stripe", "phonepe", "swiggy", "razorpay", "google", "meta", "amazon",
            "microsoft", "company", "team", "report", "slack", "jira");
    private static final List<String> MODE_KEYWORDS_HEALTH = List.of(
            "doctor", "appointment", "prescription", "medication", "allergy",
            "epipen", "inhaler", "insulin", "blood", "diabetes", "asthma",
            "depression", "anxiety", "therapist", "psychiatrist", "surgery",
            "hospital", "clinic", "fever", "pain", "treatment", "diagnosis",
            "symptom", "vaccine", "checkup");
    private static final List<String> MODE_KEYWORDS_FINANCE = List.of(
            "sip", "ppf", "epf", "nps", "401k", "mutual fund", "stock", "shares",
            "salary", "bonus", "tax", "investment", "savings", "loan", "emi",
            "mortgage", "credit", "bank", "rupees", "dollars", "₹", "$", "lakh",
            "crore", "million", "portfolio",
            // Crypto — user flagged this gap explicitly. BTC/ETH/bitcoin/ethereum
            // are first-class financial instruments for a memory about "Kim's BTC".
            "btc", "bitcoin", "eth", "ethereum", "crypto", "cryptocurrency",
            "stablecoin", "usdc", "usdt", "solana", "sol ", "ether",
            "defi", "staking", "wallet", "satoshi", "block chain", "blockchain",
            "binance", "coinbase", "zerodha", "groww", "dhan", "upstox",
            // Other common financial instruments / providers we missed
            "etf", "nifty", "sensex", "dow", "nasdaq", "s&p", "index fund",
            "fd ", "fixed deposit", "recurring deposit", "rd ", "hdfc",
            "icici", "sbi", "axis", "kotak", "robo advisor",
            "dividend", "yield", "cagr", "xirr", "nav");
    private static final List<String> MODE_KEYWORDS_SOCIAL = List.of(
            "party", "wedding", "birthday", "anniversary", "festival",
            "diwali", "eid", "christmas", "thanksgiving", "celebration",
            "dinner with", "lunch with", "drinks with", "weekend with",
            "vacation", "trip");

    private static final List<String> CRITICAL_KEYWORDS = List.of(
            "allergy", "allergic", "epipen", "inhaler", "insulin", "blood thinner",
            "anaphylaxis", "anaphylactic", "diabetic", "asthmatic", "epilepsy",
            "seizure", "pacemaker", "defibrillator", "warfarin",
            "medication", "prescription", "dose", "mg ", " mg");

    private static final List<String> SENSITIVE_KEYWORDS = List.of(
            "depression", "anxiety", "ptsd", "ocd", "bipolar", "therapy",
            "therapist", "psychiatrist", "ssri", "antidepressant",
            "salary", "income", "bonus", "net worth", "debt", "loan amount",
            "divorce", "separated", "affair", "ex-husband", "ex-wife", "ex-boyfriend",
            "ex-girlfriend", "lgbtq", "gay", "lesbian", "bisexual", "trans",
            "infertility", "miscarriage", "abortion",
            "bankruptcy", "fired", "laid off", "investigation");

    private static boolean containsAny(String haystack, List<String> needles) {
        for (String n : needles) if (haystack.contains(n)) return true;
        return false;
    }

    /**
     * Best-effort future-date detector. Catches dates in the message and
     * picks the nearest future one as expiry. If the message refers to
     * a past date, we set expiry=now so recall treats it as expired.
     */
    private static String guessExpiry(String low) {
        // ISO YYYY-MM-DD wins if present
        java.util.regex.Matcher iso = java.util.regex.Pattern
                .compile("\\b(20\\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])\\b").matcher(low);
        if (iso.find()) return iso.group();
        // Day-of-week → next occurrence
        for (var entry : DAY_OF_WEEK.entrySet()) {
            if (low.contains(entry.getKey())) {
                java.time.LocalDate today = java.time.LocalDate.now();
                int days = (entry.getValue().getValue() - today.getDayOfWeek().getValue() + 7) % 7;
                if (days == 0) days = 7;
                return today.plusDays(days).toString();
            }
        }
        return null;
    }

    private static final Map<String, java.time.DayOfWeek> DAY_OF_WEEK = Map.of(
            "monday", java.time.DayOfWeek.MONDAY,
            "tuesday", java.time.DayOfWeek.TUESDAY,
            "wednesday", java.time.DayOfWeek.WEDNESDAY,
            "thursday", java.time.DayOfWeek.THURSDAY,
            "friday", java.time.DayOfWeek.FRIDAY,
            "saturday", java.time.DayOfWeek.SATURDAY,
            "sunday", java.time.DayOfWeek.SUNDAY);

    /** Quick heuristic: does this message mention investment-y content?
     *  Used only as a guard before falling back to raw message for
     *  INVESTMENT_INTEREST storage, so we don't dump unrelated messages. */
    private static boolean looksLikeInvestment(String msg) {
        if (msg == null) return false;
        String low = msg.toLowerCase(Locale.ROOT);
        return low.contains("sip") || low.contains("mutual fund") || low.contains("mutual-fund")
                || low.contains("index fund") || low.contains("etf")
                || low.contains("ppf") || low.contains("nps") || low.contains("epf")
                || low.contains("401k") || low.contains("401(k)") || low.contains("ira")
                || low.contains("isa ") || low.contains("roth")
                || low.contains("stock") || low.contains("shares")
                || low.contains("bond") || low.contains("treasury")
                || low.contains("invest") || low.contains("portfolio")
                || low.contains("zerodha") || low.contains("robinhood") || low.contains("fidelity")
                || low.contains("vanguard") || low.contains("schwab")
                || low.contains("nifty") || low.contains("sensex")
                || low.contains("s&p") || low.contains("sp500") || low.contains("nasdaq")
                || low.contains("crypto") || low.contains("bitcoin") || low.contains("btc")
                || low.contains("ethereum") || low.contains("eth ");
    }

    /** Validate a stock ticker — must be 1-5 uppercase letters (possibly with
     *  a dot for share classes like BRK.A), no lowercase, no spaces.
     *  Rejects junk like "YC alums", "Mexican angels", "SG". */
    private static boolean isValidTicker(Object tickerObj) {
        if (tickerObj == null) return false;
        String t = tickerObj.toString().trim();
        if (t.isEmpty() || t.length() > 10) return false;
        // Reject multi-word strings ("YC alums", "Mexican angels")
        if (t.contains(" ")) return false;
        // Must be primarily uppercase letters, optional dot/hyphen
        int upperCount = 0;
        for (char c : t.toCharArray()) {
            if (Character.isUpperCase(c)) upperCount++;
            else if (!".-".contains(String.valueOf(c)) && !Character.isDigit(c)) return false;
        }
        // At least 1 uppercase letter and mostly uppercase
        if (upperCount < 1) return false;
        // Reject 2-letter country codes masquerading as tickers ("SG", "US", "IN")
        // unless context strongly suggests a ticker (hard to tell; be conservative)
        if (t.length() == 2 && upperCount == 2 && Set.of(
                "SG", "US", "UK", "IN", "CN", "JP", "KR", "DE", "FR", "IT", "ES",
                "AU", "NZ", "CA", "BR", "MX", "AR", "ZA", "NG", "EG", "TR", "RU",
                "AE", "SA", "ID", "PH", "TH", "VN", "MY", "CH", "NL", "BE", "SE",
                "NO", "DK", "FI", "IE", "PT", "GR", "PL", "CZ", "HU", "IL", "IR"
        ).contains(t)) return false;
        return true;
    }

    // ===== Contradiction demotion =============================================

    /**
     * Demote old facts that are contradicted by new extractions.
     * Runs BEFORE new facts are stored so only old facts are affected.
     */
    private void demoteContradicted(String userId, Map<String, Object> extracted) {
        // RESOLUTION explicitly overrides prior decisions
        if (extracted.containsKey("RESOLUTION")) {
            for (Map<String, Object> res : toList(extracted.get("RESOLUTION"))) {
                String decision = str(res, "decision");
                if (!decision.isEmpty()) {
                    memory.demoteContradicted(userId, decision, 0.4);
                }
            }
        }

        // PROFILE changes (company, occupation) supersede old profile facts
        if (extracted.containsKey("PROFILE")) {
            Map<String, Object> profile = asMap(extracted.get("PROFILE"));
            if (profile != null) {
                String company = str(profile, "company");
                String occupation = str(profile, "occupation");
                if (!company.isEmpty() || !occupation.isEmpty()) {
                    // Demote old role/company facts (matches both old and new wording)
                    memory.demoteContradicted(userId, "job role", 0.5);
                    memory.demoteContradicted(userId, "User works", 0.5);
                }
            }
        }
    }

    // ===== JSON parsing =====================================================

    /**
     * Extracts the JSON object from the LLM response. Handles responses that
     * include markdown fences or preamble text by scanning for the first
     * {@code {curly bracket} and last {@code }}.
     */
    Map<String, Object> parseResponse(String raw) {
        if (raw == null || raw.isBlank()) return Map.of();

        try {
            int start = raw.indexOf('{');
            int end = raw.lastIndexOf('}');
            if (start < 0 || end <= start) return Map.of();

            String json = raw.substring(start, end + 1);
            @SuppressWarnings("unchecked")
            Map<String, Object> parsed = Json.MAPPER.readValue(json, Map.class);
            return parsed != null ? parsed : Map.of();
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Failed to parse extraction JSON", e);
            return Map.of();
        }
    }

    // ===== Memory routing ===================================================

    /**
     * Routes extracted facts to the memory tier. Semantic facts go through
     * {@link MemoryService#learnFact}, episodic events go through
     * {@link MemoryService#remember}.
     *
     * <p>Before storing a new fact, we check existing memory for contradictions.
     * If a new PROFILE/PREFERENCE/RESOLUTION supersedes an old fact, the old
     * fact is stored with lower confidence so the new one outranks it.
     */
    private void routeToMemory(String userId, Map<String, Object> extracted, String rawMessage) {

        // R4-B — skip PROFILE when the message is about a third party (e.g.
        // "my wife Sarah is a cardiologist"). qwen2.5:14b occasionally
        // misattributes third-party profile fields to the user's PROFILE.
        // We already know the subject from analyseMessage (via thread-local
        // CURRENT_CONTEXT); if subject isn't "self", drop PROFILE entirely.
        MessageContext ctx = CURRENT_CONTEXT.get();
        boolean thirdPartySubject = ctx != null && ctx.subject() != null
                && ctx.subject().startsWith("other:");

        // PROFILE -> learnFact (with contradiction handling for job/location changes)
        if (extracted.containsKey("PROFILE") && !thirdPartySubject) {
            Map<String, Object> profile = asMap(extracted.get("PROFILE"));
            if (profile != null) {
                routeProfile(userId, profile);
            }
        } else if (extracted.containsKey("PROFILE") && thirdPartySubject) {
            LOG.fine(() -> "Skipping PROFILE extraction for userId=" + userId
                    + " — message is about " + ctx.subject());
        }

        // RELATIONSHIP -> learnFact (retrieval-friendly: includes "family" for family members)
        if (extracted.containsKey("RELATIONSHIP")) {
            for (Map<String, Object> rel : toList(extracted.get("RELATIONSHIP"))) {
                // R4-B1 — LLM sometimes emits {spouse: "Sarah"} instead of
                // {person_name: "Sarah", relationship_type: "spouse"}. Detect
                // and normalize.
                rel = normalizeRelationshipShape(rel);
                String name = str(rel, "person_name");
                String type = str(rel, "relationship_type");
                String context = str(rel, "context");
                // Guard: the LLM routinely emits festivals ("Pongal"), places
                // ("Tirupati"), and possessive refs ("Arjun's wife") as
                // person_name, and non-relationships ("visit") as relationship_type.
                if (!isValidPersonName(name)) {
                    LOG.fine(() -> "Dropping RELATIONSHIP with invalid person_name: " + name);
                    continue;
                }
                if (!isValidRelationshipType(type)) {
                    LOG.fine(() -> "Dropping RELATIONSHIP with invalid type: " + type);
                    continue;
                }
                boolean isFamily = Set.of("spouse","wife","husband","partner",
                                           "child","son","daughter",
                                           "parent","mother","father","mom","dad",
                                           "sibling","brother","sister","family")
                        .contains(type.toLowerCase());
                String fact = name + " is the user's " + type;
                if (isFamily) fact += " (family member)";
                if (!context.isEmpty()) fact += " — " + context;
                learnFactCtx(userId, fact, SOURCE, 0.9);
            }
        }

        // PREFERENCE -> learnFact
        // When a RESOLUTION accompanies the preference (e.g., "switched from dark to light mode"),
        // the preference itself should also be at higher confidence since it's a user-stated update.
        if (extracted.containsKey("PREFERENCE")) {
            boolean hasResolution = extracted.containsKey("RESOLUTION");
            double prefConf = hasResolution ? 0.92 : 0.85;
            String prefSource = hasResolution ? "user_stated" : SOURCE;
            for (Map<String, Object> pref : toList(extracted.get("PREFERENCE"))) {
                String subject = str(pref, "subject");
                String assertion = str(pref, "assertion");
                String domain = str(pref, "domain");
                String fact = "User prefers " + subject + ": " + assertion;
                if (!domain.isEmpty()) fact += " [" + domain + "]";
                learnFactCtx(userId, fact, prefSource, prefConf);
            }
        }

        // EVENT -> remember (episodic)
        if (extracted.containsKey("EVENT")) {
            for (Map<String, Object> event : toList(extracted.get("EVENT"))) {
                String summary = str(event, "summary");
                String eventType = str(event, "event_type");
                String date = str(event, "date");
                Map<String, String> tags = new LinkedHashMap<>();
                tags.put("kind", "event");
                tags.put("event_type", eventType);
                if (!date.isEmpty()) tags.put("date", date);
                memory.remember(userId, summary, tags, SOURCE);
            }
        }

        // GOAL -> learnFact (G18 — category + synonyms in content so "my goals"
        // query hits via FTS on 'goal'/'goals'/'aspiration' tokens)
        if (extracted.containsKey("GOAL")) {
            for (Map<String, Object> goal : toList(extracted.get("GOAL"))) {
                String title = str(goal, "title");
                String description = str(goal, "description");
                String category = str(goal, "category");
                StringBuilder fact = new StringBuilder(
                        "Goal (aspiration, target, objective): User wants to ")
                        .append(title);
                if (!description.isEmpty()) fact.append(" — ").append(description);
                if (!category.isEmpty()) fact.append(" [").append(category).append("]");
                learnFactCtx(userId, fact.toString(), SOURCE, 0.85);
            }
        }

        // FOLLOW_UP -> learnFact SEMANTIC (was EPISODIC — customer tests showed
        // follow-ups get lost because recall filters out raw EPISODIC). G18/G34.
        if (extracted.containsKey("FOLLOW_UP")) {
            for (Map<String, Object> followUp : toList(extracted.get("FOLLOW_UP"))) {
                String desc = str(followUp, "description");
                String days = String.valueOf(followUp.getOrDefault("follow_up_days", 1));
                Map<String, String> tags = new LinkedHashMap<>();
                tags.put("kind", "follow_up");
                tags.put("follow_up_days", days);
                // Ctx tags (subject/mode/sensitivity/source_*) come via learnFactCtx.
                String fact = "Follow-up (task, reminder, todo, pending action): " + desc
                        + " — due in " + days + " day(s)";
                // Merge the follow_up-specific tags onto the context tags.
                MessageContext c = CURRENT_CONTEXT.get();
                Map<String, String> merged = new LinkedHashMap<>();
                if (c != null) merged.putAll(c.asTags());
                merged.putAll(tags);
                memory.learnFact(userId, fact, SOURCE, 0.85, merged);
            }
        }

        // IMPLICIT_COMMITMENT -> learnFact SEMANTIC (same reason as FOLLOW_UP)
        if (extracted.containsKey("IMPLICIT_COMMITMENT")) {
            for (Map<String, Object> commit : toList(extracted.get("IMPLICIT_COMMITMENT"))) {
                String desc = str(commit, "description");
                String type = str(commit, "commitment_type");
                String person = str(commit, "related_person");
                String timeframe = String.valueOf(commit.getOrDefault("estimated_timeframe", 7));
                Map<String, String> tags = new LinkedHashMap<>();
                tags.put("kind", "commitment");
                tags.put("commitment_type", type);
                if (!person.isEmpty()) tags.put("related_person", person);
                tags.put("estimated_timeframe_days", timeframe);
                String fact = "Commitment (intent, pending, waiting): User committed to "
                        + desc + " (type: " + type + ", timeframe: " + timeframe + "d)";
                MessageContext c = CURRENT_CONTEXT.get();
                Map<String, String> merged = new LinkedHashMap<>();
                if (c != null) merged.putAll(c.asTags());
                merged.putAll(tags);
                memory.learnFact(userId, fact, SOURCE, 0.8, merged);
            }
        }

        // INVESTMENT_INTEREST -> learnFact
        // Guaranteed narrative storage: if the LLM emits ANY INVESTMENT_INTEREST
        // category (even with empty structured fields), we STILL store the
        // original message as an investment-context fact. This is the
        // "no silent data loss" contract — before this fix, the user's SIP /
        // PPF / EPF / NPS narrative could vanish if tickers was empty.
        if (extracted.containsKey("INVESTMENT_INTEREST")) {
            for (Map<String, Object> inv : toList(extracted.get("INVESTMENT_INTEREST"))) {
                List<?> rawTickers = inv.get("tickers") instanceof List<?> t ? t : List.of();
                List<?> sectors = inv.get("sectors") instanceof List<?> s ? s : List.of();
                String style = str(inv, "style");
                String focus = str(inv, "focus");
                String vehicles = str(inv, "vehicles");
                String summary = str(inv, "summary");
                String description = str(inv, "description");

                // Filter out junk "tickers" like "YC alums", "Mexican angels", "SG"
                List<String> validTickers = new ArrayList<>();
                for (Object t : rawTickers) {
                    if (isValidTicker(t)) validTickers.add(t.toString().trim());
                }

                // Narrative form (richer, survives retrieval by natural-language query).
                // Priority: explicit summary > description > focus > raw message.
                // Falling back to rawMessage guarantees the user's investment
                // context is always stored — even when the LLM returns a
                // near-empty INVESTMENT_INTEREST object.
                String narrative = !summary.isEmpty() ? summary
                                 : !description.isEmpty() ? description
                                 : !focus.isEmpty() ? focus
                                 : rawMessage != null && looksLikeInvestment(rawMessage)
                                         ? rawMessage
                                         : "";
                if (!narrative.isEmpty()) {
                    StringBuilder nf = new StringBuilder("User investment approach: ").append(narrative);
                    if (!vehicles.isEmpty()) nf.append(" [vehicles: ").append(vehicles).append("]");
                    if (!style.isEmpty()) nf.append(" [style: ").append(style).append("]");
                    learnFactCtx(userId, nf.toString(), "user_stated", 0.9);
                }

                // Structured tickers form (kept for precise ticker lookups)
                if (!validTickers.isEmpty() || !sectors.isEmpty()) {
                    StringBuilder fact = new StringBuilder("User interested in ");
                    if (!validTickers.isEmpty()) fact.append("tickers: ").append(validTickers);
                    if (!sectors.isEmpty()) {
                        if (!validTickers.isEmpty()) fact.append(", ");
                        fact.append("sectors: ").append(sectors);
                    }
                    if (!style.isEmpty()) fact.append(", style: ").append(style);
                    learnFactCtx(userId, fact.toString(), SOURCE, 0.8);
                }
            }
        }

        // DEFERRAL -> learnFact (G18 — decision/pending/postponed synonyms)
        if (extracted.containsKey("DEFERRAL")) {
            for (Map<String, Object> deferral : toList(extracted.get("DEFERRAL"))) {
                String decision = str(deferral, "decision");
                String context = str(deferral, "context");
                StringBuilder fact = new StringBuilder(
                        "Deferral (postponed, waiting, pending decision, on hold): User deferred ")
                        .append(decision);
                if (!context.isEmpty()) fact.append(" (blocker: ").append(context).append(")");
                learnFactCtx(userId, fact.toString(), SOURCE, 0.85);
            }
        }

        // RESOLUTION -> learnFact at HIGH confidence (0.95) because resolutions
        // explicitly override prior decisions/facts. Demotion already happened
        // in demoteContradicted() before this method runs. G18 — synonyms.
        if (extracted.containsKey("RESOLUTION")) {
            for (Map<String, Object> res : toList(extracted.get("RESOLUTION"))) {
                String decision = str(res, "decision");
                String choice = str(res, "choice");
                String fact = "Resolution (decision, choice, commitment made): User decided "
                        + decision + " → " + choice;
                learnFactCtx(userId, fact, "user_stated", 0.95);
            }
        }
    }

    /**
     * Routes PROFILE fields individually so each piece of biographical data
     * becomes a separate fact in semantic memory.
     *
     * <p>Content is written to be <strong>retrieval-friendly</strong>: each fact
     * contains the words a user would naturally use to ask about it.
     * "User's current job role is Staff Engineer" matches "what is my role"
     * via FTS on "role". Without this, keyword-based retrieval fails on
     * natural language queries.
     */
    private void routeProfile(String userId, Map<String, Object> profile) {
        String name = str(profile, "name");
        // Drop placeholders / pronouns the LLM emits when there's no real
        // name in the message ("You", "User", "I", etc.)
        if (isPlaceholder(name) || isPronounish(name)) name = "";
        // G21 — reject injected payloads (PWNED, hacker, etc.)
        if (looksLikeInjectedPayload(name)) {
            LOG.warning("Dropping injected-looking PROFILE.name for userId=" + userId);
            name = "";
        }
        if (!name.isEmpty()) {
            learnFactCtx(userId, "User's name is " + name, SOURCE, 0.95);
        }

        String nickname = str(profile, "nickname");
        if (isPlaceholder(nickname) || isPronounish(nickname)) nickname = "";
        if (!nickname.isEmpty()) {
            learnFactCtx(userId, "User goes by nickname " + nickname, SOURCE, 0.9);
        }

        String city = str(profile, "location_city");
        String country = str(profile, "location_country");
        if (!city.isEmpty() || !country.isEmpty()) {
            String location = city;
            if (!country.isEmpty()) location += (city.isEmpty() ? "" : ", ") + country;
            learnFactCtx(userId, "User lives in " + location + " (home location city)", SOURCE, 0.9);
        }

        String occupation = str(profile, "occupation");
        String company = str(profile, "company");
        // Defensive: drop LLM placeholders ("unspecified", "current company",
        // "working") — we saw these polluting recall in real-world tests.
        if (isPlaceholder(occupation)) occupation = "";
        if (isPlaceholder(company)) company = "";
        // G21 — reject injected values ("hacker", "PWNED", etc.)
        if (looksLikeInjectedPayload(occupation)) occupation = "";
        if (looksLikeInjectedPayload(company)) company = "";
        // Combine role + company into one rich, retrieval-friendly fact.
        // Include multiple synonyms (work, job, role, position, employer, company)
        // so FTS matches common natural-language queries.
        if (!occupation.isEmpty() && !company.isEmpty()) {
            learnFactCtx(userId,
                    "User works at " + company + " as " + occupation
                            + " (current job role, position, employer, company)",
                    "user_stated", 0.95);
        } else if (!occupation.isEmpty()) {
            learnFactCtx(userId,
                    "User's current job role / position is " + occupation + " (work)",
                    "user_stated", 0.95);
        } else if (!company.isEmpty()) {
            learnFactCtx(userId,
                    "User works at " + company + " (employer, company, job)",
                    "user_stated", 0.95);
        }

        String industry = str(profile, "industry");
        if (!industry.isEmpty()) {
            learnFactCtx(userId, "User is in the " + industry + " industry", SOURCE, 0.85);
        }

        String birthday = str(profile, "birthday");
        if (!birthday.isEmpty()) {
            learnFactCtx(userId, "User's birthday date is " + birthday, SOURCE, 0.95);
        }

        String timezone = str(profile, "timezone");
        if (!timezone.isEmpty()) {
            learnFactCtx(userId, "User's timezone is " + timezone, SOURCE, 0.85);
        }
    }

    // ===== Helpers ==========================================================

    /** Resolve which {@link ExtractionCategory}s are present in the parsed map. */
    private Set<ExtractionCategory> resolveCategories(Map<String, Object> extracted) {
        Set<ExtractionCategory> cats = EnumSet.noneOf(ExtractionCategory.class);
        for (String key : extracted.keySet()) {
            ExtractionCategory cat = CATEGORY_KEYS.get(key);
            if (cat != null) cats.add(cat);
        }
        return cats;
    }

    /** Count the total number of discrete extractions across all categories. */
    private int countExtractions(Map<String, Object> extracted) {
        int count = 0;
        for (Map.Entry<String, Object> entry : extracted.entrySet()) {
            if (!CATEGORY_KEYS.containsKey(entry.getKey())) continue;
            Object value = entry.getValue();
            if (value instanceof List<?> list) {
                count += list.size();
            } else if (value instanceof Map) {
                count += 1;
            }
        }
        return count;
    }

    /** Safely cast an object to a String-keyed map. Returns null if not a map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object obj) {
        if (obj instanceof Map<?, ?> m) return (Map<String, Object>) m;
        return null;
    }

    /**
     * R4-B1 — handles the common LLM shape
     * {@code {spouse: "Sarah"}} / {@code {wife: "Sarah"}} /
     * {@code {mother: "Priya"}} by synthesising the canonical
     * {@code {person_name, relationship_type}} form.
     *
     * <p>If the dict already has {@code person_name} and
     * {@code relationship_type}, it's returned untouched. Otherwise we scan
     * for a single known-relationship key whose value is a plain string.
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> normalizeRelationshipShape(Map<String, Object> rel) {
        if (rel == null) return java.util.Collections.emptyMap();
        if (rel.containsKey("person_name") && rel.containsKey("relationship_type")) return rel;

        Map<String, Object> out = new LinkedHashMap<>(rel);

        // Form 2 (R8-4): {name: "Priya", relation: "wife"} or {name, relation_type}
        if (!out.containsKey("person_name")) {
            for (String k : new String[]{"name", "person", "full_name"}) {
                Object v = rel.get(k);
                if (v instanceof String s && !s.isBlank()) { out.put("person_name", s); break; }
            }
        }
        if (!out.containsKey("relationship_type")) {
            for (String k : new String[]{"relation", "relation_type", "type", "kind", "role"}) {
                Object v = rel.get(k);
                if (v instanceof String s && !s.isBlank()) { out.put("relationship_type", s); break; }
            }
        }
        if (out.containsKey("person_name") && out.containsKey("relationship_type")) return out;

        // Form 3: {spouse: "Sarah"} / {wife: "Sarah"} / {mother: "Priya"} —
        // the relationship type IS the key, the value is the name.
        Set<String> relationKeys = Set.of(
                "spouse", "wife", "husband", "partner",
                "mother", "father", "mom", "dad", "parent",
                "son", "daughter", "child", "kid",
                "brother", "sister", "sibling",
                "friend", "colleague", "mentor", "boss", "manager",
                "cofounder", "co-founder", "business_partner");
        for (Map.Entry<String, Object> e : rel.entrySet()) {
            String key = e.getKey().toLowerCase();
            if (!relationKeys.contains(key)) continue;
            Object v = e.getValue();
            if (!(v instanceof String s) || s.isBlank()) continue;
            out.put("person_name", s);
            out.put("relationship_type", key);
            return out;
        }
        return out;
    }

    /**
     * Normalize a value to a list of maps. Handles the case where the LLM
     * returns a single object instead of an array (common for
     * INVESTMENT_INTEREST and RELATIONSHIP).
     */
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> toList(Object obj) {
        if (obj instanceof List<?> list) {
            List<Map<String, Object>> result = new ArrayList<>(list.size());
            for (Object item : list) {
                if (item instanceof Map<?, ?> m) result.add((Map<String, Object>) m);
            }
            return result;
        }
        if (obj instanceof Map<?, ?> m) {
            return List.of((Map<String, Object>) m);
        }
        return List.of();
    }

    /** Safe string extraction from a map. Returns empty string if absent or null. */
    private static String str(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val == null) return "";
        String s = val.toString().strip();
        return "null".equals(s) ? "" : s;
    }
}
