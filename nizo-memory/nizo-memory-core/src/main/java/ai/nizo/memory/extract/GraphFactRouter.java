package ai.nizo.memory.extract;

import ai.nizo.memory.api.graph.Edge;
import ai.nizo.memory.api.graph.GraphService;
import ai.nizo.memory.api.graph.Node;

import java.time.Instant;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Routes extracted facts into the personal knowledge graph.
 *
 * <p>Creates nodes for every extractable entity (people, organizations,
 * locations, preferences, goals, events, topics) and wires them together
 * with typed edges. Handles cross-message self-resolution: if a user's
 * name becomes known in message N, relationships from messages 1..N-1
 * are retroactively linked via {@link #retroactiveLinking}.
 *
 * <p>De-Springified port from Kimaya's {@code GraphFactExtractor} (~670 LOC).
 * Qdrant sync dropped. No Spring, no JPA, no Lombok -- plain Java 21.
 */
public final class GraphFactRouter {

    private static final Logger LOG = Logger.getLogger(GraphFactRouter.class.getName());

    private static final String SOURCE = "extraction";

    // ── Node category constants ─────────────────────────────────────────

    static final String CAT_PERSON       = "person";
    static final String CAT_LOCATION     = "location";
    static final String CAT_ORGANIZATION = "organization";
    static final String CAT_PREFERENCE   = "preference";
    static final String CAT_EVENT        = "event";
    static final String CAT_GOAL         = "goal";
    static final String CAT_TOPIC        = "topic";

    // ── Relationship strength (higher = stronger bond) ──────────────────

    private static final Map<String, Integer> RELATIONSHIP_STRENGTH = Map.ofEntries(
            Map.entry("spouse",      10),
            Map.entry("partner",     10),
            Map.entry("parent",       9),
            Map.entry("child",        9),
            Map.entry("sibling",      9),
            Map.entry("family",       8),
            Map.entry("mentor",       7),
            Map.entry("manager",      7),
            Map.entry("colleague",    5),
            Map.entry("coworker",     5),
            Map.entry("friend",       4),
            Map.entry("acquaintance", 2),
            Map.entry("knows",        1)
    );

    // ── Dependencies ────────────────────────────────────────────────────

    private final GraphService graph;

    public GraphFactRouter(GraphService graph) {
        this.graph = Objects.requireNonNull(graph, "graph");
    }

    // ===== Public entry point ===============================================

    /**
     * Routes all extracted facts into the knowledge graph. Creates nodes,
     * then wires edges, then runs retroactive linking for orphans.
     *
     * @param userId    owner
     * @param extracted parsed extraction map (keys are category names)
     * @return total number of graph operations performed
     */
    public int routeToGraph(String userId, Map<String, Object> extracted) {
        if (extracted == null || extracted.isEmpty()) return 0;

        int ops = 0;
        try {
            ops += mirrorToGraph(userId, extracted);
            ops += createImmediateEdges(userId, extracted);
            ops += retroactiveLinking(userId);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "Error during graph routing for userId=" + userId, e);
        }
        int totalOps = ops;
        LOG.fine(() -> "Graph routing completed: " + totalOps + " ops for userId=" + userId);
        return ops;
    }

    // ===== Node creation (mirrorToGraph) ====================================

    /**
     * Creates or merges graph nodes for every entity found in the extraction.
     */
    int mirrorToGraph(String userId, Map<String, Object> extracted) {
        int ops = 0;

        // PROFILE -> person (is_self=true) + organization + location
        if (extracted.containsKey("PROFILE")) {
            Map<String, Object> profile = asMap(extracted.get("PROFILE"));
            if (profile != null) {
                ops += mirrorProfile(userId, profile);
            }
        }

        // RELATIONSHIP -> person nodes (with disambiguation)
        if (extracted.containsKey("RELATIONSHIP")) {
            for (Map<String, Object> rel : toList(extracted.get("RELATIONSHIP"))) {
                String name = str(rel, "person_name");
                if (name.isEmpty()) continue;

                Map<String, Object> props = new LinkedHashMap<>();
                copyIfPresent(rel, props, "relationship_type", "context");
                graph.createOrMergePersonNode(userId, name, props, SOURCE);
                ops++;
            }
        }

        // EVENT -> event nodes
        if (extracted.containsKey("EVENT")) {
            for (Map<String, Object> event : toList(extracted.get("EVENT"))) {
                String summary = str(event, "summary");
                if (summary.isEmpty()) continue;

                Map<String, Object> props = new LinkedHashMap<>();
                copyIfPresent(event, props, "event_type", "date", "emotional_valence");
                Object participants = event.get("participants");
                if (participants != null) props.put("participants", participants);

                Node node = new Node(
                        UUID.randomUUID().toString(), userId, CAT_EVENT, summary,
                        props, 0.85, SOURCE, "active", 1, Instant.now(), Instant.now()
                );
                graph.createOrMergeNode(node);
                ops++;
            }
        }

        // GOAL -> goal nodes
        if (extracted.containsKey("GOAL")) {
            for (Map<String, Object> goal : toList(extracted.get("GOAL"))) {
                String title = str(goal, "title");
                if (title.isEmpty()) continue;

                Map<String, Object> props = new LinkedHashMap<>();
                copyIfPresent(goal, props, "description", "category", "target_date", "priority");

                Node node = new Node(
                        UUID.randomUUID().toString(), userId, CAT_GOAL, title,
                        props, 0.85, SOURCE, "active", 1, Instant.now(), Instant.now()
                );
                graph.createOrMergeNode(node);
                ops++;
            }
        }

        // PREFERENCE -> preference nodes (label = "subject: assertion")
        if (extracted.containsKey("PREFERENCE")) {
            for (Map<String, Object> pref : toList(extracted.get("PREFERENCE"))) {
                String subject = str(pref, "subject");
                String assertion = str(pref, "assertion");
                if (subject.isEmpty() && assertion.isEmpty()) continue;

                String label = subject + ": " + assertion;
                Map<String, Object> props = new LinkedHashMap<>();
                copyIfPresent(pref, props, "domain");
                props.put("subject", subject);
                props.put("assertion", assertion);

                Node node = new Node(
                        UUID.randomUUID().toString(), userId, CAT_PREFERENCE, label,
                        props, 0.85, SOURCE, "active", 1, Instant.now(), Instant.now()
                );
                graph.createOrMergeNode(node);
                ops++;
            }
        }

        // INVESTMENT_INTEREST -> topic nodes (one per ticker)
        if (extracted.containsKey("INVESTMENT_INTEREST")) {
            for (Map<String, Object> inv : toList(extracted.get("INVESTMENT_INTEREST"))) {
                List<?> tickers = inv.get("tickers") instanceof List<?> t ? t : List.of();
                for (Object ticker : tickers) {
                    String tickerStr = ticker.toString().strip();
                    if (tickerStr.isEmpty()) continue;

                    Map<String, Object> props = new LinkedHashMap<>();
                    props.put("type", "investment");
                    props.put("ticker", tickerStr);
                    copyIfPresent(inv, props, "style", "risk_appetite", "focus");

                    Node node = new Node(
                            UUID.randomUUID().toString(), userId, CAT_TOPIC, tickerStr,
                            props, 0.8, SOURCE, "active", 1, Instant.now(), Instant.now()
                    );
                    graph.createOrMergeNode(node);
                    ops++;
                }
            }
        }

        // IMPLICIT_COMMITMENT -> topic nodes
        if (extracted.containsKey("IMPLICIT_COMMITMENT")) {
            for (Map<String, Object> commit : toList(extracted.get("IMPLICIT_COMMITMENT"))) {
                String desc = str(commit, "description");
                if (desc.isEmpty()) continue;

                Map<String, Object> props = new LinkedHashMap<>();
                props.put("type", "commitment");
                copyIfPresent(commit, props, "commitment_type", "related_person", "estimated_timeframe");

                Node node = new Node(
                        UUID.randomUUID().toString(), userId, CAT_TOPIC, desc,
                        props, 0.8, SOURCE, "active", 1, Instant.now(), Instant.now()
                );
                graph.createOrMergeNode(node);
                ops++;
            }
        }

        return ops;
    }

    /**
     * Mirrors PROFILE data into graph nodes: a self person node, plus optional
     * organization and location nodes.
     */
    private int mirrorProfile(String userId, Map<String, Object> profile) {
        int ops = 0;

        // Self person node
        String name = str(profile, "name");
        if (name.isEmpty()) name = str(profile, "nickname");
        if (name.isEmpty()) name = "Me";

        Map<String, Object> selfProps = new LinkedHashMap<>();
        selfProps.put("is_self", true);
        copyIfPresent(profile, selfProps,
                "name", "nickname", "occupation", "industry", "birthday", "timezone",
                "location_city", "location_country", "company");

        graph.createOrMergePersonNode(userId, name, selfProps, "user_stated");
        ops++;

        // Organization node
        String company = str(profile, "company");
        if (!company.isEmpty()) {
            Map<String, Object> orgProps = new LinkedHashMap<>();
            copyIfPresent(profile, orgProps, "industry");

            Node orgNode = new Node(
                    UUID.randomUUID().toString(), userId, CAT_ORGANIZATION, company,
                    orgProps, 0.85, "user_stated", "active", 1, Instant.now(), Instant.now()
            );
            graph.createOrMergeNode(orgNode);
            ops++;
        }

        // Location node
        String city = str(profile, "location_city");
        String country = str(profile, "location_country");
        String locationLabel = city;
        if (!country.isEmpty()) {
            locationLabel += (city.isEmpty() ? "" : ", ") + country;
        }
        if (!locationLabel.isEmpty()) {
            Map<String, Object> locProps = new LinkedHashMap<>();
            if (!city.isEmpty()) locProps.put("city", city);
            if (!country.isEmpty()) locProps.put("country", country);

            Node locNode = new Node(
                    UUID.randomUUID().toString(), userId, CAT_LOCATION, locationLabel,
                    locProps, 0.85, "user_stated", "active", 1, Instant.now(), Instant.now()
            );
            graph.createOrMergeNode(locNode);
            ops++;
        }

        return ops;
    }

    // ===== Edge creation ====================================================

    /**
     * Creates edges between nodes created in this extraction pass.
     * Resolves the self node for the user and links relationships, goals,
     * preferences, and other entities to it.
     */
    int createImmediateEdges(String userId, Map<String, Object> extracted) {
        int ops = 0;

        // Resolve the self node
        Node self = resolveSelfNode(userId, extracted);
        if (self == null) {
            // Without a self node we can still link event participants
            ops += linkEventParticipants(userId, extracted, null);
            return ops;
        }

        // RELATIONSHIP -> self <-> person edges (with hierarchy check)
        if (extracted.containsKey("RELATIONSHIP")) {
            for (Map<String, Object> rel : toList(extracted.get("RELATIONSHIP"))) {
                String name = str(rel, "person_name");
                String relType = str(rel, "relationship_type");
                if (name.isEmpty() || relType.isEmpty()) continue;

                Optional<Node> personOpt = graph.resolveEntity(userId, name);
                if (personOpt.isEmpty()) continue;
                Node person = personOpt.get();

                // Only create edge if no stronger relationship already exists
                if (!hasStrongerRelationship(userId, self.id(), person.id(), relType)) {
                    Edge edge = new Edge(
                            UUID.randomUUID().toString(), userId,
                            self.id(), person.id(), relType,
                            Map.of(), Instant.now(), null, 0.9, SOURCE, null
                    );
                    graph.createEdgeIfNotExists(edge);
                    ops++;
                }
            }
        }

        // GOAL -> self -[has_goal]-> goal
        if (extracted.containsKey("GOAL")) {
            for (Map<String, Object> goal : toList(extracted.get("GOAL"))) {
                String title = str(goal, "title");
                if (title.isEmpty()) continue;

                Optional<Node> goalNode = graph.resolveEntity(userId, title);
                if (goalNode.isEmpty()) continue;

                Edge edge = new Edge(
                        UUID.randomUUID().toString(), userId,
                        self.id(), goalNode.get().id(), "has_goal",
                        Map.of(), Instant.now(), null, 0.85, SOURCE, null
                );
                graph.createEdgeIfNotExists(edge);
                ops++;
            }
        }

        // PROFILE -> self -[works_at]-> company, self -[lives_in]-> location
        if (extracted.containsKey("PROFILE")) {
            Map<String, Object> profile = asMap(extracted.get("PROFILE"));
            if (profile != null) {
                String company = str(profile, "company");
                if (!company.isEmpty()) {
                    Optional<Node> orgNode = graph.resolveEntity(userId, company);
                    if (orgNode.isPresent()) {
                        Edge edge = new Edge(
                                UUID.randomUUID().toString(), userId,
                                self.id(), orgNode.get().id(), "works_at",
                                Map.of(), Instant.now(), null, 0.9, "user_stated", null
                        );
                        graph.createEdgeIfNotExists(edge);
                        ops++;
                    }
                }

                String city = str(profile, "location_city");
                if (!city.isEmpty()) {
                    Optional<Node> locNode = graph.resolveEntity(userId, city);
                    if (locNode.isPresent()) {
                        Edge edge = new Edge(
                                UUID.randomUUID().toString(), userId,
                                self.id(), locNode.get().id(), "lives_in",
                                Map.of(), Instant.now(), null, 0.9, "user_stated", null
                        );
                        graph.createEdgeIfNotExists(edge);
                        ops++;
                    }
                }
            }
        }

        // PREFERENCE -> self -[has_preference]-> preference
        if (extracted.containsKey("PREFERENCE")) {
            for (Map<String, Object> pref : toList(extracted.get("PREFERENCE"))) {
                String subject = str(pref, "subject");
                String assertion = str(pref, "assertion");
                String label = subject + ": " + assertion;

                Optional<Node> prefNode = graph.resolveEntity(userId, label);
                if (prefNode.isEmpty()) continue;

                Edge edge = new Edge(
                        UUID.randomUUID().toString(), userId,
                        self.id(), prefNode.get().id(), "has_preference",
                        Map.of(), Instant.now(), null, 0.85, SOURCE, null
                );
                graph.createEdgeIfNotExists(edge);
                ops++;
            }
        }

        // INVESTMENT_INTEREST -> self -[interested_in]-> ticker
        if (extracted.containsKey("INVESTMENT_INTEREST")) {
            for (Map<String, Object> inv : toList(extracted.get("INVESTMENT_INTEREST"))) {
                List<?> tickers = inv.get("tickers") instanceof List<?> t ? t : List.of();
                for (Object ticker : tickers) {
                    String tickerStr = ticker.toString().strip();
                    if (tickerStr.isEmpty()) continue;

                    Optional<Node> tickerNode = graph.resolveEntity(userId, tickerStr);
                    if (tickerNode.isEmpty()) continue;

                    Edge edge = new Edge(
                            UUID.randomUUID().toString(), userId,
                            self.id(), tickerNode.get().id(), "interested_in",
                            Map.of(), Instant.now(), null, 0.8, SOURCE, null
                    );
                    graph.createEdgeIfNotExists(edge);
                    ops++;
                }
            }
        }

        // EVENT participants
        ops += linkEventParticipants(userId, extracted, self);

        return ops;
    }

    /**
     * Links event participants to event nodes. If a self node is available,
     * adds the self node as a participant too.
     */
    private int linkEventParticipants(String userId, Map<String, Object> extracted, Node self) {
        int ops = 0;
        if (!extracted.containsKey("EVENT")) return ops;

        for (Map<String, Object> event : toList(extracted.get("EVENT"))) {
            String summary = str(event, "summary");
            if (summary.isEmpty()) continue;

            Optional<Node> eventNode = graph.resolveEntity(userId, summary);
            if (eventNode.isEmpty()) continue;

            // Link each participant to the event
            Object participantsObj = event.get("participants");
            List<?> participants = participantsObj instanceof List<?> p ? p : List.of();
            for (Object participant : participants) {
                String pName = participant.toString().strip();
                if (pName.isEmpty()) continue;

                Optional<Node> personNode = graph.resolveEntity(userId, pName);
                if (personNode.isEmpty()) continue;

                Edge edge = new Edge(
                        UUID.randomUUID().toString(), userId,
                        personNode.get().id(), eventNode.get().id(), "participated_in",
                        Map.of(), Instant.now(), null, 0.8, SOURCE, null
                );
                graph.createEdgeIfNotExists(edge);
                ops++;
            }

            // Self participated too, if available
            if (self != null) {
                Edge selfEdge = new Edge(
                        UUID.randomUUID().toString(), userId,
                        self.id(), eventNode.get().id(), "participated_in",
                        Map.of(), Instant.now(), null, 0.85, SOURCE, null
                );
                graph.createEdgeIfNotExists(selfEdge);
                ops++;
            }
        }

        return ops;
    }

    // ===== Retroactive linking ==============================================

    /**
     * Finds orphaned nodes (person, goal, org, location) that have no edges
     * to the self node and creates appropriate edges. Runs after every
     * extraction pass so that earlier facts get linked once a self node
     * is available.
     */
    int retroactiveLinking(String userId) {
        int ops = 0;

        Node self;
        try {
            self = graph.findSelfNode(userId);
        } catch (Exception e) {
            LOG.fine(() -> "No self node found for retroactive linking, userId=" + userId);
            return 0;
        }

        // Build set of already-connected node IDs
        List<Edge> existingEdges = graph.getEdgesForNode(userId, self.id());
        Set<String> connectedIds = new HashSet<>();
        for (Edge e : existingEdges) {
            connectedIds.add(e.sourceNodeId());
            connectedIds.add(e.targetNodeId());
        }

        // Orphaned person nodes with relationship_type
        List<Node> persons = graph.getNodesByCategory(userId, CAT_PERSON);
        for (Node person : persons) {
            if (!person.isAccessible()) continue;
            if (person.id().equals(self.id())) continue;
            if (connectedIds.contains(person.id())) continue;

            String relType = person.properties() != null
                    ? (String) person.properties().get("relationship_type")
                    : null;
            if (relType == null || relType.isBlank()) continue;

            Edge edge = new Edge(
                    UUID.randomUUID().toString(), userId,
                    self.id(), person.id(), relType,
                    Map.of(), Instant.now(), null, 0.8, SOURCE, null
            );
            graph.createEdgeIfNotExists(edge);
            ops++;
        }

        // Orphaned goal nodes
        List<Node> goals = graph.getNodesByCategory(userId, CAT_GOAL);
        for (Node goal : goals) {
            if (!goal.isAccessible()) continue;
            if (connectedIds.contains(goal.id())) continue;

            Edge edge = new Edge(
                    UUID.randomUUID().toString(), userId,
                    self.id(), goal.id(), "has_goal",
                    Map.of(), Instant.now(), null, 0.8, SOURCE, null
            );
            graph.createEdgeIfNotExists(edge);
            ops++;
        }

        // Orphaned organization matching self.company
        String selfCompany = self.properties() != null
                ? str(self.properties(), "company")
                : "";
        if (!selfCompany.isEmpty()) {
            List<Node> orgs = graph.getNodesByCategory(userId, CAT_ORGANIZATION);
            for (Node org : orgs) {
                if (!org.isAccessible()) continue;
                if (connectedIds.contains(org.id())) continue;
                if (org.label().equalsIgnoreCase(selfCompany)) {
                    Edge edge = new Edge(
                            UUID.randomUUID().toString(), userId,
                            self.id(), org.id(), "works_at",
                            Map.of(), Instant.now(), null, 0.8, SOURCE, null
                    );
                    graph.createEdgeIfNotExists(edge);
                    ops++;
                }
            }
        }

        // Orphaned location matching self.location_city
        String selfCity = self.properties() != null
                ? str(self.properties(), "location_city")
                : "";
        if (!selfCity.isEmpty()) {
            List<Node> locations = graph.getNodesByCategory(userId, CAT_LOCATION);
            for (Node loc : locations) {
                if (!loc.isAccessible()) continue;
                if (connectedIds.contains(loc.id())) continue;
                String locCity = loc.properties() != null ? str(loc.properties(), "city") : "";
                if (locCity.equalsIgnoreCase(selfCity) || loc.label().toLowerCase().contains(selfCity.toLowerCase())) {
                    Edge edge = new Edge(
                            UUID.randomUUID().toString(), userId,
                            self.id(), loc.id(), "lives_in",
                            Map.of(), Instant.now(), null, 0.8, SOURCE, null
                    );
                    graph.createEdgeIfNotExists(edge);
                    ops++;
                }
            }
        }

        if (ops > 0) {
            int retroOps = ops;
            LOG.fine(() -> "Retroactive linking created " + retroOps + " edges for userId=" + userId);
        }
        return ops;
    }

    // ===== Relationship strength ============================================

    /**
     * Returns {@code true} if any existing edge between source and target
     * has a higher relationship strength than the proposed one.
     */
    boolean hasStrongerRelationship(String userId, String sourceId, String targetId, String proposedRelType) {
        int proposedStrength = RELATIONSHIP_STRENGTH.getOrDefault(proposedRelType, 0);
        if (proposedStrength == 0) return false;

        List<Edge> edges = graph.getEdgesBetween(userId, sourceId, targetId);
        for (Edge edge : edges) {
            if (!edge.isCurrent()) continue;
            int currentStrength = RELATIONSHIP_STRENGTH.getOrDefault(edge.relationship(), 0);
            if (currentStrength > proposedStrength) return true;
        }
        return false;
    }

    // ===== Self-node resolution =============================================

    /**
     * Resolves the self node using the current extraction's PROFILE name if
     * available, falling back to {@link GraphService#findSelfNode}.
     */
    private Node resolveSelfNode(String userId, Map<String, Object> extracted) {
        // Try resolving by name from current PROFILE extraction
        if (extracted.containsKey("PROFILE")) {
            Map<String, Object> profile = asMap(extracted.get("PROFILE"));
            if (profile != null) {
                String name = str(profile, "name");
                if (!name.isEmpty()) {
                    Optional<Node> resolved = graph.resolveEntity(userId, name);
                    if (resolved.isPresent()
                            && resolved.get().properties() != null
                            && Boolean.TRUE.equals(resolved.get().properties().get("is_self"))) {
                        return resolved.get();
                    }
                }
            }
        }

        // Fallback to findSelfNode
        try {
            return graph.findSelfNode(userId);
        } catch (Exception e) {
            LOG.fine(() -> "Could not resolve self node for userId=" + userId);
            return null;
        }
    }

    // ===== Utility helpers ==================================================

    /** Safely cast an object to a String-keyed map. Returns null if not a map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object obj) {
        if (obj instanceof Map<?, ?> m) return (Map<String, Object>) m;
        return null;
    }

    /**
     * Normalize a value to a list of maps. Handles the case where the LLM
     * returns a single object instead of an array.
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

    /** Copy non-null, non-blank values for the given keys from source to destination. */
    private static void copyIfPresent(Map<String, Object> src, Map<String, Object> dst, String... keys) {
        for (String key : keys) {
            Object val = src.get(key);
            if (val == null) continue;
            if (val instanceof String s && s.isBlank()) continue;
            dst.put(key, val);
        }
    }
}
