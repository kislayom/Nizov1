package ai.nizo.tools.finance;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * India sector view — Phase 3 of the "India Picks" build.
 *
 * <p>Pulls the major NSE sector indices, computes per-sector composite scores
 * (momentum + valuation + breadth), and returns a ranked list of sectors.
 * Used by {@link ai.nizo.skills.IndiaTopPicksTool} to bias picks toward
 * sectors with favorable top-down conditions.
 *
 * <p>Tracked sectors (NSE-published thematic indices, all free):
 * <ul>
 *   <li>{@code NIFTY BANK}        — banking heavyweights</li>
 *   <li>{@code NIFTY IT}          — IT services exporters</li>
 *   <li>{@code NIFTY PHARMA}      — pharma generics + branded</li>
 *   <li>{@code NIFTY FMCG}        — FMCG / consumer staples</li>
 *   <li>{@code NIFTY AUTO}        — passenger + commercial vehicles</li>
 *   <li>{@code NIFTY ENERGY}      — oil refining + power gen</li>
 *   <li>{@code NIFTY METAL}       — steel + non-ferrous</li>
 *   <li>{@code NIFTY REALTY}      — real estate / construction</li>
 *   <li>{@code NIFTY PSU BANK}    — PSU banks</li>
 *   <li>{@code NIFTY MEDIA}       — media + broadcasting</li>
 * </ul>
 *
 * <p>Per-sector composite (0-100): {@code 0.5 * momentum + 0.3 * breadth + 0.2 * valueProxy}.
 * Momentum from index's perChange365; breadth from advance-decline ratio in the
 * NSE response; valueProxy = inverse of 1y price-percentile range (lower = better).
 *
 * <p>Cached 4h on disk at {@code ~/.nizo/cache/india-sectors.json}.
 */
public final class IndiaSectorViewTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaSectorViewTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long CACHE_TTL_MS = 4L * 60L * 60L * 1000L;

    private static final String[] SECTORS = {
            "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY AUTO",
            "NIFTY ENERGY", "NIFTY METAL", "NIFTY REALTY", "NIFTY PSU BANK", "NIFTY MEDIA"
    };

    /** Map NSE sector-index label → human-readable "sector" string used by industries field. */
    private static final Map<String, String[]> SECTOR_INDUSTRY_KEYWORDS = Map.of(
            "NIFTY BANK",     new String[]{"private sector bank", "bank"},
            "NIFTY IT",       new String[]{"computers", "it ", "software", "consulting"},
            "NIFTY PHARMA",   new String[]{"pharma", "pharmaceutical"},
            "NIFTY FMCG",     new String[]{"fmcg", "diversified fmcg", "personal care", "cigarettes"},
            "NIFTY AUTO",     new String[]{"auto", "automobile", "commercial vehicle", "passenger"},
            "NIFTY ENERGY",   new String[]{"refineries", "oil", "gas", "petroleum", "power"},
            "NIFTY METAL",    new String[]{"metal", "steel", "iron", "aluminium"},
            "NIFTY REALTY",   new String[]{"realty", "construction", "real estate"},
            "NIFTY PSU BANK", new String[]{"public sector bank", "psu bank"},
            "NIFTY MEDIA",    new String[]{"media", "broadcasting", "entertainment"}
    );

    private final NseIndiaClient nse;
    private final Path cacheDir;

    public IndiaSectorViewTool() { this(new NseIndiaClient()); }

    public IndiaSectorViewTool(NseIndiaClient nse) {
        this.nse = nse;
        String home = System.getProperty("user.home", ".");
        this.cacheDir = Paths.get(home, ".nizo", "cache");
        try { Files.createDirectories(cacheDir); } catch (Exception e) { /* best-effort */ }
    }

    @Override public String name() { return "india_sector_view"; }

    @Override
    public String description() {
        return "Per-sector momentum + valuation + breadth rank for the 10 major NSE sector "
                + "indices (Bank, IT, Pharma, FMCG, Auto, Energy, Metal, Realty, PSU Bank, Media). "
                + "Returns composite score per sector + a sector-tilt map that india_top_picks "
                + "uses to bias picks. Cached 4h on disk.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "refresh": { "type": "boolean", "description": "Bypass 4h cache, re-fetch fresh." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            boolean force = args.path("refresh").asBoolean(false);

            if (!force) {
                ObjectNode cached = readCache();
                if (cached != null) return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(cached));
            }

            ExecutorService pool = Executors.newFixedThreadPool(SECTORS.length, r -> {
                Thread t = new Thread(r, "sector-view");
                t.setDaemon(true);
                return t;
            });
            Map<String, CompletableFuture<SectorScore>> futs = new HashMap<>();
            for (String s : SECTORS) {
                futs.put(s, CompletableFuture.supplyAsync(() -> scoreSector(s), pool));
            }
            List<SectorScore> scored = new ArrayList<>();
            for (Map.Entry<String, CompletableFuture<SectorScore>> e : futs.entrySet()) {
                try {
                    SectorScore s = e.getValue().get(20, TimeUnit.SECONDS);
                    if (s != null) scored.add(s);
                } catch (Exception ex) {
                    LOG.debug("sector {} failed: {}", e.getKey(), ex.toString());
                }
            }
            pool.shutdown();
            try { pool.awaitTermination(2, TimeUnit.SECONDS); }
            catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

            // Normalize value/momentum/breadth to percentile within the set, then composite.
            applyPercentiles(scored);
            for (SectorScore s : scored) {
                s.composite = 0.50 * s.momentumScore + 0.30 * s.breadthScore + 0.20 * s.valueScore;
            }
            scored.sort(Comparator.comparingDouble((SectorScore s) -> s.composite).reversed());

            ObjectNode out = MAPPER.createObjectNode();
            out.put("asOf", Instant.now().toString());
            out.put("source", "nseindia.com (live)");
            ArrayNode arr = out.putArray("sectors");
            int rank = 1;
            // Tilt map: human-readable industry keyword -> bias (-10..+10)
            ObjectNode tilts = MAPPER.createObjectNode();
            for (SectorScore s : scored) {
                ObjectNode r = MAPPER.createObjectNode();
                r.put("rank", rank++);
                r.put("index", s.indexName);
                r.put("composite", round1(s.composite));
                ObjectNode comp = r.putObject("components");
                comp.put("momentum", round1(s.momentumScore));
                comp.put("breadth", round1(s.breadthScore));
                comp.put("value", round1(s.valueScore));
                ObjectNode raw = r.putObject("rawSignals");
                putIfFinite(raw, "perChange365", s.perChange365);
                putIfFinite(raw, "perChange30", s.perChange30);
                putIfFinite(raw, "advanceRatio", s.advanceRatio);
                putIfFinite(raw, "lastPrice", s.lastPrice);
                arr.add(r);
                // Bias: rank 1 → +10, rank 10 → -10 linearly (subject to dispersion)
                double bias = 10.0 * (s.composite - 50.0) / 50.0;   // map 0..100 score → -10..+10
                for (String kw : SECTOR_INDUSTRY_KEYWORDS.getOrDefault(s.indexName, new String[0])) {
                    tilts.put(kw, bias);
                }
            }
            out.set("sectorTiltByKeyword", tilts);

            writeCache(out);
            return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
        } catch (Exception e) {
            LOG.warn("india_sector_view failed: {}", e.toString());
            return ToolResult.error("india_sector_view failed: " + e.getMessage());
        }
    }

    /* ─────────────────────────────────────────────────────────── scoring */

    private SectorScore scoreSector(String indexName) {
        SectorScore s = new SectorScore();
        s.indexName = indexName;
        JsonNode raw = nse.indexConstituents(indexName);
        if (raw == null) return null;
        // Index summary row (priority=1) carries perChange30/365 + lastPrice for the index itself.
        JsonNode summary = null;
        for (JsonNode row : raw.path("data")) {
            if (row.path("priority").asInt(0) == 1 || row.path("symbol").asText("").equalsIgnoreCase(indexName)) {
                summary = row; break;
            }
        }
        if (summary == null && raw.path("data").size() > 0) summary = raw.path("data").get(0);
        if (summary != null) {
            s.lastPrice    = summary.path("lastPrice").asDouble(Double.NaN);
            s.perChange30  = summary.path("perChange30d").asDouble(Double.NaN);
            s.perChange365 = summary.path("perChange365d").asDouble(Double.NaN);
        }
        // Breadth: advance / (advance + decline) ratio from the NSE response's advance/declines.
        JsonNode adv = raw.path("advance");
        double advances = adv.path("advances").asDouble(0);
        double declines = adv.path("declines").asDouble(0);
        if (advances + declines > 0) s.advanceRatio = advances / (advances + declines);
        return s;
    }

    private static void applyPercentiles(List<SectorScore> all) {
        // Raw signals → 0-100 percentile rank within the set.
        // - momentum from perChange365 (with perChange30 as tie-breaker)
        // - breadth from advanceRatio
        // - value from inverse perChange365 (cheaper sectors tend to have lagged) — approximation
        double[] mom = all.stream().mapToDouble(x -> x.perChange365).toArray();
        double[] br  = all.stream().mapToDouble(x -> x.advanceRatio).toArray();
        for (SectorScore s : all) {
            s.momentumScore = percentile(mom, s.perChange365);
            s.breadthScore  = percentile(br, s.advanceRatio);
            // Value heuristic: sectors that DIDN'T rocket are "cheaper" relative to peers.
            // 0% return → value=50 baseline. >+30% → value penalty. <-10% → value bonus.
            if (Double.isNaN(s.perChange365)) s.valueScore = 50;
            else if (s.perChange365 > 30) s.valueScore = clamp(50 - (s.perChange365 - 30), 0, 50);
            else if (s.perChange365 < -10) s.valueScore = clamp(50 + (-s.perChange365 - 10), 50, 100);
            else s.valueScore = 50;
        }
    }

    private static double percentile(double[] sorted, double v) {
        if (Double.isNaN(v)) return 50;
        double[] copy = sorted.clone();
        Arrays.sort(copy);
        int idx = 0;
        for (double x : copy) { if (Double.isNaN(x)) continue; if (x <= v) idx++; }
        int valid = 0;
        for (double x : copy) if (!Double.isNaN(x)) valid++;
        return valid == 0 ? 50 : 100.0 * idx / valid;
    }

    private static double clamp(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

    private static void putIfFinite(ObjectNode o, String k, double v) {
        if (!Double.isNaN(v) && !Double.isInfinite(v)) o.put(k, v);
    }

    private static double round1(double v) { return Math.round(v * 10.0) / 10.0; }

    /* ─────────────────────────────────────────────────────── disk cache */

    private Path cachePath() { return cacheDir.resolve("india-sectors.json"); }

    private ObjectNode readCache() {
        Path p = cachePath();
        if (!Files.isRegularFile(p)) return null;
        try {
            long age = System.currentTimeMillis() - Files.getLastModifiedTime(p).toMillis();
            if (age > CACHE_TTL_MS) return null;
            JsonNode node = MAPPER.readTree(Files.readAllBytes(p));
            if (!(node instanceof ObjectNode obj)) return null;
            obj.put("source", "nseindia.com (cached " + (age / 60_000L) + "m ago)");
            return obj;
        } catch (Exception e) { return null; }
    }

    private void writeCache(ObjectNode payload) {
        try {
            Files.write(cachePath(), MAPPER.writeValueAsBytes(payload),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (Exception e) { /* best-effort */ }
    }

    private static final class SectorScore {
        String indexName;
        double lastPrice = Double.NaN;
        double perChange30 = Double.NaN;
        double perChange365 = Double.NaN;
        double advanceRatio = Double.NaN;
        double momentumScore = 50;
        double breadthScore = 50;
        double valueScore = 50;
        double composite = 50;
    }
}
