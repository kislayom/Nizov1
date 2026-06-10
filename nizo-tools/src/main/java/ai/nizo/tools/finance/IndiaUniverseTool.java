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
import java.util.Locale;

/**
 * Indian equity universe — constituents of an NSE-published index (NIFTY 500 by default).
 *
 * <p>Used by the {@code india_top_picks} skill as the candidate pool to rank.
 * Authoritative source is NSE's {@code /api/equity-stockIndices?index=...} endpoint;
 * we cache the response for 24h on disk so subsequent runs don't pound NSE.
 *
 * <p>Returned JSON shape (one example row):
 * <pre>{@code
 * {
 *   "index": "NIFTY 500",
 *   "asOf":  "2026-05-19T18:34:21Z",
 *   "source": "nseindia.com (cached 4h)",
 *   "constituents": [
 *     {
 *       "symbol":       "RELIANCE",
 *       "ticker":       "RELIANCE.NS",      // Yahoo-style — usable downstream
 *       "companyName":  "Reliance Industries Limited",
 *       "industry":     "Refineries",
 *       "lastPrice":    1335.90,
 *       "freeFloatMc":  9.3e12,
 *       "yearHigh":     1620.10,
 *       "yearLow":      1140.45,
 *       "perChange365": -2.6,
 *       "perChange30":   1.4
 *     }, ...
 *   ],
 *   "count": 500
 * }
 * }</pre>
 *
 * <p>Supported {@code index} values (anything NSE publishes; common ones):
 * <ul>
 *   <li>{@code "NIFTY 500"} — broad universe (default)</li>
 *   <li>{@code "NIFTY 50"} — large-cap blue chips</li>
 *   <li>{@code "NIFTY MIDCAP 150"} — mid-caps</li>
 *   <li>{@code "NIFTY SMALLCAP 250"} — small-caps</li>
 *   <li>{@code "NIFTY BANK"} — banking sector</li>
 *   <li>{@code "NIFTY IT"} — IT sector</li>
 *   <li>{@code "NIFTY PHARMA"} — pharma sector</li>
 *   <li>{@code "NIFTY FMCG"} — FMCG sector</li>
 *   <li>{@code "NIFTY AUTO"} — autos</li>
 *   <li>{@code "NIFTY ENERGY"} — energy / refining</li>
 *   <li>{@code "NIFTY METAL"} — metals</li>
 * </ul>
 *
 * <p>Disabled gracefully if {@code NIZO_NSEINDIA_DISABLED=1} or NSE is unreachable —
 * tool returns {@code "error":"universe-unavailable"} rather than crashing the
 * orchestrator.
 */
public final class IndiaUniverseTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaUniverseTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Default cache TTL — NSE rebalances NIFTY 500 semi-annually; 24h is conservative. */
    private static final long CACHE_TTL_MS = 24L * 60L * 60L * 1000L;

    /** Per-instance cookie warmup for NSE. */
    private final NseIndiaClient nse;
    private final Path cacheDir;

    public IndiaUniverseTool() {
        this(new NseIndiaClient());
    }

    public IndiaUniverseTool(NseIndiaClient nse) {
        this.nse = nse;
        // ~/.nizo/cache/  — same convention as HistoricalPriceTool's disk cache
        String home = System.getProperty("user.home", ".");
        this.cacheDir = Paths.get(home, ".nizo", "cache");
        try { Files.createDirectories(cacheDir); } catch (Exception e) { /* best-effort */ }
    }

    @Override public String name() { return "india_universe"; }

    @Override
    public String description() {
        return "Constituents of an NSE-published Indian index (NIFTY 500 by default, also "
                + "NIFTY 50 / MIDCAP 150 / SMALLCAP 250 / sector indices). Used to build the "
                + "candidate pool for india_top_picks. Returns symbol, Yahoo-style ticker, "
                + "company name, industry, last price, free-float market cap, 52w range, "
                + "30d/365d return. Cached 24h on disk; authoritative source nseindia.com.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "index":   { "type": "string", "description": "NSE index name (default \\"NIFTY 500\\"). E.g. NIFTY 50, NIFTY MIDCAP 150, NIFTY BANK." },
                "refresh": { "type": "boolean", "description": "If true, bypass the 24h cache and re-fetch from NSE." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) throws Exception {
        JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
        String indexName = args.path("index").asText("NIFTY 500").trim();
        if (indexName.isBlank()) indexName = "NIFTY 500";
        boolean force = args.path("refresh").asBoolean(false);

        // Try cache first unless caller asked for a fresh hit.
        if (!force) {
            ObjectNode cached = readCache(indexName);
            if (cached != null) {
                return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(cached));
            }
        }

        if (!nse.isEnabled()) {
            return ToolResult.error("universe-unavailable: NseIndiaClient disabled "
                    + "(NIZO_NSEINDIA_DISABLED=1 or warmup failed)");
        }

        JsonNode raw = nse.indexConstituents(indexName);
        if (raw == null) {
            // Last resort — try to serve a stale cache even past TTL rather than fail outright.
            ObjectNode stale = readCache(indexName, /*ignoreTtl=*/true);
            if (stale != null) {
                stale.put("source", "nseindia.com (stale — current fetch failed)");
                return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(stale));
            }
            return ToolResult.error("universe-unavailable: NSE returned no data for index '"
                    + indexName + "' (rate-limited or geo-blocked?)");
        }

        ObjectNode out = MAPPER.createObjectNode();
        out.put("index", indexName);
        out.put("asOf", Instant.now().toString());
        out.put("source", "nseindia.com (live)");

        ArrayNode constituents = MAPPER.createArrayNode();
        JsonNode data = raw.path("data");
        int kept = 0;
        if (data.isArray()) {
            for (JsonNode row : data) {
                // NSE's response includes the index row itself first (symbol == index name);
                // skip it — we want only the constituent stocks.
                String symbol = row.path("symbol").asText("").trim();
                if (symbol.isBlank()) continue;
                if (symbol.equalsIgnoreCase(indexName)) continue;
                // Some NSE responses include "NIFTY 500" or similar as a meta row at top
                // even when the symbol field is filled — skip rows whose `priority` is 1
                // (NSE marks the aggregate row that way).
                if (row.path("priority").asInt(0) == 1) continue;

                ObjectNode c = MAPPER.createObjectNode();
                c.put("symbol", symbol);
                c.put("ticker", symbol + ".NS");
                JsonNode meta = row.path("meta");
                c.put("companyName", meta.path("companyName").asText(symbol));
                c.put("industry",    meta.path("industry").asText("Unknown"));
                c.put("isin",        meta.path("isin").asText(""));

                putNum(c, "lastPrice",    row.path("lastPrice"));
                putNum(c, "previousClose", row.path("previousClose"));
                putNum(c, "change",       row.path("change"));
                putNum(c, "pChange",      row.path("pChange"));
                putNum(c, "freeFloatMc",  row.path("ffmc"));    // ₹ Cr-scaled by NSE
                putNum(c, "yearHigh",     row.path("yearHigh"));
                putNum(c, "yearLow",      row.path("yearLow"));
                putNum(c, "perChange365", row.path("perChange365d"));
                putNum(c, "perChange30",  row.path("perChange30d"));
                putNum(c, "totalTradedVolume", row.path("totalTradedVolume"));
                putNum(c, "totalTradedValue",  row.path("totalTradedValue"));

                constituents.add(c);
                kept++;
            }
        }
        out.set("constituents", constituents);
        out.put("count", kept);

        // Persist cache (best-effort).
        writeCache(indexName, out);

        return ToolResult.ok(MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(out));
    }

    /* ─────────────────────────────────────────────────────────────── helpers */

    private static void putNum(ObjectNode dst, String key, JsonNode src) {
        if (src == null || src.isNull() || src.isMissingNode()) return;
        if (src.isNumber()) {
            dst.set(key, src);
        } else if (src.isTextual()) {
            String t = src.asText().trim();
            if (t.isEmpty() || "-".equals(t) || "NaN".equalsIgnoreCase(t)) return;
            try {
                dst.put(key, Double.parseDouble(t.replace(",", "")));
            } catch (NumberFormatException ignored) { /* leave key absent */ }
        }
    }

    private Path cachePath(String indexName) {
        String safe = indexName.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_");
        return cacheDir.resolve("india-universe-" + safe + ".json");
    }

    private ObjectNode readCache(String indexName) { return readCache(indexName, false); }

    private ObjectNode readCache(String indexName, boolean ignoreTtl) {
        Path p = cachePath(indexName);
        if (!Files.isRegularFile(p)) return null;
        try {
            long age = System.currentTimeMillis() - Files.getLastModifiedTime(p).toMillis();
            if (!ignoreTtl && age > CACHE_TTL_MS) return null;
            JsonNode node = MAPPER.readTree(Files.readAllBytes(p));
            if (!(node instanceof ObjectNode obj)) return null;
            long ageMin = age / 60_000L;
            obj.put("source", "nseindia.com (cached " + ageMin + "m ago)");
            return obj;
        } catch (Exception e) {
            LOG.debug("universe cache read failed for {}: {}", indexName, e.toString());
            return null;
        }
    }

    private void writeCache(String indexName, ObjectNode payload) {
        try {
            Files.write(cachePath(indexName),
                    MAPPER.writeValueAsBytes(payload),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (Exception e) {
            LOG.debug("universe cache write failed for {}: {}", indexName, e.toString());
        }
    }
}
