package ai.nizo.skills;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Top-N Indian stocks orchestrator — Phase 1 of the "India Picks" build.
 *
 * <p>Deterministic Java pipeline (no LLM in the hot path) that:
 * <ol>
 *   <li>Calls {@code india_universe} → fetches NIFTY 500 (or chosen index) constituents.</li>
 *   <li>Filters to the top {@code candidateN} by free-float market cap (default 60).
 *       Skipping the long tail of small-caps cuts the deep-pass time from ~5 min
 *       (500 stocks) to ~40s (60 stocks) without materially affecting quality —
 *       NIFTY 50 + NIFTY Next 50 captures ~80% of free-float market cap.</li>
 *   <li>Calls {@code india_screener} on those candidates with quality/value/growth/
 *       momentum/Buffett factor scoring.</li>
 *   <li>Returns markdown + a {@code ```chart-india-picks JSON} fence the front-end
 *       can later render as a table widget (Phase 6).</li>
 * </ol>
 *
 * <p>Phases 2-4 (macro / sector / political overlays) will plug in here as
 * additional steps before the final ranking. For now the output is pure
 * bottom-up factor scoring — already useful, fast, deterministic.
 *
 * <p>Disclaimer: output is research / education, not SEBI-registered investment
 * advice. Always do your own due diligence before any equity purchase.
 */
public final class IndiaTopPicksTool implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(IndiaTopPicksTool.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Supplier<ToolRegistry> registry;

    public IndiaTopPicksTool(Supplier<ToolRegistry> registry) {
        this.registry = registry;
    }

    @Override public String name() { return "india_top_picks"; }

    @Override
    public String description() {
        return "Top-N Indian stocks ranker. Pulls NIFTY 500 constituents, narrows to top "
                + "~60 by market cap, runs the multi-factor india_screener (quality / value / "
                + "growth / momentum / Buffett-Munger), and returns a ranked list with rationales. "
                + "Output: markdown table + chart-india-picks JSON fence (front-end renderable).";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "universe":     { "type": "string", "description": "NSE index to use as universe. Default 'NIFTY 500'." },
                "topN":         { "type": "integer", "description": "How many final picks to return (default 10)." },
                "candidateN":   { "type": "integer", "description": "How many top-by-market-cap candidates to deep-score. Default 60." },
                "weights":      { "type": "object", "description": "Factor weights — keys: quality, value, growth, momentum, buffett. Defaults: 0.35/0.20/0.20/0.15/0.10." },
                "concurrency":  { "type": "integer", "description": "Parallel scorers passed to india_screener (default 6)." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        long t0 = System.currentTimeMillis();
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            String universe   = args.path("universe").asText("NIFTY 500").trim();
            int topN          = Math.max(1, Math.min(50,  args.path("topN").asInt(10)));
            int candidateN    = Math.max(topN, Math.min(200, args.path("candidateN").asInt(60)));
            int concurrency   = Math.max(1, Math.min(10, args.path("concurrency").asInt(6)));
            JsonNode weights  = args.path("weights");

            ToolRegistry reg = registry.get();
            if (reg == null) return ToolResult.error("tool registry not available");

            Tool universeT = reg.byName("india_universe").orElse(null);
            Tool screenerT = reg.byName("india_screener").orElse(null);
            Tool macroT    = reg.byName("india_macro_dashboard").orElse(null);
            Tool sectorT   = reg.byName("india_sector_view").orElse(null);
            Tool eventT    = reg.byName("india_event_calendar").orElse(null);
            if (universeT == null) return ToolResult.error("india_universe tool not registered");
            if (screenerT == null) return ToolResult.error("india_screener tool not registered");

            // 0. Top-down overlays (Phases 2, 3, 4) — run in parallel ──────────
            // All three are cheap (4h-cached or no-network). Run concurrently so they don't add
            // serial latency to the picks pipeline. Each is fully optional — picks ranking still
            // produces a valid result if any overlay tool fails.
            java.util.concurrent.ExecutorService overlayPool = java.util.concurrent.Executors.newFixedThreadPool(3, r -> {
                Thread t = new Thread(r, "india-overlay");
                t.setDaemon(true);
                return t;
            });
            java.util.concurrent.CompletableFuture<JsonNode> macroF  = runOverlay(overlayPool, macroT,  "{}");
            java.util.concurrent.CompletableFuture<JsonNode> sectorF = runOverlay(overlayPool, sectorT, "{}");
            java.util.concurrent.CompletableFuture<JsonNode> eventF  = runOverlay(overlayPool, eventT,  "{}");
            JsonNode macroJson  = awaitOverlay(macroF,  "macro");
            JsonNode sectorJson = awaitOverlay(sectorF, "sector");
            JsonNode eventJson  = awaitOverlay(eventF,  "event");
            overlayPool.shutdown();
            try { overlayPool.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS); }
            catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

            // 1. Universe ─────────────────────────────────────────────────────
            ObjectNode uArgs = MAPPER.createObjectNode();
            uArgs.put("index", universe);
            ToolResult uRes = universeT.execute(MAPPER.writeValueAsString(uArgs));
            if (uRes == null || !uRes.ok() || uRes.content() == null) {
                return ToolResult.error("india_universe failed: "
                        + (uRes == null ? "null" : uRes.content()));
            }
            JsonNode uJson = MAPPER.readTree(uRes.content());
            JsonNode constituents = uJson.path("constituents");
            if (!constituents.isArray() || constituents.isEmpty()) {
                return ToolResult.error("india_universe returned empty constituents for '"
                        + universe + "' — NSE may be geo-blocked or rate-limited");
            }
            LOG.info("india_top_picks: universe '{}' fetched {} constituents", universe, constituents.size());

            // 2. Sort by free-float market cap descending, take candidateN ────
            List<Candidate> candidates = new ArrayList<>();
            for (JsonNode c : constituents) {
                String ticker = c.path("ticker").asText("");
                if (ticker.isBlank()) continue;
                double mc = c.path("freeFloatMc").asDouble(0);
                String industry = c.path("industry").asText("Unknown");
                String name = c.path("companyName").asText(ticker);
                candidates.add(new Candidate(ticker, name, industry, mc));
            }
            candidates.sort(Comparator.comparingDouble((Candidate x) -> x.marketCap).reversed());
            List<Candidate> top = candidates.size() <= candidateN
                    ? candidates : candidates.subList(0, candidateN);
            LOG.info("india_top_picks: narrowed to {} candidates by market cap", top.size());

            // 3. Run screener on candidates ──────────────────────────────────
            ObjectNode sArgs = MAPPER.createObjectNode();
            ArrayNode tickersA = sArgs.putArray("tickers");
            for (Candidate c : top) tickersA.add(c.ticker);
            sArgs.put("topN", topN);
            sArgs.put("concurrency", concurrency);
            if (weights.isObject()) sArgs.set("weights", weights);

            // Pre-compute momentum signals from the NSE universe response — saves the
            // per-ticker Yahoo historical_price calls (which are heavily rate-limited).
            // Shape: { ticker: { perChange365: x, perChange30: y } }.
            ObjectNode priorMomentum = MAPPER.createObjectNode();
            for (JsonNode c : constituents) {
                String tk = c.path("ticker").asText("");
                if (tk.isBlank()) continue;
                boolean hasY = c.path("perChange365").isNumber();
                boolean hasM = c.path("perChange30").isNumber();
                if (!hasY && !hasM) continue;
                ObjectNode entry = priorMomentum.putObject(tk);
                if (hasY) entry.put("perChange365", c.path("perChange365").asDouble());
                if (hasM) entry.put("perChange30",  c.path("perChange30").asDouble());
            }
            if (priorMomentum.size() > 0) sArgs.set("priorMomentum", priorMomentum);

            ToolResult sRes = screenerT.execute(MAPPER.writeValueAsString(sArgs));
            if (sRes == null || !sRes.ok() || sRes.content() == null) {
                return ToolResult.error("india_screener failed: "
                        + (sRes == null ? "null" : sRes.content()));
            }
            JsonNode sJson = MAPPER.readTree(sRes.content());
            JsonNode ranked = sJson.path("ranked");
            if (!ranked.isArray() || ranked.isEmpty()) {
                return ToolResult.error("india_screener returned no ranked stocks");
            }

            // 4. Enrich each pick with sector/company name from candidates ───
            //    Then apply sector tilt (from india_sector_view) + event tilt (from
            //    india_event_calendar) as POST-SCORE adjustments. Final score:
            //       finalScore = bottomUpScore + sectorTilt + eventTilt
            //    where tilts are typically in the range -10..+10.
            Map<String, Candidate> byTicker = new java.util.HashMap<>();
            for (Candidate c : top) byTicker.put(c.ticker, c);
            JsonNode sectorTilts = (sectorJson != null) ? sectorJson.path("sectorTiltByKeyword") : MAPPER.createObjectNode();
            JsonNode eventTilts  = (eventJson  != null) ? eventJson.path("eventTiltByKeyword")  : MAPPER.createObjectNode();
            ArrayNode enriched = MAPPER.createArrayNode();
            for (JsonNode pick : ranked) {
                ObjectNode p = pick.deepCopy();
                String tk = p.path("ticker").asText("");
                Candidate c = byTicker.get(tk);
                String sector = "Unknown";
                if (c != null) {
                    p.put("companyName", c.name);
                    p.put("sector", c.industry);
                    sector = c.industry;
                }
                // Apply tilts based on sector industry-keyword match
                double bottomUp = p.path("score").asDouble(50);
                double sTilt = lookupTilt(sectorTilts, sector);
                double eTilt = lookupTilt(eventTilts,  sector);
                double finalScore = bottomUp + sTilt + eTilt;
                // Bound to 0..100
                if (finalScore < 0)   finalScore = 0;
                if (finalScore > 100) finalScore = 100;
                p.put("bottomUpScore", round1(bottomUp));
                p.put("sectorTilt", round1(sTilt));
                p.put("eventTilt",  round1(eTilt));
                p.put("score", round1(finalScore));   // overwrite with the tilt-adjusted score
                enriched.add(p);
            }
            // Re-sort by the adjusted score
            List<JsonNode> enrichedList = new ArrayList<>();
            enriched.forEach(enrichedList::add);
            enrichedList.sort((a, b2) -> Double.compare(b2.path("score").asDouble(0), a.path("score").asDouble(0)));
            enriched = MAPPER.createArrayNode();
            enrichedList.forEach(enriched::add);

            // 5. Apply sector diversification — max 3 per sector unless topN<=5 ─
            ArrayNode diversified = MAPPER.createArrayNode();
            java.util.Map<String, Integer> sectorCounts = new java.util.HashMap<>();
            int sectorCap = (topN <= 5) ? topN : 3;
            for (JsonNode pick : enriched) {
                String sec = pick.path("sector").asText("Unknown");
                int n = sectorCounts.getOrDefault(sec, 0);
                if (n >= sectorCap) continue;
                diversified.add(pick);
                sectorCounts.put(sec, n + 1);
                if (diversified.size() >= topN) break;
            }
            // If diversification dropped picks below topN, fill from un-used enriched picks.
            if (diversified.size() < topN) {
                java.util.Set<String> chosen = new java.util.HashSet<>();
                diversified.forEach(p -> chosen.add(p.path("ticker").asText()));
                for (JsonNode pick : enriched) {
                    if (chosen.contains(pick.path("ticker").asText())) continue;
                    diversified.add(pick);
                    if (diversified.size() >= topN) break;
                }
            }
            // Re-rank 1..N after diversification reshuffles.
            for (int i = 0; i < diversified.size(); i++) {
                ((ObjectNode) diversified.get(i)).put("rank", i + 1);
            }

            // 6. Assemble output: markdown + chart fence ─────────────────────
            long elapsed = System.currentTimeMillis() - t0;
            ObjectNode payload = MAPPER.createObjectNode();
            payload.put("universe", universe);
            payload.put("asOf", java.time.Instant.now().toString());
            payload.put("topN", topN);
            payload.put("candidateN", candidateN);
            payload.put("elapsedMs", elapsed);
            payload.set("picks", diversified);
            payload.set("weights", sJson.path("summary").path("weights"));
            payload.set("summary", sJson.path("summary"));
            if (macroJson  != null) payload.set("macro",  macroJson);
            if (sectorJson != null) payload.set("sectors", sectorJson);
            if (eventJson  != null) payload.set("events",  eventJson);
            payload.put("disclaimer",
                    "Research output — not SEBI-registered advice. Past performance does not guarantee "
                  + "future returns. Always do independent due diligence.");

            String md = renderMarkdown(payload, diversified, universe, elapsed, macroJson);
            String fence = "```chart-india-picks\n"
                    + MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(payload)
                    + "\n```";
            String out = md + "\n\n" + fence + "\n";

            return ToolResult.ok(out);
        } catch (Exception e) {
            LOG.warn("india_top_picks failed: {}", e.toString());
            return ToolResult.error("india_top_picks failed: " + e.getMessage());
        }
    }

    /** Markdown table view. Front-end can later replace this with a chart widget. */
    private static String renderMarkdown(JsonNode payload, JsonNode picks, String universe, long elapsedMs, JsonNode macro) {
        StringBuilder b = new StringBuilder();
        b.append("# India Top Picks — ").append(picks.size()).append(" stocks\n\n");
        b.append("**Universe:** ").append(universe)
         .append(" · **As of:** ").append(payload.path("asOf").asText())
         .append(" · **Compute:** ").append(elapsedMs).append(" ms\n\n");
        b.append("> ").append(payload.path("disclaimer").asText()).append("\n\n");

        // Macro context block (Phase 2)
        if (macro != null && !macro.isMissingNode()) {
            b.append("## Macro context\n\n");
            JsonNode r = macro.path("rates");
            JsonNode fx = macro.path("fx");
            JsonNode cm = macro.path("commodities");
            JsonNode mc = macro.path("macro");
            JsonNode rg = macro.path("regime");
            JsonNode in = macro.path("interpretation");
            b.append("| Signal | Value | Regime |\n|---|---|---|\n");
            if (r.path("rbiRepo").isNumber())
                b.append("| RBI repo rate | ").append(fmt1(r.path("rbiRepo").asDouble())).append("% | ").append(rg.path("rate").asText("?")).append(" |\n");
            if (fx.path("usdInr").isNumber())
                b.append("| USD/INR | ").append(fmt2(fx.path("usdInr").asDouble())).append(" (").append(fmtPctChg(fx.path("usdInrChange30d").asDouble(0))).append(" 30d) | ").append(rg.path("inr").asText("?")).append(" |\n");
            if (cm.path("wtiUsd").isNumber())
                b.append("| Crude (WTI) | $").append(fmt1(cm.path("wtiUsd").asDouble())).append(" (").append(fmtPctChg(cm.path("wtiChange30d").asDouble(0))).append(" 30d) | ").append(rg.path("commodity").asText("?")).append(" |\n");
            if (mc.path("gdpGrowthYoY").isNumber())
                b.append("| India GDP YoY | ").append(fmt1(mc.path("gdpGrowthYoY").asDouble())).append("% | ").append(rg.path("growth").asText("?")).append(" |\n");
            if (mc.path("cpiInflationYoY").isNumber())
                b.append("| India CPI YoY | ").append(fmt1(mc.path("cpiInflationYoY").asDouble())).append("% | ").append(rg.path("inflation").asText("?")).append(" |\n");
            b.append("\n**Composite tilt:** ").append(rg.path("equityTilt").asText("neutral")).append("\n\n");
            // Interpretive bullets
            if (in.isObject()) {
                if (!in.path("inrSignal").asText("").isEmpty())       b.append("- ").append(in.path("inrSignal").asText()).append("\n");
                if (!in.path("oilSignal").asText("").isEmpty())       b.append("- ").append(in.path("oilSignal").asText()).append("\n");
                if (!in.path("growthSignal").asText("").isEmpty())    b.append("- ").append(in.path("growthSignal").asText()).append("\n");
                if (!in.path("inflationSignal").asText("").isEmpty()) b.append("- ").append(in.path("inflationSignal").asText()).append("\n");
                b.append("\n");
            }
        }

        b.append("## Ranked picks\n\n");

        b.append("| # | Ticker | Company | Sector | Score | Quality | Value | Growth | Momentum | Buffett | Rationale |\n");
        b.append("|---|--------|---------|--------|-------|---------|-------|--------|----------|---------|-----------|\n");
        for (JsonNode p : picks) {
            JsonNode c = p.path("components");
            b.append("| ").append(p.path("rank").asInt())
             .append(" | ").append(escapeMd(p.path("ticker").asText()))
             .append(" | ").append(escapeMd(p.path("companyName").asText("")))
             .append(" | ").append(escapeMd(p.path("sector").asText("")))
             .append(" | ").append(fmt1(p.path("score").asDouble()))
             .append(" | ").append(fmt1(c.path("quality").asDouble()))
             .append(" | ").append(fmt1(c.path("value").asDouble()))
             .append(" | ").append(fmt1(c.path("growth").asDouble()))
             .append(" | ").append(fmt1(c.path("momentum").asDouble()))
             .append(" | ").append(c.path("buffett").isNull() ? "—" : fmt1(c.path("buffett").asDouble()))
             .append(" | ").append(escapeMd(p.path("rationale").asText("")))
             .append(" |\n");
        }
        b.append("\n");
        b.append("**Factor weights:** ");
        JsonNode w = payload.path("weights");
        if (w.isObject()) {
            boolean first = true;
            for (java.util.Iterator<Map.Entry<String, JsonNode>> it = w.fields(); it.hasNext(); ) {
                if (!first) b.append(" · "); first = false;
                Map.Entry<String, JsonNode> e = it.next();
                b.append(e.getKey()).append(" ").append(fmtPct(e.getValue().asDouble()));
            }
        }
        b.append("\n\n");
        return b.toString();
    }

    private static String fmt1(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.US, "%.1f", v);
    }
    private static String fmt2(double v) {
        return Double.isNaN(v) ? "—" : String.format(Locale.US, "%.2f", v);
    }
    private static String fmtPctChg(double v) {
        if (Double.isNaN(v)) return "—";
        return (v >= 0 ? "+" : "") + String.format(Locale.US, "%.1f%%", v);
    }

    private static String fmtPct(double v) {
        return String.format(Locale.US, "%.0f%%", v * 100);
    }

    private static String escapeMd(String s) {
        if (s == null) return "";
        return s.replace("|", "\\|").replace("\n", " ");
    }

    private record Candidate(String ticker, String name, String industry, double marketCap) {}

    /* ─────────────────────────────────────────────────────── overlay helpers */

    private static java.util.concurrent.CompletableFuture<JsonNode> runOverlay(
            java.util.concurrent.ExecutorService pool, Tool tool, String argsJson) {
        if (tool == null) return java.util.concurrent.CompletableFuture.completedFuture(null);
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> {
            try {
                ToolResult r = tool.execute(argsJson);
                if (r == null || !r.ok() || r.content() == null) return null;
                return MAPPER.readTree(r.content());
            } catch (Exception e) {
                LOG.warn("overlay tool failed: {}", e.toString());
                return null;
            }
        }, pool);
    }

    private static JsonNode awaitOverlay(java.util.concurrent.CompletableFuture<JsonNode> f, String label) {
        try { return f.get(20, java.util.concurrent.TimeUnit.SECONDS); }
        catch (Exception e) {
            LOG.warn("overlay {} timed out / failed: {}", label, e.toString());
            return null;
        }
    }

    /**
     * Look up tilt for a stock's sector by matching its industry string against any of the
     * keyword keys in the tilt map. Case-insensitive substring match. Returns 0 if no match.
     */
    private static double lookupTilt(JsonNode tiltMap, String industry) {
        if (tiltMap == null || !tiltMap.isObject() || industry == null) return 0;
        String lower = industry.toLowerCase(Locale.ROOT);
        double best = 0;
        java.util.Iterator<Map.Entry<String, JsonNode>> it = tiltMap.fields();
        while (it.hasNext()) {
            Map.Entry<String, JsonNode> e = it.next();
            if (lower.contains(e.getKey().toLowerCase(Locale.ROOT))) {
                double v = e.getValue().isNumber() ? e.getValue().asDouble() : 0;
                // Prefer the largest-magnitude tilt when multiple keywords match.
                if (Math.abs(v) > Math.abs(best)) best = v;
            }
        }
        return best;
    }

    private static double round1(double v) { return Math.round(v * 10.0) / 10.0; }
}
