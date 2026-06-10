package ai.nizo.memory.llm;

import ai.nizo.memory.api.model.EmbeddingClient;
import ai.nizo.memory.util.Http;
import ai.nizo.memory.util.Json;
import com.fasterxml.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link EmbeddingClient} backed by an Ollama HTTP endpoint.
 *
 * <p>POSTs to {@code {baseUrl}/api/embeddings} with body
 * {@code {"model": "...", "prompt": "..."}} and reads back
 * {@code {"embedding": [0.1, 0.2, ...]}}.
 *
 * <p>Ollama has no batch endpoint, so {@link #embedBatch(List)} just calls
 * {@link #embed(String)} sequentially. Dimensions are discovered from the
 * first successful embed call (or hard-wired for a few well-known models).
 */
public final class OllamaEmbeddingClient implements EmbeddingClient {

    private static final Logger log = LoggerFactory.getLogger(OllamaEmbeddingClient.class);

    /** Hard-coded dims for common models so {@link #dimensions()} doesn't have
     *  to make a network call before the first embed. Falls through to
     *  on-demand discovery for anything else. */
    private static final Map<String, Integer> KNOWN_DIMS = Map.of(
            "nomic-embed-text",        768,
            "nomic-embed-text:latest", 768,
            "nomic-embed-text-v1.5",   768,
            "mxbai-embed-large",       1024,
            "all-minilm",              384,
            "snowflake-arctic-embed",  1024
    );

    private final String baseUrl;
    private final String model;
    private final Duration timeout;
    private final HttpClient http;
    private volatile int dims = -1;

    public OllamaEmbeddingClient(String baseUrl, String model, Duration timeout) {
        this.baseUrl = Http.normaliseEndpoint(baseUrl == null ? "http://localhost:11434" : baseUrl);
        this.model = model == null || model.isBlank() ? "nomic-embed-text" : model;
        this.timeout = timeout == null ? Duration.ofSeconds(30) : timeout;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        Integer known = KNOWN_DIMS.get(this.model.toLowerCase(java.util.Locale.ROOT));
        if (known != null) {
            this.dims = known;
        }
    }

    @Override
    public int dimensions() {
        if (dims < 0) {
            // Probe once to discover.
            embed("dimension probe");
        }
        return dims;
    }

    @Override
    public float[] embed(String text) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("prompt", text == null ? "" : text);

        URI uri = URI.create(baseUrl + "/api/embeddings");
        HttpRequest req = HttpRequest.newBuilder(uri)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.stringify(body)))
                .build();

        try {
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() / 100 != 2) {
                throw new RuntimeException("Ollama /api/embeddings returned HTTP "
                        + resp.statusCode() + ": " + truncate(resp.body()));
            }
            JsonNode root = Json.tree(resp.body());
            JsonNode arr = root.path("embedding");
            if (!arr.isArray() || arr.size() == 0) {
                throw new RuntimeException("Ollama returned no embedding for model " + model);
            }
            float[] out = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++) {
                out[i] = (float) arr.get(i).asDouble();
            }
            if (dims < 0) dims = out.length;
            return out;
        } catch (java.io.IOException ioe) {
            throw new RuntimeException(
                    "Ollama /api/embeddings call failed: " + ioe.getMessage(), ioe);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Ollama /api/embeddings call interrupted", ie);
        }
    }

    @Override
    public List<float[]> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) return List.of();
        List<float[]> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(embed(t));
        return out;
    }

    private static String truncate(String s) {
        if (s == null) return "";
        return s.length() <= 200 ? s : s.substring(0, 200) + "…";
    }
}
