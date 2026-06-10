package ai.nizo.memory.llm;

import ai.nizo.memory.api.Modality;
import ai.nizo.memory.api.model.Message;
import ai.nizo.memory.api.model.ModelCapability;
import ai.nizo.memory.api.model.ModelClient;
import ai.nizo.memory.api.model.ModelRequest;
import ai.nizo.memory.api.model.ModelResponse;
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
import java.util.Set;

/**
 * {@link ModelClient} backed by an Ollama HTTP endpoint.
 *
 * <p>POSTs to {@code {baseUrl}/api/chat} with a JSON body of:
 * <pre>{@code
 *   {
 *     "model":   "qwen2.5:7b",
 *     "stream":  false,
 *     "messages": [{"role": "user", "content": "..."}],
 *     "options": {"temperature": 0.1}
 *   }
 * }</pre>
 * and parses the {@code message.content} field of the response.
 *
 * <p>Uses the JDK {@link HttpClient} — no external HTTP dependency. Tools are
 * <em>not</em> wired (the consolidator/extractor/verifier never call them).
 */
public final class OllamaModelClient implements ModelClient {

    private static final Logger log = LoggerFactory.getLogger(OllamaModelClient.class);

    private final String baseUrl;
    private final String model;
    private final double temperature;
    private final Duration timeout;
    private final HttpClient http;

    public OllamaModelClient(String baseUrl, String model, double temperature, Duration timeout) {
        this.baseUrl = Http.normaliseEndpoint(baseUrl == null ? "http://localhost:11434" : baseUrl);
        this.model = model == null || model.isBlank() ? "qwen2.5:7b" : model;
        this.temperature = temperature;
        this.timeout = timeout == null ? Duration.ofSeconds(60) : timeout;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    @Override
    public ModelCapability capability() {
        return new ModelCapability(
                model,
                "ollama",
                Set.of(Modality.TEXT),
                Set.of(Modality.TEXT),
                /* maxContextTokens */ 32_768,
                /* supportsTools     */ false,
                /* isLocal           */ true,
                /* usdPerMInput      */ 0.0,
                /* usdPerMOutput     */ 0.0,
                /* latencyHintMs     */ 200);
    }

    @Override
    public ModelResponse complete(ModelRequest request) {
        // Build wire-format messages.
        List<Map<String, Object>> wireMessages = new ArrayList<>();
        for (Message m : request.messages()) {
            Map<String, Object> wm = new LinkedHashMap<>();
            wm.put("role", m.role().wire());
            wm.put("content", m.text());
            wireMessages.add(wm);
        }

        Map<String, Object> options = new LinkedHashMap<>();
        options.put("temperature", temperature);
        // A handful of caller options are top-level in Ollama's /api/chat payload
        // (e.g. "think" — toggles Qwen3 / DeepSeek-R1 chain-of-thought). Hoist
        // them out so they don't get nested under "options" where Ollama ignores
        // them. Everything else flows into "options" as a model parameter.
        Map<String, Object> topLevelExtras = new LinkedHashMap<>();
        Set<String> topLevelKeys = Set.of("think", "format", "keep_alive");
        if (request.options() != null) {
            for (Map.Entry<String, Object> e : request.options().entrySet()) {
                if (topLevelKeys.contains(e.getKey())) {
                    topLevelExtras.put(e.getKey(), e.getValue());
                } else {
                    options.put(e.getKey(), e.getValue());
                }
            }
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model);
        body.put("stream", false);
        body.put("messages", wireMessages);
        body.put("options", options);
        body.putAll(topLevelExtras);

        String payload = Json.stringify(body);
        URI uri = URI.create(baseUrl + "/api/chat");

        HttpRequest httpReq = HttpRequest.newBuilder(uri)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .build();

        try {
            HttpResponse<String> resp = http.send(httpReq, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() / 100 != 2) {
                throw new RuntimeException("Ollama /api/chat returned HTTP " + resp.statusCode()
                        + ": " + truncate(resp.body()));
            }
            JsonNode root = Json.tree(resp.body());
            String content = "";
            JsonNode messageNode = root.path("message");
            if (messageNode.isObject()) {
                content = messageNode.path("content").asText("");
            }

            int promptTokens = root.path("prompt_eval_count").asInt(0);
            int completionTokens = root.path("eval_count").asInt(0);
            String finishReason = root.path("done_reason").asText("stop");

            return new ModelResponse(
                    content,
                    List.of(),
                    new ModelResponse.Usage(promptTokens, completionTokens),
                    finishReason);
        } catch (java.io.IOException ioe) {
            throw new RuntimeException(
                    "Ollama /api/chat call failed: " + ioe.getMessage(), ioe);
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Ollama /api/chat call interrupted", ie);
        }
    }

    private static String truncate(String s) {
        if (s == null) return "";
        return s.length() <= 200 ? s : s.substring(0, 200) + "…";
    }
}
