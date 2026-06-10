package ai.nizo.llm;

import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.ChatStreamHandler;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.ToolCall;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * OpenAI-compatible {@code /v1/chat/completions} client.
 *
 * <p>Talks to vLLM, llama.cpp's {@code llama-server}, Ollama, LM Studio, etc. Supports:
 * <ul>
 *   <li>Blocking and SSE-streaming completions</li>
 *   <li>Tool calls (via standard {@code tools}/{@code tool_calls})</li>
 *   <li>Multimodal input — packs {@link ChatMessage#images()} as {@code image_url} parts</li>
 *   <li>{@code reasoning_content} extraction (Qwen3.6 with {@code --reasoning-format deepseek})</li>
 * </ul>
 */
public final class OpenAiCompatibleClient implements LlmClient {

    private static final Logger LOG = LoggerFactory.getLogger(OpenAiCompatibleClient.class);

    private final URI endpoint;
    private final String authToken;
    private final HttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();
    private final Duration timeout;

    public OpenAiCompatibleClient(String baseUrl, String authToken) {
        // Long default. Local Qwen3.6-27B with reasoning + tools + a fat context can stream for
        // 5-15 minutes per turn. The JDK HttpClient timeout applies to the WHOLE response — for
        // an SSE stream that means the entire stream must finish within this window. 30 min is
        // generous enough to cover legitimate slow turns; if the LLM is genuinely deadlocked,
        // surfacing an error after 30 min beats hanging forever.
        // Override with NIZO_LLM_TIMEOUT_MIN.
        this(baseUrl, authToken, Duration.ofMinutes(envIntMinutes("NIZO_LLM_TIMEOUT_MIN", 30)));
    }

    public OpenAiCompatibleClient(String baseUrl, String authToken, Duration timeout) {
        if (baseUrl == null || baseUrl.isBlank()) {
            throw new IllegalArgumentException("baseUrl required");
        }
        String trimmed = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.endpoint = URI.create(trimmed + "/v1/chat/completions");
        this.authToken = authToken;
        this.timeout = timeout;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    private static int envIntMinutes(String name, int dflt) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) return dflt;
        try { return Math.max(1, Integer.parseInt(v.trim())); } catch (NumberFormatException e) { return dflt; }
    }

    // ===================== blocking =====================

    @Override
    public ChatResponse chat(ChatRequest request) {
        try {
            String body = encodeRequest(request, /*stream=*/ false);
            HttpResponse<String> resp = http.send(buildRequest(body, false), HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() / 100 != 2) {
                throw new RuntimeException("LLM HTTP " + resp.statusCode() + ": " + resp.body());
            }
            return parseResponse(resp.body());
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("LLM request failed: " + e.getMessage(), e);
        }
    }

    // ===================== streaming =====================

    @Override
    public void streamChat(ChatRequest request, ChatStreamHandler handler) {
        try {
            String body = encodeRequest(request, /*stream=*/ true);
            HttpResponse<InputStream> resp = http.send(
                    buildRequest(body, true), HttpResponse.BodyHandlers.ofInputStream());
            if (resp.statusCode() / 100 != 2) {
                String err;
                try (InputStream is = resp.body()) { err = new String(is.readAllBytes(), StandardCharsets.UTF_8); }
                handler.onError(new RuntimeException("LLM HTTP " + resp.statusCode() + ": " + err));
                return;
            }
            parseSseStream(resp.body(), handler);
        } catch (Exception e) {
            handler.onError(e);
        }
    }

    private void parseSseStream(InputStream body, ChatStreamHandler handler) throws Exception {
        StringBuilder content = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        // toolCallId -> {name, argsBuilder}
        Map<String, ToolBuf> toolCalls = new HashMap<>();
        String finishReason = null;
        ChatResponse.Usage usage = ChatResponse.Usage.EMPTY;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(body, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                // User-stop integration: ChatExecution.stopCurrent() interrupts the worker
                // thread. We were blocked in readLine() but it returns at chunk boundaries
                // (every 25-50ms during streaming) — so checking interrupt here gives
                // ~50ms-or-better stop latency without any extra plumbing. Closing body
                // explicitly forces any in-flight read to fail fast on the next iteration.
                if (Thread.currentThread().isInterrupted()) {
                    try { body.close(); } catch (Exception ignored) {}
                    throw new InterruptedException("LLM stream interrupted by user");
                }
                if (line.isEmpty() || !line.startsWith("data:")) continue;
                String data = line.substring(5).trim();
                if (data.equals("[DONE]")) break;
                if (data.isEmpty()) continue;
                JsonNode chunk;
                try {
                    chunk = mapper.readTree(data);
                } catch (Exception e) {
                    LOG.debug("skip non-json SSE line: {}", data);
                    continue;
                }
                JsonNode choice = chunk.path("choices").path(0);
                JsonNode delta = choice.path("delta");

                JsonNode contentNode = delta.path("content");
                if (contentNode.isTextual() && !contentNode.asText().isEmpty()) {
                    String t = contentNode.asText();
                    content.append(t);
                    handler.onToken(t);
                }

                JsonNode reasoningNode = delta.path("reasoning_content");
                if (reasoningNode.isTextual() && !reasoningNode.asText().isEmpty()) {
                    String t = reasoningNode.asText();
                    thinking.append(t);
                    handler.onThinking(t);
                }

                // tool_calls deltas — OpenAI streams them as partial chunks indexed by position.
                JsonNode tcs = delta.path("tool_calls");
                if (tcs.isArray()) {
                    for (JsonNode tc : tcs) {
                        String idx = tc.path("index").asText("0");
                        ToolBuf buf = toolCalls.computeIfAbsent(idx, k -> new ToolBuf());
                        if (tc.has("id")) buf.id = tc.get("id").asText();
                        JsonNode fn = tc.path("function");
                        if (fn.has("name")) buf.name = fn.get("name").asText();
                        if (fn.has("arguments")) {
                            String chunkArgs = fn.get("arguments").asText("");
                            buf.args.append(chunkArgs);
                            handler.onToolCallDelta(buf.id, buf.name, chunkArgs);
                        }
                    }
                }

                if (choice.has("finish_reason") && !choice.get("finish_reason").isNull()) {
                    finishReason = choice.get("finish_reason").asText();
                }

                JsonNode u = chunk.path("usage");
                if (u.isObject() && u.has("total_tokens")) {
                    usage = new ChatResponse.Usage(
                            u.path("prompt_tokens").asInt(0),
                            u.path("completion_tokens").asInt(0),
                            u.path("total_tokens").asInt(0));
                }
            }
        }

        List<ToolCall> finalCalls = new ArrayList<>();
        for (ToolBuf b : toolCalls.values()) {
            if (b.id != null && b.name != null) {
                finalCalls.add(new ToolCall(b.id, b.name, b.args.toString()));
            }
        }
        ChatResponse resp = new ChatResponse(content.toString(), finalCalls, finishReason, usage);
        handler.onComplete(resp);
    }

    private static final class ToolBuf {
        String id;
        String name;
        final StringBuilder args = new StringBuilder();
    }

    // ===================== encode =====================

    private HttpRequest buildRequest(String body, boolean stream) {
        HttpRequest.Builder bld = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .header("Accept", stream ? "text/event-stream" : "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body));
        if (authToken != null && !authToken.isBlank()) {
            bld.header("Authorization", "Bearer " + authToken);
        }
        return bld.build();
    }

    private String encodeRequest(ChatRequest request, boolean stream) throws Exception {
        ObjectNode root = mapper.createObjectNode();
        root.put("model", request.model());
        root.put("stream", stream);
        if (request.temperature() != null) root.put("temperature", request.temperature());
        if (request.maxTokens() != null) root.put("max_tokens", request.maxTokens());

        ArrayNode msgs = root.putArray("messages");
        for (ChatMessage m : request.messages()) {
            ObjectNode mn = msgs.addObject();
            mn.put("role", m.role().name().toLowerCase());

            if (m.role() == ai.nizo.api.llm.Role.USER && m.hasImages()) {
                ArrayNode parts = mn.putArray("content");
                if (m.content() != null && !m.content().isBlank()) {
                    ObjectNode tp = parts.addObject();
                    tp.put("type", "text");
                    tp.put("text", m.content());
                }
                for (String dataUri : m.images()) {
                    ObjectNode ip = parts.addObject();
                    ip.put("type", "image_url");
                    ObjectNode iu = ip.putObject("image_url");
                    iu.put("url", dataUri);
                }
            } else if (m.content() != null) {
                mn.put("content", m.content());
            }

            if (m.toolCallId() != null) mn.put("tool_call_id", m.toolCallId());
            if (m.name() != null) mn.put("name", m.name());

            if (!m.toolCalls().isEmpty()) {
                ArrayNode tcs = mn.putArray("tool_calls");
                for (ToolCall tc : m.toolCalls()) {
                    ObjectNode tcn = tcs.addObject();
                    tcn.put("id", tc.id());
                    tcn.put("type", "function");
                    ObjectNode fn = tcn.putObject("function");
                    fn.put("name", tc.name());
                    fn.put("arguments", tc.argumentsJson());
                }
            }
        }

        if (!request.tools().isEmpty()) {
            ArrayNode tools = root.putArray("tools");
            for (var td : request.tools()) {
                ObjectNode tn = tools.addObject();
                tn.put("type", "function");
                ObjectNode fn = tn.putObject("function");
                fn.put("name", td.name());
                if (td.description() != null) fn.put("description", td.description());
                if (td.parametersJsonSchema() != null) {
                    fn.set("parameters", mapper.readTree(td.parametersJsonSchema()));
                }
            }
        }

        // Merge any extra JSON fields the caller wants on the wire (e.g. Qwen-specific
        // chat_template_kwargs.enable_thinking=false to force content output, or other
        // provider knobs). Values can be primitives, Maps (→ ObjectNode), or Lists.
        if (request.extraBody() != null && !request.extraBody().isEmpty()) {
            for (var entry : request.extraBody().entrySet()) {
                root.set(entry.getKey(), mapper.valueToTree(entry.getValue()));
            }
        }

        return mapper.writeValueAsString(root);
    }

    private ChatResponse parseResponse(String json) throws Exception {
        JsonNode root = mapper.readTree(json);
        JsonNode choice = root.path("choices").path(0);
        JsonNode msg = choice.path("message");

        String content = (msg.path("content").isMissingNode() || msg.path("content").isNull())
                ? null : msg.path("content").asText();
        String finish = choice.path("finish_reason").asText(null);

        List<ToolCall> calls = new ArrayList<>();
        for (JsonNode tc : msg.path("tool_calls")) {
            String id = tc.path("id").asText(null);
            JsonNode fn = tc.path("function");
            String name = fn.path("name").asText(null);
            String args = fn.path("arguments").asText(null);
            calls.add(new ToolCall(id, name, args));
        }

        JsonNode usageNode = root.path("usage");
        ChatResponse.Usage usage = usageNode.isMissingNode()
                ? ChatResponse.Usage.EMPTY
                : new ChatResponse.Usage(
                        usageNode.path("prompt_tokens").asInt(0),
                        usageNode.path("completion_tokens").asInt(0),
                        usageNode.path("total_tokens").asInt(0)
                );
        return new ChatResponse(content, calls, finish, usage);
    }
}
