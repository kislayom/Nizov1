package ai.nizo.llm;

public record LlmConfig(
        String baseUrl,
        String model,
        String authToken,
        Double defaultTemperature,
        Integer defaultMaxTokens
) {
    public static LlmConfig fromEnv() {
        return new LlmConfig(
                System.getenv().getOrDefault("NIZO_LLM_URL", "http://localhost:8080"),
                System.getenv().getOrDefault("NIZO_LLM_MODEL", "Qwen/Qwen3.6-27B"),
                System.getenv("NIZO_LLM_TOKEN"),
                envDouble("NIZO_LLM_TEMP"),
                envInt("NIZO_LLM_MAX_TOKENS")
        );
    }

    private static Double envDouble(String key) {
        String v = System.getenv(key);
        return (v == null || v.isBlank()) ? null : Double.parseDouble(v);
    }

    private static Integer envInt(String key) {
        String v = System.getenv(key);
        return (v == null || v.isBlank()) ? null : Integer.parseInt(v);
    }
}
