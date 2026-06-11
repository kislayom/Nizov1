package ai.nizo.agent.deepwork;

import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolResult;
import ai.nizo.api.tool.UserContext;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;

/** Inspect deep-work jobs: list the caller's recent jobs, or drill into one by id. */
public final class JobStatusTool implements Tool {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private final JobStore store;

    public JobStatusTool(JobStore store) { this.store = store; }

    @Override public String name() { return "job_status"; }

    @Override
    public String description() {
        return "Check on deep-work background jobs. Without arguments: lists the user's recent jobs "
                + "and statuses. With job_id: full step-by-step progress, per-step results, and the "
                + "final deliverable if done. Use when the user asks how a long-running job is going.";
    }

    @Override
    public String parametersJsonSchema() {
        return """
            {
              "type": "object",
              "properties": {
                "job_id": { "type": "string", "description": "Optional dw-… id for one job's details." }
              }
            }
            """;
    }

    @Override
    public ToolResult execute(String argumentsJson) {
        try {
            JsonNode args = MAPPER.readTree(argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson);
            String jobId = args.path("job_id").asText("").trim();
            if (jobId.isEmpty()) {
                String userId = UserContext.current() == null ? "web-user" : UserContext.current();
                List<JobStore.Job> jobs = store.recent(userId, 10);
                if (jobs.isEmpty()) return ToolResult.ok("No deep-work jobs yet for this user.");
                StringBuilder sb = new StringBuilder("Recent deep-work jobs:\n");
                for (JobStore.Job j : jobs) {
                    sb.append("- ").append(j.id()).append(" [").append(j.status()).append("] ")
                      .append(j.goal().length() > 80 ? j.goal().substring(0, 80) + "…" : j.goal())
                      .append('\n');
                }
                return ToolResult.ok(sb.toString());
            }
            JobStore.Job j = store.get(jobId).orElse(null);
            if (j == null) return ToolResult.error("no job " + jobId);
            StringBuilder sb = new StringBuilder();
            sb.append("Job ").append(j.id()).append(" [").append(j.status()).append("]\n")
              .append("Goal: ").append(j.goal()).append('\n');
            for (JobStore.Step s : store.steps(jobId)) {
                sb.append(s.idx() + 1).append(". [").append(s.status()).append("] ").append(s.title());
                if (s.resultSummary() != null) {
                    String r = s.resultSummary();
                    sb.append("\n   → ").append(r.length() > 200 ? r.substring(0, 200) + "…" : r);
                }
                if (s.verifyNote() != null && !s.verifyNote().isBlank()
                        && JobStore.StepStatus.FAIL.name().equals(s.status())) {
                    sb.append("\n   ✗ ").append(s.verifyNote());
                }
                sb.append('\n');
            }
            if (j.finalText() != null) sb.append("\nFinal deliverable:\n").append(j.finalText());
            if (j.error() != null) sb.append("\nError: ").append(j.error());
            return ToolResult.ok(sb.toString());
        } catch (Exception e) {
            return ToolResult.error("job_status failed: " + e.getMessage());
        }
    }
}
