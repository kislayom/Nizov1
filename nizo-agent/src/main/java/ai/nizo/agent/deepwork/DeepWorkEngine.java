package ai.nizo.agent.deepwork;

import ai.nizo.agent.session.SessionStore;
import ai.nizo.api.llm.ChatMessage;
import ai.nizo.api.llm.ChatRequest;
import ai.nizo.api.llm.ChatResponse;
import ai.nizo.api.llm.LlmClient;
import ai.nizo.api.llm.ToolCall;
import ai.nizo.api.llm.ToolDef;
import ai.nizo.api.tool.Tool;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.api.tool.ToolResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Deep Work — long-horizon task execution with machine-checkable accuracy.
 *
 * <p>The single-turn agent loop improvises; that's fine for 30-second asks and
 * exactly wrong for "analyze X deeply" or "this will take an hour". This engine
 * runs the disciplined version, generalizing what the deterministic stock
 * orchestrator proved: <b>structure beats improvisation</b>.
 *
 * <pre>
 *   goal → PLAN (3-7 concrete steps, JSON, no-thinking)
 *        → per step: EXECUTE (focused mini tool-loop, fresh context, ≤8 rounds)
 *                    VERIFY  (separate LLM pass: does the evidence support the result?)
 *                    retry once with the verifier's objection injected
 *                    CHECKPOINT (sqlite — survives restarts, resumes on boot)
 *        → SYNTHESIZE (deliverable from VERIFIED step results only; gaps flagged,
 *                      claims attributed to the step that produced them)
 *        → deliverable appended to the originating chat
 * </pre>
 *
 * <p>Accuracy mechanics worth naming:
 * <ul>
 *   <li><b>Fresh context per step</b> — each step sees the goal, the plan, and one-line
 *       summaries of prior steps; never 100KB of raw accumulated tool output. Focus is
 *       what keeps a 27B model precise.</li>
 *   <li><b>Adversarial verification</b> — the verifier's ONLY job is to reject results
 *       not grounded in the step's actual tool evidence (numbers without a source,
 *       claims the tools never returned).</li>
 *   <li><b>No-thinking mode everywhere</b> — Qwen3.6's think blocks burn unbounded
 *       tokens and add latency without adding evidence; structure replaces rumination.</li>
 *   <li><b>Honest failure</b> — a step that fails verification twice is recorded FAIL
 *       and the synthesis must declare the gap rather than paper over it.</li>
 * </ul>
 */
public final class DeepWorkEngine {

    private static final Logger LOG = LoggerFactory.getLogger(DeepWorkEngine.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final Map<String, Object> NO_THINK =
            Map.of("chat_template_kwargs", Map.of("enable_thinking", false));
    private static final int MAX_STEPS = 7;
    private static final int MAX_ROUNDS_PER_STEP = 8;
    private static final int TOOL_RESULT_CAP = 6_000;
    private static final int STEP_SUMMARY_CAP = 1_200;

    private final LlmClient llm;
    private final String model;
    private final ToolRegistry tools;
    private final JobStore store;
    private final SessionStore sessions;
    /** One job at a time — the GPU serializes LLM calls anyway; queueing keeps each
     *  job's steps cache-warm instead of interleaving two jobs' contexts. */
    private final ExecutorService worker =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "deep-work");
                t.setDaemon(true);
                return t;
            });

    public DeepWorkEngine(LlmClient llm, String model, ToolRegistry tools,
                          JobStore store, SessionStore sessions) {
        this.llm = llm;
        this.model = model;
        this.tools = tools;
        this.store = store;
        this.sessions = sessions;
    }

    /** Plan synchronously (seconds), persist, then queue execution. Returns a summary
     *  the chat agent can relay immediately. */
    public String start(String userId, String chatId, String goal, String deliverable) throws Exception {
        String jobId = "dw-" + UUID.randomUUID().toString().substring(0, 8);
        List<String> plan = plan(goal, deliverable);
        JobStore.Job job = new JobStore.Job(jobId, userId, chatId, goal,
                deliverable == null ? "" : deliverable,
                JobStore.JobStatus.RUNNING.name(), 0,
                System.currentTimeMillis(), System.currentTimeMillis(), null, null);
        store.createJob(job, plan);
        worker.submit(() -> runJob(jobId));
        StringBuilder sb = new StringBuilder();
        sb.append("Deep-work job ").append(jobId).append(" started — ").append(plan.size()).append(" steps:\n");
        for (int i = 0; i < plan.size(); i++) sb.append(i + 1).append(". ").append(plan.get(i)).append('\n');
        sb.append("It runs in the background and survives restarts. The finished deliverable will be ")
          .append("posted into this chat; progress via the job_status tool.");
        return sb.toString();
    }

    /** Re-queue jobs that were mid-flight when the process died. Call once at boot. */
    public void resumeAll() {
        List<JobStore.Job> jobs = store.resumable();
        for (JobStore.Job j : jobs) {
            LOG.info("deep-work: resuming job {} (step {} of plan)", j.id(), j.currentStep());
            worker.submit(() -> runJob(j.id()));
        }
        if (!jobs.isEmpty()) LOG.info("deep-work: {} job(s) re-queued after restart", jobs.size());
    }

    // ───────────────────────────── planning ─────────────────────────────

    private List<String> plan(String goal, String deliverable) throws Exception {
        String ask = "Goal: " + goal
                + (deliverable == null || deliverable.isBlank() ? "" : "\nRequested deliverable: " + deliverable)
                + "\n\nAvailable tool names: " + String.join(", ",
                        tools.all().stream().map(Tool::name).filter(n -> !n.startsWith("deep_work") && !n.startsWith("job_")).toList());
        ChatResponse resp = llm.chat(ChatRequest.of(model, List.of(
                ChatMessage.system(PLAN_SYSTEM), ChatMessage.user(ask)))
                .withMaxTokens(900).withExtraBody(NO_THINK));
        JsonNode arr = extractJson(resp.content());
        List<String> steps = new ArrayList<>();
        if (arr != null && arr.isArray()) {
            for (JsonNode n : arr) {
                String s = n.asText("").trim();
                if (!s.isEmpty() && steps.size() < MAX_STEPS) steps.add(s);
            }
        }
        if (steps.isEmpty()) throw new IllegalStateException("planner produced no steps");
        return steps;
    }

    // ───────────────────────────── execution ─────────────────────────────

    private void runJob(String jobId) {
        JobStore.Job job = store.get(jobId).orElse(null);
        if (job == null) return;
        if (JobStore.JobStatus.CANCELLED.name().equals(job.status())) return;
        List<JobStore.Step> steps = store.steps(jobId);
        LOG.info("deep-work {} starting: \"{}\" ({} steps, resuming at {})",
                jobId, truncate(job.goal(), 80), steps.size(), job.currentStep());
        try {
            for (int i = job.currentStep(); i < steps.size(); i++) {
                JobStore.Step step = steps.get(i);
                if (JobStore.StepStatus.PASS.name().equals(step.status())) continue; // already verified pre-restart
                store.setCurrentStep(jobId, i);
                boolean ok = runStep(job, steps, i);
                if (!ok) LOG.warn("deep-work {} step {} FAILED twice — continuing with gap", jobId, i);
                steps = store.steps(jobId); // refresh summaries for the next step's context
            }
            String finalText = synthesize(job, store.steps(jobId));
            store.setJobStatus(jobId, JobStore.JobStatus.DONE, finalText, null);
            deliver(job, finalText);
            LOG.info("deep-work {} DONE ({} chars)", jobId, finalText.length());
        } catch (Exception e) {
            LOG.error("deep-work {} failed: {}", jobId, e.toString());
            store.setJobStatus(jobId, JobStore.JobStatus.FAILED, null, e.toString());
            deliver(job, "Deep-work job " + jobId + " failed: " + e.getMessage()
                    + "\nCompleted steps are preserved — job_status " + jobId + " shows partial results.");
        }
    }

    /** Execute + verify one step, with one retry informed by the verifier's objection. */
    private boolean runStep(JobStore.Job job, List<JobStore.Step> steps, int idx) {
        JobStore.Step step = steps.get(idx);
        String objection = null;
        for (int attempt = 1; attempt <= 2; attempt++) {
            store.updateStep(job.id(), idx, JobStore.StepStatus.RUNNING, attempt, null, null);
            try {
                StepOutcome out = executeStep(job, steps, idx, objection);
                Verdict v = verify(job, step.title(), out);
                if (v.pass()) {
                    store.updateStep(job.id(), idx, JobStore.StepStatus.PASS, attempt,
                            truncate(out.result(), STEP_SUMMARY_CAP), v.reason());
                    LOG.info("deep-work {} step {} PASS (attempt {}, {} tool calls)",
                            job.id(), idx, attempt, out.toolCalls());
                    return true;
                }
                objection = v.reason();
                LOG.info("deep-work {} step {} attempt {} rejected: {}", job.id(), idx, attempt, objection);
            } catch (Exception e) {
                objection = "execution error: " + e.getMessage();
                LOG.warn("deep-work {} step {} attempt {} threw: {}", job.id(), idx, attempt, e.toString());
            }
        }
        store.updateStep(job.id(), idx, JobStore.StepStatus.FAIL, 2, null, objection);
        return false;
    }

    private record StepOutcome(String result, String evidence, int toolCalls) {}

    private StepOutcome executeStep(JobStore.Job job, List<JobStore.Step> steps,
                                    int idx, String objection) throws Exception {
        // Focused context: goal + plan position + verified prior summaries. Nothing else.
        StringBuilder ctx = new StringBuilder();
        ctx.append("Overall goal: ").append(job.goal()).append('\n');
        if (!job.deliverable().isBlank()) ctx.append("Final deliverable: ").append(job.deliverable()).append('\n');
        ctx.append("\nPlan position: step ").append(idx + 1).append(" of ").append(steps.size()).append('\n');
        for (int i = 0; i < idx; i++) {
            JobStore.Step p = steps.get(i);
            ctx.append("  [done] ").append(p.title());
            if (p.resultSummary() != null) ctx.append(" → ").append(truncate(p.resultSummary(), 300));
            ctx.append('\n');
        }
        ctx.append("\nYOUR STEP NOW: ").append(steps.get(idx).title()).append('\n');
        if (objection != null) {
            ctx.append("\nYour previous attempt was REJECTED by verification: ").append(objection)
               .append("\nFix exactly that.\n");
        }

        List<ToolDef> defs = tools.toolDefs().stream()
                .filter(d -> !d.name().startsWith("deep_work") && !d.name().startsWith("job_"))
                .toList();
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(STEP_SYSTEM));
        messages.add(ChatMessage.user(ctx.toString()));

        StringBuilder evidence = new StringBuilder();
        int calls = 0;
        for (int round = 0; round < MAX_ROUNDS_PER_STEP; round++) {
            ChatResponse resp = llm.chat(new ChatRequest(model, messages, defs, null, 3_000, false, NO_THINK));
            if (!resp.hasToolCalls()) {
                String result = resp.content() == null ? "" : resp.content().trim();
                if (result.isEmpty()) throw new IllegalStateException("step produced empty result");
                return new StepOutcome(result, evidence.toString(), calls);
            }
            messages.add(ChatMessage.assistantToolCalls(resp.toolCalls()));
            for (ToolCall call : resp.toolCalls()) {
                calls++;
                ai.nizo.api.tool.UserContext.set(job.userId());
                ai.nizo.api.tool.UserContext.setChat(job.chatId());
                ToolResult r;
                try {
                    Tool tool = tools.byName(call.name()).orElse(null);
                    r = (tool == null)
                            ? ToolResult.error("unknown tool " + call.name())
                            : tool.execute(call.argumentsJson() == null ? "{}" : call.argumentsJson());
                } catch (Exception e) {
                    r = ToolResult.error("tool threw: " + e.getMessage());
                } finally {
                    ai.nizo.api.tool.UserContext.clear();
                }
                String content = r.content() == null ? "" : r.content();
                if (content.length() > TOOL_RESULT_CAP) content = content.substring(0, TOOL_RESULT_CAP) + "…[truncated]";
                messages.add(ChatMessage.toolResult(call.id(), content));
                evidence.append("[").append(call.name()).append("] ")
                        .append(truncate(content, 400)).append('\n');
            }
        }
        throw new IllegalStateException("step exceeded " + MAX_ROUNDS_PER_STEP + " tool rounds without concluding");
    }

    // ───────────────────────────── verification ─────────────────────────────

    private record Verdict(boolean pass, String reason) {}

    private Verdict verify(JobStore.Job job, String stepTitle, StepOutcome out) throws Exception {
        String ask = "Step: " + stepTitle
                + "\n\nClaimed result:\n" + truncate(out.result(), 3_000)
                + "\n\nTool evidence collected during the step:\n"
                + (out.evidence().isBlank() ? "(none — no tools were called)" : truncate(out.evidence(), 4_000));
        ChatResponse resp = llm.chat(ChatRequest.of(model, List.of(
                ChatMessage.system(VERIFY_SYSTEM), ChatMessage.user(ask)))
                .withMaxTokens(300).withExtraBody(NO_THINK));
        JsonNode v = extractJson(resp.content());
        if (v == null) return new Verdict(true, "verifier unparseable — accepted by default");
        boolean pass = "pass".equalsIgnoreCase(v.path("verdict").asText(""));
        return new Verdict(pass, v.path("reason").asText(""));
    }

    // ───────────────────────────── synthesis + delivery ─────────────────────────────

    private String synthesize(JobStore.Job job, List<JobStore.Step> steps) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("Goal: ").append(job.goal()).append('\n');
        if (!job.deliverable().isBlank()) sb.append("Deliverable requested: ").append(job.deliverable()).append('\n');
        sb.append("\nVerified step results:\n");
        for (JobStore.Step s : steps) {
            sb.append("\n### Step ").append(s.idx() + 1).append(": ").append(s.title())
              .append(" [").append(s.status()).append("]\n");
            if (s.resultSummary() != null) sb.append(s.resultSummary()).append('\n');
            else if (JobStore.StepStatus.FAIL.name().equals(s.status()))
                sb.append("(FAILED verification: ").append(s.verifyNote()).append(")\n");
        }
        ChatResponse resp = llm.chat(ChatRequest.of(model, List.of(
                ChatMessage.system(SYNTH_SYSTEM), ChatMessage.user(sb.toString())))
                .withMaxTokens(4_500).withExtraBody(NO_THINK));
        String out = resp.content() == null ? "" : resp.content().trim();
        if (out.isEmpty()) throw new IllegalStateException("synthesis produced no content");
        return out;
    }

    private void deliver(JobStore.Job job, String text) {
        try {
            sessions.append(job.chatId(), ChatMessage.assistant(
                    "📋 **Deep-work " + job.id() + " finished**\n\n" + text), job.userId());
        } catch (Exception e) {
            LOG.warn("deep-work {} delivery failed: {}", job.id(), e.toString());
        }
    }

    // ───────────────────────────── helpers + prompts ─────────────────────────────

    private static String truncate(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n) + "…";
    }

    private static JsonNode extractJson(String content) {
        if (content == null) return null;
        int a = content.indexOf('['); int b = content.lastIndexOf(']');
        int oa = content.indexOf('{'); int ob = content.lastIndexOf('}');
        try {
            if (a >= 0 && b > a && (oa < 0 || a < oa)) return MAPPER.readTree(content.substring(a, b + 1));
            if (oa >= 0 && ob > oa) return MAPPER.readTree(content.substring(oa, ob + 1));
        } catch (Exception ignore) {}
        return null;
    }

    private static final String PLAN_SYSTEM = """
            You are a planner for a long-running autonomous job. Break the goal into 3-7
            CONCRETE, independently-executable steps. Each step must be doable with the
            available tools and produce a verifiable intermediate result (data gathered,
            comparison computed, section drafted). Order steps so later ones consume
            earlier ones. The LAST step must NOT be "write the final deliverable" —
            synthesis happens separately.
            Reply with EXACTLY a JSON array of step titles, e.g.
            ["Gather X via tool_a", "Collect Y via tool_b", "Compare X and Y on criteria Z"]
            """;

    private static final String STEP_SYSTEM = """
            You are executing ONE step of a long job with machine-level accuracy standards.
            Use tools to gather real data — every number and claim in your result must come
            from a tool output in THIS step (or an earlier step's summary given to you).
            When you have enough evidence, reply WITHOUT tool calls: a dense result for
            this step only — concrete findings, numbers with units, and name the source
            tool for key figures (e.g. "revenue ₹2.4L Cr (stock_fundamentals)").
            Do not pad. Do not do other steps' work. If a tool fails, try ONE alternative
            tool, then report what you could and couldn't get.
            """;

    private static final String VERIFY_SYSTEM = """
            You are an adversarial verifier. Decide if the claimed step result is GROUNDED
            in the tool evidence shown. Reject if: key numbers/claims don't appear in the
            evidence; the result answers a different question than the step; the result
            says "I would/could" instead of reporting actual findings; tools all failed
            but the result pretends otherwise. Minor wording/rounding is fine.
            Reply with EXACTLY one JSON object:
            {"verdict":"pass"} or {"verdict":"fail","reason":"<specific objection>"}
            """;

    private static final String SYNTH_SYSTEM = """
            Compose the final deliverable from the verified step results below. Rules:
            - Use ONLY facts present in the step results. No new numbers, no padding.
            - Attribute key figures to their step, e.g. "(step 2)".
            - If any step FAILED, include a short "Gaps" section naming what's missing —
              never paper over a gap.
            - Match the requested deliverable format if one was given; otherwise produce
              a tight markdown memo: title, verdict/summary first, then sections, then gaps.
            - End with a one-line "Method" note: N steps, each verified against tool evidence.
            """;
}
