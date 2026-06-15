package ai.nizo.agent.eval;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * One machine-checkable evaluation task, loaded from {@code /eval/gold-tasks.json}.
 *
 * <p>The point of these is to turn "machine-level accuracy" from a claim into a number: each task
 * has a deterministically-checkable answer, so a regression in the prompts, tools, or engine shows
 * up as a falling score rather than a vibe. The starter set leans on exact computation precisely
 * because that is where an ungrounded LLM drifts — and therefore where {@code code_exec} should
 * earn its keep.
 *
 * @param id        short stable id; also used as the chatId so each task runs in an isolated thread
 * @param prompt    the user message POSTed to {@code /api/chat}
 * @param check     {@code "numeric"} | {@code "contains"} | {@code "regex"}
 * @param expected  numeric → target value as a string; contains → {@code "||"}-separated required
 *                  substrings; regex → a pattern (case-insensitive, DOTALL)
 * @param tolerance numeric only → absolute tolerance around {@code expected} (ignored otherwise)
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record GoldTask(String id, String prompt, String check, String expected, double tolerance) {}
