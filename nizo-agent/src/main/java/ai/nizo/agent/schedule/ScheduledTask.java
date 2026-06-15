package ai.nizo.agent.schedule;

import ai.nizo.scheduler.ScheduleKind;

/**
 * A persisted schedule: a prompt to run for a chat/user, either once or on a recurring cron.
 *
 * @param id         stable id
 * @param kind       ONCE or CRON
 * @param spec       ONCE → ISO instant; CRON → 5-field UNIX cron
 * @param prompt     the instruction to run through the agent when it fires
 * @param chatId     the chat to deliver the result back to
 * @param userId     the owning user
 * @param nextFireMs epoch-millis of the next firing
 * @param lastFireMs epoch-millis of the last firing (0 if never)
 * @param enabled    false once a ONCE task has fired or a task is cancelled
 * @param createdMs  creation time
 */
public record ScheduledTask(String id, ScheduleKind kind, String spec, String prompt,
                            String chatId, String userId, long nextFireMs, long lastFireMs,
                            boolean enabled, long createdMs) {}
