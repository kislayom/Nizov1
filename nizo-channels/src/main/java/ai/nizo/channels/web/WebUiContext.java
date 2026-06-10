package ai.nizo.channels.web;

import ai.nizo.agent.cache.StockReportStore;
import ai.nizo.agent.condense.CondenseEngine;
import ai.nizo.agent.exec.ChatExecutor;
import ai.nizo.agent.session.SessionStore;
import ai.nizo.api.tool.ToolRegistry;
import ai.nizo.mcp.client.McpClientPool;
import ai.nizo.mcp.config.McpServersFile;
import ai.nizo.skills.SkillManifest;
import ai.nizo.tools.registry.UsageTracker;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;

/**
 * Read-only snapshot context the web UI needs to render its inspector panes (sessions,
 * tools, skills, workspace, status). Decouples the channel from {@code Bootstrap} —
 * Bootstrap implements this in {@code nizo-app} without leaking app-layer types here.
 *
 * <p>{@code condense} and {@code mcpPool} / {@code mcpConfig} are nullable for callers that
 * don't wire those subsystems.
 */
public record WebUiContext(
        String modelName,
        String llmEndpoint,
        Path home,
        Path workspaceDir,
        SessionStore sessions,
        ToolRegistry tools,
        Supplier<List<SkillManifest>> skills,
        CondenseEngine condense,
        McpClientPool mcpPool,
        McpServersFile mcpConfig,
        ChatExecutor chatExecutor,
        UsageTracker usage,
        StockReportStore stockReports
) {
    /** Backwards-compat constructor that defaults the optional pieces to null. */
    public WebUiContext(String modelName, String llmEndpoint, Path home, Path workspaceDir,
                        SessionStore sessions, ToolRegistry tools,
                        Supplier<List<SkillManifest>> skills) {
        this(modelName, llmEndpoint, home, workspaceDir, sessions, tools, skills, null, null, null, null, null, null);
    }

    public WebUiContext(String modelName, String llmEndpoint, Path home, Path workspaceDir,
                        SessionStore sessions, ToolRegistry tools,
                        Supplier<List<SkillManifest>> skills, CondenseEngine condense) {
        this(modelName, llmEndpoint, home, workspaceDir, sessions, tools, skills, condense, null, null, null, null, null);
    }

    public WebUiContext(String modelName, String llmEndpoint, Path home, Path workspaceDir,
                        SessionStore sessions, ToolRegistry tools,
                        Supplier<List<SkillManifest>> skills, CondenseEngine condense,
                        McpClientPool mcpPool, McpServersFile mcpConfig) {
        this(modelName, llmEndpoint, home, workspaceDir, sessions, tools, skills,
                condense, mcpPool, mcpConfig, null, null, null);
    }

    public WebUiContext(String modelName, String llmEndpoint, Path home, Path workspaceDir,
                        SessionStore sessions, ToolRegistry tools,
                        Supplier<List<SkillManifest>> skills, CondenseEngine condense,
                        McpClientPool mcpPool, McpServersFile mcpConfig, ChatExecutor chatExecutor) {
        this(modelName, llmEndpoint, home, workspaceDir, sessions, tools, skills,
                condense, mcpPool, mcpConfig, chatExecutor, null, null);
    }

    public WebUiContext(String modelName, String llmEndpoint, Path home, Path workspaceDir,
                        SessionStore sessions, ToolRegistry tools,
                        Supplier<List<SkillManifest>> skills, CondenseEngine condense,
                        McpClientPool mcpPool, McpServersFile mcpConfig, ChatExecutor chatExecutor,
                        UsageTracker usage) {
        this(modelName, llmEndpoint, home, workspaceDir, sessions, tools, skills,
                condense, mcpPool, mcpConfig, chatExecutor, usage, null);
    }
}
