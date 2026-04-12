import { describe, it, expect, vi, beforeEach } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";
import { Pushable } from "../utils.js";
import type { AgentSideConnection, SessionNotification } from "@agentclientprotocol/sdk";
import type { SDKMessage, SDKUserMessage, Query } from "@anthropic-ai/claude-agent-sdk";
import { SettingsManager } from "../settings.js";

/**
 * Create a controllable async generator that acts as a mock Query.
 * push() enqueues messages; end() signals the generator is done.
 */
function createMockQuery(): {
  query: Query;
  push: (msg: SDKMessage) => void;
  end: () => void;
} {
  const pushable = new Pushable<SDKMessage>();
  // Cast to Query — we only need the async iterator + a few stubs
  const query = Object.assign(pushable[Symbol.asyncIterator](), {
    [Symbol.asyncIterator]() {
      return this;
    },
    interrupt: vi.fn(async () => {}),
    setPermissionMode: vi.fn(async () => {}),
    setModel: vi.fn(async () => {}),
    setMaxThinkingTokens: vi.fn(async () => {}),
    supportedCommands: vi.fn(async () => []),
    supportedModels: vi.fn(async () => []),
    initializationResult: vi.fn(async () => ({ models: [] })),
    return: vi.fn(async () => ({ value: undefined, done: true as const })),
    throw: vi.fn(async () => ({ value: undefined, done: true as const })),
  }) as unknown as Query;

  return {
    query,
    push: (msg: SDKMessage) => pushable.push(msg),
    end: () => pushable.end(),
  };
}

function createMockClient(): {
  client: AgentSideConnection;
  updates: SessionNotification[];
} {
  const updates: SessionNotification[] = [];
  const client = {
    sessionUpdate: vi.fn(async (notification: SessionNotification) => {
      updates.push(notification);
    }),
    requestPermission: vi.fn(async () => ({
      outcome: { outcome: "selected", optionId: "allow" },
    })),
    readTextFile: vi.fn(async () => ({ content: "" })),
    writeTextFile: vi.fn(async () => ({})),
  } as unknown as AgentSideConnection;
  return { client, updates };
}

function makeSuccessResult(): SDKMessage {
  return {
    type: "result",
    subtype: "success",
    duration_ms: 100,
    duration_api_ms: 50,
    is_error: false,
    num_turns: 1,
    result: "done",
    stop_reason: "end_turn",
    total_cost_usd: 0.01,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
    modelUsage: {},
    permission_denials: [],
    uuid: "test-uuid",
    session_id: "test-session",
  } as unknown as SDKMessage;
}

function makeTaskCompletedMessage(taskId: string): SDKMessage {
  return {
    type: "system",
    subtype: "task_notification",
    task_id: taskId,
    status: "completed",
    output_file: "",
    summary: "task completed",
    uuid: "task-uuid-" + taskId,
    session_id: "test-session",
  } as unknown as SDKMessage;
}

const silentLogger = {
  log: () => {},
  error: () => {},
};

describe("drain loop: prompt returns on first result", () => {
  let agent: ClaudeAcpAgent;
  let mockClient: ReturnType<typeof createMockClient>;
  let mockQuery: ReturnType<typeof createMockQuery>;
  const SESSION_ID = "test-session-drain";

  beforeEach(() => {
    mockClient = createMockClient();
    agent = new ClaudeAcpAgent(mockClient.client, silentLogger);
    mockQuery = createMockQuery();

    // Install a mock session directly (bypasses createSession which needs the real SDK)
    agent.sessions[SESSION_ID] = {
      query: mockQuery.query,
      input: new Pushable<SDKUserMessage>(),
      cancelled: false,
      permissionMode: "default",
      settingsManager: {} as SettingsManager,
      pendingPrompt: null,
      activeTasks: new Set(),
      drainPromise: Promise.resolve(),
    };

    // Start the drain loop
    agent.sessions[SESSION_ID].drainPromise = (agent as any).drainSession(SESSION_ID);
  });

  it("resolves prompt on first result message", async () => {
    // Call prompt (it returns a promise that the drain loop resolves)
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // Simulate the SDK emitting a result
    mockQuery.push(makeSuccessResult());

    const response = await promptPromise;
    expect(response.stopReason).toBe("end_turn");

    // Clean up
    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("resolves prompt before idle — does not wait for task completion", async () => {
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "start background task" }],
    });

    // Emit the result (main turn done) — prompt should resolve immediately
    mockQuery.push(makeSuccessResult());

    const response = await promptPromise;
    expect(response.stopReason).toBe("end_turn");

    // Now emit task notification (happens AFTER prompt returned)
    mockQuery.push(makeTaskCompletedMessage("task-1"));

    // Emit another result (auto-turn from background task completion)
    mockQuery.push(makeSuccessResult());

    // The prompt already resolved — this just exercises between-turn processing
    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("handles multiple sequential prompts", async () => {
    // First prompt
    const p1 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 1" }],
    });
    mockQuery.push(makeSuccessResult());
    const r1 = await p1;
    expect(r1.stopReason).toBe("end_turn");

    // Second prompt (on same session, drain loop still running)
    const p2 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 2" }],
    });
    mockQuery.push(makeSuccessResult());
    const r2 = await p2;
    expect(r2.stopReason).toBe("end_turn");

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("forwards stream_event messages as session updates", async () => {
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // Emit a stream event with text content
    mockQuery.push({
      type: "stream_event",
      event: {
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "text",
          text: "Hello back!",
        },
      },
      parent_tool_use_id: null,
      uuid: "stream-uuid",
      session_id: SESSION_ID,
    } as unknown as SDKMessage);

    // Then the result
    mockQuery.push(makeSuccessResult());

    await promptPromise;

    // Check that the stream event was forwarded
    const textUpdates = mockClient.updates.filter(
      (u) => u.update.sessionUpdate === "agent_message_chunk",
    );
    expect(textUpdates.length).toBe(1);
    expect(textUpdates[0].update).toMatchObject({
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: "Hello back!" },
    });

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("tracks task_notification — removes from activeTasks set", async () => {
    // Pre-populate an active task
    agent.sessions[SESSION_ID].activeTasks.add("task-42");
    expect(agent.sessions[SESSION_ID].activeTasks.has("task-42")).toBe(true);

    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // Emit task_notification (task completed)
    mockQuery.push({
      type: "system",
      subtype: "task_notification",
      task_id: "task-42",
      status: "completed",
      output_file: "",
      summary: "done",
      uuid: "notif-uuid",
      session_id: SESSION_ID,
    } as unknown as SDKMessage);

    // Then the result
    mockQuery.push(makeSuccessResult());

    await promptPromise;

    expect(agent.sessions[SESSION_ID].activeTasks.has("task-42")).toBe(false);

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("cancel resolves pending prompt with cancelled", async () => {
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // Cancel the session
    await agent.cancel({ sessionId: SESSION_ID });

    // The SDK will emit a result after interrupt
    mockQuery.push(makeSuccessResult());

    const response = await promptPromise;
    expect(response.stopReason).toBe("cancelled");

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("rejects pending prompt on error result", async () => {
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    mockQuery.push({
      type: "result",
      subtype: "error_during_execution",
      duration_ms: 100,
      duration_api_ms: 50,
      is_error: true,
      num_turns: 1,
      stop_reason: null,
      total_cost_usd: 0.01,
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
      modelUsage: {},
      permission_denials: [],
      errors: ["something went wrong"],
      uuid: "err-uuid",
      session_id: SESSION_ID,
    } as unknown as SDKMessage);

    await expect(promptPromise).rejects.toThrow("something went wrong");

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("rejects pending prompt when generator exhausts without result", async () => {
    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // End the generator without emitting a result
    mockQuery.end();

    await expect(promptPromise).rejects.toThrow("Session did not end in result");

    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("between-turn result messages do not resolve", async () => {
    // First: send a prompt and resolve it
    const p1 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 1" }],
    });
    mockQuery.push(makeSuccessResult());
    await p1;

    // Now emit a between-turn result (e.g., from auto-turn after bg task)
    // No prompt is pending — this should NOT cause any errors
    mockQuery.push(makeSuccessResult());

    // Now send a second prompt — it should still work
    const p2 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 2" }],
    });
    mockQuery.push(makeSuccessResult());
    const r2 = await p2;
    expect(r2.stopReason).toBe("end_turn");

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("concurrent prompt rejects the first pending prompt", async () => {
    const p1 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 1" }],
    });

    // Send a second prompt before the first resolves
    const p2 = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "prompt 2" }],
    });

    // The first prompt should be rejected
    await expect(p1).rejects.toThrow("Prompt superseded by a new prompt");

    // The second prompt should resolve normally
    mockQuery.push(makeSuccessResult());
    const r2 = await p2;
    expect(r2.stopReason).toBe("end_turn");

    mockQuery.end();
    await agent.sessions[SESSION_ID].drainPromise;
  });

  it("drainSession error propagates to pending prompt", async () => {
    // Make sessionUpdate throw on the next call
    const error = new Error("sessionUpdate failed");
    (mockClient.client.sessionUpdate as ReturnType<typeof vi.fn>).mockRejectedValueOnce(error);

    const promptPromise = agent.prompt({
      sessionId: SESSION_ID,
      prompt: [{ type: "text", text: "hello" }],
    });

    // Emit a stream event that triggers sessionUpdate (which will throw)
    mockQuery.push({
      type: "stream_event",
      event: {
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "text",
          text: "boom",
        },
      },
      parent_tool_use_id: null,
      uuid: "stream-uuid",
      session_id: SESSION_ID,
    } as unknown as SDKMessage);

    await expect(promptPromise).rejects.toThrow("sessionUpdate failed");

    // drainPromise should also complete (not hang)
    await agent.sessions[SESSION_ID].drainPromise;
  });
});
