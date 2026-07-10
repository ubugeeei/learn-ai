# 35 — A bounded agent loop and audit trace

## What you will build

Implement `model -> tool -> observation -> model` with guaranteed termination
through final answer, hard limits, or typed failure. Source:
`src/main/scala/learnai/agent/AgentRuntime.scala`.

## Agent loop as a state machine

```text
InvokeModel
  | FinalAnswer ----------------------> Completed
  | RequestTools
  v
ValidateAndExecute
  | append observations
  +-----------------------------------> InvokeModel

Any state -> model error / step limit / tool limit -> terminal failure
```

Explicit state, counters, and terminal reasons are safer than an unbounded loop
over an ambiguous text buffer.

## One iteration

1. Build `ModelRequest` from history and granted tools.
2. Accumulate usage.
3. On final answer, append assistant text and finish.
4. On tool calls, append the assistant call item.
5. Validate capability, schema, and budget.
6. Execute within the timeout.
7. Append structured observations.
8. Begin the next model step.

The model sees effects only through observations.

## Hard limits

`AgentConfig` includes:

- `maximumModelSteps`;
- `maximumToolCalls`;
- `toolTimeoutMillis`.

These are liveness and safety invariants, not quality suggestions. A repeating
model, hanging tool, or invalid-call correction loop still reaches a terminal
state. Production systems also need whole-run deadlines and cost/token budgets.

## Timeout and cancellation

Tools run on JDK virtual threads and are awaited with a timeout. On timeout, the
task is interrupted and a retryable `tool_timeout` observation is returned.

Interrupt is cooperative, not guaranteed forced termination. Tool code and
network clients need their own deadlines. A side effect may have succeeded just
before timeout, so blind retry can duplicate it.

## Idempotency

Call ID acts as an idempotency key:

- same ID and same call: reuse the cached result;
- same ID with different contents: reject as conflict.

For payments, sends, and creates, pass the same key to the external service.
Durable deduplication is required across process crashes.

## Audit events

Every run returns success or failure with:

- model invocation/decision and usage;
- tool start/finish/reuse;
- terminal status and reason.

Production events may add durations, trace ID, principal, policy decision, and
redacted argument hashes. Do not log secrets or unnecessary personal data.

## Terminal states are distinct

- `Completed`;
- `ModelFailed`;
- `ModelStepLimitExceeded`;
- `ToolCallLimitExceeded`.

Limit exhaustion is not disguised as a successful answer. A caller can choose
retry, more budget, user approval, or manual handoff.

## Trajectory tests

Tests inspect more than final text:

- valid call becomes the next request's observation;
- invalid schema never invokes a tool;
- unknown capabilities never execute;
- identical IDs produce one side effect;
- conflicts and timeouts remain typed;
- budget prevents the next execution;
- failures retain a complete trace.

Real-model evaluation should separately measure final correctness, tool
selection, argument accuracy, steps, cost, latency, and unsafe attempts.

## Implementation walkthrough

Read `AgentRuntime.run` as an interpreter for a small state machine. It uses
local mutable variables internally, but exposes only the immutable `AgentRun`
snapshot when it stops.

### 1. Identify the complete run state

The loop owns six evolving values:

| Variable | Meaning |
| --- | --- |
| `history` | model-visible conversation items |
| `events` | host-visible audit trajectory |
| `usage` | sum of every returned model decision |
| `executedToolCalls` | logical new call IDs charged to the budget |
| `step` | number of completed model/tool iterations |
| `cached` | call ID to original proposal and observation |

`stop` closes over those values. Every return path therefore produces the same
complete shape, including failures. There is no exception-only result format
that discards the trace.

### 2. Trace a successful tool round trip

Assume the initial history contains one user message and the model first
returns `RequestTools(Vector(call), usage0)`.

1. `ModelInvoked(0, 1)` is appended.
2. `ModelRequest(history, definitions, 0)` is sent.
3. `usage0` is added to the run total.
4. `ModelReturned(0, "tool_calls", usage0)` and
   `AssistantToolCalls` are appended.
5. The call ID is absent from `cached`, so the logical budget is checked and
   incremented.
6. `executeCall` returns one observation plus authorization/attempt events.
7. The observation enters both history and the ID cache.
8. `step` becomes 1, and the next request contains user message, assistant
   call, and tool observation.

If the second decision is `FinalAnswer`, the runtime appends assistant text and
returns `Completed`. A two-decision interaction therefore invokes the model
twice but executes one logical tool call.

### 3. Understand why limits are checked in different places

The model-step bound is the outer `while` condition. The tool-call bound is
checked only for a new call ID, immediately before physical execution. A cached
duplicate consumes neither another logical call nor another physical attempt.
A conflicting duplicate also does not execute, but becomes a typed
`conflicting_call_id` observation.

This placement prevents an off-by-one error: with `maximumToolCalls = 1`, the
first new call executes, while the second new call stops the run before any
side effect.

### 4. Separate logical calls from physical attempts

`executedToolCalls` counts accepted new IDs. `ToolAttemptStarted` counts actual
invocations, including retries. One logical read call may therefore produce
three physical attempts. This distinction is necessary for billing,
idempotency analysis, and evaluation.

### 5. Follow the timeout boundary

`invokeWithTimeout` creates a virtual-thread executor, submits the tool, and
waits using `future.get(timeout, MILLISECONDS)`. A normal `Right` becomes
`ToolSucceeded`; a normal `Left` becomes `ToolFailed`. A timeout cancels with
interrupt and returns retryable `tool_timeout`. An exception from the worker is
unwrapped from `ExecutionException` and becomes non-retryable `tool_exception`.
The `finally` block always shuts down the executor.

This contains the waiting thread, not the external world. An HTTP request or
database transaction needs its own deadline and idempotency key.

## Reading the tests

Use `AgentRuntimeSuite` as a trajectory specification. Each fake has one
purpose: scripted model decisions control the route, counting tools prove
whether execution happened, and blocking tools exercise timeout behavior.

For every new terminal path, assert four layers:

1. `AgentStatus` and final answer;
2. exact history correspondence;
3. relevant audit-event ordering;
4. usage and logical/physical counters.

The model-step-limit test is especially useful: it proves an otherwise
infinite correction loop terminates and still emits `AgentStopped`.

## Debugging checklist

- If the second model request lacks a result, check that the observation is
  appended to `history`, not only to `events`.
- If usage is low, add usage before matching on decision type so final and tool
  decisions follow the same accounting path.
- If a limit permits one extra side effect, locate the comparison relative to
  the increment and `executeCall`.
- If a duplicate executes twice, compare the entire cached `ToolCall`, not only
  its ID or name.
- If a timed-out operation keeps running, add cooperative interruption and
  transport deadlines inside the tool; executor cancellation alone is not a
  rollback mechanism.
- If a test checks only final text, inspect the event trace as well; unsafe
  attempts can be hidden behind a plausible answer.

## Exercises

1. Add a whole-run deadline and token budget.
2. Add bounded retry with exponential backoff and a fake clock.
3. Design parallel read-only calls with deterministic observation ordering.
4. Add a human-approval state for write tools.
5. Aggregate success, p95 steps, and tool-error rate from events.
6. Design a durable idempotency store.

## Completion criteria

- Describe the loop as a finite state machine.
- Explain how limits guarantee liveness.
- Separate timeout from side-effect success.
- Explain end-to-end idempotency.
- List trajectory metrics beyond final answer.
- `AgentRuntimeSuite` passes.
