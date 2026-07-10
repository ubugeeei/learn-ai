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
