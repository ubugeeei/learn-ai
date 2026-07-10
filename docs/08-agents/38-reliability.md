# 38 — Reliability, approval, and retries

## What you will build

A model can propose an action, but it must not decide whether that action is
authorized. A practical agent separates three boundaries:

```text
model proposal -> schema/capability check -> host authorization -> execution
```

In this chapter you will extend the bounded runtime with:

- an explicit effect classification for every tool;
- fail-closed host approval for effectful calls;
- timeout and exception isolation on every attempt;
- bounded retries only when both error and operation permit them;
- distinct logical call IDs and physical attempt numbers;
- audit events for authorization, attempts, retries, and suppression.

## 1. Capability is not authorization

Passing a tool definition to the model means:

> This capability exists and the model may propose arguments for it.

It does **not** mean:

> Every proposed invocation is approved.

A file-writing tool may be available because a workflow sometimes needs it,
while a particular path, diff, or moment still requires human or policy
approval. The runtime therefore validates the model proposal and then asks a
host-owned `ToolApprover` before an effectful invocation.

The default approver denies effectful tools. Callers must opt in explicitly.

## 2. Classify operational effects

`ToolEffect` has three cases:

| Effect | Approval | Automatic retry |
| --- | --- | --- |
| `ReadOnly` | automatic | allowed |
| `IdempotentWrite` | required | allowed |
| `NonIdempotentWrite` | required | forbidden |

The categories describe behavior, not intent.

- Reading a public document is normally `ReadOnly`.
- Setting a resource to a specified final value may be `IdempotentWrite`.
- Charging a card, sending an email, or appending a record is normally
  `NonIdempotentWrite`.

Misclassifying an operation as idempotent can duplicate external effects.
Classification belongs to the host/tool implementation, never to model
arguments.

## 3. Approval fails closed

For an effectful call, the runtime asks:

```scala
approver.decide(call, definition, ToolContext(call.id, step))
```

The decision contains a reason and is recorded as
`ToolAuthorizationChecked`. A denial becomes a model-visible
`approval_denied` observation; it does not invoke the tool.

If the approver throws, times out in an adapter, or is unavailable, the runtime
converts that failure to a denial. Treating an approval-system failure as
permission would turn an availability incident into a security incident.

Read-only operations are allowed without interactive approval, but the policy
decision is still represented in the trace.

## 4. Logical calls and physical attempts

One `ToolCall.id` identifies the model's logical request. A retry is another
physical attempt of that same request:

```text
call ID = search-17
  attempt 1 -> retryable timeout
  attempt 2 -> success
```

`ToolContext.attempt` starts at one. `executedToolCalls` counts logical calls
against the agent budget, while `ToolAttemptStarted` and
`ToolAttemptFinished` expose physical attempts.

This distinction matters for cost, latency, debugging, and idempotency keys.
An HTTP adapter could forward the stable call ID as an idempotency key and the
attempt number only as diagnostic metadata.

## 5. Retry requires two independent permissions

The runtime retries only when:

\[
\text{error.retryable}
\land
\text{effect.safeToRetry}
\land
\text{attempt} < \text{maximumToolAttempts}
\]

The error describes the observed failure. The effect describes the operation.
Neither fact alone is sufficient.

For example, a network timeout is often retryable, but a payment request may
have committed before its response was lost. Retrying a non-idempotent payment
could charge twice. The runtime emits `ToolRetrySuppressed` instead.

`maximumToolAttempts` includes the first attempt. A value of one disables
automatic retry.

## 6. Timeouts do not prove rollback

The existing runtime interrupts a virtual thread when an attempt exceeds its
deadline. That protects local liveness, but interruption does not prove that a
remote system rolled back.

```text
local timeout
  != remote cancellation acknowledgement
  != proof that no side effect happened
```

This is why timeout errors are marked retryable while retry policy still checks
the tool effect. Production adapters should also use remote idempotency keys,
deadlines, and operation-status lookup when the service supports them.

## 7. Duplicate call IDs

The runtime caches the final observation for each call ID.

- Repeating the identical call returns the cached observation.
- Reusing the ID with different contents returns `conflicting_call_id`.
- Authorization and execution are not repeated for an identical cached call.

This protects against a model repeating a request after losing track of its
own conversation state. It does not replace remote idempotency for process
crashes; durable call records are a later extension.

## 8. Audit event sequence

A successful effectful call produces a trace similar to:

```text
ToolStarted
ToolAuthorizationChecked(approved=true)
ToolAttemptStarted(attempt=1)
ToolAttemptFinished(attempt=1, retryable failure)
ToolRetryScheduled(nextAttempt=2)
ToolAttemptStarted(attempt=2)
ToolAttemptFinished(attempt=2, success)
ToolFinished
```

The outer events describe one logical call. Inner events describe authorization
and physical attempts. Terminal `AgentStopped` remains present for every run.

## Implementation walkthrough

The reliability policy is split between declarations in `Protocol.scala` and
control flow in `AgentRuntime.scala`. Read both: a correct enum with an
incorrect branch condition is still an unsafe runtime.

### 1. Encode effect policy in host-owned metadata

Each `ToolEffect` case carries two values:

```scala
enum ToolEffect(val requiresApproval: Boolean, val safeToRetry: Boolean)
```

The model receives the resulting `ToolDefinition`, but it cannot manufacture a
new executable `Tool` or change the effect stored by the host implementation.
`ToolApprover.denyEffectful` is the constructor default for `AgentRuntime`, so
omitting configuration does not grant writes.

### 2. Follow `executeCall` as a security pipeline

For a known tool with valid arguments, `executeCall` calls `authorize` and
immediately records `ToolAuthorizationChecked`. A denial is converted into a
non-retryable observation:

```scala
ToolFailed(ToolError("approval_denied", reason, retryable = false))
```

No `ToolAttemptStarted` event is produced. This absence is a testable guarantee
that authorization happened before implementation code.

Read-only calls produce an internal `ApprovalGranted` without invoking the
external approver. Effectful calls invoke `approver.decide` with host-created
`ToolContext`. The `try/catch` converts every approver exception into
`ApprovalDenied`; the fallback is never permission.

### 3. Execute the retry predicate literally

`invokeWithRetries` starts at attempt 1 and records start/finish around every
physical invocation. After a failed outcome it evaluates:

```scala
error.retryable &&
attempt < config.maximumToolAttempts &&
tool.definition.effect.safeToRetry
```

When true, it increments `attempt` and records `ToolRetryScheduled` with the
next number. When only the effect condition is false, it records
`ToolRetrySuppressed`. When the error is non-retryable or the attempt budget is
exhausted, it simply finishes with the last outcome.

Notice that `maximumToolAttempts = 3` permits attempts 1, 2, and 3—not three
retries after the first call.

### 4. Preserve a stable logical identity

Every retry receives `ToolContext(call.id, step, attempt)`. `call.id` is stable;
only `attempt` changes. An external adapter should send the call ID as its
idempotency key if the service supports one. Generating a new idempotency key
per retry would defeat the runtime's logical identity.

The run-level `cached` map stores `(ToolCall, ToolObservation)`. Equality checks
all case-class fields. Identical repeat requests reuse the final observation;
same ID with changed name or arguments is rejected. Cached reuse occurs before
the logical call budget and before authorization.

### 5. Interpret timeout and exception outcomes precisely

`invokeWithTimeout` has three result channels:

| Tool behavior | Runtime outcome |
| --- | --- |
| returns `Right(value)` | `ToolSucceeded(value)` |
| returns `Left(error)` | `ToolFailed(error)` |
| exceeds deadline | retryable `tool_timeout` |
| throws | non-retryable `tool_exception` |

An `ExecutionException` wraps a throwable from the worker thread, so the code
unwraps its cause before creating the safe error. In production, do not expose
raw exception messages until they have been classified and redacted.

The executor is shut down in `finally` for every branch. That is resource
cleanup, not proof of remote cancellation or transaction rollback.

## 9. Reading the tests as executable policy

The suite verifies:

- effectful tools are denied by default;
- an explicitly approved idempotent write executes;
- approver exceptions fail closed;
- read-only retryable failures recover within the attempt limit;
- attempts receive monotonically increasing numbers;
- non-idempotent writes suppress automatic retry;
- timeout remains visible as a retryable observation;
- duplicate call IDs reuse outcomes instead of repeating effects;
- conflicting call IDs are rejected.

These are policy tests, not only code-coverage tests. A change that alters an
authorization or retry rule must change an explicit expectation.

`AgentRuntimeSuite` uses tools with counters and recorded attempt numbers. For
the denial test, assert zero executions and zero `ToolAttemptStarted` events.
For retries, assert the exact attempt vector and event order, not only eventual
success. For a non-idempotent timeout, assert `ToolRetrySuppressed` and a single
execution.

This test style catches policy regressions that final-answer assertions miss.
A model may gracefully describe a denial even if unsafe code was invoked before
the denial event, so event order and physical counters are part of the oracle.

## 10. Remaining production boundaries

This implementation deliberately leaves several concerns visible:

- the approver interface is synchronous;
- approval decisions are not durably persisted;
- retry uses no delay, jitter, or global deadline;
- tools do not yet expose per-resource scopes;
- remote cancellation acknowledgement is adapter-specific;
- secrets and personally identifiable data need separate handling;
- multi-process idempotency needs a durable shared store.

The small runtime establishes where those mechanisms belong without pretending
they are already solved.

## Debugging checklist

- If a write executes without approval, inspect the tool's host-owned
  `ToolEffect` and confirm the runtime did not receive `allowAll` accidentally.
- If an approver outage permits work, ensure every exception path constructs
  `ApprovalDenied` and produces no attempt event.
- If a retry count is off by one, remember that `maximumToolAttempts` includes
  attempt 1 and inspect the comparison before incrementing.
- If a payment-like call retries, correct the effect classification first;
  changing only the error's `retryable` flag is insufficient policy.
- If duplicate calls consume budget, verify cache lookup occurs before the
  logical-call limit and increment.
- If a timeout duplicates remote state, inspect external idempotency and status
  lookup. Local thread interruption cannot repair a committed remote effect.
- If audit order is confusing, group events by call ID, then separate outer
  logical-call events from numbered attempt events.

## Exercises

1. Add exponential backoff with deterministic injected time for tests.
2. Add a total call deadline shared across attempts.
3. Define path-scoped approval for a file tool.
4. Persist call IDs and final outcomes in an append-only journal.
5. Add an approval decision that rewrites arguments to a safer subset, and
   explain why the rewritten call needs its own identity.
6. Simulate a timeout after a remote commit and verify that a non-idempotent
   operation is not repeated.

## Completion criteria

You are done when you can:

- distinguish model capability, host authorization, and execution;
- classify read-only, idempotent, and non-idempotent operations;
- derive the three-part retry predicate;
- explain why local timeout does not prove remote rollback;
- trace one logical call across multiple physical attempts;
- explain which guarantees require a durable external store.

## Primary evaluation references

- [AgentBench](https://arxiv.org/abs/2308.03688)
- [WebArena](https://arxiv.org/abs/2307.13854)
- [SWE-bench](https://arxiv.org/abs/2310.06770)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
