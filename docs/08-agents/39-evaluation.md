# 39 — Agent evaluation

## What you will build

A fluent final answer is not enough to show that an agent worked correctly. A
practical evaluation must inspect outcome, trajectory, safety boundaries, and
operational cost.

In this chapter you will build a deterministic harness that records:

- terminal status and exact outcome checks;
- required and forbidden tool requests or execution attempts;
- expected policy failures such as approval denial;
- token, model-step, and logical tool-call budgets;
- end-to-end latency per case;
- estimated token cost from an explicit price snapshot;
- per-check failures and aggregate metrics without discarding traces.

## 1. Define the unit of evaluation

An `AgentEvalCase` contains:

```text
stable case name
fresh LanguageModel factory
initial conversation state
named checks over the final AgentRun
```

The model is created fresh for every case. Reusing a stateful fake or provider
session can leak decisions, caches, or random state between cases.

Case and check names must be unique. Stable identity is needed for regression
history, failure grouping, and comparing reports across revisions.

## 2. Outcome checks

`ExpectedStatusCheck` verifies the terminal state:

```text
Completed
ModelFailed
ModelStepLimitExceeded
ToolCallLimitExceeded
```

`ExactAnswerCheck` compares a deterministic protocol result byte for byte. It
is appropriate for fixed strings, IDs, structured renderings, and known fake-
model answers.

Exact matching is usually too strict for open-ended natural language. Future
semantic or model-graded checks should be separate, versioned components with
documented uncertainty. Do not silently replace a deterministic oracle with a
judge model.

## 3. Trajectory checks

Two tool questions that sound similar have different security meanings.

### Did the model request the tool?

`ForbiddenToolRequestsCheck` inspects `AssistantToolCalls`. It fails even when
the runtime denied execution.

Use this for behavioral requirements such as:

> The model must not propose sending a message for a read-only task.

### Did the runtime physically attempt the tool?

`ForbiddenToolAttemptsCheck` and `RequiredToolAttemptsCheck` inspect
`ToolAttemptStarted` events.

Use this for effect requirements such as:

> No write may reach its implementation without approval.

A denied write can therefore pass the forbidden-attempt check while failing the
forbidden-request check. Keeping both prevents a safe runtime from hiding a
poor model policy and prevents a well-behaved model from hiding an unsafe
runtime.

## 4. Expected failure checks

Some scenarios succeed by producing a controlled failure. For example, a model
may propose an effectful tool and the host must deny it.

`RequiredToolErrorCheck("approval_denied")` verifies that the model observed the
expected policy result. Similar cases can test:

- invalid schema arguments;
- unavailable capabilities;
- timeouts;
- conflicting idempotency IDs;
- retry exhaustion.

Checking only final `Completed` status would miss whether the safety mechanism
actually activated.

## 5. Resource checks

The harness includes:

```text
MaximumTokensCheck
MaximumToolCallsCheck
MaximumModelStepsCheck
```

Budgets are part of correctness for agents. An answer that consumes unbounded
tokens or loops through tools is not operationally equivalent to a bounded
answer.

Logical tool-call count differs from physical attempts. Chapter 38 may retry
one logical call several times. Add an attempt-count check when a scenario must
constrain retry behavior specifically.

## 6. Cost is a versioned estimate

`TokenPricing` stores input and output dollars per million tokens as
`BigDecimal`. Cost is:

\[
\text{cost} =
\frac{n_{in}p_{in} + n_{out}p_{out}}{10^6}
\]

Pricing changes over time and differs by provider, model, cache status, batch,
and service tier. The harness therefore accepts a caller-provided snapshot and
defaults to zero. A report is reproducible only when it records the pricing
source and effective date outside the pure calculation.

Do not hard-code a current vendor price into the learning core.

## 7. Latency is an observation, not a benchmark claim

Every case records elapsed `System.nanoTime`. This helps find large regressions
inside a fixed environment, but one sample is not a benchmark.

For a performance claim, also control and report:

- warmup and repeated measurements;
- percentile or variation, not only mean;
- model/provider revision;
- network location and service tier;
- tool/environment reset time;
- concurrency and rate limits;
- hardware and JVM;
- prompt, seed, and cache state.

The report retains raw case duration so a separate benchmark runner can add
those controls.

## 8. Aggregate without hiding cases

`AgentEvalReport` calculates:

- passed cases and pass rate;
- total input/output token usage;
- total model steps;
- total logical tool calls;
- total elapsed time;
- total estimated token cost;
- failed check counts grouped by stable name.

It also retains every `AgentEvalResult`, complete `AgentRun`, and check detail.
An aggregate pass rate cannot explain why a case failed; the trace can.

Always inspect at least these slices:

```text
task category
tool/effect category
terminal status
failure check
model/provider revision
prompt or policy revision
latency and token bucket
```

## 9. Deterministic fakes before live providers

Runtime and policy regression tests use scripted models and deterministic tools.
This isolates orchestration behavior from remote model drift.

Live-provider evaluation is still necessary, but it answers a different
question:

```text
fake-model eval: is the runtime/policy implementation correct?
live-model eval: how often does this model produce successful trajectories?
```

Combining them into one suite makes failures hard to diagnose.

## 10. Environment-state oracles

The `AgentRunCheck` trait is extensible. A check may inspect a controlled test
environment after the run:

- file content or hash;
- database row state;
- sent-message outbox;
- browser application state;
- test-suite exit code;
- citation IDs against returned evidence.

Prefer executable state oracles over judging the final prose. Reset the
environment before every case and verify the reset itself.

## 11. Run the lab

```console
$ nix develop -c sbt 'runMain learnai.agent.runAgentEvaluationLab'
```

The lab runs three deterministic scenarios:

1. a direct exact answer;
2. an answer that must physically execute a read-only lookup;
3. a proposed non-idempotent write that must be denied before execution.

It prints each check, tokens, latency, aggregate pass rate, and logical calls.

## 12. Tests

The suite verifies:

- outcome, required-tool, resource, usage, and cost aggregation;
- failed checks remain visible by case and category;
- requested and attempted forbidden tools are distinguished;
- a fresh model is created for every case;
- case order remains deterministic;
- duplicate case/check identity fails before execution.

## 13. Remaining production boundaries

- durable result storage and schema versioning;
- provider/model/prompt/policy fingerprints;
- environment reset and isolation;
- statistical confidence over repeated stochastic samples;
- semantic, citation-faithfulness, and calibrated judge checks;
- adversarial prompt-injection and poisoned-tool suites;
- planning-level DAG metrics and checkpoint recovery evals;
- human review protocols and inter-annotator agreement;
- continuous drift detection and release gates.

## Exercises

1. Add physical tool-attempt and retry-count budgets.
2. Add a file-state oracle with setup, check, and teardown phases.
3. Run each stochastic case over fixed seeds and compute a confidence interval.
4. Add citation precision and recall checks for Chapter 36.
5. Add a planning check that identifies the first task that blocks a goal.
6. Define a versioned JSON report and round-trip it with the strict parser.

## Completion criteria

- Distinguish final outcome, request trajectory, and physical effects.
- Explain why deterministic fakes and live-provider evals answer different
  questions.
- Make token, tool, step, latency, and price assumptions explicit.
- Preserve per-case traces behind aggregate metrics.
- Prefer environment-state oracles when functional correctness is available.
- Identify which checks are deterministic and which require statistical or
  human uncertainty reporting.

## Primary sources

- [AgentBench](https://arxiv.org/abs/2308.03688)
- [WebArena](https://arxiv.org/abs/2307.13854)
- [SWE-bench](https://arxiv.org/abs/2310.06770)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
