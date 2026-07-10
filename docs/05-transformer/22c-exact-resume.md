# 22c — Exact training resume

## What you will build

Chapter 22b's trainer is deterministic: run it twice from update zero with
the same seed and you get identical metrics. But determinism is not
resumability. If the process dies at update 700 of 1000, replaying from
zero wastes 70% of the compute, and "approximately resuming" — reloading
weights but resetting the optimizer and data stream — silently trains a
*different* run whose loss curve can no longer be compared with anything.

In this chapter you will:

- enumerate the five pieces of state a training run actually owns;
- replace a hidden-state RNG with `SplitMix64`, whose entire state is one
  observable `Long`;
- capture and restore complete AdamW moments with validation;
- implement `ResumableMiniGptTraining` whose resume contract is *bitwise*
  equality, verified at every split point of a run.

## Prerequisites

You should understand Chapter 13 (AdamW state), Chapter 22a (experiment
identity), and Chapter 22b (the batch training loop being made resumable).

## 1. What state does a training run own?

Everything the next update reads that the previous update wrote. For the
Chapter 22b loop that inventory has exactly five entries:

```text
model:      every parameter tensor's values
optimizer:  AdamW first/second moments and the step count
scheduler:  the global update index (schedules are pure functions of it)
data:       the batch sampling cursor (here: the RNG stream position)
randomness: the generator state itself
```

Miss any one and resume is approximate. The classic failure is restoring
weights but recreating the optimizer: AdamW's bias correction
$1 - \beta^t$ restarts at $t = 1$, the effective step size jumps, and the
loss curve kinks exactly at the resume point. The second classic is the
data cursor — a reseeded generator replays early batches, so the model
sees some data twice and some never.

## 2. Why the RNG had to change

`java.util.SplittableRandom` is deterministic for a fixed seed but offers
no way to read its position mid-stream, so a run using it can be replayed
from zero and never resumed from the middle.

SplitMix64 — the mixer published by Steele, Lea, and Flood, used inside
the JDK for seeding — has a 64-bit counter as its *entire* state:

$$
s_{n+1} = s_n + \gamma, \qquad
\text{output}_n = \operatorname{mix}(s_{n+1})
$$

with the golden-gamma constant $\gamma = \lfloor 2^{64}/\varphi \rfloor$
forced odd, and `mix` two multiply-xorshift avalanche rounds. Capture
`rng.state`, later call `SplitMix64.fromState(state)`, and the future of
the stream is reproduced exactly. The class implements
`java.util.random.RandomGenerator`, so `CausalDataset.sampleBatch` accepts
it unchanged; `nextInt`/`nextDouble` are overridden with explicitly
specified algorithms (top-31-bit rejection sampling, top-53-bit doubles)
so the stream does not depend on JDK default-method details. Rejection
sampling consumes a variable number of outputs — that consumption is part
of the stream contract, which is why the *counter*, not the draw count, is
the checkpointed cursor.

## 3. Optimizer state as a first-class value

`AdamWSnapshot` records the step count and both moment arrays in
parameter order. Two design points matter more than the copying:

- *position, not identity.* The live optimizer keys its state by tensor
  identity, which cannot survive a process restart. The snapshot flattens
  to parameter order, and the restore contract is "same architecture, same
  parameter ordering" — the same contract Chapter 25's checkpoint loader
  already enforces by label.
- *validate before mutating.* Restore checks sizes, finiteness, and
  second-moment non-negativity for every tensor before touching any state,
  so a corrupted snapshot leaves the optimizer usable and the error names
  the offending tensor index.

## 4. The resume contract

`MiniGptTrainingState` carries the five ingredients plus summary
bookkeeping (initial/best validation, tokens seen). The contract, tested
literally, is: for every split point $k$ of an $n$-update run,

```text
train(model, s0, n)
  ==  train(model, s0, k) ; train(freshModel, sk, n - k)
```

with `==` meaning *bitwise equality* of every metric and every final
parameter value — not tolerance-based closeness. Bitwise is achievable
because every operation in this course is deterministic and executed in a
fixed order, and it is the right bar: a tolerance would hide exactly the
class of bug (a reset bias correction, a replayed batch) this chapter
exists to prevent.

Note what resuming into `freshModel` proves: the test restores into a
model constructed with *different* random weights, so if any information
needed by update $k+1$ lives outside the state value, the sweep fails.

## 5. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `ResumableMiniGptTraining` suite runs one straight 8-update run and
seven chunked runs (split at every interior update), comparing metrics,
weights, and final states exactly. A 30-update run then checks learning
progress and validation cadence under the SplitMix64 stream.

## 6. Implementation walkthrough

`freshState` validates the datasets against the model, computes the same
pre-update validation loss Chapter 22b records, and captures the origin:
zero updates, zero tokens, `randomState = config.batchSeed`, zero
optimizer moments (via `AdamWSnapshot.zero`), and the initial parameter
values.

`train(model, split, config, state, updates)` first checks that the chunk
fits inside `config.totalUpdates`, then *restores before running*: every
parameter tensor is overwritten from the state via
`assignParameterValues`, a fresh AdamW is built from the configured
hyperparameters and given the snapshot, and the RNG is rebuilt with
`SplitMix64.fromState`. The loop body is Chapter 22b's algorithm verbatim
— replacement-sampled batches, weighted microbatch accumulation with
`backwardAccumulating`, the schedule evaluated at the *global* update
index, interval validation with a final-update override. Nothing about the
loop knows it is a chunk; only the loop bounds do.

The returned state is captured after the last update: new counter, new
moments, new parameter values, carried-forward best-validation
bookkeeping. `trainFromStart` composes `freshState` and one full-length
`train` call into the familiar `MiniGptTrainingRun` summary.

One honest limitation is documented in the code: `MiniGptTraining` (22b)
and `ResumableMiniGptTraining` use different generators, so their batch
sequences differ; each is deterministic under its own contract, but their
metrics are not mutually comparable. The state itself is an in-memory
value here — persisting it with Chapter 25's versioned, checksummed,
`doubleToLongBits`-exact discipline is Exercise 1, and decimal text
formats are *not* acceptable there because they round.

## 7. Reading the tests

- the split-point sweep is the contract: metrics concatenation, final
  weights, and the full final state must equal the straight run at every
  interior split, with resume targets seeded differently on purpose;
- the fresh-state test pins the origin semantics (seed as cursor, zero
  moments, initial-equals-best validation);
- the cadence test shows validation intervals spanning a chunk boundary
  fire exactly once each, on the straight run's schedule;
- failure tests cover zero-length chunks, overrunning the configured
  total, restoring into a wider architecture (rejected by shape, with the
  model untouched), and corrupt bookkeeping in the state constructor;
- `SplitMix64` has its own suite, including an independent transcription
  of the published algorithm and a state-capture/continue oracle, and
  `AdamW.snapshot/restore` has a three-plus-two-equals-five continuation
  test in the optimizer suite.

## 8. Debugging checklist

1. If resumed loss kinks at the boundary, print the optimizer step count
   on both sides; a reset bias correction is the usual culprit.
2. If early batches repeat after resume, you checkpointed the seed instead
   of the stream position.
3. Compare the learning rate at update $k+1$ in straight and resumed runs;
   a chunk-local schedule index is an off-by-`k` you can read directly.
4. Assert bitwise equality in tests, not closeness — tolerances hide
   resume bugs by construction.
5. When equality fails, bisect the five state ingredients: overwrite each
   resumed component with the straight run's value until the diff
   disappears; the last one you replaced was incomplete.

## 9. Failure modes to test

- optimizer moments or step count silently reset at resume;
- RNG reseeded from the original seed (replayed data);
- schedule evaluated at the chunk-local index;
- weights restored into the wrong tensor ordering;
- state accepted for a model of a different architecture;
- validation bookkeeping (best loss/update) lost across the boundary;
- a "resume" test that starts from the same live objects and therefore
  proves nothing about the state value.

## Exercises

1. Persist `MiniGptTrainingState` with Chapter 25's format discipline:
   magic, version, SHA-256, exact bit-level doubles, and truncation tests.
2. Extend the state with the Chapter 22a experiment identity and refuse to
   resume under a different config hash.
3. Add periodic automatic checkpointing every $m$ updates and measure its
   wall-clock overhead honestly (Chapter 22 rules).
4. Make `MiniGptTraining` and `ResumableMiniGptTraining` share one loop
   implementation and prove 22b's tests still pass unchanged.

## Completion criteria

You are done when you can:

- list the five state ingredients from memory and name the symptom of
  omitting each one;
- explain why replay-determinism does not imply resumability, and what
  property SplitMix64 adds;
- explain identity-keyed versus position-keyed optimizer state and where
  each is valid;
- defend bitwise equality as the resume test bar;
- state why the RNG counter, not the draw count, is the data cursor.

## Primary sources

- [Fast Splittable Pseudorandom Number Generators (Steele, Lea, Flood)](https://doi.org/10.1145/2660193.2660195)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
