# 22b — Batch training, validation, accumulation, and schedules

## What you will build

Chapter 21 proves that MiniGPT can lower loss on one fixed sequence. That is a
graph-connectivity test, not an experiment-grade training loop. This chapter
implements:

- replacement-sampled training batches with a dedicated seed;
- exact example-weighted microbatch gradient accumulation;
- constant and linear-warmup/cosine-decay learning rates;
- schedule-provided AdamW updates and gradient clipping telemetry;
- deterministic, sequential held-out validation;
- initial and periodic validation, best-step selection, and final validation;
- token, loss, learning-rate, gradient-norm, and clipping metrics;
- exact replay from update zero under equal inputs;
- a canonical experiment ID from Chapter 22a.

Sources:

- `src/main/scala/learnai/training/MiniGptTraining.scala`;
- `src/main/scala/learnai/training/MiniGptTrainingLab.scala`;
- accumulation support in `Tensor.backwardAccumulating`;
- dynamic optimizer rates in `Optimizers.scala`;
- `src/test/scala/learnai/training/MiniGptTrainingSuite.scala`.

Exact mid-run resume is deliberately not claimed. It requires serializing
optimizer moments, update index, data/random state, and metrics atomically; that
is Chapter 22c.

## 1. Define the training objective over examples

One causal example has $T$ target tokens and MiniGPT returns their mean cross
entropy:

$$
L_i=\frac{1}{T}\sum_{t=1}^{T}-\log p_\theta(y_{i,t}\mid x_{i,\le t})
$$

For a batch of $B$ equal-length examples:

$$
L_{batch}=\frac{1}{B}\sum_{i=1}^{B}L_i
$$

`MiniGptTraining.meanLoss` builds one fresh graph per example, adds the scalar
losses, and scales by `1 / batchSize`. The shared model parameter Tensors appear
as leaves in every graph branch, so reverse mode accumulates contributions from
all examples.

This implementation is intentionally transparent but allocates one graph per
example. A production tensor engine would represent batch as an explicit axis
and execute batched kernels. The objective and weighting must remain the same.

## 2. Why microbatches exist

A full batch may not fit activation memory. Divide $B$ examples into $M$
microbatches with sizes $b_m$:

$$
B=\sum_{m=1}^{M}b_m
$$

The correct full-batch gradient is:

$$
\nabla L_{batch}
=\sum_{m=1}^{M}\frac{b_m}{B}\nabla L_m
$$

where $L_m$ is the mean loss inside microbatch $m$.

The factor $b_m/B$ is essential. Adding unscaled microbatch means multiplies
gradient magnitude by the number of microbatches. Averaging each microbatch
equally is wrong when the final microbatch is smaller.

This chapter requires `batchSize` to be divisible by `microBatchSize`, so all
training microbatches are equal. The implementation still uses exact
`microBatch.batchSize / batchSize` weighting, making the invariant visible and
ready for uneven future batches.

## 3. Accumulate only parameter leaves

Ordinary `Tensor.backward()` clears every node in its graph, including model
parameters. Calling it on microbatch 2 would erase microbatch 1's gradients.

`backwardAccumulating()` changes one rule:

```text
operation/intermediate gradients -> always clear
trainable leaf gradients         -> preserve and add
```

The training loop owns the boundary:

```scala
parameters.foreach(_.clearGradients())
microBatches.foreach { microBatch =>
  meanLoss(model, microBatch)
    .scale(microBatch.batchSize.toDouble / batchSize.toDouble)
    .backwardAccumulating()
}
optimizer.stepAtLearningRate(parameters, learningRate)
parameters.foreach(_.clearGradients())
```

Clearing intermediates prevents stale values if a graph node is reused during
its own traversal. Preserving only trainable leaves makes separate graphs act
like branches of one full-batch graph.

The focused Tensor test compares two accumulated microbatch graphs with one
combined full graph and expects identical `[3,1]` gradients. It then calls
ordinary `backward` and proves replacement semantics still work.

## 4. Batch sampling and random-state ownership

Training samples example indices with replacement using one
`SplittableRandom(batchSeed)`. Model initialization uses a separate seed.
Separating streams prevents an architecture change that consumes more random
numbers during initialization from silently changing data order.

Replacement sampling means an example may appear multiple times before another
appears once. It provides a small stochastic training baseline, not an epoch
definition. A streaming professional data path will instead own a deterministic
shuffle permutation, shard cursor, worker partition, and restart position.

Validation never consumes the training random stream. It uses consecutive
batches from the immutable validation dataset. Therefore changing validation
frequency does not change subsequent training examples.

## 5. Keep train and validation responsibilities separate

Training loss answers:

> How well did the current parameters fit the sampled batch before this update?

Validation loss answers:

> How well do parameters after this update predict a fixed held-out slice?

The loop records validation before update 1, then after every configured
interval and after the final update. `bestValidationUpdate = 0` is possible: it
means no measured update improved the initial model.

Training and validation examples come from disjoint contiguous token regions
created before windowing. This prevents a causal window from crossing the
boundary. It does not prevent duplicate text or repeated phrases from appearing
on both sides; Chapter 30's contamination work addresses content overlap.

## 6. Weight validation by examples, not batches

Suppose validation has five examples and batch size four. Sequential batches
have sizes four and one. If their mean losses are $L_4$ and $L_1$, the correct
dataset mean is:

$$
L_{val}=\frac{4L_4+1L_1}{5}
$$

The incorrect `(L4 + L1) / 2` gives the final example four times the weight of
each earlier example.

`validationLoss` multiplies each batch mean by its example count, adds counts,
and divides once. The test independently evaluates all five example losses and
compares their arithmetic mean.

`maximumValidationBatches` bounds validation cost. The resulting metric
describes only that deterministic prefix, so the configuration and experiment
manifest record the limit. Changing it changes experiment identity.

## 7. Derive learning-rate schedules exactly

### Constant

$$
\eta_t=\eta
$$

`ConstantLearningRate` accepts a finite non-negative value. Zero is a valid
no-op update and useful at a schedule endpoint.

### Linear warmup

For zero-based update index $t<W$ and $W$ warmup updates:

$$
\eta_t=\eta_{peak}\frac{t+1}{W}
$$

With peak `1` and two warmup updates, values are `0.5, 1.0`.

Warmup limits early updates while parameter scales and Adam moments are still
forming. It does not guarantee stability; peak rate, initialization, data, and
precision still matter.

### Cosine decay

After warmup, normalized progress $q\in[0,1]$ gives:

$$
\eta(q)=\eta_{min}
+\frac{1}{2}(\eta_{peak}-\eta_{min})(1+\cos(\pi q))
$$

The decay region includes peak at progress zero and reaches minimum at its final
update. For total six updates, warmup two, peak one, and minimum zero:

```text
0.5, 1.0, 1.0, 0.75, 0.25, 0.0
```

The duplicate peak marks the boundary between the last warmup point and first
decay point. This is a deliberate, tested convention. Other libraries use
different inclusive/exclusive endpoints; compare actual sequences rather than
schedule names.

## 8. Apply a scheduled rate without recreating AdamW

AdamW owns first and second moment arrays plus a one-based step. Recreating it
each update would erase those states and bias correction.

`stepAtLearningRate(parameters, effectiveLearningRate)` keeps optimizer state
and changes only the rate used for this update. `OptimizerStats` records the
actual rate alongside gradient norm and clipping scale.

Both SGD and AdamW retain `step(parameters)` for constant-rate callers. It
delegates to `stepAtLearningRate` with the constructor rate, so there is one
update implementation.

The AdamW update uses the effective rate in both decoupled weight decay and
adaptive update:

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

## 9. Interpret gradient telemetry

Before clipping, global norm is:

$$
G=\sqrt{\sum_p\sum_i g_{p,i}^2}
$$

For maximum norm $C$, the applied scale is:

$$
s=\min(1,C/G)
$$

Metrics record both. If `gradientNorm=1.2` and `gradientScale=0.8333`, clipping
reduced the effective norm to approximately one.

Repeated severe clipping may indicate an excessive learning rate, unstable
loss, bad data, or merely a deliberately low threshold. The telemetry locates
when it happened; it does not identify cause by itself.

## 10. Count tokens with an explicit convention

Each training example supplies `contextLength` targets. After one update:

$$
\Delta tokens=batchSize\times contextLength
$$

Validation tokens are excluded from `tokensSeen` because they do not update
parameters. Prompt/input tokens overlap targets in causal shifting, so this is
an objective-token convention—not unique source bytes read.

The loop uses exact `Long` arithmetic. Overflow throws rather than wrapping to
a negative count.

Tokens, not just updates, allow comparison when batch or sequence length
changes. Comparisons still need data mixture, optimizer, and model alignment.

## 11. Metrics retained per update

`TrainingStepMetrics` stores:

| Field | Observation point |
| --- | --- |
| `update` | after optimizer update, one-based |
| `learningRate` | rate applied to that update |
| `trainingLoss` | sampled batch before update |
| `validationLoss` | fixed held-out data after update, when scheduled |
| `gradientNorm` | accumulated gradient before clipping |
| `gradientScale` | multiplier applied during optimizer update |
| `tokensSeen` | cumulative training target tokens after update |

`MiniGptTrainingRun` retains every step, initial validation, best validation
value/update, final measured validation, and total tokens. Aggregates do not
delete the trajectory.

There is no wall-clock field in the deterministic core result. Timing belongs
in an execution record because it changes across machines and runs. Chapter
22's benchmark and runtime fingerprint provide that evidence model.

## Implementation walkthrough

### Configuration construction

`MiniGptTrainingConfig` validates all counts and requires exact batch/microbatch
divisibility. It derives `microBatchesPerUpdate`, although the loop groups the
sampled examples directly. `AdamWTrainingConfig` validates betas, epsilon,
weight decay, and optional clipping before model mutation.

Schedule objects validate their scalar parameters at construction and update
indices at lookup. A warmup count greater than total updates fails before the
first optimizer step.

### Training preconditions

`validateDatasets` requires non-empty training and validation datasets, equal
context lengths, and a context no larger than the model maximum. These checks
happen before creating the optimizer or sampling a batch.

The loop creates its random generator from `batchSeed`, constructs one AdamW,
and measures initial validation. Best validation begins at update zero.

### One update

1. Clear parameter gradients.
2. Sample exactly `batchSize` examples from training.
3. Partition them into `microBatchSize` groups.
4. Build each microbatch mean loss.
5. Add its scalar value weighted by number of examples for training metrics.
6. Scale its graph by `microBatchSize / batchSize` and accumulate backward.
7. Evaluate the schedule at the zero-based update index.
8. Apply one AdamW step with retained moments.
9. Clear gradients so the returned model has no stale training state.
10. Add target-token count with exact arithmetic.
11. Optionally evaluate deterministic validation after the update.
12. Update best validation and append one immutable metric.

The model is mutated only at step 8. Any validation failure occurs after an
update and would need a run-failure record in production. Exact transaction and
resume behavior remains future work.

### Validation

`sequentialBatches(..., dropLast=false)` retains the final short batch. The
method takes only the configured prefix, requires at least one batch, evaluates
each batch mean without backward, and computes an example-weighted total.

Validation graph objects become unreachable after their scalar values are read.
This transparent Tensor engine relies on garbage collection; a production
engine uses explicit no-gradient mode and releases activation storage earlier.

### Final run

After all updates, `MiniGptTrainingRun` validates one metric per configured
update. Final validation walks backward to the most recent recorded value; the
loop always validates the final update, but the defensive fallback is initial
validation.

## Reading the tests

Schedule tests assert every endpoint and interior value for a six-update
example. This catches off-by-one conventions that a monotonicity-only test
would miss.

Batch loss compares the graph's mean with three independently evaluated scalar
losses. Validation uses five examples split `4+1`, catching accidental equal
batch weighting.

Determinism constructs two models from the same seed, trains them independently,
and requires exact equality of the full run records and every parameter vector.
It also verifies gradients are zero when training returns.

The learning test uses a deterministic four-token cycle. Sixty updates must
produce scheduled validation at `[5,10,...,60]`, exactly 720 training target
tokens, finite gradient telemetry, valid clipping scales, and a final validation
loss below half its initial value. The threshold is tied to this fixed fixture;
it is not a general model-quality claim.

Failure tests reject incompatible batch/microbatch sizes, empty training data,
and warmup longer than the run before any accepted update.

## Run and interpret the lab

```console
$ nix develop -c sbt 'runMain learnai.training.runMiniGptTrainingLab'
```

The lab uses a fixed byte tokenizer, contiguous split, model seed, batch seed,
and warmup/cosine schedule. It prints:

- experiment ID;
- training/validation example counts;
- initial validation loss;
- every scheduled update with tokens, rate, train/validation loss, norm, and
  clipping scale;
- best/final validation and total training tokens.

On the validated local run, held-out loss falls substantially and the final
updates use the minimum learning rate. Exact timing is intentionally not
documented as a portable claim. Inspect your printed trajectory and runtime
context.

The corpus repeats one phrase and tokenizer training is avoided by using bytes.
This demonstrates control flow, not generalization to natural language.

## Debugging checklist

### Microbatch result differs from full batch

1. Clear parameter gradients exactly once before the first microbatch.
2. Use `backwardAccumulating`, not ordinary `backward`.
3. Scale each microbatch mean by `microbatchSize / batchSize`.
4. Ensure optimizer step occurs only after all microbatches.
5. Disable stochastic layers or replay identical random masks.
6. Compare one parameter coordinate before checking the complete model.

### Validation changes when frequency changes

1. Confirm validation does not consume the training random generator.
2. Confirm model has no train/eval stochastic behavior in this chapter.
3. Check validation data and ordering are immutable.
4. Check validation performs no backward or optimizer step.

### Loss is unstable or `NaN`

1. Locate the first non-finite loss/update, not only the final failure.
2. Inspect input/target range and stable cross entropy.
3. Plot learning rate, gradient norm, and clipping scale together.
4. Overfit one fixed batch before reintroducing sampling.
5. Reduce peak rate only after confirming graph and data invariants.
6. Add parameter/activation norm telemetry at the first unstable block.

### Replay differs

1. Compare experiment IDs and canonical specifications.
2. Compare model and batch seeds independently.
3. Compare exact data tokens and split boundary.
4. Check iteration order over every collection.
5. Check JVM/environment revision and any added stochastic component.
6. Remember that mid-run restart is not supported yet; only replay from zero is
   guaranteed here.

## Run the complete workflow on your own corpus

`runMiniGptTrainingLab` is a deterministic fixture for observing the algorithm.
The next layer, `TrainingWorkflow`, connects the existing components across real
file boundaries:

```console
$ ./learn-ai train --input data/corpus.txt --output runs/first \
    --context 32 --channels 32 --heads 4 --hidden 64 --layers 2 \
    --updates 100 --batch-size 8 --microbatch 2
```

The command reads UTF-8, tokenizes bytes, creates non-crossing train/validation
splits, runs resumable training, and persists the result.

| Artifact | Purpose | Verification boundary |
| --- | --- | --- |
| `manifest.json` | corpus/config/code/runtime identity | canonical JSON and SHA-256 ID |
| `metrics.jsonl` | loss, LR, gradient, and tokens per update | one machine-readable row per update |
| `model.laigpt` | inference architecture and weights | version, shape, label, and SHA-256 |
| `training.laibnd` | complete exact-resume state | optimizer, scheduler position, RNG, data cursor, and SHA-256 |

The CLI rejects unknown, duplicate, and valueless options. Missing inputs,
insufficient split windows, incompatible head/channel counts, and write failures
also produce a `training workflow failed` diagnostic and a nonzero exit. The
models remain deliberately small, but this is an end-to-end path from user data
to reusable, inspectable artifacts rather than a fixed-string printing demo.

## Limitations and next connection

This training system still lacks:

- epoch/shard data cursor and distributed shuffle;
- batched Tensor kernels;
- dropout and train/eval modes;
- mixed precision and loss scaling;
- activation memory accounting/checkpointing;
- asynchronous logging and a large-scale run registry;
- early stopping policy and artifact retention;
- distributed gradient reduction.

Chapter 22c will make interrupted state resumable. Data and distributed stages
will replace replacement sampling with versioned streaming order and partition
the same accumulated gradient semantics across workers.

## Exercises

1. Support uneven final training microbatches and prove example weighting.
2. Add linear decay and inverse-square-root schedules with exact endpoint tests.
3. Add an initial training-batch overfit diagnostic before a full run.
4. Record parameter/update norms by named module without changing optimization.
5. Add early stopping as an explicit policy while always retaining the complete
   validation trajectory.
6. Add dropout with a separately owned random stream and train/eval mode.
7. Implement a batched MiniGPT path and prove loss/gradient equivalence with the
   per-example reference.

## Completion criteria

- Derive the batch and weighted microbatch objectives.
- Explain which gradients accumulate and which graph nodes reset.
- Enumerate the exact learning-rate sequence for a given configuration.
- Distinguish training loss before update from validation loss after update.
- Compute example-weighted validation with a short final batch.
- Explain every per-step metric and token-count convention.
- Reproduce a run exactly from update zero.
- State why this does not yet guarantee exact interrupted resume.
- Interpret loss reduction as a fixture-specific systems check, not frontier
  model quality.

## Primary sources

- [Adam](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- [Course reading map](../09-papers/40-primary-reading-map.md)
