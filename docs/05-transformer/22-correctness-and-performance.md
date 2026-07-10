# 22 — Correctness diagnostics and performance measurement

## What you will build

A model that compiles and lowers its training loss can still be wrong. A model
that returns the expected tokens can still be needlessly slow. This chapter
adds three reusable diagnostic layers:

1. a dense parameter inventory with explicit byte accounting;
2. central-difference gradient checks over deterministic parameter coordinates;
3. a local benchmark harness with warmup, batching, raw samples, descriptive
   statistics, an observable checksum, and a runtime fingerprint.

Production sources:

- `src/main/scala/learnai/diagnostics/GradientCheck.scala`;
- `src/main/scala/learnai/diagnostics/Benchmark.scala`;
- `src/main/scala/learnai/transformer/MiniGptDiagnosticsLab.scala`.

Executable specifications:

- `src/test/scala/learnai/diagnostics/GradientCheckSuite.scala`;
- `src/test/scala/learnai/diagnostics/BenchmarkSuite.scala`.

The goal is not to manufacture a green score. The goal is to retain enough
evidence to answer *what was compared, where it differed, how much timing
varied, and which environment produced the observation*.

## 1. Separate three kinds of correctness

### Mathematical correctness

The implementation must compute the intended function and derivative. Examples:

- matrix multiplication contracts the correct inner dimension;
- causal attention assigns no probability or gradient to future positions;
- cross entropy differentiates to `softmax - oneHot`;
- a tied output projection accumulates gradient into the shared embedding.

Hand calculations, finite differences, algebraic properties, and independent
reference paths test this layer.

### Numerical correctness

An algebraically valid formula may fail in finite precision. Examples include
overflowing `exp(logit)`, dividing by a nearly zero norm, or subtracting two
nearly equal values. Numerical tests need realistic scales, explicit
tolerances, and checks for `NaN` and infinity.

### Systems correctness

The right number may be computed with the wrong operational behavior. A cache
may return equivalent logits while allocating on every token. A retry may
eventually succeed after duplicating a write. A benchmark may report an
impressive time because the result was unused and optimized away.

Systems tests inspect ownership, work counters, bytes, calls, event order,
latency samples, and environment state—not only final values.

No single test covers all three layers.

## 2. Use a correctness ladder

When implementing a new kernel or model path, validate from smallest to largest:

```text
hand-computable scalar/vector case
  -> operation property
  -> finite-difference local gradient
  -> reference/optimized equivalence
  -> full-block causality and shape
  -> one-batch overfit
  -> deterministic end-to-end run
  -> statistical evaluation on held-out data
```

If an end-to-end run fails first, the search space is the entire system. If a
three-element property fails, the responsible operation is usually obvious.
Professional debugging moves upward only after lower invariants pass.

## 3. Derive central differences

For scalar loss $L(\theta)$ and one parameter coordinate $\theta_i$, Taylor
expansion around the current point gives:

$$
L(\theta_i+h)
=L(\theta_i)+hL'(\theta_i)+\frac{h^2}{2}L''(\theta_i)
+\frac{h^3}{6}L'''(\theta_i)+O(h^4)
$$

and:

$$
L(\theta_i-h)
=L(\theta_i)-hL'(\theta_i)+\frac{h^2}{2}L''(\theta_i)
-\frac{h^3}{6}L'''(\theta_i)+O(h^4)
$$

Subtracting cancels the constant and even-order terms:

$$
L'(\theta_i)
\approx
\frac{L(\theta_i+h)-L(\theta_i-h)}{2h}
$$

The truncation error is $O(h^2)$, better than the $O(h)$ error of a one-sided
difference. But `h` cannot shrink without limit. Floating-point subtraction of
nearly equal losses introduces rounding error that behaves approximately like
$O(\epsilon_{machine}/h)$. Gradient checking therefore needs a moderate step,
commonly near `1e-5` for these small `Double` computations, followed by scale-
aware tolerance rather than an assumption of exact equality.

## 4. Work one gradient probe by hand

Let:

$$
L(x_0,x_1)=x_0^2+x_1^2
$$

at $(x_0,x_1)=(1.5,-2)$. Reverse-mode autodiff should report:

$$
\nabla L=(2x_0,2x_1)=(3,-4)
$$

For $x_0$, use $h=10^{-5}$:

$$
g_{num}
=\frac{(1.5+h)^2+(-2)^2-((1.5-h)^2+(-2)^2)}{2h}
\approx3
$$

The test constructs exactly this parameter Tensor, runs `backward`, probes all
three coordinates, compares the analytical values with the numerical values,
and finally asserts that the original parameter vector is unchanged.

The restoration assertion matters. A diagnostic that silently perturbs a model
can corrupt the training run it is supposed to investigate.

The checker also clears the analytical gradients it created. Run it between
training steps: it intentionally does not preserve gradients that were already
present when checking began, because silently combining training gradients with
diagnostic gradients would be more dangerous than establishing an empty
boundary.

## 5. Define a scale-aware pass condition

For analytical gradient $g_a$ and numerical gradient $g_n$:

$$
e_{abs}=|g_a-g_n|
$$

$$
s=\max(|g_a|,|g_n|)
$$

The probe passes when:

$$
e_{abs}\le a_{tol}+r_{tol}s
$$

The absolute term handles gradients near zero. The relative term allows error
to scale with a large gradient. Using only relative error makes a harmless
`1e-10` versus `0` comparison look infinite. Using only absolute error can be
too strict for a gradient of magnitude one million.

`GradientProbe` retains both gradients, flat and multidimensional coordinates,
absolute error, relative error, allowed error, and the ratio between observed
and allowed error. A failed report therefore identifies the exact parameter
element rather than returning one Boolean.

## 6. Sample coordinates without hiding ownership

Checking every coordinate costs two forward evaluations per element. Even this
tiny MiniGPT has enough parameters to make a full check slow. The checker
therefore selects at most `maximumCoordinatesPerParameter` indices, evenly
including the first and final coordinates.

For a parameter of size 10 and a maximum of 3, the selected flat indices are:

```text
0, 4, 9
```

This is deterministic and covers both ends, which often reveal offset and
shape-boundary bugs. It is not random statistical coverage and does not prove
every coordinate. During development, combine sampled full-model checks with
exhaustive checks on each small operation.

Every supplied parameter reference and label must be unique. Duplicate
references would probe and count tied storage twice. MiniGPT's `parameters`
method deliberately returns its tied token embedding only once.

## 7. Treat the loss as a fresh deterministic graph

The checker accepts `lossFactory: () => Tensor`, not a previously evaluated
loss Tensor. Every perturbation must build a new graph using the changed
parameter value:

```scala
GradientChecker.check(
  model.parameters,
  () => model.loss(inputs, targets),
  GradientCheckConfig(maximumCoordinatesPerParameter = 2)
)
```

The factory must be deterministic. Dropout, random data sampling, mutable
batch order, or asynchronous state changes make the plus/minus loss difference
measure two sources of variation at once. Freeze random seeds and data before
interpreting a gradient mismatch.

Non-smooth points also need care. ReLU has no unique derivative at zero. If a
perturbation crosses the boundary, the finite difference averages behavior on
both sides while this Tensor engine chooses the positive-side derivative only
for inputs greater than zero. That mismatch may describe the mathematical
point, not an implementation defect.

## 8. Inventory parameters before estimating memory

`ParameterInventory` records label, shape, elements, and dense payload bytes for
every owned trainable Tensor. With $P$ `Double` elements:

$$
M_{parameters}=8P\text{ bytes}
$$

That is *not* total training memory. In the current all-`Double` AdamW path, a
rough persistent-state decomposition is:

```text
parameters       8P
gradients        8P
Adam first       8P
Adam second      8P
-------------------
subtotal        32P bytes
```

This still excludes activations retained for backward, temporary operation
buffers, object/array headers, graph nodes, tokenizer/data memory, JVM runtime,
and allocator fragmentation. A payload inventory is useful precisely because
it names what is counted. Calling it “model memory” would be misleading.

Later distributed chapters will use the same decomposition to show which
states data parallelism duplicates and which ZeRO stages partition.

## 9. Benchmark an operation rather than a story

A defensible benchmark defines:

- the operation and input shapes;
- setup performed outside the timed region;
- warmup behavior;
- operations per measurement;
- number of raw samples;
- correctness/checksum evidence;
- runtime and hardware identity;
- statistics and variation;
- claims the experiment cannot support.

`Benchmark.measure` executes warmup batches with the same shape as measured
batches. Warmups are excluded from statistics, allowing JVM class loading and
just-in-time compilation to begin before sampling. Warmup is not a guarantee
that compilation has stabilized; inspect longer runs when making a serious
performance claim.

Each measured region can contain several operations. If a batch of three
forwards takes 900,000 ns, the stored sample is:

$$
900{,}000/3=300{,}000\text{ ns/op}
$$

Batching reduces timer-resolution overhead, but too much batching can hide
garbage-collection pauses or thermal behavior. Raw samples remain available.

## 10. Prevent unused work from disappearing

The benchmarked operation returns a `Long`. The MiniGPT lab computes logits,
sums their values, converts the `Double` bits into a `Long`, and lets the
harness mix that value into a checksum.

```scala
val logits = model.logits(inputs)
java.lang.Double.doubleToRawLongBits(logits.values.sum)
```

This makes the result observable and catches accidental workload changes. It
does not prove that every possible JVM optimization is defeated; JMH uses more
sophisticated machinery and isolated forks. The course harness exists to make
the necessary concepts visible before introducing that tool.

## 11. Read timing statistics correctly

The harness retains normalized samples and reports:

- minimum and maximum;
- median;
- nearest-rank p95;
- arithmetic mean;
- population standard deviation;
- coefficient of variation;
- derived operations per second.

For samples `[1,2,3,4]` ns/op:

$$
\operatorname{median}=\frac{2+3}{2}=2.5
$$

The nearest-rank p95 index is $\lceil0.95\times4\rceil-1=3$, so p95 is `4`.
Mean is `2.5`; population standard deviation is $\sqrt{1.25}$.

The coefficient of variation is:

$$
CV=\frac{\sigma}{\mu}
$$

A large `CV` warns that the environment or workload is unstable. It does not
explain why. Possible causes include JIT compilation, garbage collection,
other processes, CPU frequency changes, allocation bursts, input-dependent
branches, and insufficient batch size.

One local p95 over 30 samples is descriptive evidence for that run—not a
service-level percentile, a confidence interval, or a cross-machine claim.

## 12. Record the runtime fingerprint

`RuntimeFingerprint.current` records Java runtime version, VM name/version,
operating system/version, architecture, and available processor count. This is
the minimum context needed to avoid comparing unlabeled numbers.

A stronger performance artifact also records:

- repository commit;
- Nix flake revision;
- exact model configuration and seed;
- CPU model, frequency policy, and memory;
- JVM flags and heap size;
- process affinity and competing load;
- warmup/measurement sample vectors;
- profiler or hardware-counter evidence.

The chapter lab prints only what it actually observes. It makes no GPU or
frontier-throughput claim.

## Implementation walkthrough

### `BenchmarkStatistics.from`

The constructor rejects empty, non-positive, infinite, and `NaN` samples. It
sorts a copy for order statistics but calculates mean and variance from the
original vector. The report retains the original measurement order so a reader
can notice warmup drift or a late pause.

Median selects the middle element for odd counts and averages the two central
elements for even counts. Nearest-rank p95 uses `ceil(0.95 * n) - 1`, producing
a valid zero-based index for every positive `n`. Variance divides by `n`
because the value describes this observed vector; the harness does not claim
an unbiased population estimator.

### `Benchmark.measure`

The by-name `operation` is evaluated inside `runBatch` on every repetition.
Warmup calls `runBatch` without reading the clock. Each measurement reads the
monotonic clock immediately before and after one batch, rejects zero or
negative elapsed time, normalizes by batch size, and appends one sample.

`NanoClock` is a boundary rather than a direct static call. Tests supply exact
timestamps and prove that warmup is untimed, batching executes the expected
operation count, and durations normalize correctly.

### `GradientChecker.check`

The checker snapshots every parameter vector before doing any work. Inside a
`try` block it clears gradients, builds one scalar loss, calls `backward`, and
captures analytical gradients as it visits selected coordinates.

For each coordinate, `evaluatePerturbation` constructs a new vector with one
changed element, assigns it through the Tensor's validated mutation boundary,
builds a fresh loss, reads its scalar value, and restores the parameter in its
own `finally` block. The outer `finally` restores *all* snapshots as a second
line of defense if any factory or validation step fails, then clears every
diagnostic gradient.

The numerical derivative, errors, tolerance, coordinates, and labels become a
`GradientProbe`. `GradientCheckReport.worstProbe` selects the largest ratio of
observed error to allowed error, which is more informative than selecting only
the largest absolute difference.

### `ParameterInventory.from`

The inventory rejects empty input, non-positive element sizes, duplicate Tensor
references, and duplicate labels. Payload multiplication uses
`Math.multiplyExact`, so integer overflow becomes an explicit failure rather
than a wrapped negative byte count.

### `runMiniGptDiagnostics`

The lab constructs one seeded MiniGPT and one fixed token/target pair. It prints
parameter ownership, runs two probes per Tensor, then benchmarks deterministic
forward passes. Gradient checking happens before timing; the checker's final
restoration ensures the measured model retains its original weights.

## Reading the tests

`GradientCheckSuite` begins with the hand-computable quadratic. This is an
independent oracle: its gradient is known without trusting Tensor autodiff.
Another test deliberately uses a large `h` and zero tolerance on a cubic. The
expected `0.01` mismatch proves failed reports retain evidence rather than only
throwing or returning false.

The injected-failure test throws during the first perturbed loss construction,
then asserts exact parameter restoration. The sampled MiniGPT test traverses
the complete embedding, attention, feed-forward, normalization, tied-head, and
cross-entropy graph. It complements, but does not replace, exhaustive operation
checks in earlier suites.

`BenchmarkSuite` calculates statistics from `[4,1,3,2]` by hand. Its scripted
clock yields timed batches of 60 and 80 ns with two operations each, so samples
must be `[30,40]`. The operation counter proves one untimed warmup batch plus
two measured batches executed exactly six operations.

Failure tests reject invalid iteration counts and a non-increasing clock. A
zero-duration result is not silently converted into infinite throughput.

## Run and interpret the lab

```console
$ nix develop -c sbt 'runMain learnai.transformer.runMiniGptDiagnostics'
```

Inspect these fields in order:

1. **Parameter entries:** labels and shapes should correspond to Chapter 21's
   architecture and contain the tied embedding once.
2. **Gradient probes:** `passed=true`, maximum absolute error, and maximum
   error/tolerance ratio show sampled agreement under the printed
   configuration; they do not prove all inputs.
3. **Worst probe:** identify its parameter and coordinate before changing a
   tolerance.
4. **Median/p95/CV:** treat them as one local timing observation. Re-run and
   compare raw samples before making a regression claim.
5. **Checksum:** equal setup should retain equal work/output identity.
6. **Runtime:** never compare timings after discarding this context.

A passing gradient check establishes local agreement between reverse mode and
central differences for sampled coordinates. It does not establish model
quality, optimizer correctness over many steps, data validity, or production
performance.

## Debugging checklist

### Gradient mismatch

1. Reduce the problem to the smallest operation containing the parameter.
2. Print analytical, plus loss, minus loss, and numerical gradient.
3. Sweep `h` across `1e-2` to `1e-7`; look for a stable region rather than
   choosing the step with the smallest displayed error.
4. Check for ReLU or other non-smooth boundaries.
5. Freeze seeds, dropout, data order, and mutable caches.
6. Verify the perturbed parameter participates in the fresh graph.
7. Verify shared paths accumulate rather than overwrite gradients.
8. Change tolerance only after identifying the numerical error source.

### Suspicious timing

1. Confirm checksum and model configuration match the reference run.
2. Increase operations per measurement if samples approach timer resolution.
3. Increase warmup and plot samples in execution order.
4. Inspect allocation and garbage collection with JFR or a dedicated profiler.
5. Separate setup, tokenization, model forward, sampling, and output rendering.
6. Compare deterministic work counters before wall-clock time.
7. Use JMH with forks for a publishable JVM microbenchmark.
8. Report `No measurements found` if the required environment evidence is
   absent; do not infer speed from asymptotic complexity alone.

### Incorrect memory total

1. Decide whether the question concerns parameter payload, persistent training
   state, peak live tensors, process RSS, or device allocator reservations.
2. Check tied/shared parameter identity before summing.
3. Confirm bytes per element for each dtype.
4. Add gradients, optimizer moments, master weights, and activations as
   separate named terms.
5. Measure object and allocator overhead instead of hiding it in payload math.

## Professional workflow connection

Every later optimized or distributed path must keep two implementations:

```text
clear reference path
  + independent correctness oracle
  + optimized path
  + equivalence tests
  + workload counters
  + environment-labeled measurements
```

FlashAttention, quantized kernels, grouped-query caches, tensor-parallel
matmuls, and speculative decoding change execution structure. None may delete
the reference path merely because the new path appears faster.

Continuous integration should run deterministic correctness checks. Stable
dedicated runners should own performance gates. Mixing noisy laptop timing into
ordinary unit tests creates flaky builds and teaches the wrong evidence model.

## Exercises

1. Add a forward-difference mode and plot its error against central difference
   across powers of ten for `h`.
2. Add a random coordinate sampler with an explicit seed and retain sampled
   indices in the report.
3. Add a physical-byte training-state estimator for SGD and AdamW, keeping
   activations separate.
4. Export benchmark raw samples and runtime metadata as versioned JSON.
5. Run Java Flight Recorder around the lab and identify the largest allocation
   sites without changing the timed operation.
6. Compare full-prefix and cached forward latency only after asserting token and
   logit equivalence.
7. Implement bootstrap confidence intervals in a separate analysis layer and
   explain why correlated JVM samples violate simple independence assumptions.

## Completion criteria

You are done when you can:

- distinguish mathematical, numerical, and systems correctness;
- derive central difference and explain its competing error sources;
- select absolute and relative tolerances from scale and precision;
- diagnose a failed probe from retained coordinate-level evidence;
- prove parameter restoration on success and failure;
- separate dense parameter payload from total training memory;
- explain warmup, batching, observable results, raw samples, median, p95, and
  variation;
- state what the local benchmark does not prove;
- require reference equivalence before accepting an optimization.

## Primary sources and professional tools

- [FlashAttention](https://arxiv.org/abs/2205.14135) — an example of preserving
  exact attention mathematics while changing IO behavior and measuring the
  resulting kernels.
- [OpenJDK JMH](https://github.com/openjdk/jmh) — the production JVM benchmark
  harness to use after understanding the visible mechanics implemented here.
- [Course reading map](../09-papers/40-primary-reading-map.md) — architecture,
  systems, data, post-training, and evaluation sources connected to later
  milestones.
