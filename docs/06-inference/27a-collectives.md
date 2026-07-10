# 27a — Collectives and data parallelism

## What you will build

One device stops being enough long before frontier scale: the batch that
saturates a model's quality does not fit one machine's throughput. Data
parallelism is the first and simplest answer — replicate the model, shard
every batch, and agree on one gradient — and it introduces the two
concepts every later distribution chapter builds on: *collectives* (the
agreement primitives) and *replica equivalence* (the invariant that makes
distributed bugs detectable).

In this chapter you will:

- implement deterministic simulated collectives with byte-level traces;
- train MiniGPT synchronously across simulated ranks with one fused
  gradient all-reduce per update;
- prove replicas stay bitwise identical, and learn precisely why
  equality with the single-process trainer is rounding-level, not
  bitwise — a floating-point lesson this chapter's tests encode;
- account for communication with the ring all-reduce formula.

## Prerequisites

You should understand Chapter 22b/22c (the loop being parallelized and
its microbatch scaling), Chapter 13 (why identical gradients imply
identical AdamW updates), and Chapter 4 (floating-point non-associativity,
which returns here with consequences).

## 1. The synchronous step

With world size $N$, every update does:

```text
each rank r:  forward/backward on its shard  ->  partial gradient g_r
all ranks:    g = allReduceSum(g_0 .. g_{N-1})     (one fused bucket)
each rank r:  identical AdamW step with g
```

The scaling discipline from Chapter 22b carries over unchanged: each
microbatch loss is scaled by `microBatchSize / batchSize` *before*
backward, so the all-reduce **sum** of partial gradients is already the
mean-over-batch gradient. Where the $1/B$ lives is the classic
data-parallel bug: scale in both places and you train at $1/N$-th the
intended learning rate; scale in neither and at $N$ times.

## 2. Replica identity is the invariant

Floating-point addition is not associative, and real networks deliver
contributions in nondeterministic order — which is why production
frameworks fix a reduction order. This simulation does the same:
contributions combine in rank order, always. The consequence is the
chapter's central invariant, asserted bitwise by the suite: after any
number of updates, every replica's weights are *bit-for-bit identical*.
Divergence between replicas is never noise; it is a bug in the reduction,
the scaling, or the optimizer state, and bitwise testing catches it on
the first update instead of after a week of silent drift.

## 3. Why single-process equality is only rounding-level

The suite deliberately does *not* assert bitwise equality against the
Chapter 22c trainer, and the reason is instructive. Sequential microbatch
accumulation interleaves additions at the *element* level: within one
backward pass a parameter element receives many contributions, and the
next microbatch's backward adds its contributions into the same running
value. Ranks instead complete their partial gradients first, and the
reduction adds finished partials:

$$
\underbrace{(((a_1 + a_2) + b_1) + b_2)}_{\text{sequential}}
\;\ne\;
\underbrace{(a_1 + a_2) + (b_1 + b_2)}_{\text{reduced}}
\quad\text{in floating point}
$$

Same terms, different association, last-ulp differences that compound
through training. Chasing bitwise equality *across* parallelism
strategies is a losing game; the honest contract — encoded in the tests —
is rounding-level equivalence across strategies and bitwise identity
*within* one. This is exactly the reasoning you will need again when
tensor and pipeline parallelism reorder reductions further.

## 4. Communication accounting

A chunked ring all-reduce splits the buffer into $N$ chunks and runs
reduce-scatter then all-gather, each moving $N - 1$ chunks per rank:

$$
\text{bytes per rank} = 2\,(N-1)\,\left\lceil \tfrac{E}{N} \right\rceil \cdot 8
\;\xrightarrow{N \to \infty}\; 2 \cdot E \cdot 8
$$

Per-rank cost is nearly independent of world size — the property that
makes ring collectives scale. Every simulated collective records a trace
with both the logical payload and this ring figure, and the training loop
performs exactly one fused gradient bucket (all parameters flattened in
order) plus one scalar loss reduction per update: latency amortization by
bucketing, the same reason real frameworks fuse gradients into
tens-of-megabytes buckets.

## 5. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `DataParallel` suite trains world sizes 1-equivalent and 2 against
the Chapter 22c trainer and asserts the trace ledger update by update.

## 6. Implementation walkthrough

`Collectives` validates world size and contribution shapes, combines in
rank order with plain `while` loops, and appends a `CollectiveTrace` per
call. `ringAllReduceBytesPerRank` implements the chunked formula with
ceiling division; world size one moves zero bytes, which the suite pins.

`DataParallelMiniGpt.train` builds `worldSize` replicas from one seed —
determinism makes replication trivial — plus one AdamW per rank and one
shared `SplitMix64` standing in for identically seeded data loaders. Per
update it samples the global batch exactly as Chapter 22c does, slices
contiguous microbatch runs per rank, and lets each rank accumulate its
partial gradient with the Chapter 22b scaling. Gradients are flattened
into a single bucket in parameter order, all-reduced once, written back
with `Tensor.assignGradients` (added for exactly this purpose: the
reduction happens outside the graph), and every replica steps its own
optimizer with the same learning rate from the shared schedule.

The `microBatchesPerUpdate % worldSize == 0` requirement keeps shards
balanced; unbalanced sharding is a real topic (stragglers) but a
different chapter's.

## 7. Reading the tests

- the equivalence test compares losses, gradient norms, and final
  weights against the single-process trainer at tolerance $10^{-12}$ /
  $10^{-10}$, with the association argument written in the comment —
  a deliberate non-bitwise assertion with its reason attached;
- the replica-identity test uses several microbatches per rank and
  demands bitwise equality across replicas;
- collectives are unit-tested with hand values, rank-order semantics,
  and trace field checks;
- the ring formula is pinned by hand at $N = 4, E = 10$ (144 bytes),
  at $N = 1$ (zero), and bounded near $2E$ bytes for large $N$;
- the ledger test counts exactly two traces per update and matches the
  gradient bucket's element count to the model's parameter count;
- failure tests cover invalid world sizes, ragged contributions,
  indivisible shards, and misused gradient assignment.

## 8. Debugging checklist

1. Assert replica identity bitwise after *one* update before anything
   else; every later symptom is downstream of it.
2. If replicas diverge, print the reduction order; "sum as they arrive"
   is the classic nondeterminism.
3. If loss scales wrongly with world size, find where $1/B$ is applied —
   it must appear exactly once.
4. Compare the gradient bucket's element count to the parameter count;
   a missing tensor trains locally and silently diverges.
5. When comparing against a single-process run, decide the tolerance
   *from the association argument*, not from what happens to pass.

## 9. Failure modes to test

- reduction order dependent on arrival (replica divergence);
- double or missing $1/B$ scaling;
- a parameter tensor omitted from the bucket;
- optimizer state allowed to differ across ranks;
- per-collective dispatch without bucketing (trace ledger explodes);
- asserting bitwise cross-strategy equality and "fixing" the code until
  it passes.

## Exercises

1. Add a `reduceScatter` and `allGather` pair and rebuild `allReduceSum`
   from them, keeping the trace ledger consistent.
2. Simulate a straggler: delay one rank's contribution and measure the
   synchronous step's idle fraction across a distribution of delays.
3. Implement gradient-bucket splitting (two buckets instead of one) and
   verify replica identity still holds bitwise.
4. Extend the Chapter 22c training bundle to restore a data-parallel
   run: what extra state, if any, must be saved per rank?

## Completion criteria

You are done when you can:

- write the synchronous step and say where $1/B$ lives and why sum, not
  mean, is reduced;
- state the replica-identity invariant and use it as the first debugging
  step;
- explain, with the association argument, why cross-strategy equality is
  rounding-level and within-strategy identity is bitwise;
- derive the ring all-reduce byte formula and its large-$N$ limit;
- explain why gradients are bucketed before reduction.

## Primary sources

- [Bringing HPC Techniques to Deep Learning (ring all-reduce, Baidu)](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
