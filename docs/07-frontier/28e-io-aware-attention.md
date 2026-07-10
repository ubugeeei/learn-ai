# 28e — IO-aware attention

## What you will build

Chapter 29a identified the quadratic term: a full forward at length $t$
materializes $L H t^2$ attention weights. FlashAttention's contribution
is not a new attention — the outputs are mathematically identical — but a
new *execution order* that never holds a full score row, because softmax
can be computed as a streaming recurrence. The insight generalizes far
beyond GPUs: whenever a reduction looks like it needs all its inputs at
once, look for a running state that makes it incremental.

In this chapter you will:

- derive the online softmax recurrence and prove its exactness;
- implement tiled causal attention that materializes at most one tile of
  scores plus one accumulator per query row;
- verify equality against two independent references, including at
  scores that overflow a naive implementation;
- quantify the memory trade with Chapter 29a style accounting.

## Prerequisites

You should understand Chapter 19 (the materializing attention being
reproduced), Chapter 4 (floating-point overflow and stabilization), and
Chapter 29a (the quadratic term being removed).

## 1. Softmax as a running state

The stable softmax over scores $s_1 \dots s_n$ needs the global maximum
$m$ and the denominator $d = \sum_i e^{s_i - m}$. Both look global —
but both fold. Maintain $(m, d)$ over any prefix; when a new tile with
local maximum $\tilde m$ and local sum $\tilde d$ arrives:

$$
m'' = \max(m, \tilde m), \qquad
d'' = d\,e^{m - m''} + \tilde d\,e^{\tilde m - m''}
$$

Both exponents are $\le 0$, so both rescales are in $(0, 1]$: *nothing is
ever exponentiated above zero*, which is why scores of $10^4$ — instant
overflow for naive $e^{s}$ — pass through exactly. The recurrence is
associative in the sense the suite tests literally: folding one tile,
tiles of two, or single scores yields the same state.

## 2. From softmax to attention without the score row

Attention needs $\sum_i \operatorname{softmax}(s)_i\, v_i$, and the
weights depend on the not-yet-final denominator. The fix is to keep the
value accumulator *unnormalized* alongside $(m, d)$:

$$
a'' = a\,e^{m - m''} + \sum_{i \in \text{tile}} e^{s_i - m''} v_i
$$

and divide once at the end: $o = a / d$. Every earlier contribution is
corrected by the same rescale factor the denominator receives, so the
final quotient equals the materializing computation exactly — not
approximately. Per query row, the peak live floats are

```text
tileSize (scores) + channels (accumulator) + 2 (m, d)
```

independent of context length, versus $t$ scores for the materializing
row — and $L H t^2$ once multiplied out across rows, heads, and layers.

## 3. What this simulation is and is not

`TiledAttention` executes the exact algorithm on the CPU in plain Scala.
What it demonstrates: the recurrence, its exactness, its stability, and
its memory structure. What it does not demonstrate: the speedup. On real
hardware the win comes from executing the recurrence inside on-chip
SRAM instead of round-tripping the score matrix through HBM — an IO
argument, hence the name. Claiming a wall-clock win from this Scala
simulation would violate Chapter 22's measurement rules; the honest
claim is algorithmic equivalence plus a bounded materialization count,
and both are tested.

The module is also forward-only on purpose. The backward pass of
FlashAttention recomputes tiles instead of storing them (a second
insight — trading FLOPs for memory), and the trainable path in this
course remains Chapter 19's; equality with it is the oracle, and a
tiled backward is an exercise.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `TiledAttention` suite includes the overflow case directly: scores
around $10^4$ with tile boundaries splitting them, compared against a
two-pass stable reference to $10^{-14}$.

## 5. Implementation walkthrough

`OnlineSoftmaxState` holds $(m, d)$ with an `empty` state of
$(-\infty, 0)$; `foldTile` implements the Section 1 recurrence with the
zero-denominator first-tile case handled explicitly rather than relying
on `exp(-∞)` conveniently evaluating to zero. `softmaxStreamed` folds
tiles then normalizes in a second pass — it exists to isolate and test
the normalizer recurrence on its own.

`attendRowTiled` is the fused version: per tile it computes at most
`tileSize` scores, merges the maximum, rescales both the denominator
*and* the channel accumulator by one shared factor, then adds the
tile's weighted values. The array writes use `while` loops in the same
style as Chapter 19's cached path. `causalAttentionTiled` calls the row
routine with `keys.take(t + 1)` — causality by construction, so there is
no masked matrix and no mask to get wrong; the suite still verifies it
behaviorally by perturbing future rows.

`materializedFloatsPerRow` is the accounting hook, matching Chapter
29a's convention of anchoring memory claims to countable quantities.

## 6. Reading the tests

- streamed softmax versus an in-test two-pass reference across tile
  sizes 1, 2, 3, 5, 11, and 64 — including tiles larger than the input;
- the overflow test uses scores a naive implementation cannot survive
  and additionally asserts finiteness;
- the fold-associativity test compares whole/pairs/singles folding of
  one score vector into identical states;
- one tiled row versus a naive full-row reference for every tile size
  from 1 to length + 2;
- the full tiled attention versus the *Tensor graph* — matmul, scale,
  causal mask, row softmax, matmul — a genuinely independent
  implementation from Chapters 12 and 19;
- behavioral causality and materialization accounting;
- rejection of empty tiles, NaN scores, zero tile sizes, and mismatched
  widths and row counts.

## 7. Debugging checklist

1. Test the normalizer recurrence alone (`softmaxStreamed`) before the
   fused accumulator; two moving parts hide each other's bugs.
2. If results drift only across tile boundaries, the accumulator and
   denominator are being rescaled by different factors.
3. If large scores produce infinities, something exponentiates before
   subtracting the merged maximum.
4. Check the first-tile case explicitly: $(-\infty, 0)$ must not leak a
   `0 * exp(-∞)` NaN into the accumulator.
5. Compare against tile size 1 and tile size $\ge$ length first — the
   two degenerate schedules bracket every indexing bug.

## 8. Failure modes to test

- accumulator rescaled by the old maximum instead of the merged one;
- denominator updated before the accumulator rescale (order matters);
- the query's own position dropped from its causal prefix;
- tile boundaries off by one at exactly `tileSize` positions;
- NaN scores silently propagating instead of being rejected;
- performance claims made from this CPU simulation.

## Exercises

1. Implement the tiled *backward* pass with tile recomputation and
   verify its gradients against Chapter 19's graph by finite
   differences.
2. Extend the row routine to grouped-query attention (Chapter 28c):
   share key/value tiles across a query-head group and re-run both
   oracles.
3. Count exact FLOPs of tiled versus materializing attention (Chapter
   29a conventions) and show the tiled version pays a small
   recomputation premium for its memory bound.
4. Simulate a two-level memory (fast tile buffer, slow main array) by
   counting array reads, and reproduce the paper's IO-complexity
   argument numerically.

## Completion criteria

You are done when you can:

- write the $(m, d, a)$ recurrence from memory and prove the final
  quotient is exact;
- explain why no positive number is ever exponentiated and what that
  buys;
- state what this simulation demonstrates and what only hardware can;
- give the peak-materialization formula and connect it to Chapter 29a's
  quadratic term;
- explain why the causal prefix construction needs no mask.

## Primary sources

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Online normalizer calculation for softmax (Milakov, Gimelshein)](https://arxiv.org/abs/1805.02867)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
