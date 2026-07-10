# 19 — Scaled dot-product causal self-attention

## What you will build

Query, key, and value projections; scaled scores; causal masking; row softmax;
multi-head split/concatenation; and output projection. Source:
`src/main/scala/learnai/transformer/Attention.scala`.

Tests verify normalized rows, zero future weights, prefix invariance, and zero
future-input gradient from a prefix-only loss.

## What attention solves

An initial token representation contains only its token and position. Context
requires selecting earlier positions based on content. From each hidden vector
\(x_t\), create:

\[
q_t=x_tW_Q,\qquad k_t=x_tW_K,\qquad v_t=x_tW_V
\]

- query: what this position seeks;
- key: what a position can match;
- value: information transferred after selection.

Queries and keys choose where to read; values determine what is read.

## Trace the shapes

For one sequence, time \(T\), and channels \(C\):

```text
X: [T,C]
Wq, Wk, Wv: [C,C]
Q, K, V: [T,C]
scores = Q K^T: [T,T]
weights = softmax(mask(scores)): [T,T]
output = weights V: [T,C]
```

A score row is the reading query position; a column is the candidate key
position.

## Dot-product similarity

\[
s_{ij}=q_i\cdot k_j
\]

Training can align query and key directions for useful relationships. Dot
product includes magnitude and therefore is not identical to cosine similarity.

## Why divide by \(\sqrt d\)?

For head width \(d\), independent unit-variance channel products make dot-
product variance grow with \(d\). Large scores saturate softmax and shrink useful
gradients. Scale them:

\[
s_{ij}=\frac{q_i\cdot k_j}{\sqrt d}
\]

This keeps initial score scale more stable across head widths.

## Causal mask

Position \(i\) must not read future position \(j>i\):

```text
query 0: [0,-,-]
query 1: [0,1,-]
query 2: [0,1,2]
```

\[
\tilde s_{ij}=\begin{cases}
s_{ij}&j\le i\\
-\infty&j>i
\end{cases}
\]

The implementation uses finite `-1e9` to preserve finite-value invariants. It
underflows to zero probability after stable softmax. Backward also blocks the
upper triangle.

Without the mask, training can read the target token already present at the
next input position, creating direct leakage.

## Row softmax and value aggregation

\[
a_{ij}=\frac{e^{\tilde s_{ij}}}{\sum_r e^{\tilde s_{ir}}}
\]

Each row is a distribution over allowed keys. Output is a weighted sum:

\[
o_i=\sum_j a_{ij}v_j
\]

Softmax backward avoids building a Jacobian. For output probabilities \(y_i\)
and upstream gradient \(g_i\):

\[
\frac{\partial L}{\partial x_i}
=y_i\left(g_i-\sum_jg_jy_j\right)
\]

The row gradient sums to zero because adding a constant to all logits changes
no softmax probability.

## Multi-head attention

Split channels into \(H\) heads with width \(d=C/H\):

```text
project Q/K/V: [T,C]
split:         H x [T,d]
attention:     H x [T,d]
concatenate:   [T,C]
project:       [T,C]
```

Different heads can learn different relationships. The architecture does not
preassign human-readable roles. `C` must be divisible by `H`.

The teaching engine slices columns explicitly. Production kernels reshape to
`[batch,head,time,d]` and use batched matrix multiplication.

## Three causality tests

1. **Weight:** strict upper triangle is zero.
2. **Forward:** changing future inputs cannot change earlier outputs.
3. **Backward:** a prefix-only loss gives zero future-input gradient.

Each catches a different class of masking defect.

## Complexity

Attention costs \(O(T^2C)\) time and \(O(T^2)\) weight memory per head/batch.
Doubling context quadruples score elements.

Flash Attention preserves exact attention while tiling and computing softmax
online so the full score matrix need not be materialized in high-bandwidth
memory.

## Implementation walkthrough

`CausalSelfAttention.forwardWithWeights` begins with shape and non-empty-time
validation. It applies four dense layers shared across positions: query, key,
value, and final output projections. Q, K, and V all have shape `[T,C]`.

For each head, `from` and `until` select one contiguous channel slice of width
`D = C/H`. The head calculation is:

```scala
val scores = headQuery.matmul(headKey.transpose2D).scale(scale)
val weights = scores.causalMask().softmaxRows
val headOutput = weights.matmul(headValue)
```

Trace shapes rather than memorizing code:

```text
Qh [T,D] x Kh^T [D,T] -> scores [T,T]
weights [T,T] x Vh [T,D] -> head output [T,D]
H head outputs concatenated -> [T,C]
output projection -> [T,C]
```

The causal mask replaces entries where key position `j > i` before softmax.
Its backward passes gradients only through allowed cells. Softmax is row-wise
because each query position needs one distribution over key positions.

`Tensor.concatenateColumns` restores original channel order. Concatenating rows
instead would produce a plausible element count with the wrong semantics. The
output projection mixes information across heads after concatenation.

The cached method implements the same equation for one query row. It appends
current key/value first, computes per-head scores against cache positions,
subtracts the maximum, normalizes, writes weighted head channels, and applies
the same output projection. It intentionally detaches cached arrays from
autodiff because generation is forward-only.

## Reading the tests

Row sums and exact future zeros verify mask/softmax order. Changing future input
while comparing an earlier output tests causality without inspecting weights.
The prefix-only loss gradient test is stronger: future input gradients must be
exactly zero. Head split tests use a width divisible by three. Cached attention
is compared at every prefix against the independent full path. Invalid head
division and cache overflow cover construction/runtime boundaries.

## Debugging checklist

1. Write `[T,C]`, `[T,D]`, and `[T,T]` beside every intermediate.
2. Inspect one attention row before combining heads.
3. Verify masking precedes softmax and uses key index greater than query index.
4. Check every row sum and future cell.
5. If cached/full results differ, compare projected Q, appended K/V, score
   order, position length, and output projection in that order.

## Exercises

1. Write the causal mask for `T=4`.
2. Compute one head by hand from chosen Q/K/V.
3. Remove scaling and measure softmax entropy as head width grows.
4. Show why masking after softmax breaks row normalization.
5. Derive softmax backward from its Jacobian.
6. Explain why changing head count need not change Q/K/V parameter count.
7. Explain why attention weights are not a complete model explanation.

## Completion criteria

- Explain Q, K, and V roles and shapes.
- Interpret both axes of `[T,T]` scores.
- Explain variance scaling by \(1/\sqrt d\).
- Explain why masking occurs before softmax.
- Trace multi-head split and concatenation.
- All attention causality and gradient tests pass.

## Primary source

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Course reading map and critical summary](../09-papers/40-primary-reading-map.md)
