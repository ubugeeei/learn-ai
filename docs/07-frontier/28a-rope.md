# 28a — Rotary position encoding (RoPE)

## What you will build

Chapter 18 gave MiniGPT a learned absolute position table, and Chapter 24
showed the operational price of that choice: when a full context window
slides, every retained hidden state is encoded at the wrong absolute
position, so the KV cache must be rebuilt. RoPE removes that coupling by
encoding position *inside* the attention score instead of inside the residual
stream.

In this chapter you will:

- derive why rotating query/key channel pairs produces relative-position
  attention scores;
- implement `RotaryPositionEncoding` from existing differentiable Tensor
  operations, with no new backward rule;
- implement `RotaryCausalSelfAttention`, a drop-in attention variant with
  exactly the same trainable parameters as Chapter 19;
- prove norm preservation, offset invariance, and gradient correctness with
  independent oracles.

## Prerequisites

You should understand Chapters 18–19 (embeddings and causal attention),
Chapter 12 (Tensor autodiff), and Chapter 24 (why absolute positions force a
cache rebuild).

## 1. Position as rotation

Group a head's channels into pairs and read each pair as a point in the
plane. RoPE rotates pair $j$ of the row at absolute position $m$ by the angle
$m\,\theta_j$, where

$$
\theta_j = b^{-2j/d}, \qquad j \in \{0, \dots, d/2 - 1\}
$$

for head dimension $d$ and frequency base $b$ (conventionally $10000$). Low
$j$ pairs spin quickly and resolve nearby offsets; high $j$ pairs spin slowly
and resolve distant offsets, exactly like digits of a positional numeral
system.

Writing the rotation of pair $j$ at position $m$ as the matrix
$R_{m,j} \in \mathbb{R}^{2\times 2}$, the rotated query/key rows are

$$
\tilde q_m = R_m q_m, \qquad \tilde k_n = R_n k_n
$$

and the attention score becomes

$$
\tilde q_m^\top \tilde k_n = q_m^\top R_m^\top R_n\, k_n
  = q_m^\top R_{n-m}\, k_n
$$

because rotations compose by adding angles. The score depends on positions
only through the offset $n - m$. Two consequences follow immediately:

1. shifting an entire window by a constant offset changes no score, so a
   sliding KV cache never needs the Chapter 24 rebuild;
2. RoPE has zero trainable parameters, so swapping it into a model changes
   position behavior without changing parameter count.

Values are never rotated. Position must influence *where* attention looks,
not *what* it retrieves; rotating values would corrupt the retrieved content.

## 2. Pair layout

This implementation pairs channel $j$ with channel $j + d/2$ (the
"rotate-half" layout used by GPT-NeoX and Llama). The original RoFormer paper
pairs adjacent channels $(2j, 2j+1)$. The two layouts differ only by a fixed
channel permutation, so they are equivalent as long as queries and keys use
the same one. Mixing layouts between query and key is a silent correctness
bug; the offset-invariance test would catch it.

## 3. Composition instead of a new operator

`rotate` needs a backward rule, but not a new one. For input halves $x_1$,
$x_2$ (each `[time, d/2]`) and constant tables $\cos$, $\sin$ of the same
shape, the rotated output is

$$
\left(\, x_1 \odot \cos - x_2 \odot \sin,\;\;
   x_1 \odot \sin + x_2 \odot \cos \,\right)
$$

which is a composition of `sliceColumns`, `hadamard`, addition, and
`concatenateColumns` — all existing Tensor operations with tested backward
rules. Gradients flow to the input automatically, and the finite-difference
test verifies the composition end to end. This is a recurring engineering
lesson: prefer composing verified primitives over writing new mutable
kernels, and only specialize once a measurement demands it.

## 4. Shapes

```text
input row (one head):   [time, d]
first half x1:          [time, d/2]
second half x2:         [time, d/2]
cos/sin tables:         [time, d/2]   constants, no gradient
rotated output:         [time, d]
```

`rotate(input, startPosition)` treats row `t` as absolute position
`startPosition + t`. A cached decoder can therefore rotate one appended row
at its true position without touching the past.

## 5. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `RotaryPositionEncoding` suite prints one line per oracle. To see the
offset invariance directly, evaluate any input at `startPosition = 0` and
`startPosition = 100` and compare outputs; they agree to floating-point
rounding.

## 6. Implementation walkthrough

`RotaryPositionEncoding` validates an even head dimension and a positive
finite base at construction, then precomputes `frequencies` as the vector
$b^{-2j/d}$. `angle(position, pair)` exposes the raw angle so tests can
verify the schedule against hand calculations rather than trusting the
rotation code.

`rotate` first validates rank, width, and a non-negative start position, and
uses `Math.addExact` so a pathological `startPosition` near `Int.MaxValue`
fails loudly instead of overflowing silently. It then builds the cosine and
sine tables as *constant* tensors with `Tensor.tabulate`; because constants
are not trainable and have no inputs, the autodiff graph treats them as
leaves and sends no gradient their way. The two input halves come from
`sliceColumns`, the four elementwise products from `hadamard`, and the final
reassembly from `concatenateColumns`. Nothing in this path mutates the input.

`RotaryCausalSelfAttention` mirrors Chapter 19's attention exactly — same
four `Linear` projections, same per-head channel slices, same causal mask and
row softmax — with one insertion: each head's query and key slices pass
through `rope.rotate` before scores. The value path is untouched. Because the
rotation happens per head after slicing, every head shares one
`RotaryPositionEncoding` sized to `headChannels`, and the constructor rejects
a RoPE whose width disagrees with the head width.

The `startPosition` parameter threads through `apply` and
`forwardWithWeights` so later chapters can decode with a cache at absolute
positions without re-deriving the layer.

## 7. Reading the tests

The suite deliberately uses independent oracles rather than comparing two
implementations that could share a bug:

- *hand calculation*: with $d = 2$ and $\theta_0 = 1$, the unit vector
  rotated at position $m$ must equal $(\cos m, \sin m)$ from `math.cos`;
- *schedule check*: $b = 100$, $d = 4$ gives $\theta_1 = 0.1$ exactly;
- *identity*: position zero must return the input unchanged;
- *isometry*: every pair's `hypot` norm is preserved at arbitrary positions;
- *relative property*: rotated dot products at positions $(9+s, 4+s)$ match
  $(9, 4)$ for several shifts $s$;
- *gradient check*: reverse-mode gradients through the full composition match
  central finite differences per element;
- *attention level*: causal masking still holds, a common offset leaves the
  full output unchanged, and the parameter count equals standard attention;
- *failure modes*: odd head width, non-positive base, negative positions,
  wrong input width, and head/RoPE width mismatch are all rejected with
  specific messages.

## 8. Debugging checklist

1. Print `frequencies` first; a wrong exponent sign makes distant pairs spin
   fastest and ruins long-context behavior while short tests still pass.
2. Verify position zero is an exact identity before anything else.
3. Check pair layout symmetry: query and key must use the same pairing.
4. Test one pair at $d = 2$ against `math.cos`/`math.sin` by hand.
5. If attention output changes under a common offset, a path is rotating with
   absolute-only semantics — usually a table built from row index instead of
   `startPosition + row`.
6. If gradients explode, confirm the cos/sin tables are constants and not
   accidentally trainable parameters.

## 9. Failure modes to test

- odd head dimension (pairs undefined);
- zero, negative, or non-finite frequency base;
- negative or overflowing start position;
- input width that is not the RoPE width;
- values accidentally rotated (offset invariance still holds, but retrieval
  content is corrupted — compare against Chapter 19 outputs at position 0);
- query rotated with a different layout than key.

## Exercises

1. Implement the interleaved $(2j, 2j+1)$ layout and show both layouts give
   identical attention scores after a fixed permutation of projection rows.
2. Add RoPE to the Chapter 24 inference session and demonstrate window
   sliding without a rebuild, using cached/full-prefix equivalence as the
   oracle.
3. Implement position interpolation: evaluate at fractional positions
   $m / s$ and measure long-context loss versus extrapolation.
4. Measure how much of the rotation cost disappears when cos/sin tables are
   cached across layers with the same head width.

## Completion criteria

You are done when you can:

- derive $\tilde q_m^\top \tilde k_n = q_m^\top R_{n-m} k_n$ from rotation
  composition;
- explain the frequency schedule and what low/high pairs each resolve;
- explain why values are not rotated and what breaks if they are;
- state why RoPE adds no parameters and what that means for ablations;
- connect offset invariance to KV-cache window sliding in Chapter 24.

## Primary sources

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
