# 28b — SwiGLU gated feed-forward

## What you will build

Chapter 20's feed-forward network is the classic two-layer block:
expand, clip at zero with ReLU, project back. Nearly every frontier model
since PaLM and Llama replaces it with a *gated linear unit* variant, most
commonly SwiGLU. The change looks small on paper and is easy to get subtly
wrong in three places: the activation itself, the extra weight matrix, and —
most often — the ablation, because SwiGLU at the same hidden width is simply
a bigger network and any comparison at equal width is unfair.

In this chapter you will:

- implement SiLU by composition, without a new autodiff rule;
- implement `SwiGluFeedForward` with separate gate, up, and down projections;
- derive the exact parameter-matching formula behind the "two-thirds" rule;
- verify the activation, the gate mechanics, gradients, and end-to-end
  learning with independent oracles.

## Prerequisites

You should understand Chapters 11–12 (MLPs and Tensor autodiff) and
Chapter 20 (the ReLU feed-forward inside a Transformer block).

## 1. From clipping to gating

The baseline network computes, for each time row independently,

$$
\mathrm{FFN}(x) = W_2\,\max(0,\; W_1 x + b_1) + b_2
$$

ReLU makes a binary decision per hidden channel: keep or delete. A gated
linear unit splits the hidden computation into two parallel projections and
multiplies them:

$$
\mathrm{SwiGLU}(x) = W_3\,\bigl(\operatorname{SiLU}(W_1 x + b_1)
  \odot (W_2 x + b_2)\bigr) + b_3
$$

The *up* path $W_2 x$ carries content linearly. The *gate* path decides,
per hidden channel and per token, how much of that content flows through —
a continuous, learned, input-dependent volume knob rather than a hard clip.
The GLU-variants paper reports consistent quality gains at matched compute,
with the honest admission that "we offer no explanation as to why these
architectures seem to work"; the mechanism above is the standard reading.

SiLU (also called swish) is

$$
\operatorname{SiLU}(z) = z\,\sigma(z) = \frac{z}{1 + e^{-z}}
$$

which is smooth, non-monotonic near zero, and unbounded above — unlike ReLU
it passes small negative values with small weight instead of erasing them.

## 2. Composition instead of a new operator

The Tensor library has `tanh` but no `sigmoid`. Rather than adding a new
backward rule, this chapter uses the identity

$$
z\,\sigma(z) = \frac{z}{2}\left(1 + \tanh\frac{z}{2}\right)
$$

so SiLU is `half + half.hadamard(half.tanh)` with `half = x.scale(0.5)`.
Every piece already has a tested gradient, and `tanh` saturates gracefully
where a naive `exp(-z)` would overflow for large negative `z`. The test
suite still checks the composition against the direct
$z / (1 + e^{-z})$ formula at hand-picked points, because an identity you
did not verify is just a hope.

## 3. Parameter accounting and the two-thirds rule

Counting every trainable element for input/output width $C$ and hidden
width $H$ (ReLU) or $S$ (SwiGLU), biases included:

$$
N_{\mathrm{relu}} = 2CH + H + C, \qquad
N_{\mathrm{swiglu}} = 3CS + 2S + C
$$

Setting them equal gives

$$
S = H\,\frac{2C + 1}{3C + 2} \;\xrightarrow{\;C \to \infty\;}\; \frac{2H}{3}
$$

This is where the folklore "use two-thirds of the hidden width" comes from:
it is the exact solution's large-width limit. The implementation exposes the
exact formula so ablations are parameter-matched by construction, and the
test asserts the rounded result beats its neighbors $S \pm 1$.

## 4. Shapes

```text
input:            [time, C]
gate(x), up(x):   [time, S]
silu(gate) ⊙ up:  [time, S]
down(...):        [time, C]
```

All three projections are ordinary `Linear` layers; the network preserves
`[time, C]` exactly like the ReLU version, so it can replace the
feed-forward inside a Transformer block without touching anything else.

## 5. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `SwiGluFeedForward` suite includes a small end-to-end optimization: an
XOR-patterned regression that linear layers cannot represent, trained with
`TensorSgd` for 300 steps, asserting at least a tenfold loss reduction from
a fixed seed. It is a learning-signal smoke test, not a benchmark.

## 6. Implementation walkthrough

`SwiGluFeedForward` validates positive widths at construction and the input
shape `[time, C]` at every call, matching the failure discipline of the rest
of the codebase. `apply` evaluates the gate projection, passes it through
the composed SiLU, multiplies elementwise with the up projection via
`hadamard` (which requires exactly equal shapes — a silent-broadcast bug is
impossible here), and finishes with the down projection.

`silu` lives on the companion object because it is a pure function of its
input, not layer state. The `fromValues` constructor mirrors `Linear` and
exists so tests can pin exact weights: the "zero gate" test builds a network
whose gate weights and biases are all zero, which forces
$\operatorname{SiLU}(0) = 0$, annihilates the up path, and leaves exactly
the down-projection bias — a hand-computable output that fails if the gate
is applied to the wrong path.

`parameterMatchedHiddenChannels` implements the Section 3 formula with
rounding and a floor of one. It intentionally returns a width, not a
network, so callers can log the achieved parameter difference before
committing to an ablation.

## 7. Reading the tests

- *activation oracle*: the tanh composition equals $z/(1+e^{-z})$ at six
  points including a large positive value;
- *gate mechanics*: a zero gate reduces the network to its down bias, by
  hand calculation;
- *shape and count*: output shape is preserved and `parameterCount` equals
  $3CS + 2S + C$;
- *matched ablation*: the matched width for $C = 8$, $H = 32$ is 21, its
  count is closer to the real `FeedForward` baseline than 20 or 22, and the
  ratio lands near two-thirds;
- *gradient check*: central finite differences over every input element
  match reverse-mode gradients through the full gate-times-up composition;
- *end-to-end learning*: the XOR regression loss drops more than tenfold;
- *failure modes*: zero widths and wrong input width are rejected.

## 8. Debugging checklist

1. Verify $\operatorname{SiLU}(0) = 0$ and $\operatorname{SiLU}(z) \to z$
   for large $z$ before testing anything downstream.
2. Confirm SiLU is applied to the *gate* path, not the up path; both choices
   type-check and train, but they are different architectures.
3. Check the hidden widths of gate and up agree; `hadamard` will refuse
   mismatched shapes loudly — let it.
4. When an ablation shows SwiGLU "winning", print both parameter counts
   first; an unmatched comparison is the most common false positive.
5. If gradients vanish early in training, inspect the gate pre-activations:
   a badly scaled gate initialization saturates SiLU's flat region.

## 9. Failure modes to test

- zero or negative channel/hidden widths;
- input width that does not match the network;
- SiLU applied to both paths (output no longer matches the zero-gate hand
  calculation);
- gate and up projections accidentally sharing one weight tensor (parameter
  count drops; the count test catches it);
- comparing SwiGLU and ReLU networks at equal hidden width and calling the
  result an ablation.

## Exercises

1. Swap `SwiGluFeedForward` into `TransformerBlock` behind a constructor
   flag and train both MiniGPT variants at matched parameter counts on the
   same corpus, seeds, and schedule; report validation loss with the
   Chapter 22a experiment manifest.
2. Implement GeGLU (GELU gate) and reuse this chapter's tests; only the
   activation oracle should change.
3. Derive the FLOP count of both networks at matched parameters and check
   whether "matched parameters" also means "matched compute".
4. Measure the distribution of gate activations after training; how many
   hidden channels are effectively closed?

## Completion criteria

You are done when you can:

- write the SwiGLU forward pass from memory, including where SiLU sits;
- derive the tanh-based SiLU identity and say why it is numerically safer;
- derive $S = H(2C+1)/(3C+2)$ and its two-thirds limit;
- explain what the gate adds over ReLU in mechanism terms;
- design a parameter-matched ablation and defend it against the equal-width
  version.

## Primary sources

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
