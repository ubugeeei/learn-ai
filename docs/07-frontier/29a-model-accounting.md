# 29a — Model accounting

## What you will build

Frontier-model engineering runs on envelope arithmetic: how many
parameters, how many FLOPs per token, how many bytes per request. People
who can do this arithmetic read a technical report and immediately know
whether a claim is plausible; people who cannot must take every number on
faith. The arithmetic is not hard — the discipline is *stating exactly
what you count* and anchoring formulas to a real implementation so they
cannot drift into folklore.

In this chapter you will:

- derive the exact parameter count of the Chapter 21 architecture;
- derive decode-step and prefill FLOPs with declared conventions;
- account for training-resident memory, quadratic attention weights, and
  KV cache bytes;
- anchor every formula to the live implementation as an oracle.

## Prerequisites

You should understand Chapters 18–21 (every tensor the formulas count),
Chapter 13 (AdamW's moment arrays), and Chapter 24 (the KV cache whose
bytes reappear here).

## 1. Parameters, exactly

For vocabulary $V$, context $T$, width $C$, feed-forward width $F$, and
$L$ layers:

$$
N = VC + TC + L\left(\underbrace{4(C^2 + C)}_{\text{attention}}
  + \underbrace{2CF + F + C}_{\text{feed-forward}}
  + \underbrace{2C}_{\text{norms}}\right) + C
$$

The tied logit head contributes nothing — it reuses the token embedding,
which is why Chapter 21 counts parameters through ownership rather than
through layers. This formula is *exact*, and the suite holds it to
`MiniGpt.random(config).parameterCount` for three configurations. That
oracle is the chapter's method in miniature: a closed form that
disagrees with the code it describes is a bug in one of them, never a
rounding story.

## 2. FLOPs, with the conventions on the table

Counting rules (chosen once, used everywhere): a multiply-add is 2 FLOPs;
matrix products are counted (projections, attention scores, weighted
values, feed-forward, logits); bias adds, norms, softmax, and residuals
are excluded as $O(C)$ against $O(C^2)$. Under those rules, evaluating
one new token whose attention covers $t$ cached positions costs

$$
\text{decode}(t) = L\left(8C^2 + 4CF + 4Ct\right) + 2CV
$$

and prefilling an $n$-token prompt sums the decode steps:

$$
\text{prefill}(n) = n\left(L(8C^2 + 4CF) + 2CV\right)
  + 4CL\,\frac{n(n+1)}{2}
$$

The two terms behave differently and that difference drives serving
design: the linear term dominates at short contexts (the model is
"compute bound on width"), the quadratic term takes over as $n$ grows
(the model is "attention bound on length"). The suite checks the closed
form against literally summing `decodeStepFlops` — a self-consistency
oracle — and against a longhand hand calculation for a tiny
configuration.

## 3. Memory: the four-times rule and the quadratic term

Training-resident state at `Double` precision, independent of batch
size:

```text
parameters:      N * 8 bytes
gradients:       N * 8 bytes
AdamW moments:   2N * 8 bytes
total:           4N * 8 bytes
```

This is why training wants roughly four times the memory of serving the
same model at equal precision, before a single activation is stored.
Two further quantities complete the picture:

- attention weights retained by a full forward at length $t$:
  $L H t^2$ values — the quadratic memory that IO-aware attention
  (Chapter 28e, planned) exists to avoid materializing;
- KV cache payload at occupancy $t$: $L \cdot 2C \cdot 8t$ bytes per
  request — weights are shared across requests, caches are not, which is
  why Chapter 28c shrinks $C$ to `keyValueHeads * headChannels` there.

Both are anchored: the attention estimator equals the sizes of the
weight tensors `forwardWithWeights` actually returns, and the cache
estimator equals a real `AttentionKeyValueCache.allocatedPayloadBytes`
times the layer count.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

Then do the exercise that matters: compute `trainingResidentBytes` and
`kvCachePayloadBytes` for a 7B-parameter configuration by hand and check
your intuition against published serving numbers.

## 5. Implementation walkthrough

`ModelAccounting` is a stateless object of pure functions over
`MiniGptConfig`. Every arithmetic step runs in `Long` with
`Math.multiplyExact`/`addExact`, so an absurd configuration overflows
loudly instead of returning a small plausible number — the same
discipline as the cache accounting in Chapter 28c.

`parameterCount` mirrors the ownership walk of `MiniGpt.parameters`
term by term, and deliberately parallels the size guard in Chapter 25's
checkpoint loader — three independent expressions of the same
architecture that the tests force to agree.

`decodeStepFlops` validates the attended-position bound against the
configured context, then evaluates the per-layer sum. `prefillFlops`
evaluates the closed form with the Gauss sum for the attention term; the
equivalent step-summing loop lives in the *test*, not the
implementation, so the two derivations stay independent.

`attentionWeightValues` and `kvCachePayloadBytes` are one multiplication
each; their value is the anchoring tests, not the arithmetic.

## 6. Reading the tests

- the parameter oracle instantiates real models for three differently
  shaped configurations and demands exact equality;
- the decode hand calculation writes every term longhand (32 + 32 + 16 +
  12 = 92) so a reviewer can re-derive it without running anything, and
  separately checks that one extra attended position costs exactly
  $4C$;
- prefill-versus-sum is a self-consistency check across all
  configurations;
- the attention and KV estimators are compared against live tensor sizes
  and a live cache allocation respectively;
- the memory test pins the 1x/1x/2x/4x structure;
- boundary tests reject zero and over-context arguments.

## 7. Debugging checklist

1. When an estimator disagrees with the implementation, print both sides
   per layer first; the diverging term names the misunderstood component.
2. State your FLOP convention before comparing against any paper —
   multiply-add-as-2 versus as-1 is a silent factor of two.
3. Check units at every step: values versus bytes versus FLOPs; most
   accounting errors are unit errors.
4. If a memory estimate misses reality by roughly 2x or 4x, you forgot
   gradients or optimizer moments.
5. Distrust any accounting that was never anchored to a running
   implementation.

## 8. Failure modes to test

- formulas that drift after an architecture change (the model oracle
  catches this on the next run);
- `Int` overflow in size arithmetic for large configurations;
- prefill formulas that miss the Gauss-sum attention term;
- comparing FLOP counts computed under different multiply-add
  conventions;
- KV cache estimates that use query width after adopting GQA.

## Exercises

1. Add `decodeStepBytes`: the parameter and cache bytes *read* per decode
   step, and compute the arithmetic intensity (FLOPs per byte) as context
   grows; find where decoding turns memory-bound.
2. Extend the accounting to Chapter 28b/28c variants (SwiGLU widths,
   grouped KV) and re-anchor against those implementations.
3. Estimate training FLOPs as `3x` forward (backward ≈ 2x forward) and
   compare your total for a small run against the tokens-seen telemetry
   from Chapter 22b.
4. Reproduce the "6ND" approximation from the scaling-laws literature and
   state exactly which of this chapter's terms it keeps and drops.

## Completion criteria

You are done when you can:

- write the parameter formula from memory and say which term dominates
  at which shapes;
- compute decode and prefill FLOPs for any configuration, stating your
  conventions;
- explain the 4x training-memory rule and what it excludes;
- name the two quantities that scale with $t^2$ and $t$ respectively and
  the systems built to tame each;
- anchor a new estimator to a live implementation before trusting it.

## Primary sources

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
