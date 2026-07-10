# 28c — Grouped-query and multi-query attention

## What you will build

Chapter 24 measured the KV cache and found that during autoregressive
serving, cached keys and values — not weights, not activations — dominate a
request's memory. Chapter 19's multi-head attention sizes that cache at
`2 * channels` values per token per layer, because every query head owns a
private key head and value head. GQA asks a blunt question: do all of those
key/value heads earn their bytes?

In this chapter you will:

- implement `GroupedQueryAttention` where several query heads share one
  key/value head;
- recover multi-head attention (MHA) and multi-query attention (MQA) as the
  two boundary configurations;
- prove correctness with two equivalence oracles instead of eyeballing text;
- compute cache and parameter savings from closed formulas, not folklore.

## Prerequisites

You should understand Chapter 19 (causal multi-head attention, channel-slice
head views) and Chapter 24 (KV cache ownership and payload accounting).

## 1. Head ownership

Let $H_q$ be the query head count and $H_{kv}$ the key/value head count,
with $H_{kv} \mid H_q$. Each key/value head serves a *group* of
$g = H_q / H_{kv}$ consecutive query heads:

```text
query heads:      q0 q1 q2 q3 q4 q5      (Hq = 6)
key/value heads:  k0    k1    k2         (Hkv = 3, g = 2)
ownership:        q0,q1 -> k0   q2,q3 -> k1   q4,q5 -> k2
```

Scores for query head $h$ use the keys of head $\lfloor h/g \rfloor$:

$$
a^{(h)}_{t,i} = \operatorname{softmax}_i
\left(\frac{q^{(h)}_t \cdot k^{(\lfloor h/g\rfloor)}_i}{\sqrt{d}}\right),
\qquad
o^{(h)}_t = \sum_{i \le t} a^{(h)}_{t,i}\, v^{(\lfloor h/g\rfloor)}_i
$$

Two settings have names of their own. $H_{kv} = H_q$ is exactly MHA — each
group has one member. $H_{kv} = 1$ is MQA — one key/value head serves every
query. GQA is the dial between them, and the GQA paper's empirical claim is
that a modest $H_{kv}$ recovers almost all of MHA's quality at almost all of
MQA's memory savings.

## 2. What actually shrinks

The key and value projections now map `channels` to
$H_{kv} \cdot d$ instead of `channels`, where $d$ is the per-head width.
Per token and layer, a Chapter 24 style cache stores one key row and one
value row, so the payload becomes

$$
\underbrace{2 \cdot H_{kv} \cdot d \cdot 8}_{\text{GQA bytes}}
\quad\text{versus}\quad
\underbrace{2 \cdot H_q \cdot d \cdot 8}_{\text{MHA bytes}}
$$

a reduction by exactly the group size $g$. At serving scale this is the
difference between fitting eight concurrent requests on a device and
fitting one. Parameters shrink too (the K/V matrices are narrower), but the
cache is the point: weights are shared across requests, caches are not.

Query width is untouched, so the model's ability to *ask* $H_q$ different
questions per position survives; what is shared is only the library being
asked. That asymmetry is the design insight.

## 3. Two equivalence oracles

Generated text looking fine proves nothing. The suite pins the
implementation with two exact comparisons:

1. **MHA boundary.** A grouped layer built from a Chapter 19 layer's own
   four projections with $H_{kv} = H_q$ must reproduce its output and every
   per-head weight matrix to within $10^{-12}$.
2. **Column duplication.** For $H_{kv} < H_q$, duplicate each key/value
   head's weight columns $g$ times to build an ungrouped layer. If and only
   if group ownership indexing is correct, the narrow and expanded layers
   compute identical mathematics. This catches off-by-one group maps that
   the boundary case cannot see.

The second oracle chains to the first: the expanded configuration is the
MHA-verified one, so both tests together tie GQA to Chapter 19's
implementation and, through Chapter 19's own suite, to hand calculations.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `GroupedQueryAttention` suite prints each oracle. To feel the
accounting, evaluate `cachePayloadBytes(contextLength = 4096, layerCount =
32)` for $H_{kv} \in \{H_q, 8, 1\}$ at a fixed width and compare.

## 5. Implementation walkthrough

The constructor validates positive counts, `channels % queryHeadCount == 0`,
and `queryHeadCount % keyValueHeadCount == 0`, then derives `headChannels`,
`groupSize`, and `keyValueChannels`. It also re-validates all four projection
shapes, because the `fromProjections` factory accepts caller-provided layers
and must not trust them.

`keyValueHeadFor` is deliberately a public one-liner — `queryHead /
groupSize` — so tests can assert the ownership map directly instead of
inferring it from outputs.

`forwardWithWeights` mirrors Chapter 19's loop. The only change is two
offsets: query head `h` slices the query at `h * headChannels` but the key
and value at `keyValueHeadFor(h) * headChannels`. Scores, causal mask, row
softmax, weighted values, concatenation, and output projection are
unchanged. If you diff the two methods, the diff *is* the architecture.

`keyValuePayloadBytesPerToken` and `cachePayloadBytes` implement Section 2's
formulas with `Math.multiplyExact`, so an absurd context length overflows
loudly instead of reporting a small number. `fromProjections` exists for
the oracles; `random` builds narrow K/V projections with the same Xavier
discipline as every other layer.

## 6. Reading the tests

- the MHA-boundary oracle compares outputs *and* per-head attention
  weights, because a wrong output projection could mask correct scores;
- the duplication oracle builds both layers from explicit `fromValues`
  weights, generated once and shared, so nothing depends on RNG order;
- causality is tested twice — forward (changed future, unchanged past
  outputs) and backward (prefix-only loss, zero future gradients) — since
  grouped indexing errors can break one direction without the other;
- accounting tests assert the closed formulas and the exact factor-of-$g$
  byte ratio between MHA and MQA configurations;
- failure tests cover indivisible widths, more key/value heads than query
  heads, mismatched projection shapes, wrong input width, and degenerate
  cache arguments.

## 7. Debugging checklist

1. Print the ownership map `keyValueHeadFor` for all query heads first;
   most GQA bugs are already visible there.
2. Test the $H_{kv} = H_q$ boundary before any sharing: if that fails, the
   bug is not about grouping.
3. Run the column-duplication oracle at $g = 2$ before trying MQA; $g$ at
   its extreme hides systematic off-by-one errors inside one big group.
4. Check the key/value projection *output* width; a full-width K/V with
   grouped indexing silently wastes the entire memory saving.
5. When integrating with a KV cache, size the cache at `keyValueChannels`,
   not `channels` — a cache sized for MHA hides the bug until memory
   pressure reveals it in production.

## 8. Failure modes to test

- channels not divisible by query heads, or query heads not divisible by
  key/value heads (including $H_{kv} > H_q$);
- projections with the wrong input or output width supplied to
  `fromProjections`;
- group map off by one (duplication oracle fails, boundary oracle passes);
- cache byte accounting with zero or negative context/layer counts;
- causality broken only in the backward pass.

## Exercises

1. Add a `forwardCached` path storing only `keyValueChannels`-wide rows in
   an `AttentionKeyValueCache`, and prove cached/full-prefix equivalence as
   in Chapter 24.
2. Combine this layer with Chapter 28a: rotate queries and their owning
   key heads with RoPE and re-run both equivalence oracles.
3. Measure quality: train MiniGPT variants at $H_{kv} \in \{H_q, H_q/2,
   1\}$ with matched seeds and report validation loss against cache bytes.
4. Implement *uptraining*: initialize a GQA checkpoint by mean-pooling an
   MHA checkpoint's key/value heads within each group, as in the GQA paper.

## Completion criteria

You are done when you can:

- state the ownership map and derive it from $H_q$, $H_{kv}$, and $g$;
- compute cache bytes per token for any configuration and the exact saving
  factor versus MHA;
- explain why sharing keys/values differs from sharing queries;
- explain both oracles and what class of bug each one uniquely catches;
- place MHA, GQA, and MQA on one dial and argue where quality is lost.

## Primary sources

- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
