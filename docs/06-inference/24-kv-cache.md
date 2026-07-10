# 24 — KV cache

## What you will build

The training implementation evaluates a complete sequence because every token
contributes to one parallel loss. Autoregressive inference has a different
shape: after sampling token `t`, the model needs only the logits for token
`t + 1`. Recomputing every earlier key and value is correct, but wasteful.

In this chapter you will:

- derive which attention values can be reused;
- implement a fixed-capacity KV cache for each Transformer layer;
- build a request-owned `MiniGptInferenceSession`;
- prove cached/full-prefix equivalence with an independent reference path;
- preserve learned absolute-position behavior when a context window shifts;
- separate deterministic work counts from measured wall-clock results.

## Prerequisites

You should understand Chapters 18–23, especially the shapes in causal
self-attention and the difference between logits and sampled token IDs.

## 1. What the reference decoder repeats

For one head at time `t`, causal attention is

\[
q_t = x_t W_Q,\qquad
k_i = x_i W_K,\qquad
v_i = x_i W_V
\]

\[
a_{t,i} = \operatorname{softmax}_i
\left(\frac{q_t k_i^\top}{\sqrt{d_h}}\right),
\qquad 0 \le i \le t
\]

\[
o_t = \sum_{i=0}^{t} a_{t,i}v_i
\]

Once `k_i` and `v_i` have been computed, their inputs and parameters do not
change during the request. The next decoding step needs a new `q_t`, `k_t`, and
`v_t`, but it may reuse all earlier keys and values.

The cache does **not** retain queries. A past query is never used again.

## 2. Shapes and ownership

For `L` layers, context capacity `T`, and model width `C`, this implementation
allocates:

```text
per layer keys:    [T, C]
per layer values:  [T, C]
all cache payload: L * 2 * T * C * sizeof(Double)
```

The head dimension is a view over channel ranges, so the cache stores one
combined `[T, C]` matrix rather than one object per head.

`AttentionKeyValueCache` is mutable and belongs to one request. Model weights
remain shared and unchanged. This distinction is operationally important:

```text
shared across requests: MiniGPT parameters
owned by one request:   token history, RNG, KV caches, work counters
```

Sharing a mutable cache between concurrent requests would mix their contexts.
The API therefore creates an explicit `MiniGptInferenceSession` per request and
documents that it is not thread-safe.

## 3. Cached one-token attention

`CausalSelfAttention.forwardCached` accepts exactly `[1, C]` input.

1. Project the current row into query, key, and value.
2. Append the key/value row to fixed arrays.
3. Compute scores between the current query and cache positions `[0, length)`.
4. Apply a stable softmax by subtracting the largest score.
5. Form one weighted value row and apply the output projection.

There is no triangular mask in this path. The cache contains only the past and
current positions, so future positions do not exist yet.

## 4. Layer-by-layer state

Each Transformer block needs a different cache. Layer 1 keys are projections
of layer 1 inputs; layer 2 keys are projections of layer 2 inputs. Reusing one
cache across layers is a shape-compatible but mathematically incorrect bug.

For a token at position `t`, the session evaluates:

```text
token[t] + position[t]
  -> block[0](cache[0])
  -> block[1](cache[1])
  -> ...
  -> final RMSNorm
  -> tied token-embedding transpose
  -> logits[t]
```

## 5. The equivalence oracle

An optimization is not correct merely because it generates plausible text.
The tests compare every cached result against the existing full-prefix path:

```text
cached(token[t], cache(prefix))
    approximately equals
fullForward(prefix :+ token[t]).lastRow
```

This comparison is performed first for attention alone and then for the entire
MiniGPT. It catches wrong head offsets, keys/values stored after attention,
layer-cache sharing, incorrect position IDs, and unstable softmax code.

The tolerance accounts only for floating-point operation ordering. It is not
loosened to hide systematic error.

## 6. Context capacity and absolute positions

MiniGPT currently uses learned absolute position embeddings. The reference
decoder crops an overlong context from the left and restarts retained position
IDs at zero:

```text
old context positions: 0 1 2 3
append at capacity:      1 2 3 4
reference window IDs:    0 1 2 3
```

Simply evicting the oldest cached row would leave the retained hidden states
encoded at positions `1, 2, 3`. That silently changes model behavior.

`generateCached` therefore rebuilds the retained window when capacity is full.
This preserves exact reference semantics, but the rebuild temporarily loses
the cache's work advantage. Later, RoPE and explicit long-context policies will
make this trade-off configurable rather than accidental.

## 7. Deterministic work accounting

Suppose a prompt has `P` tokens and we generate `N` more without reaching
capacity. The reference path evaluates

\[
P + (P+1) + \cdots + (P+N-1)
= NP + \frac{N(N-1)}{2}
\]

token rows. The cached path evaluates

\[
P + (N-1)
\]

rows: prefill the prompt once, then evaluate each sampled token except the last
one, whose logits are not needed.

These are deterministic algorithmic work counts. They do not prove a runtime
speedup because allocation, JVM compilation, small matrix overhead, and memory
access also matter.

## 8. Run the lab

```console
$ nix develop -c sbt 'runMain learnai.transformer.runKvCacheLab'
```

The lab:

- verifies identical sampled token IDs before timing;
- performs JIT warmup;
- runs repeated reference and cached measurements;
- reports minimum, median, and maximum time;
- records JVM, OS, CPU, configuration, seeds, and workload;
- prints a checksum so outputs remain observable;
- reports token work and cache payload separately.

This is a transparent educational microbenchmark, not a replacement for JMH.
Do not copy its timing into a general performance claim. Re-run it on the
target hardware and retain the full environment output.

## 9. Failure modes to test

- zero or negative cache dimensions;
- append after fixed capacity;
- wrong key/value shape;
- cache/model channel mismatch;
- empty prefill;
- prefill longer than context capacity;
- one cache accidentally shared across requests or layers;
- cached output differs from the independent full-prefix path;
- window eviction changes absolute-position semantics;
- sampling uses a different seed or filtering policy in the comparison.

## Exercises

1. Add a paged cache abstraction without changing attention results.
2. Store cache payloads as `Float` and measure logit error and memory.
3. Add a batch dimension and explain why requests in a batch may have different
   logical lengths.
4. Implement cache compaction after request cancellation.
5. After implementing RoPE, compare rebuild, eviction, and position-offset
   policies explicitly.

## Completion criteria

You are done when you can:

- derive why keys and values are reusable but queries are not;
- calculate cache payload size from layer, context, and channel counts;
- explain request ownership and concurrency boundaries;
- prove cached/full-prefix equivalence at attention and model levels;
- explain why learned absolute positions require a rebuild on window shift;
- report work counts separately from a qualified timing experiment.

## Primary sources

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
