# 21 — Train and generate with MiniGPT

## What you will build

Combine embeddings, Transformer blocks, final RMSNorm, tied output projection,
and cross entropy into a decoder-only language model that trains end to end and
generates autoregressively.

Source: `src/main/scala/learnai/transformer/MiniGpt.scala`.

## Architecture

```text
token IDs [T]
  -> token + position embeddings       [T,C]
  -> Transformer block x L             [T,C]
  -> final RMSNorm                      [T,C]
  -> tied token-embedding transpose     [T,V]
  -> cross entropy with targets [T]     scalar
```

`MiniGptConfig` determines every parameter and activation shape:

| Field | Meaning |
| --- | --- |
| `vocabularySize` \(V\) | tokenizer categories |
| `maximumContextLength` \(T_{max}\) | position table and input limit |
| `channels` \(C\) | hidden width per token |
| `headCount` \(H\) | attention heads dividing \(C\) |
| `hiddenChannels` \(F\) | FFN expansion width |
| `layerCount` \(L\) | Transformer depth |

Config, tokenizer, and checkpoint form one compatibility contract.

## Final normalization

Pre-norm blocks leave the final residual stream outside a post-sublayer norm.
Normalize before vocabulary projection:

\[
H_{final}=\operatorname{RMSNorm}(H_L)
\]

The presence and type of final normalization are checkpoint architecture.

## Weight tying

The token table is \(E\in\mathbb{R}^{V\times C}\). Instead of allocating a
separate output matrix, use its transpose:

\[
Z=H_{final}E^{\mathsf T}
\]

Benefits:

- removes another \(CV\) parameters;
- shares input and output token space;
- lets embedding weights receive gradients from lookup and logits paths.

`parameters` contains the shared Tensor once so the optimizer cannot update it
twice.

## Next-token loss

```text
inputs:  [x0,x1,x2,x3]
targets: [x1,x2,x3,x4]
```

Causal masking ensures each logit row depends only on its prefix. Mean cross
entropy produces one scalar whose backward reaches embeddings, all blocks, and
final norm.

## End-to-end step

```scala
val loss = model.loss(inputs, targets)
loss.backward()
optimizer.step(model.parameters)
```

Those lines now invoke everything built so far: BPE IDs, embedding gather,
position addition, RMSNorm, Q/K/V projections, causal attention, residuals,
FFN, tied logits, stable cross entropy, Tensor reverse mode, clipping, and
AdamW.

A falling end-to-end loss does not replace focused operation tests. It verifies
integration and trainability.

## Autoregressive generation

Run the prompt, take only the final logits row, sample one token, append it, and
repeat:

```text
prompt -> all logits -> final row -> sample
   ^                                  |
   +------------ append --------------+
```

This reference recomputes all past Q/K/V at every step. KV caching later stores
past keys and values and computes only the new position.

## Sliding context

When generation exceeds the context limit, keep the latest tokens:

```scala
val retained = context.takeRight(maximumContextLength)
```

Absolute position IDs restart at zero for the retained window. Training must use
the same convention. Cropped information is inaccessible to the model.

## What the demo proves

The demo deliberately overfits a short repeated sequence. It proves:

- the graph is connected;
- parameters update;
- the model can reduce its objective;
- generation executes.

It does not prove generalization, knowledge, instruction following, or safety.
Those require diverse data, held-out evaluation, contamination checks, and
human/task assessment.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
```

This standard-library CPU engine is intentionally small. Profile before growing
the config.

## Relationship to frontier models

Shared principles:

- autoregressive next-token objective;
- embeddings, causal attention, FFN, residuals, normalization;
- logits, cross entropy, AdamW;
- temperature sampling.

Still missing:

- batched GPU and distributed kernels;
- mixed precision and loss scaling;
- RoPE, GQA, SwiGLU, MoE;
- learning-rate schedules and production checkpointing;
- KV cache, quantized serving, and request scheduling;
- large-scale data curation and post-training.

Later chapters scale or refine the same computation rather than replacing it
with a different principle.

## Implementation walkthrough

`MiniGpt.random` is the ownership root. One `SplittableRandom` initializes token
and position embeddings, each Transformer block in index order, and the final
RMSNorm. Labels include ownership paths such as `blocks.0.attention.query`; the
checkpoint loader later uses stable parameter order and labels as structural
evidence.

`logits` is intentionally short because earlier types own their invariants:

```scala
val embedded = embeddings(tokenIds)
val hidden = blocks.foldLeft(embedded)((current, block) => block(current))
val normalized = finalNorm(hidden)
normalized.matmul(embeddings.tokens.weight.transpose2D)
```

Trace a concrete configuration `V=5`, `T=3`, `C=4`, `L=1`:

```text
token IDs                 [3]
embedding                 [3,4]
one block                 [3,4]
final norm                [3,4]
token table transpose     [4,5]
logits                    [3,5]
targets                   [3]
cross entropy             scalar
```

`loss` validates target length and every target vocabulary ID before calling
`logits`. `Tensor.crossEntropy` then fuses stable log-sum-exp forward and
`(softmax-oneHot)/T` backward. Fusing avoids building a large one-hot graph but
keeps the exact derivative documented in Tensor.

`parameters` concatenates embeddings, all block parameters, and final norm.
The output head adds no parameter because it reuses the token table transpose.
The `distinct` test ensures this shared Tensor is returned once.

`MiniGptTrainer.trainSequence` creates AdamW once so moments persist across
steps, but constructs loss anew each iteration. It records loss before
backward/update, matching Chapter 9's trajectory convention. Gradient clipping
is enabled at norm one.

Reference generation crops context, runs `logits`, selects the final row,
softmaxes after temperature scaling, samples, and appends. It is deliberately
simple enough to serve as the correctness oracle for cached generation.

## Reading the tests

Logit shape and finite values verify the full forward path. Parameter count is
derived from embeddings, four attention projections, two norms, two FFN
projections, and final norm. Training uses a fixed sequence and checks a strong
loss decrease. Future-token replacement tests prefix causality at model level.
Equal seeds verify generation reproducibility. Cached/full tests compare every
position and context-rebuild behavior. Boundary tests cover invalid config,
context, target, and empty prompt.

## Debugging checklist

1. Trace shapes from IDs to logits with the actual config values.
2. Verify target IDs are the one-token-shifted sequence.
3. Inspect loss, global gradient norm, and one parameter update separately.
4. If training loss is flat, confirm every owned parameter reaches `parameters`.
5. If generation differs with equal seeds, compare context crop, logits row,
   probability filter, and RNG call count.
6. Never use generated text quality as the first forward-correctness test.

## Exercises

1. Compute the config's full parameter count.
2. Untie the output matrix and compare count and initial loss.
3. Compare depth `1`, `2`, and `4` for time and gradient norm.
4. Separate training and validation windows and plot overfitting.
5. Compare generation entropy and repetition across temperatures.
6. Display tokens discarded by context cropping.
7. Print parameter labels, shapes, and counts.

## Completion criteria

- Trace every shape from token IDs to logits.
- Explain final norm and weight tying.
- Explain both gradient paths into the embedding.
- Identify teacher-forced training and autoregressive generation.
- Explain context cropping and position reassignment.
- Explain why low training loss is not production quality.
- `MiniGptSuite` and all lower-level suites pass.

## Primary sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
