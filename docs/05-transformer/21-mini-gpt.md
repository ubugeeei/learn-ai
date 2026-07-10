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
