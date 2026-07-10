# 18 — Embeddings, position, linear layers, and RMSNorm

## What you will build

Implement token and position embeddings, dense channel transforms, and RMSNorm
on the Tensor autodiff engine. Source:
`src/main/scala/learnai/transformer/Layers.scala`.

## From discrete IDs to continuous vectors

A token ID is a category label. ID `100` is not ten times ID `10`. An embedding
table $E\in\mathbb{R}^{V\times C}$ maps each token to a $C$-dimensional
vector:

```text
token IDs:       [time]
embedding table: [vocabulary,channels]
output:          [time,channels]
```

$$
\boldsymbol{x}_t=E[\text{tokenId}_t]
$$

Initial vectors have no inherent semantic meaning. Next-token gradients move
them into configurations useful for prediction.

## Embedding backward is scatter-add

Forward gathers rows. Backward scatters each output gradient into its source
row. Repeated token IDs accumulate into one shared parameter row:

```text
IDs [2,0,2]
forward:  gather rows 2,0,2
backward: add positions 0 and 2 into row 2
```

This is parameter sharing in its simplest form.

## Attention needs position information

Without a positional signal, self-attention operations cannot distinguish all
reorderings. Add learned position vectors:

$$
\boldsymbol{h}_t^{(0)}=E[x_t]+P[t]
$$

```text
token embeddings:    [time,channels]
position embeddings: [time,channels]
sum:                 [time,channels]
```

Equal tokens at different positions begin with different hidden states. A
learned absolute table has only `maximumContextLength` rows and rejects longer
inputs. RoPE later introduces position by rotating query and key channels.

## Linear layers transform the channel axis

Apply the same affine transform to every time row:

$$
Y=XW+b
$$

```text
X: [time,inputChannels]
W: [inputChannels,outputChannels]
b: [outputChannels]
Y: [time,outputChannels]
```

`addRowVector` explicitly repeats bias across rows. Its backward sums all row
gradients into the bias. Query, key, value, output, feed-forward, and logit
projections are all linear transforms.

## Why normalization is needed

Residual additions can change hidden-state scale across depth. Extreme scales
can saturate softmax, damage gradient flow, and worsen numerical behavior.

RMSNorm does not subtract the mean. It divides by root mean square and applies
a learned per-channel scale $g_i$:

$$
\operatorname{RMS}(x)=
\sqrt{\frac{1}{C}\sum_i x_i^2+\epsilon}
$$

$$
y_i=g_i\frac{x_i}{\operatorname{RMS}(x)}
$$

Every `[channels]` row is normalized independently. $\epsilon$ prevents
division by zero.

## RMSNorm backward

Let $r=\operatorname{RMS}(x)$ and let $g_i$ denote upstream gradient after
scale. Then:

$$
\frac{\partial L}{\partial x_i}
=\frac{g_i}{r}
-\frac{x_i}{Cr^3}\sum_j g_jx_j
$$

Scale gradients sum over rows:

$$
\frac{\partial L}{\partial \gamma_i}
=\sum_{row}\frac{\partial L}{\partial y_{row,i}}
\frac{x_{row,i}}{r_{row}}
$$

Tests compare input gradients with finite differences.

## Parameter count

For vocabulary $V$, context $T$, and channels $C$:

- token embedding: $VC$;
- position embedding: $TC$;
- linear layer: $C_{in}C_{out}+C_{out}$;
- RMSNorm: $C$.

Token embedding often dominates at large vocabularies. Weight tying reuses its
transpose as the output classifier.

## Implementation walkthrough

`Embedding` owns a trainable weight `Tensor` with shape `[entries, channels]`.
`apply(indices)` first validates every integer against `0 <= id < entries`, then
delegates to `gatherRows`. The output shape is `[indices.size, channels]`.
Repeated IDs deliberately select the same source row, so backward accumulates
all contributions into one embedding vector.

Consider a table with three rows and two channels:

```text
E = [[1, 0],
     [0, 1],
     [2, 3]]
ids = [2, 0, 2]
output = [[2,3], [1,0], [2,3]]
```

If the loss sends gradients `[a,b]` and `[c,d]` to the two appearances of ID
`2`, row two receives `[a+c,b+d]`. This is parameter sharing, not accidental
aliasing.

`TokenPositionEmbedding` owns separate token and position tables whose channel
counts must match. For a sequence of length `T`, it gathers token rows by token
ID and position rows by `0 until T`, then adds equal-shaped tensors. The cached
inference method `at(tokenId, position)` performs the same operation for one
explicit position.

`Linear.apply` checks `[rows,inputChannels]`, computes matmul with weight
`[inputChannels,outputChannels]`, and calls `addRowVector` for bias. The named
broadcast operation is important: bias backward must sum contributions across
all rows.

`RmsNorm.apply` delegates to the fused Tensor operation with one learned scale
per channel. Its epsilon protects the inverse root when a row is all zero.

## Reading the tests

Lookup tests use explicit tables so expected rows are independent of random
initialization. Repeated-token position tests show that equal token IDs can
produce different combined vectors. Linear tests calculate affine outputs by
hand. RMSNorm checks row mean-square and uses finite differences for both input
and scale gradients. Range and context-limit tests ensure errors occur before
gathering.

## Debugging checklist

1. Separate token IDs, position IDs, token rows, and position rows.
2. Check table/output shapes before values.
3. If repeated IDs have wrong gradients, inspect `gatherRows` accumulation.
4. If bias gradients are wrong, verify reduction across rows.
5. If RMSNorm produces large values, inspect row mean square and epsilon before
   the learned scale.

## Exercises

1. Compute parameter count and fp16 bytes for `V=50,000`, `C=4,096`.
2. Verify repeated-token gradient accumulation.
3. Remove position embeddings and study reordered sequences.
4. Derive the bias gradient's time-axis sum.
5. Compare LayerNorm and RMSNorm equations.
6. Explain zero-input behavior without epsilon.

## Completion criteria

- Explain why token IDs are categorical rather than ordinal.
- Explain gather/scatter embedding gradients.
- State token and position embedding shapes.
- Trace a linear layer's channel-axis change.
- Explain RMSNorm, epsilon, and learned scale.
- `LayersSuite` passes.
