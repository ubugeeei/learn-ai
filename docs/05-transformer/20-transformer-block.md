# 20 — Transformer block

## What you will build

Compose RMSNorm, causal self-attention, residual connections, and a position-
wise feed-forward network into a shape-preserving pre-norm block. Source:
`src/main/scala/learnai/transformer/TransformerBlock.scala`.

## Block equations

$$
H=X+\operatorname{Attention}(\operatorname{RMSNorm}(X))
$$

$$
Y=H+\operatorname{FFN}(\operatorname{RMSNorm}(H))
$$

```text
X [T,C]
 |-------------------------------+
 -> RMSNorm -> causal attention -> + = H [T,C]
                                      |---------------------+
                                      -> RMSNorm -> FFN -----+ = Y [T,C]
```

Input and output have equal shape, so blocks can be stacked.

## Residual connections

Instead of replacing input with a sublayer result:

$$
y=x+F(x)
$$

Backward becomes:

$$
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
\left(I+\frac{\partial F}{\partial x}\right)
$$

The identity path carries gradient directly across depth and lets each sublayer
learn a correction rather than a complete replacement. Both operands of the
residual addition must have identical shape.

## Pre-norm versus post-norm

```text
pre-norm:  x + Sublayer(Norm(x))
post-norm: Norm(x + Sublayer(x))
```

Pre-norm leaves the residual identity path outside normalization, which often
improves gradient flow in deep models. It also changes representation scale and
requires a final normalization before logits.

Architecture choices are evaluated with training curves, gradient norms, and
final quality, not with a universal rule.

## Position-wise feed-forward network

Attention mixes information across positions. The FFN transforms each position
independently with shared parameters:

$$
\operatorname{FFN}(x)
=W_2\operatorname{ReLU}(W_1x+b_1)+b_2
$$

```text
[T,C] -> Linear -> [T,F] -> ReLU -> Linear -> [T,C]
```

Expansion width $F$ adds nonlinear capacity. Attention chooses where to read;
the FFN transforms the collected information. A later chapter replaces ReLU
with a modern gated activation.

## Parameter count

For channels $C$ and FFN width $F$:

- Q/K/V/output projections: $4(C^2+C)$;
- two RMSNorm scales: $2C$;
- FFN expansion: $CF+F$;
- FFN projection: $FC+C$.

Total:

$$
4C^2+2CF+7C+F
$$

At large widths, matrix terms dominate biases and normalization scales.

## Depth

Stacked blocks can build relationships over relationships. Greater depth also
increases parameters, FLOPs, activation memory, sequential latency, and
optimization difficulty. Residuals, normalization, initialization, and learning-
rate schedules make deep optimization practical.

## Dropout and regularization

The reference block omits dropout to preserve deterministic observation. A
dropout implementation needs:

- training/evaluation mode;
- seeded masks stored for backward;
- scaling of retained activations;
- reproducible RNG state.

Whether dropout helps depends on data and model scale and must be measured.

## Tested properties

- input and output shapes match;
- parameter count matches the formula;
- parameter references are unique;
- input and parameter gradients are finite;
- the complete block preserves prefix causality;
- invalid widths fail at construction.

Attention causality alone is insufficient if another block component mixes
time positions incorrectly.

## Implementation walkthrough

`FeedForward` owns `expansion: Linear(C,F)` and
`projection: Linear(F,C)`. It validates the final channel axis, applies
expansion, ReLU, and projection. The same two layers process every time row;
there is no interaction across positions in the FFN.

`TransformerBlock.apply` mirrors the two pre-norm equations without hidden
mutation:

```scala
val afterAttention = input + attention(attentionNorm(input))
afterAttention + feedForward(feedForwardNorm(afterAttention))
```

The first residual must add attention output to the original `input`, not to
the normalized input. The second norm must read `afterAttention`, because the
attention branch has already updated the residual stream. These distinctions
are easy to lose in a long expression and are why the intermediate has a name.

All tensors preserve `[T,C]` around residual additions. Only internal attention
and FFN shapes change. `parameters` follows forward ownership order: attention
norm, four attention projections, FFN norm, expansion, projection. Stable order
is required by checkpoint serialization and optimizer state.

`forwardCached` differs only in the attention call and the required input time
dimension of one. RMSNorm and FFN are position-wise, so they need no cache. One
layer-specific KV cache is passed to attention; sharing it between blocks would
mix projections from different parameter sets.

## Reading the tests

Shape preservation catches residual/FFN-width mistakes. Parameter count is
calculated component by component, so missing bias/norm parameters fail. A
scalar output is backpropagated to input and every parameter, verifying graph
connectivity. Whole-block causality catches any future mixing outside
attention. Invalid hidden width fails during construction.

## Debugging checklist

1. Name the residual stream before and after each branch.
2. Verify norm input and residual addend are not accidentally the same tensor.
3. Check branch output is `[T,C]` before addition.
4. If a parameter has zero gradient, trace whether its branch output reaches
   the scalar loss.
5. If causality fails, test attention and FFN row independence separately.

## Exercises

1. Count parameters for `C=768`, `F=3072`.
2. Remove residuals and compare deep input-gradient norms.
3. Implement post-norm and compare equal-seed training.
4. Replace ReLU with tanh and measure activation/gradient statistics.
5. Test that FFN rows are independent.
6. Design a dropout mode and RNG protocol.

## Completion criteria

- Write both pre-norm residual equations.
- Explain the residual identity path.
- Distinguish attention and FFN roles.
- Trace expansion and projection shapes.
- Compute block parameter count.
- Explain why whole-block causality is retested.
- `TransformerBlockSuite` passes.

## Primary sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
