# 13 — Initialization, SGD, AdamW, and gradient clipping

## What you will build

Tensor parameter update boundaries, global gradient norm, clipping, SGD,
AdamW, and Xavier/He initialization. Source:
`src/main/scala/learnai/optim/Optimizers.scala`.

## Optimizer responsibility

Autodiff computes current gradients. The optimizer combines them with state and
hyperparameters to produce new parameters.

```text
forward -> loss -> backward -> gradients
                                |
                         optimizer state
                                |
                         new parameters
```

`Tensor.updateParameter` is the mutation boundary. Optimizers do not need to
know the computation graph.

## SGD

$$
\theta_t=\theta_{t-1}-\eta g_t
$$

Here $g_t$ is commonly a mini-batch gradient estimate. SGD stores no
per-parameter state, making it memory-efficient and a useful baseline.

## Global gradient norm

Treat all parameter gradients as one long vector:

$$
\lVert g\rVert_2
=\sqrt{\sum_p\sum_i g_{p,i}^2}
$$

It gives one model-wide signal for exploding gradients. Compare it carefully
across model sizes, but track it over time within one model.

## Gradient clipping

If the norm exceeds maximum $c$, multiply all gradients by one scale:

$$
\tilde g=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
$$

For gradient `[3,4]`, norm is `5`; limit `1` yields `[0.6,0.8]`. Direction is
preserved while magnitude is bounded.

Clipping limits symptoms but can hide a root cause. Record original norm and
the fraction of clipped steps alongside loss and activations.

## Momentum intuition

An exponential moving average emphasizes persistent directions:

$$
m_t=\beta m_{t-1}+(1-\beta)g_t
$$

Adam tracks this first moment and a second moment of squared gradients.

## Adam

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Zero initialization biases early estimates toward zero, so correct them:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Then update:

$$
\theta_t=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Adaptive state costs two extra values per parameter.

## AdamW and decoupled weight decay

Weight decay shrinks parameters:

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

AdamW separates this from the adaptive gradient update:

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Decay acts even when gradient is zero. Production models often exclude biases
and normalization scales through parameter groups.

## Initialization and variance

Deep networks can amplify or erase activation and gradient scales. Xavier
uniform uses:

$$
w\sim U\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}\right]
$$

He uniform for ReLU uses fan-in:

$$
w\sim U\left[-\sqrt{\frac{6}{n_{in}}},
\sqrt{\frac{6}{n_{in}}}\right]
$$

Initialization should be selected by measuring layer activation and gradient
statistics, not by memorizing one formula.

## Reproducibility

Record:

- random seed;
- optimizer and hyperparameters;
- initialization;
- batch and data order;
- model structure;
- code revision.

The same seed can produce different corresponding values after a model change
because the random-consumption order changes.

## Implementation walkthrough

Both optimizers receive `Vector[Tensor]`; parameter ownership stays in the
model. `GradientNorm.global` sums squared gradients across every element and
takes one square root. If the norm exceeds the configured maximum, a single
scale `maximum / norm` multiplies all gradients. One shared factor preserves
the global direction.

`TensorSgd.step` computes the norm and clipping scale once, then calls each
parameter's mutation boundary. The callback receives flat index, current data,
and gradient; no optimizer reaches into Tensor's private arrays.

`AdamW` additionally owns first-moment, second-moment, and step state keyed by
parameter identity. For the first gradient `g=2`, `beta1=0.9`, and `beta2=0.999`:

```text
m1 = 0.1*2 = 0.2
v1 = 0.001*4 = 0.004
mHat = 0.2/(1-0.9) = 2
vHat = 0.004/(1-0.999) = 4
```

Bias correction removes the initialization-to-zero bias. The adaptive update
is approximately `learningRate * 2/(sqrt(4)+epsilon)`. Decoupled weight decay
is then applied to parameter data independently of the gradient; a zero-
gradient parameter still shrinks.

Initialization helpers calculate bounds from `fanIn` and `fanOut`, consume the
caller RNG, and label the returned parameter. Xavier and He encode assumptions
about how activation variance propagates; they are not arbitrary random ranges.

## Reading the tests

SGD uses a one-element hand update. Clipping verifies both final norm and
direction ratio. AdamW minimizes a scalar quadratic, while a separate zero-
gradient test specifically distinguishes decoupled decay from L2-in-loss.
Initialization tests check bounds and equal-seed determinism rather than one
particular sampled vector.

## Debugging checklist

1. Inspect raw global norm and clipping scale separately.
2. Verify optimizer state is keyed to parameter identity and shape.
3. Hand-calculate the first Adam step before debugging later steps.
4. Confirm bias-correction exponent uses the current one-based step.
5. If parameters become non-finite, locate the first gradient/update and inspect
   epsilon and learning rate only after the formula is verified.

## Exercises

1. Clip `[6,8]` to maximum norm `5`.
2. Add momentum SGD and compare a quadratic trajectory.
3. Hand-calculate Adam's first step for gradient `2.0`.
4. Remove bias correction and inspect the first ten updates.
5. Design parameter groups with and without weight decay.
6. Measure sample mean and variance of Xavier and He weights.

## Completion criteria

- Compare SGD and AdamW state requirements.
- Explain moments and bias correction.
- Show that global clipping preserves direction.
- Explain why decoupled decay acts at zero gradient.
- Explain why initialization depends on activation behavior.
- `OptimizersSuite` passes.

## Primary sources

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
