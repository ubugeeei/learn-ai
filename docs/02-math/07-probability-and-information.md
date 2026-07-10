# 07 — Probability and information

## What you will build

A validated categorical distribution, stable softmax and log-sum-exp, entropy,
cross entropy, and deterministic-seed sampling. Source:
`src/main/scala/learnai/math/Probability.scala`.

## Probability represents uncertainty

In next-token prediction, every vocabulary token is a possible outcome. A
categorical distribution assigns probability \(p_i\) to each outcome:

\[
p_i\geq0,\qquad\sum_{i=1}^{V}p_i=1
\]

`Categorical.from` rejects empty, negative, or incorrectly normalized input.
Successful construction establishes an invariant used by sampling and entropy.

## Random variables, expectation, and variance

A random variable maps outcomes to numeric values. Its expectation is a
probability-weighted average:

\[
\mathbb{E}[X]=\sum_i p_i x_i
\]

Variance is expected squared distance from that mean:

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
\]

Individual samples vary, but a sufficiently large sample mean approaches the
expectation. Mini-batch gradients rely on the same estimation idea.

## Models output logits before probabilities

A neural network emits one real-valued score \(z_i\) per token. These **logits**
may be negative and need not sum to one. Softmax turns them into probabilities:

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Exponentials are positive, normalization makes the sum one, and larger logits
receive larger probability.

## Stable softmax

`exp(10000)` overflows. Subtracting a constant \(c\) from every logit leaves
softmax unchanged:

\[
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Choose \(c=\max_jz_j\). The largest exponential becomes `1`, and every other
one is at most `1`.

Log-sum-exp uses the same transformation:

\[
\log\sum_j e^{z_j}
=m+\log\sum_j e^{z_j-m},\quad m=\max_jz_j
\]

## Information and entropy

The information content of an event with probability \(p\) is:

\[
I(p)=-\log p
\]

A certain event has zero information; rare events contain more. Expected
information is entropy:

\[
H(p)=-\sum_i p_i\log p_i
\]

The zero-probability contribution is defined as zero by a limit. A fair coin
has entropy \(\log2\); a certain coin has zero.

## Cross entropy as language-model loss

For target distribution \(q\) and model distribution \(p\):

\[
H(q,p)=-\sum_i q_i\log p_i
\]

With a one-hot target at index \(t\):

\[
\mathcal L=-\log p_t
\]

Expanding softmax gives a stable logits-only form:

\[
\mathcal L=\log\sum_j e^{z_j}-z_t
\]

A large correct logit drives loss toward zero. Assigning little probability to
the target produces a large penalty.

## Sampling

Draw \(u\) uniformly from `[0,1)` and select the first index whose cumulative
probability exceeds it.

```text
probabilities: [0.2,0.5,0.3]
cumulative:    [0.2,0.7,1.0]
u = 0.62 -> index 1
```

Equal random seeds produce equal random sequences. Fix seeds for comparison;
change them only when studying generation diversity.

## Exercises

1. Compute `softmax(VectorD(0,0,0))` by hand.
2. Compare the entropy of `[0.9,0.1]` and `[0.5,0.5]`.
3. Compute loss for target probabilities `0.5`, `0.1`, and `0.01`.
4. Sample 10,000 times and compare frequencies with probabilities.
5. Remove maximum subtraction and observe large-logit behavior.

## Completion criteria

- Distinguish logits, probabilities, and one-hot targets.
- Explain why softmax is non-negative and normalized.
- Derive maximum subtraction invariance.
- Connect cross entropy with target negative log-probability.
- `ProbabilitySuite` passes.
