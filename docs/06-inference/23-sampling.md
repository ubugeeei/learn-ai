# 23 — Temperature, top-k, and top-p sampling

## What you will build

An independent decoding policy that applies temperature, top-k, and nucleus
top-p filtering, renormalizes probabilities, and samples from a caller-owned
random generator. Source: `src/main/scala/learnai/lm/Sampling.scala`.

## Decoding is a separate decision layer

The model outputs next-token logits. A decoding policy chooses an actual token.
Equal model and prompt with different policies can produce different
determinism, diversity, repetition, and failure rates.

```text
logits
 -> temperature
 -> softmax
 -> top-k
 -> renormalize
 -> top-p
 -> renormalize
 -> sample
```

Policy and seed are part of generation reproducibility.

## Greedy decoding

\[
x_{t+1}=\operatorname*{arg\,max}_i z_i
\]

Greedy output is deterministic but can become repetitive or choose a locally
likely poor continuation. Top-k with `k=1` is equivalent to greedy decoding.

## Temperature

\[
p_i=\operatorname{softmax}(z_i/\tau),\quad\tau>0
\]

- low temperature enlarges logit differences and lowers entropy;
- `1` preserves the model distribution;
- high temperature flattens the distribution and increases diversity.

Temperature zero is undefined. Represent greedy decoding explicitly rather than
dividing by zero.

## Top-k

Keep the \(k\) highest-probability tokens, set the others to zero, and
renormalize:

\[
\tilde p_i=\begin{cases}
p_i/Z&i\in K\\
0&\text{otherwise}
\end{cases}
\]

It gives a fixed candidate count regardless of confidence. If `k` exceeds
vocabulary size, all tokens remain. Equal probabilities use lower token ID as a
deterministic tie-breaker.

## Nucleus top-p

Sort by probability and retain the smallest prefix whose cumulative probability
reaches threshold \(p\):

```text
probabilities: [0.60,0.25,0.10,0.05]
top-p = 0.80 -> keep [0.60,0.25], cumulative 0.85
```

A confident distribution keeps few candidates; a flat one keeps more. At least
one token is always retained.

## Filtering order is part of the contract

This implementation uses temperature, then top-k, then top-p. Top-k is
renormalized before nucleus filtering. Reversing operations can change results.

When comparing serving APIs, inspect:

- top-p/top-k order;
- boundary inclusion;
- logit-bias and penalty order;
- minimum retained tokens;
- random-generator and seed semantics.

## Evaluation

Measure model likelihood separately from one sampled response. Generation tasks
need several seeds and metrics such as correctness, diversity, repetition,
latency, and unsafe-output rate.

Exact tasks may prefer greedy or low temperature. Creative tasks may benefit
from moderate entropy. High-impact tool actions still require schemas and
policy gates regardless of decoding.

## Implementation walkthrough

`Sampling.distribution` applies policy in a fixed order: temperature, softmax,
top-k, renormalization, top-p, final renormalization. Changing order changes the
distribution, so the order is part of the public behavior.

For logits `[3,2,1,0]`, top-k `2` retains token IDs `0` and `1`. Probabilities
for other IDs become zero and the survivors are renormalized. Top-p instead
sorts probability/index pairs descending, accumulates until the threshold is
reached, and keeps the smallest prefix that crosses it. At least one token is
always retained, even for a very small threshold.

Temperature divides logits before softmax. Values below one magnify differences
and lower entropy; values above one flatten them. Temperature must be finite and
positive—zero is not a supported shorthand for greedy decoding. Use top-k one
for deterministic greedy behavior.

Tie ordering uses smaller token ID. Without this secondary key, equal
probabilities could be ordered by collection implementation details and fixed-
seed sampling would drift. `normalize` requires positive remaining mass and
scales by its sum.

`Sampling.sample` owns no random state. It receives a `RandomGenerator`, builds
the filtered `Categorical`, and asks it for one ID. Reproducibility therefore
depends on both policy and prior RNG calls.

## Reading the tests

Top-k one has a deterministic argmax oracle. Top-k larger than vocabulary tests
clamping. A constructed probability vector gives a hand-known nucleus. Entropy
comparison verifies temperature direction. Equal-logit ties verify ID ordering.
Equal seeds verify the complete filtered sampling path. Invalid policy values
fail at `SamplingConfig` construction.

## Debugging checklist

1. Print logits after temperature but before softmax.
2. Print retained IDs and mass after each filter.
3. Renormalize after removing tokens.
4. Specify deterministic tie ordering.
5. If seeded samples drift, compare policy order and RNG call count.

## Exercises

1. Compute probabilities and entropy for logits `[0,1,2]` at several
   temperatures.
2. Find a case where swapping top-k and top-p changes candidates.
3. Compare 10,000-sample frequencies with theoretical probabilities.
4. Design a repetition penalty and state its operation order.
5. Compare greedy, top-k, and top-p on fixed prompts and seeds.

## Completion criteria

- Separate model logits from decoding policy.
- Explain temperature's entropy effect.
- Compare top-k and top-p selection.
- Explain why filtering requires renormalization.
- Record operation order and seed for reproduction.
- `SamplingSuite` passes.
