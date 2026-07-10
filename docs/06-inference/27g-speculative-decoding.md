# 27g — Speculative decoding

## What you will build

Autoregressive decoding is serial: one target-model forward per token,
each waiting for the last. Speculative decoding breaks the serialization
with a bet — let a cheap *draft* model propose several tokens, then let
the target model score the whole block in one forward pass and keep the
prefix it agrees with. The remarkable part is the guarantee: the emitted
tokens are distributed *exactly* as if the target had decoded alone. Draft
quality affects only speed, never correctness, and that guarantee is a
small piece of sampling mathematics you can implement and test directly.

In this chapter you will:

- implement the accept/reject rule and residual resampling;
- turn the paper's correctness theorem into an executable oracle;
- verify convergence with a seeded Monte Carlo experiment;
- run draft-and-verify generation over two MiniGPTs with honest work
  accounting — and meet a real position-window bug on the way.

## Prerequisites

You should understand Chapter 23 (sampling from categorical
distributions), Chapter 7 (what a distribution over tokens is), and
Chapter 24 (why the target's per-token forward is the serial bottleneck).

## 1. The accept/reject rule

Let $p$ be the target distribution and $q$ the draft distribution at one
position. The draft samples $x \sim q$; the verifier accepts $x$ with
probability $\min(1, p(x)/q(x))$, and on rejection resamples from the
*residual*

$$
r(y) = \frac{\max(p(y) - q(y),\, 0)}{\sum_z \max(p(z) - q(z),\, 0)}
$$

The emitted token's distribution is then

$$
\underbrace{\min(p(x), q(x))}_{\text{accepted mass}}
 + \bigl(1 - \alpha\bigr)\, r(x) = p(x),
\qquad \alpha = \sum_z \min(p(z), q(z))
$$

because $\min(p,q)$ covers the part of $p$ the draft can supply and the
residual replaces exactly what it cannot. $\alpha$ — the overlap between
the distributions — is the acceptance probability, and it is the *only*
thing draft quality touches.

Rather than trusting the derivation, `outputDistribution` computes the
left side from the implementation's own accept and residual code, and the
suite asserts it equals $p$ elementwise for adversarial pairs (reversed
supports, disjoint supports, a uniform target against a spiked draft, and
identical distributions). A bug in the residual normalization or the
accepted mass breaks the identity immediately.

## 2. Blocks, bonus tokens, and the work story

Per round, the draft proposes up to $k$ tokens *sequentially* — draft
forwards are the cheap serial part — and the target scores all proposal
positions plus one more in a *single* `logits` call, because causal
attention yields every position's next-token distribution in one forward.
Proposals are verified in order; the first rejection emits the residual
sample and ends the round, and a fully accepted round emits a *bonus*
token from the target's own distribution at the block's end. Every round
therefore emits at least one target-quality token per target pass — the
worst case degrades to ordinary decoding plus wasted draft work, never
below it.

The statistics object carries the story: `targetVerificationPasses`
equals rounds, not tokens; the acceptance rate connects draft quality to
expected block progress; and `emittedTokens` must reconcile exactly with
`accepted + rejections + bonus`, which the suite checks.

## 3. A real bug: the shared window

The first version of this chapter's generation loop let the draft
condition on its own cropped context while the verifier used a shorter
window (leaving room for the proposal block). With learned absolute
positions, that shifted every position id between the two models — and
the "identical draft never rejects" test failed with real rejections.
The fix is to compute one retained window per round and condition *both*
phases on it. The test that caught this is worth internalizing: an
identical draft must never reject, because $p = q$ makes the accept test
$u < 1$, always true. Any rejection with an identical draft is a context,
position, or temperature mismatch between the two paths — precisely the
class of bug that would otherwise silently degrade acceptance rates in
production while producing perfectly valid-looking text.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The Monte Carlo test pushes 40,000 seeded draws through the verifier
against a deliberately bad draft and requires every token's empirical
frequency to match the target within one percentage point.

## 5. Implementation walkthrough

`SpeculativeSampling` validates every distribution (non-negative,
finite, sums to one) before using it. The accept test is written as
`u * q(x) < p(x)` so nothing divides; rejection walks the *unnormalized*
excess with a second uniform scaled by the residual mass, avoiding a
renormalization pass. Both uniforms come from the caller-owned generator
— that consumption order is part of the deterministic stream contract
from Chapter 22c. The numerically-identical corner (rejection fired but
the residual mass rounds to zero) keeps the drafted token, with the
reasoning in a comment.

`SpeculativeDecoding.generate` validates both models share a vocabulary
and that the lookahead leaves context room, then loops rounds: compute
the shared retained window, draft `blockSize` proposals sequentially
(recording each $q_i$), score `retained ++ proposals` with one target
forward, and verify in order via `verifyDraftToken`. Target
distributions come from the same softmax-with-temperature path the
draft uses, so the identical-models property holds bitwise. Statistics
are accumulated inline and validated by the `SpeculativeStatistics`
constructor.

## 6. Reading the tests

- the analytic oracle covers five adversarial distribution pairs,
  including both disjoint and identical supports;
- acceptance probability is pinned at the overlap sum, at one for
  identical distributions, and at zero for disjoint ones;
- the residual test checks support (only where $p > q$), normalization,
  and the undefined case;
- the identical-draft test runs 500 seeded verifications and requires
  zero rejections — the property that caught the window bug;
- Monte Carlo convergence uses an independent inverse-CDF sampler in
  the test, not the production one;
- end-to-end tests cover the never-rejects property through two full
  models, exact token counts, accounting reconciliation, and vocabulary
  bounds;
- failure tests cover unnormalized and negative distributions, size
  mismatches, impossible draft tokens, vocabulary mismatches, and
  oversized lookaheads.

## 7. Debugging checklist

1. Run the identical-draft test first; any rejection means the two
   paths disagree about context, positions, or temperature — not about
   sampling.
2. Check the residual's support by hand on a three-token example before
   trusting Monte Carlo statistics.
3. If empirical frequencies skew toward the draft, the accept test is
   comparing against the wrong side ($u < q/p$ instead of $u < p/q$).
4. Count RNG draws per verification (one on accept, two on reject);
   resume determinism depends on that consumption being stable.
5. If acceptance rates sag in production, diff the draft's and target's
   *inputs* token by token before touching the models.

## 8. Failure modes to test

- residual computed as `|p - q|` instead of `max(p - q, 0)`;
- accepted mass double-counted when `q(x) > p(x)`;
- draft conditioned on a different window than the verifier;
- temperature applied to one model only;
- bonus token emitted even after a rejection;
- work statistics counting target passes per token instead of per
  round.

## Exercises

1. Sweep draft quality: interpolate the draft's logits toward uniform
   and plot acceptance rate against interpolation weight.
2. Derive and verify the expected emitted tokens per round,
   $\frac{1 - \alpha^{k+1}}{1 - \alpha}$, for constant per-position
   acceptance $\alpha$, against measured statistics.
3. Combine with Chapter 24: keep KV caches for both models across
   rounds and roll the target's cache back on rejection.
4. Implement greedy (temperature-zero) verification, where acceptance
   becomes an argmax comparison, and re-derive the correctness claim.

## Completion criteria

You are done when you can:

- state the accept rule and derive the residual so the identity
  $\min(p,q) + (1-\alpha) r = p$ holds;
- explain why draft quality affects speed only, and through which
  quantity;
- explain the bonus token and the at-least-one-per-round guarantee;
- articulate the identical-draft property as a debugging tool and what
  its violation localizes;
- account for target passes versus emitted tokens honestly.

## Primary sources

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
