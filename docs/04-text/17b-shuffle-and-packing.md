# 17b — Shuffle and sequence packing

## What you will build

Chapter 16 turned one token sequence into fixed windows, and Chapter 22b
sampled them with replacement — simple, correct, and wasteful in two ways.
Replacement sampling revisits some examples before seeing others, and real
corpora are not one sequence: they are many documents of wildly different
lengths, and stuffing each into its own padded window burns most of the
batch on padding.

In this chapter you will:

- build a deterministic epoch shuffle whose whole iteration state is a
  two-field cursor, so a data sweep is exactly restartable;
- pack variable-length documents into fixed windows with a per-position
  loss mask;
- add masked cross entropy so masked positions contribute *exactly zero*
  loss and gradient;
- verify both with counting oracles rather than inspection.

## Prerequisites

You should understand Chapter 16 (causal windows and batches), Chapter 12
(the Tensor cross entropy being extended), and Chapter 22c's resume
contract; the SplitMix64 generator built there powers the shuffle.

## 1. Shuffling that can be checkpointed

An epoch shuffle visits every example exactly once per epoch in a
different order each epoch. The standard tool is a Fisher–Yates
permutation; the engineering question is where the iteration state lives.
This implementation makes the permutation a *pure function* of
`(size, seed, epoch)` and keeps all traversal state in a caller-owned
value:

```text
ShuffleCursor(epoch, position)
index = permutation(size, seed, epoch)(position)
```

Each epoch derives its generator as `seed + epoch * gamma` with the
SplitMix64 golden gamma — the same stream-separation device the generator
itself uses — so epoch `n + 1` needs no memory of epoch `n`. Checkpoint
two numbers and the sweep resumes exactly; the suite verifies that a
25-read straight traversal equals an 11-then-14 chunked traversal across
three epoch boundaries. Recomputing the permutation per access is O(size)
and deliberately unoptimized: the contract is the lesson, and a cached
variant is an exercise once a measurement justifies it.

## 2. Packing: the shape problem

Batches need fixed shapes; documents have arbitrary lengths. Padding each
document to the context length wastes compute proportional to the length
variance. Packing instead concatenates documents into one stream, each
terminated by a separator token, and cuts consecutive windows of
`contextLength + 1` advancing by `contextLength`, so every next-token
prediction in the stream lands in exactly one window. The tail is
completed with padding.

The cost of packing is *semantic*, and the loss mask pays it. Window
position `t` trains the prediction `inputs(t) -> targets(t)`; the mask is
false when:

- the target is padding — the stream has ended; or
- the input is the separator — predicting the next document's opening
  from the previous document's context is noise, not signal.

Predicting the separator itself stays trainable: learning where documents
end is real. Counting gives a clean invariant the tests assert literally:
every document token is predicted exactly once, so

$$
\text{unmasked targets} \;=\; \sum_d \lvert d \rvert
$$

— within-document transitions contribute $\lvert d\rvert - 1$ and the
final-token-to-separator prediction contributes one.

## 3. Masked loss with exact zeros

The mask needs support in the loss, not just the data. `crossEntropyMasked`
extends Chapter 12's fused cross entropy: the forward mean is taken over
the *unmasked count* (so masked rows do not dilute the scale), and the
backward rule skips masked rows entirely — their gradient is exactly zero,
not approximately zero. Two design details are worth noticing:

- every target index is validated even when masked, so an indexing bug
  cannot hide behind a mask;
- an all-masked batch is rejected: a mean over zero rows has no value, and
  packing *can* produce such windows (a context of one whose only input is
  a separator). `PackingResult.trainableExamples` exposes the filter
  instead of silently dropping windows.

What this chapter does **not** do, stated plainly: attention inside a
packed window still crosses document boundaries, because our attention has
no block-diagonal document mask. The contamination is bounded by the
window length; making it visible (and fixing it) is an exercise.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `DeterministicShuffle`, `SequencePacking`, and Tensor suites carry
this chapter's oracles, including a MiniGPT window trained end-to-end
through `lossMasked`.

## 5. Implementation walkthrough

`DeterministicShuffle.permutation` runs Fisher–Yates from the top index
down, swapping with `nextInt(index + 1)` — including the possible
self-swap that unbiasedness requires. `ShuffledSweep` is immutable; its
three methods (`indexAt`, `next`, `take`) all validate the cursor and
`take` returns the continuation cursor so reads compose.

`SequencePacking.pack` validates documents first (non-empty, free of the
separator and padding tokens, which must differ), builds the terminated
stream, computes the window count as a ceiling division of the
prediction count, pads to `windows * contextLength + 1`, and then cuts
windows with `slice`. The mask logic is two boolean tests against the
*absolute* stream position — real target, non-separator input — and the
result carries the stream length, padding count, and unmasked-target
count so callers can assert the arithmetic.

`Tensor.crossEntropyMasked` mirrors the unmasked op's log-sum-exp
stabilization row by row, but accumulates loss and probabilities only for
unmasked rows and divides by the unmasked count. `MiniGpt.lossMasked`
validates lengths and vocabulary range and forwards to the op, exactly as
`loss` does for the unmasked case.

## 6. Reading the tests

- shuffle: bijection per epoch, purity in all three arguments, exact
  one-epoch coverage, chunked-equals-straight resume across epoch
  boundaries, and consecutive epochs producing different orders;
- packing: a fully hand-computed two-document example (every input,
  target, and mask bit written out), the multiset oracle comparing
  observed unmasked pairs against independently enumerated
  within-document transitions plus endings, and a property test that
  every masked position has a stated reason;
- degenerate windows: the context-length-one construction that produces
  an all-masked window, exposed and filterable;
- masked loss: all-true-mask equality with `crossEntropy` (values and
  gradients), a `gatherRows` composition oracle for both value and
  full-tensor gradients, exact zero gradients on masked rows, and
  validation failures including masked-row targets;
- integration: a packed window trains MiniGPT and produces gradients.

## 7. Debugging checklist

1. Check `unmaskedTargetCount` against the total document token count
   first; the invariant catches most window and mask bugs at once.
2. If loss jumps when padding increases, the mean is dividing by the row
   count instead of the unmasked count.
3. If the model starts documents strangely, look for unmasked
   separator-input positions — you are training cross-document guesses.
4. If resumed sweeps repeat examples, the cursor's epoch and position are
   being restored into a sweep built with a different seed or size.
5. Print one full window by hand (inputs, targets, mask) before writing
   any property test; the hand example in this chapter's suite exists
   because it caught its own off-by-one during development.

## 8. Failure modes to test

- empty document lists, empty documents, separator/padding collisions,
  special tokens inside documents;
- cursors outside the sweep, negative counts, zero-size sweeps;
- all-masked windows fed directly to the loss;
- masks that silently hide invalid target indices;
- a shuffle that is deterministic but *not* a bijection (duplicated or
  dropped indices);
- packing that duplicates or drops a next-token pair at a window
  boundary — the multiset oracle exists for exactly this.

## Exercises

1. Cache permutations per epoch and measure the speedup honestly against
   the recomputing sweep (Chapter 22 rules: warmup, repeats, medians).
2. Wire `ShuffledSweep` and packed windows into the Chapter 22c resumable
   trainer, extending its state with the `ShuffleCursor`, and re-run the
   bitwise split-point sweep.
3. Add a block-diagonal attention mask so packed documents cannot attend
   to each other, and measure the loss difference on a corpus with many
   short documents.
4. Implement best-fit packing (place documents to minimize padding) and
   compare padding fractions against this chapter's sequential packing.

## Completion criteria

You are done when you can:

- explain why a pure-function permutation plus a two-field cursor is
  exactly restartable, and what a hidden-state shuffle would break;
- derive the unmasked-target invariant and use it as a first-line check;
- state both mask rules and why separator *targets* stay trainable;
- explain the difference between masking loss and masking attention, and
  what contamination simple packing accepts;
- argue why masked rows must receive exactly zero gradient rather than a
  small one.

## Primary sources

- [The Fisher–Yates shuffle (Knuth, TAOCP Vol. 2, Algorithm P)](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
- [Language Models are Few-Shot Learners (GPT-3; packed training)](https://arxiv.org/abs/2005.14165)
- [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
