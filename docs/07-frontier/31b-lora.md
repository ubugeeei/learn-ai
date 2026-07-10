# 31b — LoRA: low-rank adaptation

## What you will build

Full fine-tuning rewrites every weight of a pretrained model, which means
storing a complete model copy per task and shipping optimizer state twice
the model's size. LoRA starts from an empirical observation: the *change*
fine-tuning induces in a weight matrix has low intrinsic rank, so you can
constrain the update to a rank-$r$ factorization, train only that, and
fold it back into the base weights when you are done.

In this chapter you will:

- implement `LoraLinear`, a frozen base layer plus a trainable low-rank
  additive update;
- understand why the up projection initializes to zero and what that
  guarantees;
- implement and verify *merging*, the property that makes LoRA free at
  inference;
- pin the freezing contract with a bitwise oracle rather than trusting a
  flag.

## Prerequisites

You should understand Chapter 12 (Tensor autodiff and `matmul`
composition), Chapter 13 (what optimizer state costs), and Chapter 20's
`Linear` layer, which is what gets adapted.

## 1. The decomposition

For a base weight $W \in \mathbb{R}^{d_{in} \times d_{out}}$, LoRA learns
$A \in \mathbb{R}^{d_{in} \times r}$ and $B \in \mathbb{R}^{r \times
d_{out}}$ with $r \ll \min(d_{in}, d_{out})$ and computes

$$
h = xW + b + \frac{\alpha}{r}\,(xA)B
$$

The base ($W$, $b$) is frozen; only $A$ and $B$ train. Trainable elements
drop from $d_{in} d_{out}$ to $r(d_{in} + d_{out})$ — for a 16×8 layer at
$r = 2$ that is 48 instead of 136, and the ratio improves quadratically
with width. AdamW moments shrink by the same factor, which is usually the
real memory win.

The scale $\alpha / r$ decouples the learning rate from the rank: doubling
$r$ without the divisor would double the update's magnitude at equal
factor scale, forcing hyperparameter retuning per rank.

## 2. Initialization is the safety argument

$B$ starts at zero and $A$ starts random. Two consequences, both tested:

1. a freshly wrapped layer computes *exactly* the base function — the
   update is a product with a zero matrix, so equality is bitwise, not
   approximate. Adaptation starts from "no change";
2. the first gradient step is non-degenerate: $\partial L / \partial B
   \propto (xA)^\top \delta$ is nonzero because $A$ is random, so $B$
   moves first and $A$ follows once $B$ is nonzero. Initializing *both*
   to zero would leave every gradient zero forever — a saddle point the
   suite's learning test would expose immediately.

## 3. Freezing is an optimizer contract

Nothing in the forward graph "knows" the base is frozen: $W$ sits on the
computation path and receives gradients during backward like any other
tensor. Freezing happens at the optimizer boundary — `trainableParameters`
returns only $A$ and $B$, and only what the optimizer receives changes.

This is worth internalizing because it is how parameter freezing works in
every major framework, and it is a classic source of silent bugs: pass the
wrong parameter list and you are full fine-tuning with extra steps. The
test pins the contract the strong way: after 25 optimizer steps, the base
weight and bias values are compared *bitwise* against their pre-training
snapshot.

## 4. Merging makes inference free

Because the update is linear, it folds into the base:

$$
W' = W + \frac{\alpha}{r} A B
$$

A merged layer is an ordinary `Linear` — same shape, same cost, no adapter
matmuls at serving time. The suite verifies that the merged layer's output
matches the adapted forward pass to $10^{-12}$ on random inputs (the
association order of the floating-point operations differs, so bitwise
equality is deliberately *not* claimed here), and that the bias passes
through untouched.

Merging also frames the trade-off honestly: a merged model loses the
ability to hot-swap adapters, which is exactly the multi-task serving
trick that keeps many LoRAs resident over one base model.

## 5. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `LoraLinear` suite includes a recovery experiment: the target is the
base layer plus a known rank-1 perturbation, the student is a fresh rank-1
adapter, and 400 SGD steps must cut the loss at least twentyfold — the
target is *reachable*, so failure to approach it indicates a gradient or
scaling bug rather than an expressiveness limit.

## 6. Implementation walkthrough

`LoraLinear` validates the rank against `min(in, out)` — a full-rank
"adapter" would defeat the decomposition and usually signals a
misconfiguration — plus a positive finite `alpha` and both adapter shapes
against the base layer. `scaling` is precomputed once as `alpha / rank`.

`apply` is one line of composition: `base(input) +
input.matmul(adapterDown).matmul(adapterUp).scale(scaling)`. Every
operation already has a verified backward rule, so no new gradient code is
written for the entire method — the same composition-over-kernels
discipline as Chapters 28a and 28b. The association `(xA)B` rather than
`x(AB)` matters operationally: it costs `r(d_{in} + d_{out})` multiplies
per row instead of materializing the $d_{in} \times d_{out}$ product every
forward pass.

`merged` computes the update product once at the value level, adds it
elementwise to the base weight's values, and builds a fresh `Linear` via
`fromValues` — the merged layer shares no tensors with the adapter, so
later adapter training cannot silently mutate it.

`wrap` implements the published initialization (Xavier down, zeros up);
`fromValues` exists for deterministic tests and for loading trained
adapters.

## 7. Reading the tests

- *fresh-wrap identity* uses `Assert.equal` on value vectors — bitwise —
  because zero-times-anything is exact;
- *merge equivalence* uses a $10^{-12}$ tolerance with the reason
  documented: association order differs between the adapted and merged
  paths;
- *frozen base* trains for 25 steps and compares base values bitwise,
  while also asserting the up projection actually moved (a test that
  freezes everything passes vacuously otherwise);
- *reachable recovery* constructs a target inside the student's
  hypothesis class, so convergence failure means a bug, not a capacity
  excuse;
- *accounting* checks $r(d_{in}+d_{out})$ exactly and the scaling value;
- *failures* cover zero and full rank, non-positive alpha, mis-sized
  adapter values, and wrong input width.

## 8. Debugging checklist

1. Check the fresh-wrap identity first; if it is not exact, the up
   projection is not actually zero-initialized.
2. If nothing learns, print both adapter gradients: both zero means both
   factors were zero-initialized; only-down zero for many steps is normal
   early on.
3. If the base changed during training, audit what parameter list the
   optimizer received — freezing lives there and nowhere else.
4. If merged and adapted outputs diverge beyond rounding, check the
   scaling: merging with `alpha` while running with `alpha / r` is the
   classic factor-of-$r$ bug.
5. Compare trainable counts against the formula before celebrating a
   memory win; biases and norms you forgot to freeze show up here.

## 9. Failure modes to test

- both adapter factors initialized to zero (permanent saddle);
- optimizer given `allParameters` instead of `trainableParameters`;
- merge applied with the wrong scale or applied twice;
- adapter shapes transposed (compiles nowhere here, but the shape errors
  must say which projection is wrong);
- rank equal to or above the layer's minimum dimension;
- evaluating memory savings while still holding full-model optimizer
  state somewhere else.

## Exercises

1. Wrap MiniGPT's four attention projections (Chapter 19) with LoRA,
   freeze everything else, and fine-tune on a small corpus; compare
   validation loss and trainable-parameter counts against full
   fine-tuning at the same step budget.
2. Add `unmerge` (subtract the update from a merged layer) and prove a
   merge/unmerge round trip restores the base bitwise or explain why it
   cannot.
3. Sweep $r \in \{1, 2, 4, 8\}$ on the recovery experiment with a rank-4
   target and plot final loss against $r$; identify where capacity, not
   optimization, is the binding constraint.
4. Serve two different adapters over one shared frozen base and measure
   the memory saved versus two merged model copies.

## Completion criteria

You are done when you can:

- write the LoRA forward pass and the merge formula from memory;
- explain both initialization choices and what breaks under zero-zero;
- state where freezing is enforced and how to test it honestly;
- derive the trainable-parameter and optimizer-state savings for a given
  layer and rank;
- argue when to merge and when to keep adapters separate.

## Primary sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
