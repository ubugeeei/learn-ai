# 31a — SFT and chat templates

## What you will build

Pretraining teaches a model to continue text; it does not teach it to be
one side of a conversation. Supervised fine-tuning (SFT) closes that gap
with surprisingly little machinery: render conversations through a fixed
template, then train the same causal objective — but only on the tokens
the assistant is supposed to produce. Almost every SFT bug in practice is
a template bug or a mask bug, which is why this chapter is mostly about
making both exactly checkable.

In this chapter you will:

- define validated conversation structure (roles, ordering rules);
- render conversations through a token-level chat template;
- derive the loss mask that trains assistant spans and nothing else;
- run a held-out masked evaluation and a small end-to-end SFT experiment.

## Prerequisites

You should understand Chapter 17b (loss masks and `crossEntropyMasked`),
Chapter 21 (the MiniGPT being tuned), and Chapter 15 (tokenization — this
chapter deliberately starts *after* it, at the token level).

## 1. The template is part of the model

A chat template maps structured turns into one flat token sequence:

```text
<|system|> s... <|eot|> <|user|> u... <|eot|> <|assistant|> a... <|eot|>
```

The four reserved tokens (three role markers and an end-of-turn) are as
much a part of the trained artifact as the weights: serve the model with
a different template than it was tuned on and quality silently collapses.
This implementation keeps the template token-level — messages carry
`Vector[TokenId]`, not strings — so every rendered position is exactly
assertable and no tokenizer ambiguity can hide in the tests.

Structure is validated *before* rendering: an optional system message
first, no assistant opening, no two consecutive turns from one role.
Malformed conversations are refused as data errors, because a
double-assistant turn does not crash anything downstream — it just
corrupts the mask, which is worse.

## 2. The mask: train the assistant, condition on the rest

For a rendered sequence, position $t$ predicts token $t+1$, and the SFT
mask keeps exactly the predictions whose *target* is assistant content or
an assistant end-of-turn:

$$
\text{mask}(t) = \text{assistantSpan}(t + 1)
$$

Three deliberate choices, each with a reason the tests encode:

- *user and system tokens are conditioning, not targets* — training on
  them teaches the model to imitate users, which measurably degrades the
  assistant persona;
- *the assistant role marker is not a target* — the serving stack always
  supplies it, so the model is conditioned on "the assistant speaks
  next", never asked to decide it;
- *the assistant end-of-turn is a target* — knowing where to stop is
  learned behavior, and forgetting to train it produces models that
  ramble past their turn.

Counting gives the invariant the suite asserts: every assistant message
contributes exactly `content.size + 1` trainable predictions.

The rendering also refuses reserved tokens inside message content. A
document that embeds a fake end-of-turn would truncate the trained span —
a prompt-injection-shaped data bug caught at validation time.

## 3. Evaluation that means something

`heldOutLoss` computes the masked loss over unseen conversations,
weighted by each conversation's trainable-target count, giving a mean
loss per assistant token. Weighting matters: an unweighted mean over
conversations lets many short turns drown a few long ones, and quietly
changes when the evaluation set is re-bucketed. The evaluation is fully
deterministic — no sampling anywhere — so two runs return the identical
double, which the suite checks with exact equality.

The chapter's honest limitation: one conversation is one example here,
and conversations longer than the model context are refused rather than
truncated. Combining SFT with Chapter 17b packing (many conversations per
window, masks composed) is an exercise, not an omission.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `ChatTemplateSft` suite includes a 100-step AdamW fine-tune of a tiny
MiniGPT on one conversation, asserting at least a fivefold drop in
assistant-span loss — a memorization smoke test that fails if the mask,
alignment, or gradient path is wrong anywhere.

## 5. Implementation walkthrough

`Conversation` carries the ordering rules in its constructor, so an
invalid conversation cannot exist as a value — the same
validate-at-construction discipline as every config class in this course.
`ChatSpecialTokens` requires four distinct tokens and maps roles to
markers.

`ChatTemplate.render` builds each turn as `marker + content + endOfTurn`
alongside a parallel boolean span vector (`false` for the marker, the
message's assistant-ness for content and end-of-turn), then concatenates
both. The span is computed during rendering, not recovered from tokens
afterwards — recovering it by scanning for markers would re-introduce
exactly the ambiguity the reserved-token validation exists to prevent.

`trainingExample` is three lines of alignment: inputs are `tokens.init`,
targets are `tokens.tail`, and the mask is `assistantSpan.tail` — the
span flags of the targets. The `SftExample` constructor refuses examples
with zero trainable targets, so a user-only conversation fails fast.

`SftEvaluation.exampleLoss` delegates to `MiniGpt.lossMasked` from
Chapter 17b; SFT needed no new loss machinery, which is the payoff of
building the mask support one chapter early. `heldOutLoss` accumulates
example losses weighted by trainable-target counts and divides once.

## 6. Reading the tests

- the rendering test writes out all eleven tokens and eleven span flags
  of a three-turn conversation by hand;
- the alignment test checks that every trainable target is assistant
  content or end-of-turn, and that the role marker is never trained;
- the counting test asserts the `content + 1` invariant across a
  five-message, two-assistant conversation;
- validation tests cover assistant-first, mid-conversation system turns,
  consecutive same-role turns, empty content, assistant-free training
  examples, embedded reserved tokens, and duplicate specials;
- the fine-tuning experiment ties template, mask, loss, and optimizer
  together end to end;
- the evaluation test compares `heldOutLoss` against a hand-computed
  weighted mean, checks determinism exactly, and exercises the context
  overflow refusal.

## 7. Debugging checklist

1. Print one rendered conversation with its span flags before anything
   else; template bugs are visible to the eye and invisible to loss
   curves.
2. Check the counting invariant next: trainable targets must equal
   $\sum (\lvert \text{assistant content} \rvert + 1)$.
3. If the tuned model rambles, verify the assistant end-of-turn is in
   the span (and therefore trained).
4. If it imitates users, the mask is shifted by one or built from input
   flags instead of target flags.
5. If serving quality is far below evaluation, diff the serving template
   against the training template token by token.

## 8. Failure modes to test

- mask aligned to inputs rather than targets (off-by-one);
- assistant role marker trained as a target;
- end-of-turn excluded from the span;
- reserved tokens accepted inside content;
- unweighted held-out means that shift when the set is re-bucketed;
- conversations silently truncated to the context length;
- training on conversations with no assistant turn.

## Exercises

1. Pack multiple short conversations into one window with Chapter 17b's
   `SequencePacking` and compose both masks; verify the counting
   invariants still hold.
2. Fine-tune only LoRA adapters (Chapter 31b) on the attention
   projections instead of full weights, and compare held-out loss per
   trainable parameter.
3. Add a weight for the system turn (train it at 0.1x) by extending the
   mask from booleans to per-position weights, and re-derive the loss
   normalization.
4. Build a template-mismatch detector: render with template A, evaluate
   with template B, and quantify the loss inflation.

## Completion criteria

You are done when you can:

- write the template layout and its span rules from memory;
- justify all three mask choices (user tokens, role marker, end-of-turn)
  in behavioral terms;
- state and use the counting invariant as a first-line check;
- explain why evaluation weights by trainable targets;
- explain why the template must ship with the model.

## Primary sources

- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
