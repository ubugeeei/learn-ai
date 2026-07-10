# Progressive writing style

Every hands-on chapter must satisfy one contract: **a motivated
high-school student with no machine-learning background can start reading
it, and a working engineer finds full professional depth by the end.**
The two audiences are served by the same document through a deliberate
ramp, not by writing two documents or by lowering the ceiling. Length is
not a constraint; missing rungs on the ladder are.

## The ramp

Chapters climb through four layers, in order. Later layers never appear
before earlier ones have prepared them.

1. **Hook — the problem in plain words.** Open with what breaks or what
   question exists, stated without any field vocabulary, ideally with an
   everyday comparison. A reader who knows nothing should finish the
   first paragraphs able to say what problem this chapter solves and why
   anyone cares.
2. **Intuition — small numbers before symbols.** Work the core idea with
   a tiny concrete example the reader can check with mental arithmetic or
   a pocket calculator: three tokens, a 2×2 matrix, five probabilities.
   Show the actual numbers in, the actual numbers out.
3. **Formalization — notation earned, not assumed.** Introduce each
   equation as the *generalization of the worked example the reader just
   verified*, and define every symbol in a full sentence at first use.
   No symbol appears before its definition; no acronym appears before its
   expansion.
4. **Professional depth — the existing rigor, unchanged.** Implementation
   walkthroughs, oracles, failure modes, debugging checklists, and
   completion criteria keep their precision. This layer is never dumbed
   down; the ramp exists so more readers arrive here.

## Language rules

- Define every technical term in one plain sentence the first time it
  appears in a chapter — even terms defined in earlier chapters get a
  one-line refresher, because readers arrive mid-course.
- Refresh prerequisites inline with a recap blockquote rather than only a
  pointer: `> Recap: a *gradient* is the list of slopes telling us how
  the loss changes if each parameter moves a little (Chapter 09).`
- Prefer short sentences and active voice. One idea per sentence. If a
  sentence needs three commas, split it.
- Every analogy is followed by one sentence saying where it breaks down;
  an analogy without its limits teaches a wrong model.
- Numbers before Greek: any equation of substance is preceded by the same
  computation done with concrete values.
- Spell out what pronouns refer to. "This" never floats.

## What must not change

- **No technical content is deleted.** Rewriting for accessibility means
  adding rungs below the existing material, reorganizing at most. Every
  oracle, failure mode, caveat, and completion criterion of the original
  survives.
- Correctness language stays exact: "bitwise", "within tolerance",
  "deterministic" keep their precise meanings and their justifications.
- The chapter anatomy and its required sections (see
  [chapter anatomy](04-chapter-anatomy.md)) still apply: What you will
  build, Prerequisites, Run the experiment, Implementation walkthrough,
  Reading the tests, Debugging checklist, Failure modes to test,
  Exercises, Completion criteria, Primary sources.
- Math uses GitHub dollar delimiters only; code blocks, identifiers,
  commands, and links are preserved exactly.
- Honesty rules from [progress standards](03-progress.md) still apply:
  no performance claims without measurement discipline, no proof by
  plausibility.

## Structure devices

- **Recap blockquotes** (`> Recap: ...`) refresh a prerequisite in place.
- **Worked mini-examples** carry the numbers: introduce them early and
  reuse the *same* numbers when the equation and the implementation
  appear, so the reader can trace one calculation through all layers.
- **"Why this matters" sentences** close each major section by connecting
  it back to the chapter's opening problem.
- Section titles state claims where possible ("Softmax turns scores into
  probabilities") rather than topics ("Softmax").

## Quality bar

A chapter passes review when:

1. the first two sections contain no undefined field vocabulary;
2. every equation has a concrete-number companion;
3. a reader can state, after each section, what problem was just solved;
4. the professional back half is as rigorous as before the rewrite;
5. the word count grew only by adding explanation, never by padding.
