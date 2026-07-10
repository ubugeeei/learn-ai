# Anatomy of a complete implementation chapter

## Why this standard exists

A short conceptual summary can help someone who already knows the subject, but
it is not a zero-background hands-on. In this repository, a completed chapter
must help a learner reconstruct the implementation—not merely recognize its
name.

The prose, equations, code, tests, and runnable experiment are different views
of one mechanism:

```text
intuition
  -> concrete numbers
  -> symbols and shapes
  -> API contract
  -> loop/index implementation
  -> tests and independent oracle
  -> observed behavior
  -> limitations and next design
```

If one arrow is missing, the learner has to guess why the code works.

## Required learning layers

### 1. Motivation in plain language

State the problem before naming the technique. Explain what fails without the
new component and what responsibility the component owns.

Bad:

> Implement softmax.

Useful:

> Logits are unrestricted scores, but categorical sampling needs non-negative
> values that sum to one. Softmax performs that conversion while preserving
> score order; a maximum subtraction prevents overflow without changing the
> result.

### 2. A hand-computable example

Use the smallest numbers that exercise the important behavior. Show every
intermediate value when practical. The example should be small enough to check
with paper and a calculator.

For a matrix operation, write the actual rows, columns, products, and sums. For
an agent state machine, list the exact states and events. For serialization,
draw the exact byte layout.

### 3. Symbols, types, and shapes

Every equation must connect to executable names.

| Mathematics | Scala | Meaning |
| --- | --- | --- |
| (X\in\mathbb{R}^{T\times C}) | `input: Tensor` | one hidden row per token |
| (W_Q\in\mathbb{R}^{C\times C}) | `queryProjection.weight` | query parameters |
| (Q=XW_Q) | `queryProjection(input)` | projected queries |

State preconditions and output shapes before explaining the algorithm. Shapes
are part of correctness, not optional commentary.

### 4. Source map

Name the production and test files. Identify the public entrypoints and the
private helpers they delegate to. A learner should know where to place a
breakpoint before reading the walkthrough.

### 5. Implementation walkthrough

Follow execution order. For each non-obvious block, explain:

- what invariant is already true;
- what data is read;
- what data is allocated or mutated;
- why the loop/index order matches the equation;
- what invariant becomes true afterward;
- where a failure is reported.

Do not paste an entire source file. Quote the smallest code fragment needed,
then explain each line's responsibility and connect it back to the symbols.

### 6. Design alternatives

Explain at least one plausible alternative and why the chapter does not use it
yet. Examples include immutable versus mutable buffers, exceptions versus
`Either`, scalar graphs versus tensor kernels, and full recomputation versus a
cache.

An educational simplification must be labeled. Otherwise a learner may mistake
it for production best practice.

### 7. Reading the tests

Tests are executable explanations. For each important test category, state:

- the bug it is meant to catch;
- why the expected value is independent of the implementation;
- what a failure usually means;
- why the tolerance or seed is appropriate.

Include normal, boundary, invalid, and relevant property/numerical cases. An
end-to-end success test does not replace focused unit tests.

### 8. Debugging checklist

Give a symptom-first decision path:

```text
NaN loss
  -> inspect logits before softmax
  -> verify maximum subtraction
  -> inspect target range
  -> inspect gradient norm
  -> reduce learning rate only after locating the first invalid value
```

The checklist should teach investigation, not suggest random parameter changes.

### 9. Runnable observation

Provide exact Nix commands and explain what output proves—and what it does not
prove. A falling training loss proves graph connectivity and optimization on
that data; it does not prove generalization.

### 10. Limits and next connection

State complexity, memory, numerical, concurrency, safety, and scale limits as
relevant. End by showing which later chapter consumes the new component.

## Definition of done for documentation

For an implemented technical chapter, reviewers should be able to answer yes
to these questions:

- Can a beginner state the problem before seeing the technique name?
- Is there a hand-computable example?
- Do equations map to Scala names and shapes?
- Does the walkthrough follow the real execution path?
- Are mutation, ownership, and error channels explicit?
- Do tests have explained independent oracles?
- Is there a symptom-driven debugging section?
- Does the run command state what its output demonstrates?
- Are educational simplifications distinguished from production systems?
- Are primary sources linked where a research mechanism is involved?

## How to study a chapter

Use four passes rather than reading once:

1. **Prediction:** read the problem and worked example, then predict the API.
2. **Tracing:** follow one input through the source with the walkthrough.
3. **Falsification:** read each test and name the bug it would catch.
4. **Reconstruction:** close the source and implement the component from the
   contract, equations, and examples.

You understand the chapter when you can reconstruct and debug it—not when the
finished code merely looks familiar.

The test suite enforces a deliberately modest floor for implemented chapters:
an implementation walkthrough, a reading/test-oracle section, a debugging
checklist, and enough space to develop the explanation. Passing that structural
check does not establish teaching quality; it only prevents a detailed chapter
from silently shrinking back into a brief overview.
