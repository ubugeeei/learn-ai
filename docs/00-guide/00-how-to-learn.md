# How to learn with this repository

## Prerequisites

No programming, terminal, Git, Scala, or advanced mathematics knowledge is
assumed. Reading alone is not sufficient, so every chapter follows the same
experimental loop:

1. **Predict** the result or learning curve before running the code.
2. **Implement** the equation by mapping every symbol to a typed value.
3. **Measure** tests, errors, time, and memory.
4. **Explain** why the result occurred in your own words.
5. **Change one variable** and compare the observation with your prediction.

## Chapter structure

Each chapter uses a consistent format:

- **What you will build**: an executable deliverable;
- **Why it matters**: its place in an LLM or agent;
- **Theory**: equations, shapes, and intuition;
- **Implementation**: small, testable steps;
- **Verification**: commands and expected properties;
- **Observations**: values worth inspecting;
- **Exercises**: focused variations;
- **Completion criteria**: evidence required before moving on.

## How to read mathematics

Mathematical notation is a compact language for removing ambiguity. Every new
symbol in this course is connected to its pronunciation, type, shape, and code.

For example, a dot product is written as:

\[
y = \boldsymbol{w}\cdot\boldsymbol{x}
  = \sum_{i=1}^{n} w_i x_i
\]

| Symbol | Meaning | Type | Code |
| --- | --- | --- | --- |
| \(n\) | number of elements | positive integer | `x.size` |
| \(\boldsymbol{x}\) | input vector | \(\mathbb{R}^n\) | `VectorD` |
| \(\boldsymbol{w}\) | weight vector | \(\mathbb{R}^n\) | `VectorD` |
| \(y\) | dot-product result | scalar | `w.dot(x)` |
| \(\sum\) | sum over a range | operation | a loop or `foldLeft` |

When you see an equation, first write down the input and output types and
shapes. Shape tracing is one of the most effective Transformer debugging tools.

## Rules for experiments

- After copying an example, hide it and write it again.
- Fix random seeds when comparing two conditions.
- Change one condition at a time.
- Do not grow code whose expected result is unknown.
- Compare floating-point values with a justified tolerance.
- Record failed experiments with their hypothesis and observation.

## Time estimate

The complete 40-chapter path and capstone are expected to take roughly 250 to
400 hours with exercises. Experience and optional extensions change that range
substantially. Progress is defined by being able to answer:

> What are this function's inputs, outputs, invariants, complexity, and sources
> of error?

## Debugging order

1. Read the first causal error rather than the final cascade.
2. Inspect shapes, ranges, `NaN`, and `Infinity`.
3. Reduce the input to one element or two tokens.
4. Compare with a hand-computable reference.
5. Revert the one condition changed most recently.

Small inputs reveal broken invariants more reliably than guessing from a large
model run.

## How to read the expanded implementation chapters

The detailed chapter contract is defined in
[Anatomy of a complete implementation chapter](04-chapter-anatomy.md). Do not
begin by reading the finished source top to bottom. Use this order:

1. work through the concrete numbers without code;
2. annotate every symbol with its Scala name and shape;
3. predict public preconditions and failure behavior;
4. trace the implementation in execution order;
5. predict each test's failure before running it;
6. run the focused suite and intentionally break one invariant;
7. restore the implementation and explain the observed failure.

This turns code reading into an experiment. It also makes it obvious when a
chapter is still too thin: if the prose cannot guide the trace and predict the
tests, the documentation is incomplete.
