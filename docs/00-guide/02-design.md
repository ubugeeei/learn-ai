# Design principles

These principles apply to both the code and the explanations. Observability and
correctness take priority over convenient abstraction.

## 1. Do not hide the mechanism being taught

The main path uses Scala 3 and JDK APIs only. It does not begin with BLAS,
machine-learning, tokenizer, JSON, or HTTP helper libraries. Data layout,
complexity, allocation, numerical error, and failure behavior remain visible.

External libraries may later appear as isolated comparison adapters after the
mechanism has been implemented and tested.

## 2. Reject invalid states at constructors and typed boundaries

- Reject vector and matrix shape mismatches immediately.
- Use distinct types for token IDs, tool calls, and other semantic values.
- Represent expected I/O failures with `Either` instead of exceptions.
- Confine mutation to training steps, optimizer updates, and agent-loop state.

Numeric kernels may own arrays for performance, but mutable storage is never
exposed to callers.

## 3. Make determinism the default

Random seeds, parameter initialization, and data order are explicit. The same
environment and seed should reproduce tests and chapter observations. Chapters
that introduce nondeterministic parallelism must measure the tradeoff.

## 4. Prove correctness on small inputs

- hand-computable unit tests;
- reference-vs-optimized equivalence;
- numerical-vs-automatic gradient checks;
- cache/quantization equivalence before and after optimization;
- serialization round trips and corruption tests.

Large end-to-end tests do not replace focused invariants.

## 5. Include shapes and units

Every Tensor equation includes shape information. Time is a duration, data is
measured in bytes, and probabilities have explicit ranges.

Example embedding lookup:

```text
tokenIds:      [batch, time]
embedding:     [vocabulary, channels]
output:        [batch, time, channels]
```

## 6. Keep dependency direction one-way

```text
apps and chapter experiments
  -> agent / language model / neural network
  -> math and tensor primitives
  -> Scala/JDK standard library
```

Lower layers never depend on higher layers. Chapter numbers describe teaching
order and do not leak into reusable package names.

## 7. Separate educational simplifications from production claims

Every chapter identifies:

- what was simplified;
- which property is lost;
- what production implementations usually add;
- what measurements should decide adoption.

A correct reference implementation is not automatically fast, safe, or ready
for production traffic.

## 8. Pass agent capabilities explicitly

The agent runtime has no implicit filesystem, shell, or network authority. It
receives a list of allowed tools, validates their inputs, executes within
budgets and timeouts, and records structured observations. Stop conditions,
idempotency keys, and audit events are protocol features rather than prompt
suggestions.

## 9. Commit one learning concept at a time

Use Conventional Commit titles. Put implementation, tests, and its chapter in
the same coherent commit.

```text
feat(math): implement immutable vectors
feat(autodiff): add scalar reverse-mode differentiation
docs(attention): derive scaled dot-product attention
test(agent): cover malformed tool arguments
```
