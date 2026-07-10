# Contribution guide

## One learning concept per commit

Use Conventional Commit titles. A commit should introduce one concept that can
be compiled, tested, and explained independently. Keep implementation, tests,
and its hands-on chapter together.

Examples:

```text
feat(autodiff): implement scalar reverse-mode differentiation
test(attention): verify future tokens receive no prefix gradient
docs(inference): derive nucleus sampling
```

## Documentation comments

Public types and non-obvious public operations should have English Scaladoc
covering the contract, not a restatement of the method name. Include the
relevant items:

- mathematical meaning and input/output shapes;
- invariants established by construction;
- mutation and ownership rules;
- determinism and random-state ownership;
- failure conditions and returned error channel;
- complexity or allocation behavior when it affects design;
- educational simplifications compared with production systems.

Implementation comments should explain why a numerical, security, or indexing
choice exists. Avoid comments that merely translate a line of code into prose.

## Tests

No external test library is used in the main learning path. Register every new
suite in `learnai.testing.AllTests` and cover more than a happy path.

Depending on the concept, include:

- hand-computable examples;
- empty, zero, maximum, and invalid inputs;
- shape/range/type errors with useful messages;
- determinism under a fixed seed;
- algebraic properties and round trips;
- finite-difference gradient checks;
- optimized/reference equivalence;
- corruption, timeout, budget, and permission failures;
- end-to-end integration without replacing focused unit tests.

Never weaken a tolerance merely to hide an unexplained numerical difference.
Record the expected error source and choose a tolerance derived from the scale
of the computation.

## Required checks

Run the repository from its pinned environment:

```console
nix develop -c sbt check
git diff --check
```

When changing a runnable chapter, also execute its `runMain` command and inspect
the observed loss, output, or metrics. Do not make performance claims without a
warm benchmark and recorded environment.

## Dependency policy

The core hands-on uses only Scala 3 and JDK APIs. Implement the mechanism first.
External libraries may later appear in isolated comparison adapters, with their
cost, hidden behavior, and version pin documented.
