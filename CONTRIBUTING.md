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

## Hands-on chapter depth

Implemented chapters follow
`docs/00-guide/04-chapter-anatomy.md`. A conceptual overview alone is not
complete documentation. Keep the following with the code change:

- a hand-computable worked example;
- an equation/type/shape map;
- a source-file and execution-path walkthrough;
- explicit invariants, ownership, mutation, and error behavior;
- test-oracle explanations and a debugging checklist;
- a runnable command with interpretation of its output;
- limitations, production differences, and the next dependency.

Prefer precise explanation of the actual Scala code over broad background
prose. A learner should be able to reconstruct the implementation from the
chapter without reverse-engineering unexplained loops.

`DocumentationStructureSuite` enforces the common walkthrough, reading, and
debugging sections for implemented chapter directories. It is a regression
floor, not a substitute for reviewing the technical explanation.

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

## Translations

Japanese translations mirror `docs/` under `docs/ja/` and follow
`docs/00-guide/07-translation-policy.md`: never coin terminology, prefer
globally standard English terms as-is, and keep established Japanese
renderings. Test guards enforce the mirror structure.
## Code readability is part of correctness

All Scala sources are formatted by Scalafmt. Run `sbt scalafmtAll scalafmtSbt`
before committing; `sbt check` rejects formatting drift. Keep public domain
types and non-obvious contracts documented with Scaladoc that explains
invariants, ownership, failure behavior, and units instead of restating names.

Feature tests are executable specifications colocated with their implementation.
Declare the complete behavior list with `specify(...)`, and name cases after an
observable property. Every public boundary should normally have a success case,
a boundary case, and a rejected-input case. Numerical algorithms additionally
need an independently calculated reference or property test.
