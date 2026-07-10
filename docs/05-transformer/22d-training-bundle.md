# 22d — Persisted training bundles

## What you will build

Chapter 22c proved that a training run's complete state fits in one value
and that resuming from it is bitwise exact — inside one process. Real
interruptions kill the process. This chapter moves the state to disk
without weakening the contract: the resumed-from-file run must still equal
the uninterrupted run bit for bit, and a bundle must refuse to resume an
experiment it does not belong to.

In this chapter you will:

- serialize `MiniGptTrainingState` with exact bit-level doubles;
- reuse Chapter 25's format discipline: magic, version, bounded reads,
  trailing SHA-256, atomic replace;
- bind every bundle to a Chapter 22a `experimentId` and refuse mismatched
  resumes explicitly;
- verify the whole path with a straight-versus-disk-resume bitwise oracle.

## Prerequisites

You should understand Chapter 22a (canonical experiment identity), Chapter
22c (what the state contains and why), and Chapter 25 (the checkpoint
format discipline this file layout copies).

## 1. Exactness survives serialization only on purpose

The resume contract is bitwise, so the file format must be too.
`DataOutputStream.writeDouble` writes the exact IEEE 754 bits
(`doubleToLongBits` under the hood), and `readDouble` restores them, so
every loss, moment, and parameter survives unchanged. The tempting
alternative — a JSON or CSV text dump — silently breaks the contract:
decimal text rounds, and a value that rounds by one ulp produces a
training trajectory that diverges from the straight run a few updates
later. If you want the state human-readable, render a *report* next to
the bundle; never make the report the source of truth.

## 2. The bundle belongs to an experiment

A state file that "loads fine" can still be wrong for the run you are
about to continue: same architecture, different corpus; same corpus,
different schedule; same everything, different code revision. Chapter 22a
already solved identity — `experimentId` is the SHA-256 of the canonical
model/data/training/code/environment specification. The bundle stores that
id, and `loadForResume(path, expectedExperimentId)` compares it against
the identity the *caller derives from its current configuration*:

```text
caller: id = ExperimentSpecification(...).experimentId   (from live config)
bundle: id stored at save time
equal  -> resume
differ -> explicit refusal naming both ids
```

The refusal is an error, not a warning, because a mismatched resume does
not fail loudly on its own — it trains a plausible-looking but
incomparable run, which is the most expensive kind of wrong.

## 3. Format discipline, restated

The layout follows Chapter 25's checkpoint rules because the failure
modes are identical:

```text
magic "LAIBND01" | version | experiment id
| completedUpdates | tokensSeen | best/initial validation | RNG counter
| parameter value groups | optimizer step | first moments | second moments
| SHA-256 over everything above
```

Every count read from the file is bounded before allocation (tensor
count, scalars per tensor, string length, total file size), so a corrupt
or hostile file cannot request an absurd allocation. The checksum is
verified before any parsing. Writes go to a temporary file first and move
into place atomically, so a crash mid-save leaves the previous bundle
intact rather than a half-written one. Loaded values pass back through
the `MiniGptTrainingState` and `AdamWSnapshot` constructors, whose
invariants (finiteness, non-negative second moments, bookkeeping ranges)
reject corruption that a checksum alone would only catch probabilistically.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

The `TrainingBundle` suite trains, saves, and resumes through a real
temporary file, including the corruption and refusal paths.

## 5. Implementation walkthrough

`save` validates the experiment id format first — before any bytes are
written — so a rejected save provably leaves no file behind. It then
streams the state through a `DataOutputStream` in a fixed field order,
appends the SHA-256 of the payload, and hands the result to
`writeAtomically`, which is the same temp-file-plus-atomic-move sequence
Chapter 25 uses, including the fallback for filesystems without atomic
move support.

`load` is the mirror with paranoia added: size bounds before reading,
checksum before parsing, magic and version before fields, bounded counts
before each allocation, and a trailing-bytes check at the end so an
appended payload cannot ride along unnoticed. The state is rebuilt through
its validating constructors rather than trusted field assignment.

`loadForResume` composes `load` with the identity comparison and returns
only the state, because that is all a resuming trainer needs; the wider
`load` exists for inspection tooling. Resuming then *is* Chapter 22c:
`ResumableMiniGptTraining.train(freshModel, split, config, state, n)`.

The value-group encoding (count, then per-group count and doubles) is
shared by parameter values and both moment arrays, so one bounded
reader/writer pair covers all three sections.

## 6. Reading the tests

- the round trip compares the loaded state to the saved one with case-
  class equality — every double, moment, and counter, bitwise;
- the disk-resume oracle trains four updates, saves, *forgets everything
  except the file and the caller-side identity*, loads, resumes four more
  updates into a differently seeded model, and matches the straight
  eight-update run's metrics and final weights exactly;
- the refusal test saves under one identity and attempts resume under
  another, asserting the message names both ids and refuses;
- corruption tests flip one payload byte (checksum), truncate the file,
  and damage the magic; all must fail before any state escapes;
- the format test rejects a malformed experiment id and proves no file
  is created by a rejected save.

## 7. Debugging checklist

1. If a resumed-from-disk run diverges from a straight run, first re-run
   the in-memory Chapter 22c sweep; if that passes, the bug is in
   serialization, not training.
2. Diff the loaded state against the saved one field by field; the first
   unequal field names the mis-ordered or mis-typed write.
3. Never debug exactness through printed decimals — compare
   `doubleToLongBits` values.
4. If refusals fire unexpectedly, render both canonical specifications
   (Chapter 22a) and diff the JSON; the differing field is the drift.
5. A bundle that loads but fails a constructor invariant means the format
   version changed semantics without a version bump.

## 8. Failure modes to test

- doubles routed through text and rounded;
- fields written and read in different orders;
- checksum verified after parsing instead of before;
- unbounded counts allocated from a corrupt header;
- a crash between truncating and rewriting the destination file;
- resume accepted under a different experiment identity;
- a version bump that silently reinterprets old files.

## Exercises

1. Wire `ExperimentSpecification.experimentId` end to end: derive the id
   from live configuration on both save and resume, and demonstrate a
   refusal by changing only the corpus text.
2. Add periodic bundle writes every $m$ updates to the resumable trainer
   and measure the overhead honestly (Chapter 22 rules).
3. Keep the last $k$ bundles with a retention policy and prove the newest
   verified bundle is chosen after a simulated partial write.
4. Write a small inspection command that prints a bundle's metadata and
   summary statistics without loading parameter values.

## Completion criteria

You are done when you can:

- explain why bitwise resume forbids decimal text serialization;
- list the verification order (size, checksum, magic, version, bounded
  fields, trailing bytes) and what each step protects against;
- explain why identity refusal must be an error rather than a warning;
- describe what atomic replacement guarantees across a crash;
- state where validation lives (constructors) and why re-validating on
  load matters even with a checksum.

## Primary sources

- [Fast Splittable Pseudorandom Number Generators (Steele, Lea, Flood)](https://doi.org/10.1145/2660193.2660195)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [IEEE 754 floating-point standard](https://en.wikipedia.org/wiki/IEEE_754)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
