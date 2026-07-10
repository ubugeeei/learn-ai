# 22a — Experiment identity and reproducibility records

## What you will build

Two runs with the same label are not necessarily the same experiment. A seed,
corpus byte, optimizer option, source revision, or Nix revision can change the
result. This chapter implements a canonical experiment specification and an
execution manifest using only the strict JSON and JDK cryptography already in
the repository.

Sources:

- `src/main/scala/learnai/experiment/ExperimentManifest.scala`;
- `src/test/scala/learnai/experiment/ExperimentManifestSuite.scala`;
- `src/main/scala/learnai/training/MiniGptTrainingLab.scala`.

The deliverable is not a filename convention. It is a deterministic identity
whose inputs are explicit enough for another engineer to decide whether two
results are comparable.

## 1. Reproducibility is a dependency graph

A training result depends on more than model architecture:

```text
source data bytes
  -> tokenizer and token IDs
  -> split and examples
  -> batch order
  -> model initialization
  -> objective and optimizer state
  -> update schedule
  -> implementation revision
  -> runtime implementation and hardware
  -> observed metrics and artifacts
```

If one node is unknown, “same experiment” becomes an assumption. Professional
records distinguish at least:

- **logical inputs:** the intended data, model, training configuration, seeds,
  code, and environment definition;
- **execution environment:** the JVM and machine that executed those inputs;
- **run state:** checkpoints, data cursor, random state, and completed update;
- **results:** metrics, samples, model artifacts, and evaluation reports.

This chapter implements the first two. Chapter 22b produces metrics. Chapter
22c will add resumable run state.

## 2. Logical identity versus observed runtime

`ExperimentSpecification` contains:

| Field | Why identity depends on it |
| --- | --- |
| `modelSeed` | changes initialized parameters |
| `MiniGptConfig` | changes shapes and computation |
| `MiniGptTrainingConfig` | changes batches, optimizer, schedule, and validation |
| `CorpusFingerprint` | changes the evidence presented to the model |
| `codeRevision` | changes implementation semantics |
| `environmentRevision` | changes pinned tools and dependencies |

Its canonical JSON is hashed into `experimentId`.

`RuntimeFingerprint` is attached by `ExperimentManifest`, but excluded from
that logical ID. The same intended experiment may be repeated on two JVMs to
study portability or performance. Their experiment IDs should match while
their full execution manifests differ.

If hardware must be part of a particular study's logical treatment, add an
explicit hardware-policy field to the specification. Do not silently change ID
semantics after data has been published; increment the manifest schema.

## 3. Hash exact bytes, not descriptions

SHA-256 maps an arbitrary byte sequence to 256 bits. The hexadecimal form has
64 lowercase digits because each digit represents four bits:

$$
256/4=64
$$

The test hashes the UTF-8 bytes of `abc` and checks the standard value:

```text
ba7816bf8f01cfea414140de5dae2223
b00361a396177a9cb410ff61f20015ad
```

One changed byte produces a different digest with overwhelming probability.
The hash establishes byte identity, not truth, quality, ownership, safety, or
permission to use the content.

`CorpusFingerprint.fromText` is intentionally small. It hashes the lab's exact
UTF-8 source string and records token/training/validation counts. A production
corpus builder should hash immutable shard files, a sorted shard manifest, the
tokenizer artifact, and every transformation configuration. Hashing a mutable
directory name is not reproducibility.

## 4. Canonical JSON is part of the protocol

Hashing an in-memory case class is not stable across processes. Hashing JSON is
stable only if equivalent content has one byte representation. The repository's
`JsonObject` preserves insertion order and rejects duplicate fields; the compact
renderer normalizes number spelling.

The specification encoder uses a fixed field order:

```text
name
model_seed
model
training
corpus
code_revision
environment_revision
```

Nested model, schedule, optimizer, and corpus objects also have fixed order.
Optional gradient clipping is encoded as either a number or JSON `null`, never
as an omitted field whose meaning could change.

The identity is:

$$
id=\operatorname{SHA256}(\operatorname{UTF8}(canonicalJson))
$$

Changing field order later would change IDs even when semantic values stay the
same. That is why `schema_version` appears in the execution manifest and why a
future decoder must treat schema evolution explicitly.

## 5. Represent variants as tagged data

Learning-rate policy is a sealed trait. Canonical JSON preserves which variant
was selected:

```json
{"kind":"constant","value":0.02}
```

or:

```json
{
  "kind":"warmup_cosine",
  "peak":0.02,
  "minimum":0.002,
  "warmup_updates":5
}
```

Recording only the first learning rate would lose the future schedule. Storing
an arbitrary Scala `toString` would couple identity to compiler-generated
formatting. Tagged protocol data makes variants explicit and parseable.

The same rule applies to future optimizers, precision modes, data samplers, and
distributed strategies: encode a stable kind plus every behavior-affecting
parameter.

## 6. Source and environment revisions are not interchangeable

`codeRevision` identifies repository content, normally a full commit SHA plus a
clean/dirty state policy. `environmentRevision` identifies the Nix flake lock or
another immutable environment definition.

Pinning Scala dependencies while leaving source unrecorded is insufficient.
Recording source while using an unpinned compiler/JDK is also insufficient.
Both fields are mandatory and non-empty.

The lab defaults them to `unrecorded` unless the caller supplies JVM properties:

```console
$ nix develop -c sbt \
  -Dlearnai.codeRevision=<commit> \
  -Dlearnai.environmentRevision=<flake-lock-hash> \
  'runMain learnai.training.runMiniGptTrainingLab'
```

`unrecorded` is an honest warning, not a reproducible revision. A future run
launcher will derive and validate these values before training.

## 7. Execution manifest contents

`ExperimentManifest` renders:

```text
schema_version
experiment_id
specification
runtime
```

The runtime includes Java runtime version, VM name/version, operating system,
architecture, and available processor count. Chapter 22's benchmark uses the
same type, so quality and performance evidence share environment vocabulary.

This is still not a complete run artifact. It does not yet include start/end
time, host identity policy, metrics, checkpoint hashes, command-line arguments,
exit status, or failure reason. Those belong in an append-only run record rather
than the immutable logical specification.

## Implementation walkthrough

### `CorpusFingerprint`

Construction validates a non-empty name, a lowercase 64-digit hash, and
non-negative counts. `fromText` converts the exact Java `String` to UTF-8 bytes,
passes them to the private `Sha256` helper, then builds the validated value.

`MessageDigest.getInstance("SHA-256")` computes the digest and JDK
`HexFormat` produces lowercase hexadecimal. There is no platform-default text
encoding.

### `ExperimentSpecification`

Case-class construction validates human identity and both revisions. It then
builds `canonicalJson` once and hashes its compact rendering. The model and
training objects are already validated by their constructors, so impossible
negative dimensions or batch sizes cannot enter the manifest.

Case-class `copy` creates a new instance and therefore recalculates JSON and ID.
Tests use that behavior to show that changing seed, corpus count, code revision,
environment revision, or batch configuration changes identity.

### `ExperimentJson`

Private encoder methods map each domain type into a `JsonObject`. Keeping them
private prevents arbitrary callers from inventing another canonical order.
Double fields use `BigDecimal.decimal` before entering `JsonNumber`, avoiding a
long decimal expansion of the binary floating-point representation.

Pattern matching on `LearningRateSchedule` is exhaustive because the trait is
sealed. Adding a new schedule forces this encoder to be updated at compile time.

### `ExperimentManifest`

The manifest stores the specification and runtime as typed values. Its
`experimentId` delegates to the specification. Its `json` wraps the canonical
specification without re-encoding it, which prevents the ID payload and
displayed payload from drifting.

`render` is deterministic but currently in-memory. Atomic persistence,
checksum, safe loading, and schema migration will reuse Chapter 25's artifact
principles.

## Reading the tests

`ExperimentManifestSuite` begins with the public SHA-256 oracle for `abc`. This
tests byte encoding and hex rendering independently of the experiment model.

The equality test constructs two specifications independently, compares their
JSON and IDs, and parses rendered JSON through the strict parser. Variant tests
change one logical boundary at a time and require a new ID.

The runtime test deliberately changes only the VM name. IDs remain equal while
full manifests differ, proving logical/run separation. The required-field test
searches rendered output for every major reproducibility boundary, while
invalid hash and empty revision tests fail before a manifest exists.

These tests establish deterministic identity semantics. They do not prove that
the caller supplied a truthful commit, complete corpus, or correct count.
Acquisition and attestation are separate responsibilities.

## Debugging checklist

- If equal experiments get different IDs, diff canonical JSON byte for byte;
  inspect field order and decimal spelling first.
- If changed configuration keeps the same ID, confirm the field is encoded in
  `ExperimentJson`, not only stored in the Scala case class.
- If a corpus hash changes across machines, compare exact bytes, normalization,
  newline conventions, shard order, and compression—not directory names.
- If a JSON round trip changes numbers, inspect `BigDecimal` construction and
  renderer normalization.
- If runtime changes alter experiment ID, verify runtime is attached only by
  `ExperimentManifest`.
- If results cannot be reproduced from an ID, locate missing logical inputs:
  tokenizer, data transformation, precision, source, environment, or seed.
- If `unrecorded` reaches a serious run, fail the launcher before spending
  compute rather than treating the placeholder as valid provenance.

## Exercises

1. Add tokenizer artifact identity and a sorted shard manifest hash.
2. Implement an atomic manifest writer and strict schema-versioned reader.
3. Record dirty-worktree status without embedding an unbounded diff.
4. Add an append-only run record containing times, status, metrics, and artifact
   hashes while preserving logical experiment identity.
5. Define a migration from schema 1 to schema 2 and prove old IDs remain
   interpretable.
6. Sign a manifest and explain what signature verification proves and does not
   prove.

## Completion criteria

- Distinguish logical experiment identity, runtime observation, run state, and
  results.
- Explain why canonical serialization is part of a hashing protocol.
- State every field that changes the current experiment ID.
- Explain what a SHA-256 corpus hash proves and omits.
- Reproduce a manifest and ID from the same typed inputs.
- Show that changing runtime alone preserves logical identity.
- Identify the additional state required for exact mid-run resume.

## Primary sources

- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
  — a process reference because it releases data, code, checkpoints, logs, and
  evaluation artifacts rather than weights alone.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — a large-scale
  case study in model, data, training, evaluation, and release documentation.
- [Course reading map](../09-papers/40-primary-reading-map.md)
