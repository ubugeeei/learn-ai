# 25 — A versioned, safe checkpoint format

## What you will build

Serialize MiniGPT config, stable parameter labels, shapes, and row-major
`Double` values. Verify SHA-256 before loading and atomically replace saved
files. Source: `src/main/scala/learnai/io/MiniGptCheckpoint.scala`.

## A checkpoint is more than arrays

Raw values do not identify their meaning. A useful format includes:

- magic bytes and format version;
- architecture config;
- stable parameter names;
- rank, shape, element count, dtype, and byte order;
- values;
- integrity checksum.

Tokenizer vocabulary and merges are also required for a deployable bundle. The
teaching format intentionally stores only the inference model and documents
that limitation.

## File layout

```text
magic "LAIGPT01"                 8 bytes
format version                   Int32
MiniGptConfig                    fixed fields
parameter tensor count          Int32
repeat:
  UTF-8 label length + bytes
  rank + dimensions
  scalar count
  Float64 values, big-endian
SHA-256(payload)                 32 bytes
```

JDK data streams use big-endian numeric encoding. The format does not depend on
JVM object serialization or in-memory object layout.

## Labels and order

Names such as `blocks.0.attention.query.weight` identify semantic ownership.
The loader constructs the expected model and checks count, order, label, shape,
and element count before assignment.

Shape alone cannot detect swapping two equal-shaped weights. Save rejects
duplicate labels.

## Integrity versus authenticity

SHA-256 detects accidental corruption and incomplete transfer. Anyone can
recompute it, so it does not authenticate a publisher.

- integrity: checksum;
- authenticity: trusted digital signature, publisher identity, and provenance.

Production distribution needs both.

## Verify before parsing

The loader checks file-size limits and checksum before interpreting fields. It
also limits:

- total file bytes;
- scalar count implied by config;
- label bytes and rank;
- expected tensor shapes and counts.

Allocating an untrusted length before validation enables memory-exhaustion
attacks.

## All-or-nothing assignment

`Tensor.assignParameterValues` validates count and all finite values before
mutating any element. The loader modifies a newly constructed model, so a failed
load cannot corrupt the currently serving model.

Production rollout should verify a new model, run smoke inference, then swap a
reference atomically.

## Atomic save

Writing directly over a good checkpoint can destroy it after a crash or full
disk. Instead:

1. write a complete temporary file in the same directory;
2. include its checksum;
3. atomically move it over the destination;
4. fall back to replacement move only if atomic move is unsupported.

Stronger durability also requires syncing the file and directory.

## Inference versus resumable training

Inference needs model config and weights. Exact training resumption also needs:

- AdamW moments and step;
- scheduler and loss-scaler state;
- RNG state;
- data sampler/shuffle position;
- tokenizer, data, code, and build revisions.

Loading weights alone restarts optimizer dynamics and cannot reproduce the
original trajectory.

## Why not object serialization?

General object deserialization can execute class hooks and binds the file to
implementation details. Untrusted model files can become a code-execution risk.
This format reads only explicit primitive fields and constructs an allowed
model structure. Safetensors follows the same data-not-code principle.

## Tests

- bit-exact config, labels, values, and logits after a round trip;
- one-byte corruption rejection;
- truncated-file rejection;
- atomic replacement of an existing file;
- invalid assignment leaves old values unchanged.

## Exercises

1. Inspect magic and version in a hex viewer.
2. Design a version-2 migration.
3. Add tokenizer metadata and checksum to a bundle manifest.
4. Add optional optimizer state that inference loading can skip.
5. Place an Ed25519 signature and verification step.
6. Design publish-time load and fixed-logit smoke tests.

## Completion criteria

- Explain why config, labels, and shapes are required.
- Distinguish checksum from signature.
- Explain untrusted-length validation.
- Explain how atomic replacement protects the previous file.
- List extra state for resumable training.
- `MiniGptCheckpointSuite` passes.
