# Progress and implementation standards

## Current runnable milestone

```text
Nix / Scala 3 / dependency-free tests
  -> numerical error / vectors / matrices / probability / calculus
  -> gradient descent / scalar autodiff / XOR MLP
  -> Tensor autodiff / SGD / AdamW / clipping
  -> UTF-8 bytes / BPE / causal dataset / bigram
  -> embeddings / RMSNorm / causal attention / Transformer
  -> MiniGPT training and generation
  -> parameter inventory / sampled gradient checks / benchmark evidence
  -> sampling / KV-cached decoding / checkpoint / int8 quantization
  -> strict JSON / typed tools / approval and retry policy / bounded agent
  -> cited retrieval / task-graph planning and checkpoint recovery
  -> deterministic agent outcome/trajectory/cost evaluation
  -> primary-paper reading map and evidence template
```

The status table in `01-curriculum.md` is authoritative. A ✅ chapter means:

1. its text includes prerequisites, equations, shapes, implementation,
   observations, exercises, and completion criteria;
2. the Scala code compiles in the Nix shell;
3. normal, boundary, failure, and relevant property tests exist;
4. non-obvious public APIs have English Scaladoc;
5. `nix develop -c sbt check` succeeds;
6. runnable experiments have been executed and inspected.

## Next milestones

1. typed experiment configuration and immutable run records;
2. train/validation loops, schedules, accumulation, and telemetry;
3. exact model/optimizer/scheduler/data/random-state resume;
4. JVM profiling plus deterministic FLOP/work/memory estimators;
5. RoPE, SwiGLU, and GQA as separate reference/ablation chapters;
6. streaming shards, provenance, packing, shuffle, and deduplication;
7. collectives and data parallelism before tensor/pipeline/ZeRO simulation;
8. serving scheduler and paged KV cache;
9. SFT and LoRA before reward/preference/policy optimization;
10. model/safety evaluation and release evidence;
11. provider adapters plus durable agent/tool state;
12. model, systems, agent, and research capstones.

A chapter is not marked complete for prose alone. It requires a reference
implementation, an independent oracle, failure tests, and an experiment.
The complete M0–M4 target and exit artifacts are defined in
`05-professional-roadmap.md`.

## Language policy

- identifiers, Scaladoc, code comments, test names: English;
- hands-on chapters and repository documentation: English;
- equations, shapes, and API names: identical to the executable code.

## Test strategy

Test count is not the objective. Distinct failure modes and independent oracles
are.

```text
hand calculation
  + representation invariants
  + shape/range failures
  + numerical stability
  + gradient checks
  + algebraic properties
  + deterministic seeds
  + end-to-end learning
  + corrupt/untrusted inputs
  + timeout/budget/capability boundaries
```

Do not compare only two implementations that may share the same bug. Use
finite differences, hand calculations, reference paths, round trips, and
causality properties.

## Performance claims

Do not claim speed or memory improvements without:

- warmup and repeated iterations;
- exact input, config, and seed;
- JDK, Scala, commit, and CPU information;
- median and variation;
- correctness equivalence;
- allocation/peak-memory data, or an explicit `No measurements found` note.
