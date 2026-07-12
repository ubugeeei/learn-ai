# Curriculum

## Status legend

- ✅ The chapter, implementation, and verification path are complete.
- 🚧 The learning outcome is fixed, but text or implementation is in progress.
- ⬜ Planned.

Each part depends only on earlier deliverables. Optional extensions may be
skipped without breaking the main path.

A ✅ means the stated chapter scope is complete, not that the component is
production-complete or that the learner has reached professional maturity. The
[professional competency roadmap](05-professional-roadmap.md) tracks the M0–M4
bridge from visible mechanisms to operational judgment.

## Part 0 — Environment and program literacy

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 00 | ✅ | How to learn | Experimental loop and completion criteria |
| 01 | ✅ | Nix development environment | Reproducible Scala 3 shell |
| 02 | ✅ | Minimal Scala 3 | Values, functions, types, control flow, CLI |
| 03 | ✅ | Testing and debugging | Dependency-free test runner |
| 03a | ✅ | Complexity and representation | Arrays, locality, allocation, stack/heap accounting |
| 03b | ✅ | JVM systems | JIT, GC, profiling, concurrency, files, networking |

## Part 1 — Mathematics and numerical computing for LLMs

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 04 | ✅ | Numbers, functions, and error | Experiments with `Double` |
| 05 | ✅ | Vectors | Immutable `VectorD` |
| 06 | ✅ | Matrices and shapes | `MatrixD` and matrix multiplication |
| 07 | ✅ | Probability and information | Distributions, entropy, cross entropy |
| 08 | ✅ | Derivatives and chain rule | Numerical differentiation reference |
| 08a | ✅ | Linear algebra depth | rank, conditioning, eigensystems, SVD, low-rank approximation |
| 08b | ✅ | Statistical inference | estimators, uncertainty, tests, calibration |

**Part deliverable:** translate linear algebra and probability equations into
shape-safe code.

## Part 2 — Build learning from scratch

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 09 | ✅ | Gradient descent | Optimization trajectory and learning curve |
| 10 | ✅ | Automatic differentiation | Scalar computation graph |
| 11 | ✅ | Neurons and MLPs | Network that solves XOR |
| 12 | ✅ | Tensor autodiff | Explicit Tensor without broadcasting |
| 13 | ✅ | Optimizers and initialization | SGD, AdamW, clipping |
| 13a | ✅ | Tensor execution model | broadcasting, batches, views, graph lifetime |
| 13b | ✅ | Precision engineering | float formats, accumulation, loss scaling |

**Part deliverable:** implement and explain forward, loss, backward, and update.

## Part 3 — Turn text into training data

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 14 | ✅ | Unicode and bytes | Strict UTF-8 tokenizer |
| 15 | ✅ | Byte pair encoding | Trainable BPE encode/decode |
| 16 | ✅ | Datasets and batches | Causal language-model dataset |
| 17 | ✅ | Bigram language model | Next-token training and generation |
| 17a | ⬜ | Corpus manifests and shards | provenance, hashes, bounded streaming reads |
| 17b | ✅ | Shuffle and sequence packing | deterministic restart and loss masks |

**Part deliverable:** convert raw text into examples and sample continuations.

## Part 4 — Transformer and small GPT

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 18 | ✅ | Embeddings and position | Token/position embeddings |
| 19 | ✅ | Attention | Causal multi-head self-attention |
| 20 | ✅ | Transformer block | RMSNorm, residuals, feed-forward network |
| 21 | ✅ | MiniGPT | End-to-end training and generation CLI |
| 22 | ✅ | Correctness and performance | Gradient checks, parameter inventory, benchmark evidence |
| 22a | ✅ | Experiment records | canonical configuration identity, corpus hash, runtime manifest |
| 22b | ✅ | Training system | train/validation, schedules, accumulation, telemetry |
| 22c | ✅ | Exact resume | model, optimizer, scheduler, data cursor, random state |
| 22d | ✅ | Training bundles | persisted exact-resume state with experiment-identity refusal |

**Part deliverable:** train a small GPT and explain every parameter.

## Part 5 — Inference engine and scale

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 23 | ✅ | Sampling | Temperature, top-k, top-p, seed |
| 24 | ✅ | KV cache | Cached/reference equivalence and timing |
| 25 | ✅ | Model format | Versioned, checksummed safe loader |
| 26 | ✅ | Quantization | Symmetric int8 and measured error |
| 27a | ✅ | Collectives and data parallelism | replica equivalence and byte traces |
| 27b | ⬜ | Tensor parallelism | row/column partitions and collectives |
| 27c | ⬜ | Pipeline parallelism | stages, microbatches, bubbles, schedules |
| 27d | ⬜ | ZeRO and recovery | state partitioning and coordinated checkpoints |
| 27e | ⬜ | Serving scheduler | prefill/decode, continuous batching, overload |
| 27f | ✅ | Paged KV and prefix reuse | allocation, fragmentation, eviction |
| 27g | ✅ | Speculative decoding | distribution-preserving draft verification |

**Part deliverable:** explain training/inference differences and measure
speed-memory tradeoffs.

## Part 6 — Frontier-model training techniques

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 28a | ✅ | RoPE | rotary pairs, cache position, context tests |
| 28b | ✅ | SwiGLU | gated FFN with parameter-matched ablation |
| 28c | ✅ | GQA/MQA | query/KV head ownership and cache bytes |
| 28d | ⬜ | Mixture of experts | routing, capacity, load balance, dropped tokens |
| 28e | ✅ | IO-aware attention | online softmax reference and tiled simulation |
| 29a | ✅ | Model accounting | parameters, FLOPs, activations, optimizer and communication bytes |
| 29b | ⬜ | Scaling experiments | fitted laws, uncertainty, budget allocation |
| 30a | ⬜ | Deduplication | exact/approximate matches and precision/recall |
| 30b | ⬜ | Filters and provenance | quality, PII, policy, licenses, lineage |
| 30c | ⬜ | Mixtures and contamination | sampling weights, ablations, eval overlap |
| 31a | ✅ | SFT and chat templates | masks, data validation, held-out evaluation |
| 31b | ✅ | LoRA | frozen base, low-rank gradients, merge equivalence |
| 31c | ⬜ | Reward modeling | preference pairs, calibration, disagreement |
| 31d | ⬜ | DPO | reference-policy log-ratio objective |
| 31e | ⬜ | Policy optimization | advantage, KL control, reward hacking tests |
| 32a | ⬜ | Language-model evaluation | loss, tasks, stochastic uncertainty, slices |
| 32b | ⬜ | Safety evaluation | adversarial cases, injection, misuse, red team |
| 32c | ⬜ | Release evidence | model/data/system cards and reproducible reports |

**Part deliverable:** read public technical reports and reason quantitatively
about quality, compute, data, and operational tradeoffs.

## Part 7 — AI agents

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 33 | ✅ | Model API boundary | Provider-neutral typed protocol |
| 34 | ✅ | Tool calling | Schema, validation, execution, observation |
| 35 | ✅ | Agent loop | Reproducible loop with hard stop conditions |
| 36 | ✅ | Memory and retrieval | Chunks, embeddings, vector search, citations |
| 37 | ✅ | Planning | State machine, task graph, recovery |
| 38 | ✅ | Reliability | Timeout, retry, idempotency, permissions |
| 39 | ✅ | Evaluation | Trajectory, fake model, quality/cost metrics |
| 39a | ⬜ | Provider adapters | local/remote/streaming contract suites and failover |
| 39b | ⬜ | Durable operation | event store, transactions, replay, schema migration |
| 39c | ⬜ | Sandboxed tools | resource scopes, isolation, quotas, incident evidence |
| 39d | ⬜ | Hybrid retrieval | lexical/vector fusion, reranking, faithfulness |

**Part deliverable:** build an agent runtime whose external actions are bounded,
auditable, and recoverable.

## Capstone — Practical grounded agent

Build an agent that indexes local documents, answers with citations, and can
execute only explicitly granted operations. Completion requires:

- interchangeable fake, local, and remote model providers;
- schema validation, timeout, and audit events for every external action;
- reproducible quality, latency, and token metrics on a fixed eval set;
- a threat model covering prompt injection and poisoned tool output;
- a system card documenting design, limitations, and reproduction steps.

## Part 8 — Read and reproduce primary research

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 40 | ✅ | Primary paper reading map | Chapter-linked summaries and original sources |
| 41 | ✅ | Reading note template | Evidence and reproduction worksheet |

**Part deliverable:** connect every major implementation choice to its primary
source, evidence, simplifications, and reproducible small-scale experiment.

## Main dependency path

```text
environment
  -> Scala/types/tests
  -> vectors/matrices/probability/calculus
  -> autodiff/MLP/Tensor
  -> tokenizer/dataset/bigram
  -> attention/Transformer/MiniGPT/diagnostics
  -> experiment training and exact resume
  -> inference/distributed systems/serving
  -> modern architecture/data/post-training/evaluation
  -> tools/agent/memory/durability/operations
  -> capstone
```
