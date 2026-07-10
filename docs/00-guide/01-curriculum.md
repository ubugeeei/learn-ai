# Curriculum

## Status legend

- ✅ The chapter, implementation, and verification path are complete.
- 🚧 The learning outcome is fixed, but text or implementation is in progress.
- ⬜ Planned.

Each part depends only on earlier deliverables. Optional extensions may be
skipped without breaking the main path.

## Part 0 — Environment and program literacy

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 00 | ✅ | How to learn | Experimental loop and completion criteria |
| 01 | ✅ | Nix development environment | Reproducible Scala 3 shell |
| 02 | ✅ | Minimal Scala 3 | Values, functions, types, control flow, CLI |
| 03 | ✅ | Testing and debugging | Dependency-free test runner |

## Part 1 — Mathematics and numerical computing for LLMs

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 04 | ✅ | Numbers, functions, and error | Experiments with `Double` |
| 05 | ✅ | Vectors | Immutable `VectorD` |
| 06 | ✅ | Matrices and shapes | `MatrixD` and matrix multiplication |
| 07 | ✅ | Probability and information | Distributions, entropy, cross entropy |
| 08 | ✅ | Derivatives and chain rule | Numerical differentiation reference |

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

**Part deliverable:** implement and explain forward, loss, backward, and update.

## Part 3 — Turn text into training data

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 14 | ✅ | Unicode and bytes | Strict UTF-8 tokenizer |
| 15 | ✅ | Byte pair encoding | Trainable BPE encode/decode |
| 16 | ✅ | Datasets and batches | Causal language-model dataset |
| 17 | ✅ | Bigram language model | Next-token training and generation |

**Part deliverable:** convert raw text into examples and sample continuations.

## Part 4 — Transformer and small GPT

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 18 | ✅ | Embeddings and position | Token/position embeddings |
| 19 | ✅ | Attention | Causal multi-head self-attention |
| 20 | ✅ | Transformer block | RMSNorm, residuals, feed-forward network |
| 21 | ✅ | MiniGPT | End-to-end training and generation CLI |
| 22 | ⬜ | Correctness and performance | Gradient checks, profiler, benchmark |

**Part deliverable:** train a small GPT and explain every parameter.

## Part 5 — Inference engine and scale

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 23 | ✅ | Sampling | Temperature, top-k, top-p, seed |
| 24 | ✅ | KV cache | Cached/reference equivalence and timing |
| 25 | ✅ | Model format | Versioned, checksummed safe loader |
| 26 | ✅ | Quantization | Symmetric int8 and measured error |
| 27 | ⬜ | Parallelism | Data/tensor/pipeline simulations |

**Part deliverable:** explain training/inference differences and measure
speed-memory tradeoffs.

## Part 6 — Frontier-model training techniques

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 28 | ⬜ | Modern blocks | RoPE, GQA, SwiGLU, MoE |
| 29 | ⬜ | Scaling | Parameter/FLOP/memory estimator |
| 30 | ⬜ | Data engineering | Deduplication, filters, mixtures, contamination |
| 31 | ⬜ | Post-training | Minimal SFT, reward model, and DPO |
| 32 | ⬜ | Evaluation and safety | Eval harness, red team, model card |

**Part deliverable:** read public technical reports and reason quantitatively
about quality, compute, data, and operational tradeoffs.

## Part 7 — AI agents

| # | Status | Chapter | Deliverable |
| --- | --- | --- | --- |
| 33 | ✅ | Model API boundary | Provider-neutral typed protocol |
| 34 | ✅ | Tool calling | Schema, validation, execution, observation |
| 35 | ✅ | Agent loop | Reproducible loop with hard stop conditions |
| 36 | ✅ | Memory and retrieval | Chunks, embeddings, vector search, citations |
| 37 | ⬜ | Planning | State machine, task graph, recovery |
| 38 | ✅ | Reliability | Timeout, retry, idempotency, permissions |
| 39 | ⬜ | Evaluation | Trajectory, fake model, quality/cost metrics |

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

## Main dependency path

```text
environment
  -> Scala/types/tests
  -> vectors/matrices/probability/calculus
  -> autodiff/MLP/Tensor
  -> tokenizer/dataset/bigram
  -> attention/Transformer/MiniGPT
  -> inference/scaling/post-training
  -> tools/agent/memory/evaluation
  -> capstone
```
