# Zero-background to professional AI engineering roadmap

## Why the roadmap needs more than a small GPT

Implementing attention once is an important milestone, but it is not
professional readiness. A professional must be able to investigate a failed
training run, distinguish data problems from optimizer problems, estimate
memory before allocating a cluster, verify a faster kernel, operate a serving
system, evaluate uncertain behavior, and bound an agent's external effects.

This repository therefore uses two simultaneous paths:

1. **Mechanism path:** small dependency-free implementations expose every
   equation, index, state transition, and failure channel.
2. **Engineering path:** increasingly realistic systems add experiment state,
   streaming data, profiling, distributed simulations, serving policies,
   evaluation statistics, durability, and operational evidence.

The mechanism path makes hidden behavior visible. The engineering path prevents
toy success from being confused with production competence.

## What “professional” means here

Completion means you can independently:

- translate a paper's equations into shape-checked reference code;
- design independent oracles and diagnose numerical disagreement;
- build a deterministic training and validation pipeline;
- reason quantitatively about parameters, FLOPs, bytes, bandwidth, and cost;
- explain and implement modern decoder components;
- partition training state and computation across simulated workers;
- design an inference service with batching, caches, deadlines, and backpressure;
- implement and critique supervised and preference-based post-training;
- design statistically defensible quality and safety evaluations;
- build an agent whose tools, authorization, state, and failures are auditable;
- reproduce a small result from a primary source and state what did not
  reproduce at small scale;
- write an architecture or system card that another engineer can use to
  operate and challenge the system.

It does **not** mean reproducing a frontier training run on a laptop. Compute,
data rights, specialized kernels, networking, and operations at that scale are
real constraints. The professional skill is understanding the complete design,
implementing inspectable versions of its mechanisms, and reasoning correctly
about what changes at scale.

## Maturity levels

Every component progresses through five maturity levels. A chapter check mark
means its stated deliverable is complete; it does not imply every maturity
level.

| Level | Evidence | Example |
| --- | --- | --- |
| M0 — vocabulary | explain the problem and symbols | define logits and causal masking |
| M1 — reference | clear implementation plus hand oracle | materialized causal attention |
| M2 — engineered local | errors, reproducibility, metrics, artifacts | checksummed checkpoint and exact resume |
| M3 — scaled design | partitioning, communication, failure simulation | tensor-parallel matmul with collective traces |
| M4 — operational judgment | compare alternatives from measured evidence | choose batching/cache policy for an SLO |

Most current mathematical and model components are M1. Checkpoint, cache,
agent-policy, and evaluation components contain selected M2 behavior. The
professional roadmap closes M2 systematically, teaches M3 through faithful
simulations, and assesses M4 through capstones and design reviews.

## Competency matrix

| Area | Current foundation | Required professional bridge | Exit artifact |
| --- | --- | --- | --- |
| Computing | Nix shell, Scala syntax, tests | complexity, JVM memory/JIT/GC, concurrency, files/networking, profiling | measured concurrent pipeline with failure tests |
| Mathematics | vectors, matrices, probability, calculus | conditioning, decompositions, estimation, statistics, optimization dynamics | derivation notebook translated to tested Scala |
| Tensor engine | dense Tensor and reverse mode | broadcasting contracts, batched operations, mixed precision, module/state model, kernel boundary | reference backend with gradient/precision suite |
| Text and data | UTF-8, BPE, causal windows | streaming shards, deterministic shuffle, packing, dedup, provenance, PII/license filters | versioned corpus build report |
| Pretraining | one-sequence AdamW training | train/validation, schedules, accumulation, exact resume, observability, instability response | resumable experiment with reproducibility report |
| Architecture | small pre-norm decoder | RoPE, GQA, SwiGLU, MoE, long-context tradeoffs | modern decoder with ablation results |
| Distributed systems | single process | collectives, data/tensor/pipeline parallelism, ZeRO, topology, recovery | deterministic multi-worker simulator and byte trace |
| Inference | sampling, KV cache, int8 baseline | continuous batching, paged cache, prefix reuse, speculative decoding, service API | load-tested local serving runtime |
| Post-training | planned | SFT, templates, LoRA, reward models, DPO, policy optimization, distillation | aligned tiny model with held-out comparison |
| Evaluation and safety | deterministic agent checks | LM evals, uncertainty, contamination, calibration, judge audits, red team | versioned evaluation and model card |
| Agents | typed tools, policy, retrieval, planning | provider adapters, durable state, hybrid retrieval, human workflow, observability | grounded agent operated against fixed SLOs |
| Research practice | paper map | claim/evidence extraction, reproduction, ablation, negative results | reproduction report with code and raw artifacts |

## Stage A — Computing and Scala systems foundations

The early course currently teaches only enough Scala to read the learning
code. Professional work also requires understanding the runtime that executes
it.

Required milestones:

1. **Complexity and representation:** stack/heap, arrays, object overhead,
   row-major locality, asymptotic time versus achieved performance.
2. **JVM execution:** bytecode, interpretation, JIT compilation, escape
   analysis, allocation, garbage collection, warmup, and why microbenchmarks
   lie.
3. **Files and binary protocols:** partial reads/writes, atomic replacement,
   checksums, versions, endianness, bounded parsing, corruption recovery.
4. **Concurrency:** threads, virtual threads, executors, futures, cancellation,
   races, ownership, bounded queues, structured shutdown.
5. **Networking:** HTTP request/response, streaming, deadlines, connection
   pools, retry classification, TLS and secret boundaries.
6. **Observability:** structured logs, metrics, traces, profiles, identifiers,
   redaction, and cardinality control.

Exit criterion: diagnose a CPU, allocation, timeout, and race failure from
recorded evidence rather than changing parameters at random.

## Stage B — Mathematics, statistics, and optimization

The current scalar path is intentionally gentle. The professional bridge adds
the mathematics needed to read experiments critically:

1. vector spaces, bases, projections, norms, and cosine geometry;
2. matrix rank, conditioning, eigenvalues, singular-value decomposition, and
   low-rank approximation;
3. random variables, common distributions, expectation, covariance, and laws
   of large numbers;
4. maximum likelihood, cross entropy, KL divergence, and calibration;
5. estimators, bias/variance, sampling error, confidence intervals, hypothesis
   tests, multiple comparisons, and effect size;
6. Jacobians, vector-Jacobian products, Hessian intuition, curvature, and
   finite-difference failure modes;
7. optimization landscapes, momentum, adaptive methods, schedules, clipping,
   regularization, and generalization diagnostics.

Every topic must end in a Scala experiment. Symbol manipulation without data
and implementation does not satisfy the exit criterion.

## Stage C — Tensor and automatic-differentiation engine

The current Tensor is a transparent reference implementation. It deliberately
lacks behavior that frameworks normally hide. The bridge adds each feature only
after its contract is explicit:

1. broadcast planning and backward reduction;
2. batched matrix multiplication and attention batch dimensions;
3. view versus copy semantics, strides, contiguity, and alias safety;
4. module ownership, named parameters, buffers, train/eval mode, and state
   dictionaries;
5. graph lifetime, saved tensors, activation checkpointing, and memory release;
6. `Float`, `Double`, bfloat16/float16 simulation, loss scaling, underflow, and
   accumulation precision;
7. reference kernels versus optimized backend interfaces;
8. numerical, gradient, property, alias, and mutation test matrices.

Exit criterion: add an operation with forward/backward derivation, exhaustive
shape contract, independent gradient oracle, precision analysis, and memory
ownership explanation.

## Stage D — Tokenization and data engineering

Model quality cannot exceed the evidence and policy encoded in its data
pipeline. A professional corpus path includes:

1. Unicode normalization and language-aware pre-tokenization policies;
2. tokenizer training artifacts, special-token governance, compatibility, and
   migration;
3. immutable source manifests with content hashes, licenses, acquisition time,
   and transformation lineage;
4. streaming shard readers with bounded memory and corruption handling;
5. deterministic distributed shuffle and exact epoch/restart semantics;
6. document-aware sequence packing, boundary tokens, padding, and loss masks;
7. exact and approximate deduplication with precision/recall measurement;
8. quality, language, safety, PII, malware, and policy filters with audit
   counts;
9. domain mixtures, sampling weights, temperature sampling, and ablations;
10. train/evaluation contamination checks and documented unresolved overlap.

Exit artifact: rebuild the same corpus version from a manifest, reproduce shard
hashes and aggregate statistics, and explain every rejected document category.

## Stage E — Pretraining system

The one-sequence MiniGPT trainer proves gradient connectivity. It does not yet
constitute a training system. The professional sequence adds:

1. separate train, validation, and test responsibilities;
2. deterministic batch sampling and document masks;
3. microbatching and gradient accumulation with equivalence tests;
4. warmup, constant, linear, cosine, and decay learning-rate schedules;
5. mixed-precision state and dynamic loss scaling simulation;
6. gradient/loss/activation/parameter norm telemetry;
7. periodic validation, checkpoint retention, and best-model selection;
8. exact resume of model, optimizer, schedule, data cursor, and random states;
9. configuration schema, run ID, source revision, environment fingerprint, and
   immutable run manifest;
10. failure drills for `NaN`, loss spikes, corrupt shards, disk exhaustion,
    worker loss, and interrupted checkpoint writes.

Exit artifact: interrupt and resume a run at an arbitrary step, then prove its
subsequent parameters, optimizer state, batches, and metrics exactly match an
uninterrupted reference run.

## Stage F — Modern decoder architectures

“Modern blocks” is too broad for one chapter. Each mechanism becomes a separate
reference implementation and ablation:

1. rotary position embeddings and context-position semantics;
2. SwiGLU gating and parameter-matched feed-forward comparisons;
3. grouped-query and multi-query attention with cache-byte accounting;
4. mixture-of-experts routing, capacity, dropped tokens, load balance, and
   active versus total parameters;
5. attention implementation alternatives, including online softmax and
   IO-aware tiling simulation;
6. normalization and residual scaling variants;
7. long-context interpolation/extrapolation claims and adversarial tests;
8. multi-token prediction and speculative-training interfaces;
9. dense versus sparse architecture cost/quality accounting.

Current public reports demonstrate why the curriculum must distinguish a dense
decoder from the broader frontier design space. Llama 3 documents a large dense
training and post-training system; DeepSeek-V3 documents MoE, latent attention,
load balancing, and multi-token prediction; Qwen3 reports both dense and MoE
families plus inference-time thinking budgets. Read these as systems reports,
not shopping lists of features.

Primary anchors:

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)

Exit artifact: train parameter-matched tiny variants, report raw evaluation and
systems metrics, and explain which result cannot be extrapolated to large scale.

## Stage G — Scaling and distributed training

No GPU dependency is required to understand partition correctness. The course
first implements deterministic worker/collective simulations:

1. data-parallel replica gradients and all-reduce;
2. reduce-scatter and all-gather contracts;
3. column- and row-parallel linear layers;
4. vocabulary-parallel cross entropy;
5. pipeline stages, microbatches, bubbles, and schedules;
6. ZeRO partitioning of optimizer state, gradients, and parameters;
7. activation checkpointing and recomputation;
8. topology-aware byte/time models and communication overlap;
9. stragglers, timeouts, worker failure, checkpoint coordination, and elastic
   restart;
10. model FLOPs utilization and the difference between theoretical and achieved
    throughput.

Primary anchors include [Megatron-LM](https://arxiv.org/abs/1909.08053) and
[ZeRO](https://arxiv.org/abs/1910.02054). Simulation establishes algebra and
protocol invariants; it cannot establish CUDA/NCCL kernel throughput.

## Stage H — Inference engines and serving

Generating one prompt at a time is not a serving system. The professional path
adds:

1. prefill versus decode scheduling and separate metrics;
2. dynamic/continuous batching under arrival and completion events;
3. paged KV-cache allocation, fragmentation, eviction, and prefix reuse;
4. grouped-query cache layouts and memory bandwidth models;
5. weight-only and activation-aware quantization with calibration/eval;
6. speculative decoding with distribution-preserving acceptance;
7. streaming protocol, cancellation, deadlines, backpressure, and disconnects;
8. admission control, token budgets, fairness, rate limits, and overload;
9. model loading, warmup, health, rolling replacement, and graceful shutdown;
10. latency distributions, throughput, goodput, memory, energy/cost, and SLO
    tradeoffs.

Exit artifact: replay a deterministic arrival trace through a local serving
simulator and explain p50/p95 time-to-first-token, inter-token latency, queueing,
cache occupancy, and rejected work.

## Stage I — Post-training and reasoning

Post-training is a data, objective, evaluation, and operations pipeline:

1. chat templates and loss masks;
2. supervised fine-tuning data validation and held-out evaluation;
3. parameter-efficient adaptation with LoRA and merge equivalence;
4. preference collection, pair construction, disagreement, and annotator bias;
5. reward-model training and calibration;
6. direct preference optimization with a frozen reference policy;
7. policy-gradient fundamentals, KL control, advantage estimation, and reward
   hacking;
8. rejection sampling, distillation, synthetic-data filtering, and iteration;
9. reasoning-token budgets and accuracy/latency tradeoffs;
10. capability, safety, and regression gates before release.

Exit artifact: compare base, SFT, and preference-trained tiny models on fixed
held-out prompts, retaining data versions, objectives, seeds, raw generations,
and uncertainty.

## Stage J — Evaluation, safety, and scientific evidence

Professional evaluation begins by naming the claim and unit of analysis:

1. language-model loss and perplexity with tokenization caveats;
2. exact and executable task oracles;
3. generation metrics and their failure modes;
4. repeated stochastic samples, uncertainty, effect size, and significance;
5. calibration and selective prediction;
6. contamination and memorization probes;
7. model-judge versioning, position bias, leakage, and human audit;
8. adversarial prompts, prompt injection, tool misuse, and policy bypass;
9. slice analysis across language, domain, difficulty, and risk;
10. model cards, system cards, dataset cards, and release decisions.

An aggregate score never replaces raw cases, environment versions, failure
taxonomy, and uncertainty. Exit criterion: another engineer can reproduce the
report and challenge its conclusion from retained artifacts.

## Stage K — Production agent systems

The current agent runtime establishes useful host-side boundaries. The
professional bridge adds:

1. real local and remote provider adapters with shared contract tests;
2. streaming response assembly, rate limits, deadlines, and model failover;
3. durable append-only run/event storage and schema migration;
4. persistent idempotency and checkpoint transactions;
5. human approval queues with authentication, expiry, and argument display;
6. lexical, vector, and hybrid retrieval with reranking and provenance;
7. citation-faithfulness and environment-state evaluation;
8. sandboxed code/file/browser tools and resource scopes;
9. multi-tenant secrets, data isolation, retention, and audit access;
10. traces, budgets, incident response, replay, and safe rollout.

Exit artifact: operate a grounded agent against a fixed evaluation and load
trace, then demonstrate recovery from provider failure, process interruption,
duplicate effects, poisoned retrieval, denied writes, and budget exhaustion.

## Stage L — Capstones and professional assessment

The curriculum ends with multiple artifacts rather than one demo:

### Model capstone

Train a modern tiny decoder from a versioned corpus. Resume exactly after an
interruption. Compare architectural variants. Publish parameter/FLOP/memory
accounting, learning curves, held-out results, failures, and a model card.

### Systems capstone

Serve the model through a bounded streaming API and deterministic load trace.
Implement batching and cache policy, profile the bottleneck, and defend one
optimization with reference equivalence and environment-labeled measurements.

### Agent capstone

Build a cited local-document agent using the served model or a contract-tested
provider. Persist run state, enforce scoped tools and approval, and evaluate
outcomes, unsafe requests, effects, latency, cost, and recovery.

### Research capstone

Select one primary paper, restate a falsifiable claim, implement the smallest
valid reproduction, run an ablation, retain raw artifacts, and document both
positive and negative results.

## Active implementation order

The near-term repository work follows dependency order:

1. correctness, gradient, parameter, and benchmark diagnostics;
2. experiment configuration and immutable run records;
3. train/validation loop, batching, schedules, and accumulation;
4. optimizer/data/random-state checkpoints with exact resume;
5. JVM profiling and deterministic work/FLOP estimators;
6. RoPE, then SwiGLU, then GQA, each with isolated ablations;
7. streaming shards, packing, deterministic shuffle, and corpus manifests;
8. collectives and data-parallel simulation before tensor/pipeline partitioning;
9. SFT and LoRA before preference objectives;
10. serving scheduler and paged cache before remote-provider capstones.

This order will evolve when evidence exposes a missing prerequisite. It will
not collapse several independent mechanisms into one check-mark chapter.

## How to use this roadmap

For each competency:

1. read its prerequisite mechanism chapter;
2. reconstruct the reference implementation without copying;
3. predict normal and failure tests;
4. run the focused lab and retain raw output;
5. implement the engineering extension;
6. compare it against the reference path;
7. read the linked primary source and extract claims/evidence/limits;
8. write a design decision using measured results;
9. explain what remains unknown at the course's scale.

Professional readiness comes from repeatedly completing that evidence loop—not
from reaching the bottom of a table of contents.
