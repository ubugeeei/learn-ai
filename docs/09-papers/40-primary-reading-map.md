# 40 — Primary paper reading map

## Why papers belong in the implementation path

Code explains what a mechanism does. A paper adds the research question,
baseline, experimental evidence, scale, and limitations that made the mechanism
matter.

This curated map answers three questions for every primary source:

1. **Contribution:** What changed relative to prior work?
2. **Connection:** Which code or chapter makes the idea concrete here?
3. **Read critically:** What should not be concluded from this paper alone?

Do not read everything before coding. Use this loop:

```text
implement a minimal mechanism
  -> predict the paper's claim
  -> read method, evidence, and limitations
  -> reproduce one equation or result
  -> record what changed in your mental model
```

Use the [paper reading note template](41-reading-note-template.md) for a deep
read.

## Core reading order

| After chapter | Read first |
| --- | --- |
| 13 | Adam; AdamW |
| 15 | Subword units with BPE |
| 19–21 | Attention Is All You Need; RMSNorm; GPT-3 |
| 22 | FlashAttention method/evidence; OpenJDK JMH implementation guidance |
| 24–26 | Multi-query attention; speculative decoding; GPTQ; AWQ |
| 28–30 | RoPE; GQA; SwiGLU; scaling laws; Chinchilla; deduplication |
| 31–32 | InstructGPT; LoRA; DPO |
| 34–39 | RAG; ReAct; Toolformer; AgentBench; WebArena; SWE-bench |
| Capstone | OLMo; Llama 3; DeepSeek-V3; Qwen3; GPT-4 technical report |

---

## A. Optimization, tokenization, and normalization

### Adam

[Kingma and Ba, 2014](https://arxiv.org/abs/1412.6980)

- **Contribution:** Combines decayed first- and second-moment estimates with
  bias correction to produce adaptive per-parameter updates.
- **Connection:** Chapter 13 implements every state variable and update term.
- **Read critically:** Adoption and strong results do not mean Adam or its
  default hyperparameters dominate on every workload.

### Decoupled Weight Decay Regularization

[Loshchilov and Hutter, 2017](https://arxiv.org/abs/1711.05101)

- **Contribution:** Distinguishes L2 regularization from true weight decay under
  adaptive optimization and proposes the decoupled AdamW update.
- **Connection:** Chapter 13 applies decay separately from the loss gradient.
- **Read critically:** Empirical benefits still depend on data, architecture,
  schedule, and tuned hyperparameters.

### Neural Machine Translation of Rare Words with Subword Units

[Sennrich, Haddow, and Birch, 2015](https://arxiv.org/abs/1508.07909)

- **Contribution:** Applies byte-pair-style merges to subword segmentation so a
  fixed vocabulary can represent rare and unseen words.
- **Connection:** Chapter 15 trains deterministic merge rules and exact
  encode/decode round trips.
- **Read critically:** Modern byte-level tokenizers differ in normalization,
  pre-tokenization, boundary rules, and special-token handling.

### Root Mean Square Layer Normalization

[Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)

- **Contribution:** RMSNorm removes mean centering and normalizes by root mean
  square, retaining rescaling invariance with a simpler computation.
- **Connection:** Chapters 18–21 implement its forward, backward, and finite-
  difference gradient checks.
- **Read critically:** Timing results from the paper do not transfer directly
  to this educational Scala Tensor graph.

---

## B. Transformer language models

### Attention Is All You Need

[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

- **Contribution:** Replaces recurrent sequence processing with stacked
  attention and feed-forward blocks, enabling parallel training over positions.
- **Connection:** Chapters 19–21 implement scaled attention, multiple heads,
  residual paths, normalization, and a causal decoder stack.
- **Read critically:** The original model is an encoder-decoder translation
  system; MiniGPT intentionally differs in mask, positions, normalization, task,
  and scale.

### Language Models are Few-Shot Learners

[Brown et al., 2020](https://arxiv.org/abs/2005.14165)

- **Contribution:** Studies how autoregressive language-model scaling affects
  zero-, one-, and few-shot behavior without evaluation-time gradient updates.
- **Connection:** Chapter 21 implements the same next-token objective and causal
  decoder principle at an inspectable scale.
- **Read critically:** MiniGPT does not reproduce GPT-3's data, scale,
  capabilities, evaluations, or risk profile.

### GLU Variants Improve Transformer

[Shazeer, 2020](https://arxiv.org/abs/2002.05202)

- **Contribution:** Compares gated feed-forward variants, including the form
  now commonly called SwiGLU, inside Transformer blocks.
- **Connection:** Chapter 28 will replace ReLU with two projections, a gate,
  element-wise multiplication, and an output projection.
- **Read critically:** The evidence is empirical and does not prove one
  activation is optimal for every budget and training recipe.

### RoFormer

[Su et al., 2021](https://arxiv.org/abs/2104.09864)

- **Contribution:** RoPE rotates query/key channel pairs by position-dependent
  angles so dot products express relative position differences.
- **Connection:** Chapter 28 will implement RoPE and revisit Chapter 24's
  learned-absolute-position cache rebuild.
- **Read critically:** RoPE alone does not guarantee arbitrary context
  extrapolation beyond the training regime.

### Switch Transformers

[Fedus, Zoph, and Shazeer, 2021](https://arxiv.org/abs/2101.03961)

- **Contribution:** Routes each token to one expert, increasing total parameter
  capacity without activating all experts for every token.
- **Connection:** Chapter 28's planned MoE lab will expose routing, capacity,
  load balance, and active-versus-total parameters.
- **Read critically:** Sparse FLOP counts do not remove communication, memory,
  load-balance, or stability costs.

---

## C. Inference and memory efficiency

### Fast Transformer Decoding: One Write-Head is All You Need

[Shazeer, 2019](https://arxiv.org/abs/1911.02150)

- **Contribution:** Multi-query attention shares keys and values across query
  heads to reduce incremental-decoding cache bandwidth and size.
- **Connection:** Chapter 24 establishes ordinary multi-head KV-cache
  correctness before Chapter 28 changes KV ownership.
- **Read critically:** Memory reduction can trade off quality; speed depends on
  batch, context, hardware, and kernels.

### GQA

[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)

- **Contribution:** Grouped-query attention uses fewer KV heads than query
  heads and studies uptraining existing multi-head checkpoints.
- **Connection:** Chapter 28 will parameterize query heads and KV groups while
  Chapter 24 exposes the cache-memory change.
- **Read critically:** GQA is a quality/latency compromise, not an automatic
  improvement independent of workload.

### FlashAttention

[Dao et al., 2022](https://arxiv.org/abs/2205.14135)

- **Contribution:** Computes exact attention with an IO-aware tiled algorithm
  that reduces movement between GPU memory levels.
- **Connection:** Chapter 22 can compare a materialized score matrix with tiled
  online softmax while retaining current attention as the oracle.
- **Read critically:** It changes execution, not the mathematical result. Scala
  CPU loops cannot reproduce CUDA kernel throughput.

### Speculative Decoding

[Leviathan, Kalman, and Matias, 2022](https://arxiv.org/abs/2211.17192)

- **Contribution:** A draft model proposes tokens and a target model verifies
  them with a correction that preserves the target distribution.
- **Connection:** Chapters 23–24 already establish sampling, RNG ownership,
  cache state, and reference equivalence.
- **Read critically:** Acceptance rate and hardware utilization determine speed;
  accepting drafts without correction changes the distribution.

### GPTQ

[Frantar et al., 2022](https://arxiv.org/abs/2210.17323)

- **Contribution:** Uses approximate second-order information for one-shot
  low-bit weight quantization of large Transformer models.
- **Connection:** Chapter 26's symmetric int8 path is the independent baseline
  for scale, rounding, error, and bytes.
- **Read critically:** Weight error is not task quality, and compression is not
  speed without suitable packing and kernels.

### AWQ

[Lin et al., 2023](https://arxiv.org/abs/2306.00978)

- **Contribution:** Uses activation statistics to protect salient channels
  during hardware-friendly low-bit weight quantization.
- **Connection:** Extend Chapter 26 with calibration activations and compare
  activation-aware scaling against the symmetric baseline.
- **Read critically:** Calibration data, target hardware, packing, and fused
  kernels are part of the deployment claim.

---

## D. Scaling, systems, and data

### Scaling Laws for Neural Language Models

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)

- **Contribution:** Fits empirical power-law relationships between cross-
  entropy loss and model size, data size, and training compute.
- **Connection:** Chapter 29 will estimate parameters, FLOPs, and optimizer
  memory under alternative budget allocations.
- **Read critically:** Fitted laws are not timeless constants; architecture,
  data, objective, and optimization can shift them.

### Training Compute-Optimal Large Language Models

[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)

- **Contribution:** The Chinchilla study revises compute-optimal model/data
  allocation using hundreds of runs and validates a smaller, more-trained model.
- **Connection:** Chapter 29 will compare Kaplan-style and Chinchilla-style
  allocation assumptions.
- **Read critically:** The result concerns a specific compute-constrained
  training frontier; inference and adaptation costs can change the optimum.

### Megatron-LM

[Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)

- **Contribution:** Presents intra-layer tensor parallelism for efficient
  multi-billion-parameter Transformer training across GPUs.
- **Connection:** Chapter 27 will simulate tensor partitions and collectives
  against the single-device reference.
- **Read critically:** Algebraic simulation omits topology, kernel overlap,
  stragglers, recovery, and achieved utilization.

### ZeRO

[Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054)

- **Contribution:** Partitions optimizer state, gradients, and parameters across
  data-parallel workers to remove redundant memory copies.
- **Connection:** Chapter 27 will calculate payloads and simulate gather/reduce
  behavior at each stage.
- **Read critically:** Savings interact with topology, batch, offload, overlap,
  and implementation quality.

### Deduplicating Training Data Makes Language Models Better

[Lee et al., 2021](https://arxiv.org/abs/2107.06499)

- **Contribution:** Measures duplicate text and train/test overlap, then studies
  how deduplication affects memorization, training, and evaluation validity.
- **Connection:** Chapter 30 will implement fingerprints, contamination checks,
  and before/after corpus statistics.
- **Read critically:** Thresholds trade precision against recall and may remove
  legitimate repetition or minority-domain data.

---

## E. Post-training and efficient adaptation

### InstructGPT

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)

- **Contribution:** Combines supervised demonstrations, a learned reward model,
  and PPO-based optimization from human feedback.
- **Connection:** Chapter 31 will separate SFT, preference collection, reward
  modeling, policy optimization, and evaluation.
- **Read critically:** Preference data reflects a particular annotator process
  and distribution; higher preference is not complete safety or truthfulness.

### LoRA

[Hu et al., 2021](https://arxiv.org/abs/2106.09685)

- **Contribution:** Freezes base weights and trains low-rank update matrices,
  reducing trainable parameters and per-task storage.
- **Connection:** Chapter 31 can add `BA` to a frozen linear weight, test that
  gradients reach only adapters, and merge for inference.
- **Read critically:** Fewer trainable parameters do not remove forward
  activation memory or guarantee every adaptation is low rank.

### QLoRA

[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)

- **Contribution:** Backpropagates into LoRA adapters through a frozen 4-bit
  base and adds further memory-saving techniques.
- **Connection:** Joins Chapter 26 quantization with Chapter 31 efficient
  adaptation and explicit memory accounting.
- **Read critically:** Reproduction requires the quantization format, paged
  optimizer, kernels, and data recipe—not merely LoRA on any quantized matrix.

### Direct Preference Optimization

[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)

- **Contribution:** Rewrites a KL-constrained preference objective as a direct
  pairwise classification-style loss without a separate reward model and RL
  loop.
- **Connection:** Chapter 31 will derive and hand-check the log-ratio objective
  on a tiny language model.
- **Read critically:** Simpler optimization does not solve preference coverage,
  distribution shift, reward hacking, or safety evaluation.

---

## F. Retrieval, tools, and agent control

### Retrieval-Augmented Generation

[Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

- **Contribution:** Combines parametric generation with passages from a non-
  parametric index and studies sequence- versus token-level conditioning.
- **Connection:** Chapter 36 implements chunking, embeddings, cosine search,
  provenance, citations, and a search tool.
- **Read critically:** The hashing embedder and tool-mediated retrieval here
  are not the paper's trained dense retriever or end-to-end objective.

### MRKL Systems

[Karpas et al., 2022](https://arxiv.org/abs/2205.00445)

- **Contribution:** Frames language models as routers among external knowledge
  and reasoning modules rather than self-contained systems.
- **Connection:** Chapters 33–35 separate provider protocol, typed tools, and
  runtime orchestration.
- **Read critically:** Modularity creates routing, trust, authorization, and
  integration problems; it does not automatically solve reliability.

### ReAct

[Yao et al., 2022](https://arxiv.org/abs/2210.03629)

- **Contribution:** Interleaves model-generated reasoning and environment
  actions so observations can update later decisions.
- **Connection:** Chapter 35 implements action/observation; Chapter 37 keeps
  operational plan state in an explicit host-visible DAG.
- **Read critically:** This course does not require private chain-of-thought.
  Public actions, outcomes, task state, and concise rationales are auditable.

### Toolformer

[Schick et al., 2023](https://arxiv.org/abs/2302.04761)

- **Contribution:** Generates and filters tool-use training examples so a model
  learns when and how to call APIs and consume results.
- **Connection:** Chapter 34 implements the inference-time typed boundary;
  Chapter 31 can later learn the model-side decision from traces.
- **Read critically:** Learning to emit a call does not authorize it. Schema,
  approval, timeout, and idempotency remain runtime responsibilities.

### Reflexion

[Shinn et al., 2023](https://arxiv.org/abs/2303.11366)

- **Contribution:** Stores language feedback from failed trials in episodic
  memory to guide later attempts without weight updates.
- **Connection:** Chapters 36–37 provide retrieval, bounded attempts, task
  state, and checkpoints where validated feedback could attach.
- **Read critically:** Self-reflection can preserve a false diagnosis; external
  outcome checks remain necessary.

---

## G. Agent evaluation

### AgentBench

[Liu et al., 2023](https://arxiv.org/abs/2308.03688)

- **Contribution:** Evaluates models as agents across several interactive
  environments and analyzes trajectory-level failure modes.
- **Connection:** Chapter 39 measures terminal status, required/forbidden tools,
  budgets, and exact outcomes with deterministic fakes.
- **Read critically:** Aggregate scores hide environment-specific weaknesses;
  inspect trajectories and failure categories.

### WebArena

[Zhou et al., 2023](https://arxiv.org/abs/2307.13854)

- **Contribution:** Provides reproducible functional websites and long-horizon
  tasks scored by environment state rather than answer fluency.
- **Connection:** Chapter 39 follows the same small-scale principle: prefer
  executable outcome checks over model grading when possible.
- **Read critically:** Scores depend on environment fidelity, task distribution,
  available actions, and reset correctness.

### SWE-bench

[Jimenez et al., 2023](https://arxiv.org/abs/2310.06770)

- **Contribution:** Turns real repository issues into executable software-
  engineering tasks judged against tests.
- **Connection:** It motivates capstone evals that reset state, apply actions,
  and judge tests rather than final prose.
- **Read critically:** Leakage, reproducibility, patch applicability, test
  coverage, and issue selection affect interpretation.

---

## H. Representative frontier-model reports

These are case studies in disclosure and system integration, not a leaderboard.

### GPT-4 Technical Report

[OpenAI, 2023](https://arxiv.org/abs/2303.08774)

- **Contribution:** Describes a large multimodal Transformer, post-training,
  evaluations, safety work, and prediction of some large-run behavior from
  smaller experiments.
- **Connection:** Chapters 29 and 32 use it to discuss scaling forecasts and
  system-card evidence.
- **Read critically:** Many architecture, data, compute, and method details are
  undisclosed, so it is not a complete reproduction specification.

### OLMo

[Groeneveld et al., 2024](https://arxiv.org/abs/2402.00838)

- **Contribution:** Releases weights with training data, code, evaluation code,
  checkpoints, and logs to support scientific study.
- **Connection:** It is a process reference for provenance, reproducibility,
  intermediate artifacts, and code-linked claims.
- **Read critically:** Openness does not guarantee quality, representative data,
  absence of harmful content, or safe use.

### The Llama 3 Herd of Models

[Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)

- **Contribution:** Covers dense multilingual models up to 405B parameters,
  long context, coding/reasoning/tool use, post-training, safety, and evaluation.
- **Connection:** After Chapters 28–39, trace how individual mechanisms become
  one model-development program.
- **Read critically:** Coupled changes prevent attributing aggregate results to
  one design choice without ablations.

### DeepSeek-V3 Technical Report

[DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)

- **Contribution:** Combines sparse MoE, multi-head latent attention, load-
  balancing changes, multi-token prediction, large pretraining, and post-
  training with resource accounting.
- **Connection:** It is a capstone case study for MoE, attention-state
  compression, scaling, distributed execution, and post-training.
- **Read critically:** Reproduction needs data, kernels, cluster behavior,
  optimization, and evaluation—not architecture equations alone.

### Qwen3 Technical Report

[Yang et al., 2025](https://arxiv.org/abs/2505.09388)

- **Contribution:** Reports a multilingual family spanning dense and MoE
  models, post-training for both thinking and non-thinking behavior, and an
  inference-time thinking-budget interface.
- **Connection:** It is a capstone case study connecting sparse architecture,
  distillation, post-training, agent evaluation, and accuracy/latency budgets.
- **Read critically:** Family-wide benchmark results do not isolate the effect
  of one architectural or post-training choice; inspect model-specific setup,
  evaluation conditions, and ablations.

## How to compare papers without a false synthesis

Before combining results, align objective, model family, parameters, training
tokens, data mixture, compute, optimizer, evaluation, hardware, kernels,
baseline strength, seeds, and uncertainty. Never combine a quality number from
one setup with a latency number from another and call it one model design.

## Completion criteria

- Identify the primary source behind each implemented mechanism.
- Distinguish the paper's contribution from this course's simplification.
- State what evidence supports each important claim.
- Identify at least one limitation or missing disclosure per paper.
- Reproduce one equation, invariant, or small experiment in Scala.
- Compare frontier reports by disclosure and method, not benchmark score alone.
