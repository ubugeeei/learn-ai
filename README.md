# learn-ai

A from-scratch, hands-on path to large language models and AI agents in Scala 3.

The repository maps every important equation to executable code. It does not
hide vectors, automatic differentiation, tokenization, attention, training, or
tool execution behind a machine-learning framework. The main path uses only the
Scala and JDK standard libraries.

Small models make every mechanism observable. Later chapters connect those same
mechanisms to the systems, data, and scaling decisions used by frontier models.

## Learning outcomes

After completing the curriculum, you should be able to:

- explain what training minimizes using derivatives and gradient descent;
- implement tensors and reverse-mode automatic differentiation;
- train neural networks and next-token language models;
- build a byte/BPE tokenizer and causal dataset pipeline;
- derive and implement causal multi-head attention and a small GPT;
- explain checkpoints, sampling, KV caches, quantization, and parallelism;
- reason about SFT, preference optimization, evaluation, and safety;
- implement an agent from model, tool, memory, planning, and policy boundaries;
- measure quality, latency, memory, cost, reproducibility, and failure modes;
- read a frontier-model technical report and critique its design choices.

> Understanding a frontier model does not mean reproducing its full training
> run on a personal computer. This course combines minimal implementations of
> the same computations with the large-scale systems design needed to run them.

## Start here

1. Read [How to learn](docs/00-guide/00-how-to-learn.md).
2. Read the [complete chapter anatomy](docs/00-guide/04-chapter-anatomy.md).
3. Read the [professional competency roadmap](docs/00-guide/05-professional-roadmap.md)
   so the small reference implementations are not mistaken for the finish line.
4. Follow the [curriculum](docs/00-guide/01-curriculum.md) in order.
5. Run the verification command in every completed chapter.
6. Solve exercises before reading or implementing an answer.
7. Explain each part's deliverable in your own words.
8. Use the [primary paper reading map](docs/09-papers/40-primary-reading-map.md)
   after each implementation milestone.

The code is cumulative rather than disposable. Each chapter adds a tested
component that later chapters reuse. Architectural choices are recorded in the
[design principles](docs/00-guide/02-design.md).

## Reproducible environment

For a first run, use the guided wrapper:

```console
$ ./learn-ai help
$ ./learn-ai model
$ ./learn-ai training
```

It explains what each lab runs, what output matters, and what the result does
not prove. See [Running the labs without guessing](docs/00-guide/06-running-the-labs.md)
for a zero-background explanation of Nix, sbt, `runMain`, and every lab.

The lower-level environment commands are:

```console
$ nix develop
$ sbt check
```

The repository pins JDK 21, sbt, Scala 3, and the Nix package revision.

## Current implementation

The runnable path currently reaches:

```text
math -> autodiff -> neural networks -> tokenization -> language modeling
  -> shuffled/packed datasets with loss masks
  -> causal Transformer -> MiniGPT -> gradient/benchmark diagnostics
  -> canonical experiment records -> batch training and held-out validation
  -> bitwise exact mid-run resume (RNG, optimizer, scheduler, data cursor)
  -> KV-cached decoding -> inference artifacts
  -> paged KV pool / speculative decoding / traced data parallelism
  -> RoPE / SwiGLU / grouped-query / tiled online-softmax attention
  -> model accounting anchored to the implementation
  -> LoRA adapters and chat-template SFT with assistant-span masks
  -> typed tools -> approval and retry policy -> bounded agent runtime
  -> cited retrieval -> task-graph planning and recovery -> agent evaluation
```

Useful entrypoints:

```console
$ nix develop -c sbt check
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
$ nix develop -c sbt 'runMain learnai.transformer.runMiniGptDiagnostics'
$ nix develop -c sbt 'runMain learnai.training.runMiniGptTrainingLab'
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

Tensor/pipeline/ZeRO simulation, the serving scheduler, corpus curation at
scale, preference optimization, and production agent adapters remain
explicit future milestones. See the [curriculum](docs/00-guide/01-curriculum.md) for exact
chapter status, the [professional roadmap](docs/00-guide/05-professional-roadmap.md)
for the complete competency target, and [progress and quality standards](docs/00-guide/03-progress.md)
for the definition of done.

## Contribution standards

Public APIs use English Scaladoc. Every implementation chapter includes the
relevant normal, boundary, failure, property, numerical, or gradient tests.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This is a learning repository. Redistribution terms have not yet been selected.
