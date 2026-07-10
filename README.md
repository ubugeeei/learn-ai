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
2. Follow the [curriculum](docs/00-guide/01-curriculum.md) in order.
3. Run the verification command in every completed chapter.
4. Solve exercises before reading or implementing an answer.
5. Explain each part's deliverable in your own words.

The code is cumulative rather than disposable. Each chapter adds a tested
component that later chapters reuse. Architectural choices are recorded in the
[design principles](docs/00-guide/02-design.md).

## Reproducible environment

```console
$ nix develop
$ sbt check
```

The repository pins JDK 21, sbt, Scala 3, and the Nix package revision.

## Current implementation

The runnable path currently reaches:

```text
math -> autodiff -> neural networks -> tokenization -> language modeling
  -> causal Transformer -> MiniGPT -> KV-cached decoding -> inference artifacts
  -> typed tools -> approval and retry policy -> bounded agent runtime -> cited retrieval
```

Useful entrypoints:

```console
$ nix develop -c sbt check
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

Distributed training, modern blocks, post-training, agent planning, and
evaluation remain explicit future milestones. See the [curriculum](docs/00-guide/01-curriculum.md)
for exact chapter status and [progress and quality standards](docs/00-guide/03-progress.md)
for the definition of done.

## Contribution standards

Public APIs use English Scaladoc. Every implementation chapter includes the
relevant normal, boundary, failure, property, numerical, or gradient tests.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This is a learning repository. Redistribution terms have not yet been selected.
