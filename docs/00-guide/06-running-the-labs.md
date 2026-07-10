# Running the labs without guessing

## The shortest path

From the repository directory, run:

```console
$ ./learn-ai chat
```

This opens an actual read/evaluate/generate loop around the MiniGPT implemented
in this repository. It performs no network request and needs no API key. On the
first run it trains the tiny conversational checkpoint before displaying the
`you>` prompt.

To start with the non-interactive model lab instead, run:

```console
$ ./learn-ai model
```

The wrapper explains what will run, what output to inspect, what the experiment
does not prove, and the lower-level command it invokes.

Then run:

```console
$ ./learn-ai training
$ ./learn-ai diagnostics
$ ./learn-ai agent
```

Use `./learn-ai help` to list every guided command.

## What the command layers mean

Consider the lower-level command:

```console
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
```

It has four layers.

### `nix develop`

Nix reads `flake.nix` and `flake.lock`, obtains the pinned JDK and sbt tools,
and creates a temporary development environment. It does not train a model.

The purpose is reproducibility: the command should not silently select a
different Java or Scala toolchain because another machine has different global
packages.

### `-c`

This tells Nix to execute the remaining command inside that environment and
then exit. Without `-c`, `nix develop` opens an interactive shell and waits for
you to type another command.

### `sbt`

sbt is the Scala build tool. It reads `build.sbt`, compiles Scala source files,
constructs the classpath, and starts the selected program.

sbt is build infrastructure, not the AI model. The model is the Scala code
under `src/main/scala/learnai`.

### `runMain ...`

`runMain` asks sbt to execute one named `@main` entrypoint. The long name is its
Scala package plus function name:

```text
learnai.transformer.trainMiniGpt
|       package       | function |
```

The `./learn-ai` wrapper maps memorable names such as `model` and `agent` to
these implementation names.

## Which lab should I run?

| Command | Changes model weights? | Uses remote API? | Main question |
| --- | --- | --- | --- |
| `./learn-ai chat` | first run only | no | Can I send terminal text through local SFT, tokenization, generation, and decoding? |
| `./learn-ai foundations` | no | no | Can I read the Scala control flow? |
| `./learn-ai xor` | yes | no | Does scalar autodiff learn a nonlinear function? |
| `./learn-ai bigram` | yes | no | What is next-token training before a Transformer? |
| `./learn-ai model` | yes | no | Can the complete Transformer graph learn and generate? |
| `./learn-ai training` | yes | no | How do batches, validation, scheduling, and metrics interact? |
| `./learn-ai diagnostics` | no | no | Are gradients correct, and how is local timing reported? |
| `./learn-ai cache` | no | no | Does cached decoding preserve the reference result while reducing work? |
| `./learn-ai quantization` | no | no | How do int8 bytes and numerical error trade off? |
| `./learn-ai agent` | no | no | Does the runtime enforce tool and permission contracts? |
| `./learn-ai planning` | no | no | How are dependent agent tasks retried and recovered? |
| `./learn-ai test` | tests many fixtures | no | Do all implemented contracts still pass? |

Every guided command runs locally on the CPU. The agent labs use
scripted fake models so their state transitions are reproducible. They do not
send prompts to OpenAI or another provider.

## Reading `./learn-ai chat`

This is the only command that waits for your input. It is still entirely local:

```text
terminal text
  -> UTF-8 byte token IDs
  -> user role marker + content + end-of-turn + assistant marker
  -> local MiniGPT next-token probabilities
  -> greedy token selection until the learned end-of-turn token
  -> UTF-8 text printed after assistant>
```

On the first invocation, no checkpoint exists under `target/local-chat`, so the
command prints a short SFT run. The default run uses seven conversations, 80
epochs, 560 optimizer updates, a fixed initialization seed, and a 4,968-scalar
MiniGPT. A representative successful trace is:

```text
no usable checkpoint; training 7 conversations for 80 epochs on the CPU
training epoch= 20 mean_loss=0.299709
training epoch= 40 mean_loss=0.032875
training epoch= 60 mean_loss=0.001929
training epoch= 80 mean_loss=0.001005
training complete: loss 5.719473 -> 0.001005, updates=560, parameters=4968
```

The loss is computed only on assistant content and the assistant end-of-turn
token. User text is context, not a target. This is the same assistant-span SFT
contract implemented in Chapter 31a, now connected to inference rather than
stopping at a unit test.

The checkpoint is then written to:

```text
target/local-chat/mini-gpt-chat-v4.laigpt
```

`target/` is generated build state and is ignored by Git. A later invocation
verifies the file checksum and architecture before loading it. Delete that file
if you deliberately want to observe training again.

At the `you>` prompt, begin with:

```text
you> /examples
learned prompts: hello | who are you? | token? | attention? | training? | scala? | bye
you> hello
assistant> hello from scala.
[tokens: input=8, output=18; stop=end_of_turn]
```

The `input` count includes chat-control tokens as well as the UTF-8 bytes in
`hello`. The `output` count includes the learned end-of-turn token. The stop
reason matters: `end_of_turn` means the model chose to stop; `token_limit`
means the host stopped it at the configured safety bound; and
`unexpected_control_token` means it emitted a role marker where ordinary
assistant text was expected.

Available commands are:

| Input | Effect |
| --- | --- |
| `/examples` | Print the exact seven user prompts in the SFT corpus. |
| `/history` | Print successful user and assistant turns held in process memory. |
| `/reset` | Clear that display history. |
| `/help` | Explain the terminal commands. |
| `/quit` | End the process without changing the checkpoint. |

The current model intentionally uses only the latest user turn for inference.
The history is display-only. The seven SFT examples are independent one-turn
conversations; passing old turns into a 48-token context would create a format
and length distribution the model never learned. Production chat models are
trained on multi-turn corpora with much larger context windows and explicit
truncation policies. This limitation is printed at startup rather than hidden.

Prompts outside `/examples` still pass through the real model. Nothing performs
a keyword lookup or replaces generated output with a canned answer. Because
the model has only 4,968 parameters and seven training pairs, out-of-corpus
text will often produce incorrect or nonsensical output. That failure is a
measurement of the current training setup, not a terminal bug.

## Reading `./learn-ai model`

This lab:

1. constructs a repeated `to be or not to be` corpus;
2. trains a small BPE tokenizer;
3. creates one input/next-token target window;
4. initializes one-layer MiniGPT from a fixed seed;
5. performs 120 AdamW updates;
6. samples 30 new tokens from a fixed prompt and seed.

Important output:

```text
parameters:   4584
initial loss: 5.514108
final loss:   0.000220
generated:    ...
```

- `parameters` is the count of trainable scalar values.
- `initial loss` measures next-token error before updates.
- `final loss` measures the same fixed training window after updates.
- `generated` is sampled from the trained toy model.

A very low loss is expected because the corpus and window are tiny and
repetitive. The model is memorizing a pattern. This is useful for verifying
that embedding, attention, feed-forward, loss, backward, and optimizer paths
connect correctly. It is not evidence of general language ability.

## Reading `./learn-ai training`

This is the more realistic learning experiment. It:

1. uses byte tokens so tokenizer training cannot leak validation text;
2. splits raw tokens before constructing windows;
3. samples training batches from only the training region;
4. divides each batch into microbatches and accumulates gradients;
5. applies AdamW with clipping and warmup/cosine learning rates;
6. evaluates a fixed validation prefix every five updates;
7. records a canonical experiment ID and token counts.

One output row looks like:

```text
update= 10 tokens= 320 lr=0.019392 train=3.667246 \
validation=3.377318 gradient=1.3141 clip=0.7610
```

- `update`: completed optimizer updates, not individual examples.
- `tokens`: cumulative training target tokens; validation is excluded.
- `lr`: learning rate used for that update.
- `train`: loss on the sampled batch before the update.
- `validation`: held-out loss after the update.
- `gradient`: global gradient norm before clipping.
- `clip`: multiplier applied to gradients; `1.0` means no clipping.

Training loss may move up between rows because batches are sampled. Validation
is the more stable comparison because it reuses the same held-out examples.

## Reading `./learn-ai diagnostics`

This command does not optimize the model. It checks the implementation.

### Parameter inventory

Labels and shapes show which component owns every Tensor. Dense payload counts
parameter values only; it excludes gradients, optimizer moments, activations,
objects, and JVM overhead.

### Gradient probes

Reverse-mode gradients are compared with central finite differences at sampled
parameter coordinates. `gradient passed: true` means they agree under the
specified absolute/relative tolerance. It does not prove every coordinate or
every possible input.

### Benchmark

`median`, `p95`, and `variation` describe repeated local CPU forwards after
warmup. The runtime line records JVM, OS, architecture, and processor count.
Do not compare this number with a GPU service or publish it without a controlled
benchmark environment.

## Reading `./learn-ai agent`

The agent lab has three fixed scenarios:

1. return an exact answer without tools;
2. request and execute one allowed read-only lookup;
3. request a write that the host must deny before execution.

The language model is a scripted fake returning predetermined decisions. This
isolates runtime correctness from model quality.

The most important distinction is:

```text
requested tool != authorized tool != physically attempted tool
```

A denied write may appear in requested tools, but it must not appear in physical
attempt events.

## When output differs

Loss and generated text should be deterministic because the lab seeds and data
are fixed. Local timing may differ because JVM warmup, CPU, OS load, and garbage
collection differ.

If a command fails:

1. read the first error, not only the final sbt summary;
2. run `./learn-ai test`;
3. check `git status --short` for work-in-progress source changes;
4. confirm `nix --version` succeeds;
5. use the chapter's debugging checklist before changing tolerances or
   hyperparameters.

A Nix warning about an uncommitted Git tree is informational. It says the
current source differs from the last commit, which matters for reproducibility;
it does not by itself mean the program failed.

## Reading source after running

Follow the program from entrypoint to implementation:

```text
./learn-ai command
  -> printed underlying runMain name
  -> @main function in src/main/scala
  -> model/runtime public methods
  -> Tensor/tool primitives
  -> focused suite in src/test/scala
  -> corresponding chapter in docs
```

Run first, predict one output field, then place a print or breakpoint at the
method responsible for it. The purpose of the wrapper is to remove command-line
guessing—not to hide the implementation path.
