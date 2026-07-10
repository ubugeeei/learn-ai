# 39a — An interactive local chat CLI

## What you will build

This chapter connects the small pieces from earlier chapters into a program
that waits for terminal input and generates an answer with the MiniGPT written
in this repository. There is no provider SDK, hosted model, API key, or network
request. The point is to make the complete boundary observable:

```text
human text -> token IDs -> chat template -> MiniGPT -> token IDs -> human text
```

Run it from the repository root:

```console
$ ./learn-ai chat
```

The first run supervised-fine-tunes a 4,968-parameter model on seven tiny
conversations and stores a checksummed checkpoint under `target/local-chat`.
Later runs load that checkpoint. Once the `you>` prompt appears, type
`/examples`, then try `hello`, `attention?`, and `who are you?`.

This is a real language-model path, but it is not a useful general assistant.
The corpus exists to make the system small enough to read and reproduce. An
unknown prompt still goes through the model; it will often produce nonsense
because seven examples cannot provide broad language knowledge. The program
states that limitation at startup.

Source files:

- `src/main/scala/learnai/chat/LocalChat.scala` owns token layout, training,
  greedy decoding, and the bundled corpus;
- `src/main/scala/learnai/chat/InteractiveChat.scala` owns terminal commands,
  checkpoint loading, and the `@main` entrypoint;
- `src/test/scala/learnai/chat/LocalChatSuite.scala` tests codec, SFT, terminal
  state, errors, limits, and end-of-input without a network or provider.

## Why a chat template is necessary

A causal language model only sees a sequence of token IDs. It does not receive
a Scala enum saying “the user is speaking now.” Training and inference must
encode that structure into the sequence. This lab reserves four IDs above the
256 possible byte values:

```text
0 ... 255   UTF-8 bytes
256         system role
257         user role
258         assistant role
259         end of turn
```

The training pair `hello -> hello from scala.` becomes conceptually:

```text
<user> hello <eot> <assistant> hello from scala. <eot>
```

During inference the host supplies everything through the assistant marker:

```text
<user> hello <eot> <assistant>
```

MiniGPT must predict the first byte of `hello from scala.`, then the next byte,
and eventually ID 259. If serving omitted the assistant marker, swapped token
IDs, or used a different end marker, the weights would be conditioned on a
format they did not learn. A checkpoint therefore consists conceptually of
weights plus architecture plus tokenizer plus template, even though this
learning artifact keeps the last two as fixed source code.

## What the SFT objective learns

The lab reuses Chapter 31a's assistant-span mask. For target position $t$, the
masked cross-entropy is included only if the target belongs to assistant
content or the assistant end-of-turn:

$$
L = -\frac{1}{|M|}\sum_{t \in M}\log p_\theta(y_t \mid y_{<t})
$$

Here $M$ is the set of assistant target positions. User bytes are still in the
input, so they affect hidden states and predictions, but the optimizer is not
asked to reproduce the user. The assistant marker is supplied by the host and
is also not a training target. The final end-of-turn is a target because
stopping is learned behavior.

One optimizer update is performed for each conversation in each epoch. Seven
examples times 80 epochs gives 560 updates. Initialization, example ordering,
and hyperparameters are fixed, so the generated checkpoint is reproducible on
the pinned environment. The model is deliberately wider than the smallest
MiniGPT demonstration: 12 channels, three heads, a 24-channel feed-forward
layer, one Transformer block, and a 48-token maximum context. That is still
only 4,968 trainable scalars.

## Implementation walkthrough

### 1. Keep bytes and control tokens disjoint

`LocalChatCodec` uses `ByteTokenizer` for message content. UTF-8 can represent
every Unicode string as bytes 0 through 255, so no unknown-token fallback is
needed. The four chat tokens begin at 256 and the model vocabulary is exactly
260.

`decodeAssistant` refuses any value at or above 256. This is stricter than
silently skipping a leaked role token. If the model generates `<user>` inside
assistant content, the caller needs to know the sequence violated the serving
protocol.

`trainingExample` converts strings to `ChatMessage` values and delegates to
`ChatTemplate.render` and `ChatTemplate.trainingExample`. The local CLI does
not invent a second SFT layout. `inferencePrompt` uses the same renderer and
then appends the assistant role marker. The suite asserts that the inference
prompt is exactly a prefix of the aligned training inputs.

### 2. Train a model you can inspect

`LocalChatTrainer.train` validates that every rendered example fits the
configured context before allocating the model. It creates MiniGPT from a fixed
seed, measures the initial mean assistant loss, and performs deterministic
AdamW updates. Gradients are cleared before each example and after each epoch.
Progress is reported only at completed epoch boundaries so printed loss has a
stable meaning.

The bundled answers are intentionally short. Attention cost grows
quadratically with sequence length in this reference implementation, and the
goal is a seconds-long CPU lab rather than a hidden training job. The corpus
contains exact prompts about tokens, attention, training, Scala, identity,
greeting, and termination. `/examples` exposes every prompt; there is no claim
that the set is representative.

### 3. Generate one byte at a time

`MiniGptChatResponder.reply` renders the current user turn and repeatedly asks
`MiniGpt.nextDistribution` for the next-token categorical distribution. It
chooses the highest-probability token; equal probabilities prefer the smaller
ID. This greedy policy makes a checkpoint and prompt produce the same result
on every run.

Generation has three terminal conditions:

- `EndOfTurn`: MiniGPT selected ID 259, which is the expected learned stop;
- `TokenLimit`: the host reached 32 generated content tokens;
- `UnexpectedControlToken`: MiniGPT selected a role marker instead of content.

The bound is part of runtime safety. Even a model that never learns to stop
cannot make the terminal loop generate forever. The CLI prints input tokens,
output tokens, and stop reason after every successful answer.

The responder receives the whole UI history but deliberately selects only the
latest user turn. The SFT corpus contains independent one-turn conversations,
so older turns would create a training/inference mismatch and quickly exceed
the 48-token window. `/history` is therefore display-only. A later multi-turn
milestone must add multi-turn training examples, a tested context-selection
policy, and evaluation questions whose answers depend on earlier turns.

### 4. Separate terminal state from model code

`InteractiveLocalChat` depends on `LocalChatResponder` and `ChatTerminal`, not
on MiniGPT or `StdIn` directly. The production terminal prints a prompt and
uses `StdIn.readLine`. Tests provide a scripted terminal and a recording fake
responder.

The loop commits a user message to history only after inference succeeds. If
the model returns `Left(problem)`, the problem is printed and the failed user
turn is not allowed to corrupt the next request. `/reset` clears history,
`/history` renders typed roles, `/examples` prints the training support, and
unknown slash commands never reach the model. Inputs are bounded by UTF-8 byte
count rather than Java character count because bytes are the actual tokens.

### 5. Load only a verified checkpoint

`LocalChatApplication.loadOrTrain` looks for a versioned file under `target`.
`MiniGptCheckpoint.load` verifies the magic header, format version, size
limits, tensor labels and shapes, finite values, and SHA-256 checksum. The
application additionally compares the loaded architecture with the current
training configuration. A corrupt or stale artifact is explained and replaced
by a fresh deterministic run.

The cache is an optimization, not a source artifact. Delete
`target/local-chat/mini-gpt-chat-v4.laigpt` to repeat training. The checkpoint
does not contain optimizer state because the next operation is inference, not
exact training resume; Chapter 22d explains the larger bundle needed for
continuation.

## Reading an actual run

A first run should show loss falling by several orders of magnitude:

```text
training epoch= 20 mean_loss=0.299709
training epoch= 40 mean_loss=0.032875
training epoch= 60 mean_loss=0.001929
training epoch= 80 mean_loss=0.001005
```

This proves that the tiny model memorized the measured assistant targets. It
does not prove semantic understanding, held-out generalization, safety, or
useful world knowledge. There is no held-out set because all seven pairs are
exposed as interaction fixtures. A professional experiment would split train
and evaluation conversations, report uncertainty over seeds, and retain the
best checkpoint by held-out loss rather than final training loss.

After training, an in-corpus exchange should resemble:

```text
you> attention?
assistant> attention mixes context.
[tokens: input=13, output=25; stop=end_of_turn]
```

The 13 input tokens are one user marker, ten ASCII bytes, one end-of-turn, and
one assistant marker. The 25 output tokens are 24 content bytes plus the final
end-of-turn. Counting the markers by hand is a useful way to verify that the
displayed metrics come from the actual template.

Try an unknown prompt next. A malformed answer is expected evidence. If it
looks surprisingly good, inspect probabilities or repeat with adversarial
inputs before attributing generalization to a 4,968-parameter model.

## Reading the tests

`LocalChatSuite` starts at the protocol boundary. It checks the exact reserved
IDs, round-trips Unicode message content, verifies the inference prompt is the
prefix used during training, and rejects a control token embedded in decoded
assistant content.

One focused integration test constructs a still smaller real MiniGPT and
trains it on `a -> b`. It asserts that assistant loss drops by at least half and
that progress arrives at the configured epochs. This test covers Tensor
autodiff, masked loss, AdamW, chat rendering, and the local trainer without
making the complete seven-pair training run part of every repository test.

The remaining tests use scripted boundaries. They prove successful turns enter
typed history, failed turns do not, reset really clears state, help and unknown
commands are visible, UTF-8 byte limits are enforced before inference, and EOF
exits cleanly. No test result depends on network availability or a secret.

## Debugging checklist

1. If the process immediately prints `bye`, confirm
   `Compile / run / connectInput := true` remains in `build.sbt`; a forked JVM
   needs sbt to forward standard input.
2. If a checkpoint is rejected, read the first checksum or architecture error,
   delete only the versioned file under `target/local-chat`, and retrain.
3. If loss does not fall, print the assistant mask and verify it contains the
   answer bytes plus end-of-turn, but not the user bytes or assistant marker.
4. If output never stops, check that ID 259 is a masked target and that greedy
   decoding compares against the same `endOfTurn` value.
5. If output is invalid UTF-8, inspect generated byte IDs and the stop reason;
   token-limit truncation can cut a multi-byte character in the middle.
6. If the first learned prompt works but later prompts fail, confirm inference
   still selects only the latest user turn. Do not silently pass display
   history into a model trained on single-turn examples.
7. If an unknown prompt gives nonsense, do not patch in a keyword response.
   Expand data, capacity, evaluation, and training while preserving an honest
   model-only generation path.

## Next engineering steps

The curriculum marks 39a in progress because a local CLI is only one provider
surface. The next increments should introduce a provider-neutral adapter that
implements the Chapter 33 `LanguageModel` boundary, stream typed deltas under a
deadline, and run the same recorded contract suite against local and remote
implementations. Multi-turn local training should precede claims about memory.
Larger checkpoints also require tokenizer/checkpoint co-versioning, evaluation
sets, best-checkpoint selection, and resource budgets.

## Primary reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) defines the
  Transformer computation used by MiniGPT.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
  motivates supervised demonstrations before preference optimization.
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
  describes a larger-scale pretrained-to-chat pipeline and its evaluations.
- [Chapter 31a — SFT and chat templates](../07-frontier/31a-sft.md) derives the
  exact token mask reused here.
- [Course paper map](../09-papers/40-primary-reading-map.md) connects these
  sources to the rest of the implementation path.
