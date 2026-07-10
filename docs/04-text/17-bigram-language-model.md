# 17 — A bigram language model

## What you will build

A trainable model that uses the current token to predict the next token, with
Tensor lookup, fused cross entropy, AdamW training, and autoregressive sampling.

- model: `src/main/scala/learnai/lm/BigramLanguageModel.scala`
- Tensor operations: `gatherRows` and `crossEntropy`

This chapter joins tokenizer, training pairs, logits, loss, backward, optimizer,
and generation into one language-model pipeline.

## Bigram assumption

An autoregressive model factorizes sequence probability:

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

A bigram simplifies the entire past to the current token:

$$
p(x_{t+1}\mid x_0,\ldots,x_t)
\approx p(x_{t+1}\mid x_t)
$$

It cannot model long context, but it exposes the complete next-token objective.

## Transition-logit table

For vocabulary size $V$, store trainable
$W\in\mathbb{R}^{V\times V}$. Row $i$ contains next-token logits after
current token $i$.

```text
W shape: [current token, next token] = [V,V]
```

Input IDs `[N]` gather rows into logits `[N,V]`. Repeated IDs read one shared
row and accumulate all corresponding gradients during backward. Embedding
lookup uses the same gather/scatter mechanism.

The table contains $V^2$ parameters. With 50,000 tokens that is 2.5 billion,
showing why language models factor tokens through a lower-dimensional hidden
space.

## Fused cross entropy

For logits $z$ and target class $t$:

$$
L=\log\sum_j e^{z_j}-z_t
$$

Its gradient is:

$$
\frac{\partial L}{\partial z_j}=p_j-\mathbb{1}[j=t]
$$

`Tensor.crossEntropy` combines stable log-sum-exp and this derivative without
materializing a one-hot Tensor or a separate softmax graph. It is a small
example of operation fusion.

Every gradient row sums to zero:

$$
\sum_j(p_j-\mathbb{1}[j=t])=1-1=0
$$

Tests check that property, large-logit stability, and finite differences.

## Training pairs

```text
tokens:  [t0,t1,t2,t3]
inputs:  [t0,t1,t2]
targets: [t1,t2,t3]
```

One full-batch step is:

```scala
val loss = model.loss(inputs, targets)
loss.backward()
optimizer.step(model.parameters)
```

Repeated target transitions raise their logits relative to alternatives.

## Autoregressive generation

Training uses the true previous token, known as teacher forcing. Generation
feeds the model's own sample back as the next input:

```text
start -> logits row -> softmax -> sample -> use sample as next input -> repeat
```

A low-probability sample may enter a rarely trained path and compound errors.
Training and generation therefore see different input distributions.

## Temperature

$$
p_i=\operatorname{softmax}(z_i/\tau),\quad\tau>0
$$

- below 1: concentrate on high logits;
- equal to 1: unchanged learned distribution;
- above 1: flatter and more diverse;
- approaching zero: close to argmax.

## UTF-8 caveat

Arbitrary byte/BPE token combinations need not form valid UTF-8, especially in
a context-one model. The demo uses strict decoding and reports failure rather
than hiding it with replacement characters.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
```

The repeated demo corpus is an integration experiment, not a quality benchmark.

## From bigram to Transformer

A bigram gives identical predictions whenever the final token is equal, even
when earlier context changes. A Transformer creates contextual hidden vectors
by mixing earlier positions with attention. The following remain unchanged:

- tokenization and causal shift;
- logits and cross entropy;
- backward and optimizer;
- autoregressive generation.

## Implementation walkthrough

`BigramLanguageModel` owns one trainable matrix with shape `[V,V]`. Row `i`
contains logits for the next token after input token `i`. Applying the model is
therefore an embedding-style row gather, not a matrix multiplication.

For vocabulary `{A,B,C}`, suppose the desired cycle is `A->B`, `B->C`,
`C->A`. Training pairs are the three corresponding row/target selections. Cross
entropy increases the target column logit in each selected row and decreases
competitors. No hidden state exists, so two equal current tokens always produce
the same next-token distribution regardless of earlier history.

`loss` validates input/target lengths and ranges, gathers transition rows, and
calls Tensor cross entropy. Backward accumulates gradients if an input token
appears multiple times because `gatherRows` routes repeated selections into the
same matrix row.

Generation begins with a validated start token. Each step selects the last
generated ID, converts its row logits to a stable softmax distribution, samples
with the caller RNG, and appends. The returned vector includes the start token;
requesting zero new tokens returns exactly that one validated token.

`BigramTrainer` rebuilds the loss graph every step and updates the single
parameter through AdamW. This small model is the first complete path from token
IDs through loss, backward, optimizer, and seeded generation.

## Reading the tests

Logit lookup checks exact output shape and rows. The three-token cycle has a
known learnable optimum and verifies large loss reduction plus dominant target
probabilities. Equal generation seeds test RNG ownership. Zero-length
generation catches loop boundaries. Invalid IDs and temperatures fail before
indexing or softmax.

## Debugging checklist

1. Interpret a failing prediction as one row and one target column.
2. Verify targets are shifted exactly one position.
3. Check repeated input IDs accumulate into one row gradient.
4. If loss falls but generation looks repetitive, remember bigram context is
   exactly one token.
5. Compare logits, probabilities, and sampled ID as separate stages.

## Exercises

1. Count parameters for a vocabulary of 3.
2. Compare empirical transition counts with learned softmax probabilities.
3. Increase the target logit by 1 and measure loss change.
4. Compare token frequencies at several temperatures.
5. Compute a separate validation-pair loss.
6. Find two contexts the bigram cannot distinguish.
7. Replace the transition table with a low-rank product.

## Completion criteria

- Explain autoregressive factorization and the bigram assumption.
- Explain `gatherRows` forward and backward shapes.
- Derive the `softmax - oneHot` gradient.
- Distinguish teacher forcing from generation.
- Explain temperature's effect.
- `BigramLanguageModelSuite` and Tensor cross-entropy tests pass.
