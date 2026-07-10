# 16 — Causal datasets, splits, and batches

## What you will build

Create fixed-length next-token examples, non-overlapping train/validation
regions, reproducible random batches, and deterministic evaluation batches.
Source: `src/main/scala/learnai/data/CausalDataset.scala`.

## Causal language modeling

Given tokens:

```text
[t0,t1,t2,t3,t4]
```

A context-length-three example is:

```text
input:  [t0,t1,t2]
target: [t1,t2,t3]
```

The output after `input(i)` predicts `target(i)`. One window therefore provides
one next-token target per time position. The next sliding window is:

```text
input:  [t1,t2,t3]
target: [t2,t3,t4]
```

A sequence of length \(N\) and context \(T\) yields
\(\max(0,N-T)\) full windows.

## Shapes

One example:

```text
inputs:  [time]
targets: [time]
```

A batch of \(B\) examples:

```text
inputs:  [batch,time]
targets: [batch,time]
```

`CausalBatch` enforces one context length. Variable-length batching later
requires padding plus an explicit mask.

## Split raw tokens before windowing

If overlapping windows are created before random splitting, training and
validation can share nearly every token:

```text
train:      [t0,t1,t2,t3]
validation:    [t1,t2,t3,t4]
```

That leakage overstates generalization. First divide the raw sequence into
contiguous regions, then window each side independently:

```text
tokens -> [training region | validation region]
             -> windows       -> windows
```

Some boundary examples are lost, but no example crosses the split. Real
datasets should often split by document or source identity and remove
duplicates across partitions.

## Batches estimate the full gradient

Computing every example each step is expensive. A batch estimates mean loss:

\[
\hat L=\frac{1}{B}\sum_{i\in\text{batch}}L_i
\]

- small batch: less memory, more gradient noise;
- large batch: more stable estimate, more memory and step compute.

Batch size changes training dynamics, not only throughput.

## Sampling with replacement

`sampleBatch` chooses each element independently, so one example may appear
more than once. Epoch-style shuffled iteration is another common policy.

Whatever the policy, checkpoint the seed and sampler state if resumed training
must preserve data order.

## Deterministic validation

Validation uses `sequentialBatches` with no parameter updates. The caller
chooses whether to keep or drop a short final batch. Models with dropout or
training-dependent normalization also need explicit evaluation mode.

## Pipeline invariants

Tests verify:

- every target is exactly one token ahead;
- partial windows are excluded;
- no split-boundary crossing;
- equal seeds produce equal batches;
- empty sampling fails explicitly;
- all examples in a batch share shape.

Data defects can look like model defects, and leakage can produce a decreasing
loss with invalid evaluation. Test data pipelines as rigorously as model code.

## Exercises

1. List the first and final windows for length 10 and context 4.
2. Implement an epoch-based shuffle sampler.
3. Design `fromDocuments` without cross-document windows.
4. Design token-budget-based dynamic batching.
5. Report token and example counts lost at the split boundary.
6. Measure overlap leakage after random window splitting.

## Completion criteria

- Explain the one-token input/target shift.
- Name both `[batch,time]` axes.
- Demonstrate why splitting before windowing prevents leakage.
- Explain batch size versus gradient noise.
- Distinguish training sampling from validation iteration.
- `CausalDatasetSuite` passes.
