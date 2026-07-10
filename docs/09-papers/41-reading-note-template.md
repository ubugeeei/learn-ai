# Paper reading note template

Copy this template for one deep read. Write in your own words; do not copy the
abstract into the summary.

## Bibliographic record

- **Title:**
- **Authors:**
- **Year / venue:**
- **Primary-source URL:**
- **Version read:**
- **Code / data / model artifacts:**
- **Related hands-on chapters:**

## One-sentence research question

What uncertainty is the paper trying to reduce?

## Prior baseline

What did a strong system do before this paper? Describe the actual baseline,
not a weak caricature.

## Claimed contribution

Separate contributions by type:

- mathematical objective or derivation;
- architecture;
- algorithm;
- systems implementation;
- dataset;
- evaluation protocol;
- empirical observation.

## Minimal mechanism

Write the smallest equation, pseudocode block, or state transition that captures
the new mechanism. Define every symbol and shape.

## Evidence table

| Claim | Experiment/table | Baseline | Metric | Uncertainty reported? |
| --- | --- | --- | --- | --- |
| | | | | |

## Reproduction boundary

Record what is disclosed:

- model dimensions and parameter count;
- data sources, filtering, mixture, and token count;
- optimizer, schedule, batch, and seeds;
- training compute and hardware;
- inference implementation and hardware;
- evaluation prompts, code, and contamination controls;
- checkpoints, logs, code, and exact artifact versions.

Then list what is missing.

## Connection to this repository

- **Existing implementation:** Which type or method corresponds to the paper?
- **Intentional simplification:** What did the hands-on omit?
- **Independent oracle:** How can correctness be checked without copying the
  same implementation?
- **Small experiment:** What can run on one CPU in under a few minutes?

## Results you expect before running

Write a falsifiable prediction before executing the experiment.

## Observed results

Record configuration, environment, seed, raw output, and variation. If no
measurement exists, write `No measurements found`.

## Failure modes and counterexamples

When should the method fail, lose its advantage, or create a new risk?

## What the paper does not prove

List plausible statements that are stronger than the evidence.

## Updated mental model

What did you believe before reading? What changed, and which evidence changed
it?

## Follow-up implementation

Define one small commit containing implementation, tests, documentation, and a
verification command.
