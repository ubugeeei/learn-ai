# 36 — Chunking, vector search, and cited retrieval

## What you will build

Split documents into source-offset chunks, embed them with a deterministic
hashing baseline, rank by cosine similarity, and return citation metadata
through an agent tool. Source:
`src/main/scala/learnai/retrieval/Retrieval.scala`.

## Context windows and external memory

Model weights compress training-time patterns but are not an exact, current,
private-document database. Putting every document into a prompt increases
context, cost, and attention work. Retrieve only likely relevant spans:

```text
documents -> chunks -> embeddings -> index
query     -> embedding -> similarity -> top chunks -> model context
```

Retrieval does not modify model parameters. Retrieved text remains untrusted and
does not grant tool authority.

## Chunks and provenance

Each `TextChunk` retains:

- stable chunk and document IDs;
- document title;
- source start and end offsets;
- exact span text.

An answer can cite a chunk ID and let a reader inspect the source span. Copying
text without provenance makes update tracking and citation verification hard.

Fixed-size chunking is transparent but can split sentences, headings, or code
blocks. Production systems compare structure-aware chunks, token limits, and
parent-child retrieval.

## Overlap

Overlap preserves context around boundaries. For chunk size \(C\) and overlap
\(O\), step size is \(C-O\). More overlap may improve recall but increases index
size and duplicate results.

## Hashing embedding baseline

The teaching embedder lowercases tokens, hashes them into fixed buckets, uses a
sign bit, and L2-normalizes:

\[
\hat v=v/\lVert v\rVert_2
\]

It captures exact-token overlap but not semantic synonymy or word order. Hash
collisions are unavoidable; signed hashing reduces systematic positive bias.

This baseline exposes indexing, normalization, ranking, citations, and
evaluation before introducing an external neural embedding model.

## Cosine search

Normalized query and document vectors make cosine similarity a dot product:

\[
\cos(q,d)=\hat q\cdot\hat d
\]

Results sort by descending score and then chunk ID for deterministic ties. An
exact scan is a useful oracle before adding an approximate nearest-neighbor
index.

## Retrieval tool

`SearchTool` is read-only. Every result contains score, text, document identity,
and source offsets. A host can enforce:

- every citation refers to a returned chunk;
- source spans are displayed with claims;
- retrieval failure is reported instead of fabricated evidence;
- instructions inside retrieved text remain data, not policy.

## Retrieval evaluation

Measure retrieval separately from answer generation:

- Recall@k: whether the correct evidence appears in top-k;
- MRR: reciprocal rank of the first correct result;
- citation precision and coverage;
- answer faithfulness;
- latency and index bytes.

Use a fixed query/evidence set to compare chunk size, overlap, embedding model,
and result count.

## Implementation walkthrough

`Retrieval.scala` deliberately keeps the full pipeline in one file. Trace a
document from source text to tool JSON before replacing any component with a
production service.

### 1. Derive chunk boundaries by hand

Take a ten-character document, `maximumCharacters = 4`, and
`overlapCharacters = 1`. The step is (4-1=3), so starts are `0, 3, 6, 9` and
the half-open spans are:

```text
[0,4)  [3,7)  [6,10)  [9,10)
```

`substring(start, end)` uses the same half-open convention. Adjacent chunks
share one UTF-16 code unit at their boundary. The final short chunk is retained
because the loop condition is `start < text.length`.

The offsets are JVM `String` indices, not Unicode code-point indices and not
UTF-8 byte offsets. A production citation API must state its convention or it
will highlight the wrong span around non-BMP characters.

### 2. Build a deterministic hashing vector

`HashingEmbedder.embed` lowercases with `Locale.ROOT`, splits on non-letter and
non-number characters, and processes each token:

```scala
val hash = token.hashCode
val bucket = (hash & 0x7fffffff) % dimensions
val sign = if (hash & 0x80000000) == 0 then 1.0 else -1.0
values(bucket) += sign
```

Masking the sign bit makes the bucket index non-negative. The original sign
bit still chooses `+1` or `-1`. Repeated tokens add repeatedly, so this is a
signed bag-of-words frequency vector. Finally, division by `raw.norm` gives
unit length unless there are no tokens.

Normalization changes magnitude, not direction. A document containing a term
ten times does not automatically beat a shorter document solely because its
vector is longer.

### 3. Index once, score exactly

`VectorIndex.build` first validates unique document IDs. It chunks each
document independently, embeds every chunk, and stores immutable
`(TextChunk, VectorD)` entries. Search embeds the query once, rejects a zero
vector, computes one dot product per entry, then sorts:

```scala
.sortBy(result => (-result.score, result.chunk.id))
.take(resultCount)
```

Negating the score turns Scala's ascending ordering into descending relevance.
Chunk ID breaks exact ties, so snapshots and evaluations do not change with
map iteration order.

The exact scan costs (O(ND)) for (N) chunks and (D) dimensions. Keep it as
a correctness oracle when later introducing an approximate index.

### 4. Convert search into a capability

`SearchTool.definition` grants exactly one string field, `query`, and defaults
to the read-only effect. The runtime validates the schema before `execute`, so
the implementation safely extracts the already-validated `JsonString`.

Every result is returned as an object containing `chunk_id`, `document_id`,
title, offsets, score, and exact text. The score helps debugging but is not a
probability. A cosine value of `0.8` does not mean an 80% chance that the answer
is correct.

### 5. Observe the lab as separate stages

When experimenting, print and inspect:

1. chunk IDs, offsets, and exact substrings;
2. vector norms and non-zero buckets;
3. ranked chunk IDs and raw cosine scores;
4. the rendered tool JSON;
5. citations chosen by the answer layer.

This localizes failures. If the correct evidence never reaches top-k, prompt
engineering cannot recover it.

## Reading the tests

`RetrievalSuite` follows pipeline order. The chunk tests prove coverage,
overlap, and document isolation. The embedder test proves determinism,
case-folding, and unit norm. Ranking tests cover a relevant result and a score
tie. Failure tests cover an empty index and tokenless query. The final tool test
asserts structured citation metadata rather than only matching returned text.

When adding a neural embedder, retain the chunk and tool tests. Replace only
embedding-specific assertions, and add a fixed relevance fixture that compares
Recall@k against the exact baseline.

## Debugging checklist

- If characters disappear, list every half-open span and verify that the union
  covers `[0, text.length)`.
- If the chunk loop never terminates, verify `overlapCharacters <
  maximumCharacters`, making the step strictly positive.
- If equivalent casing gives different vectors, use `Locale.ROOT` rather than
  the machine's default locale.
- If cosine scores exceed the expected range, inspect normalization and vector
  dimensions before ranking.
- If tie order changes, ensure the secondary key is the stable chunk ID.
- If citations point to the wrong text, compare the stored offset convention
  with the consumer's byte/code-point/UTF-16 convention.
- If answers hallucinate despite good retrieval, evaluate citation use and
  faithfulness separately from retrieval recall.

## Exercises

1. Implement sentence-boundary-aware chunking.
2. Measure collisions and Recall@k across hashing dimensions.
3. Implement BM25 and compare it with hashing cosine.
4. Put prompt injection in a retrieved chunk and verify authority is unchanged.
5. Validate that citations reference returned chunk IDs.

## Completion criteria

- Distinguish model weights from retrieval memory.
- Explain why offsets and provenance are needed.
- Explain overlap's recall/storage tradeoff.
- Explain normalized cosine as dot product.
- Evaluate retrieval and answer quality separately.
- `RetrievalSuite` passes.

## Primary sources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
