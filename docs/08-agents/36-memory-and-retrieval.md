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
