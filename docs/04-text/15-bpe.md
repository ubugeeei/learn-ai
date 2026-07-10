# 15 — Byte pair encoding

## What you will build

Train a byte-level BPE vocabulary by repeatedly merging the most frequent
adjacent token pair, then encode and decode with the learned merge table.
Source: `src/main/scala/learnai/text/BpeTokenizer.scala`.

## Vocabulary-size versus sequence-length tradeoff

A byte vocabulary has only 256 entries but often produces long sequences. At
the opposite extreme, one token per document would be short but impossible to
generalize to unseen documents.

- smaller vocabulary: fewer embedding/output parameters, longer sequences;
- larger vocabulary: shorter sequences, more parameters and rare tokens.

BPE starts from universal byte coverage and adds only repeated corpus patterns.

## Training algorithm

Initial sequences are UTF-8 byte IDs:

```text
"banana" -> [98,97,110,97,110,97]
```

Repeat until the target vocabulary size:

1. count adjacent pairs in every sequence;
2. select the most frequent pair;
3. assign a new token ID;
4. replace non-overlapping occurrences in every sequence.

If `(97,110)`, the bytes for `an`, is most frequent:

```text
[98,97,110,97,110,97]
 -> [98,256,256,97]
```

Later merges may reference previous merged tokens, allowing `ban`, words, or
multi-word fragments to emerge.

## The merge table is part of the model artifact

```text
merge 0: (97,110) -> 256
merge 1: (98,256) -> 257
```

Merge order defines encoding. Result ID `256+i` references only bytes or older
merge results, creating a directed acyclic expansion graph.

Weights are meaningless with a different merge table because token IDs no
longer represent the same byte sequences. A deployable bundle includes
tokenizer type, merges, special tokens, normalization rules, version, and
checksums.

## Encoding

```text
UTF-8 bytes
 -> apply merge 0 everywhere
 -> apply merge 1 everywhere
 -> ...
 -> token IDs
```

The teaching implementation scans sequence and merge table directly.
Production tokenizers use merge ranks, pair indices, or priority queues.

## Decoding

Byte-token expansion is the byte itself. A merged token expands by concatenating
its left and right expansions:

```text
bytes(256) = bytes(97) ++ bytes(110) = [97,110]
```

After expanding all tokens, strict UTF-8 decoding restores text.

## Deterministic tie-breaking

Equal pair frequencies must not depend on hash-map iteration. This trainer
chooses by:

1. larger count;
2. smaller left ID;
3. smaller right ID.

Corpus content and order, preprocessing, target size, and tie policy are all
reproducibility inputs.

## Unicode boundaries

Byte-level BPE does not know code-point or grapheme boundaries. A token may
contain part of a code point or several visible symbols. The complete token
sequence still round-trips because its byte concatenation is unchanged.

A streaming decoder must keep UTF-8 state rather than decoding each token
independently.

## Complexity and production concerns

The reference trainer recounts and replaces the entire corpus for every merge,
roughly \(O(MN)\) for \(M\) merges and \(N\) corpus tokens. Large-scale
training requires incremental counts, indexed occurrences, parallelism, or
external memory.

Production designs also specify normalization, pre-tokenization, special-token
collision rules, byte fallback, streaming, and artifact compatibility.

## Exercises

1. Count all pairs in `banana` and predict the first merge.
2. Measure average token count for several vocabulary sizes.
3. Compare compression on natural language and code corpora.
4. Explain how unordered tie handling causes nondeterminism.
5. Design merge-table serialization and a round-trip test.
6. Find learned tokens that are not valid UTF-8 independently.

## Completion criteria

- Explain the vocabulary/sequence tradeoff.
- Execute count, select, and merge on a small example.
- Explain why merge ordering supports encode and decode.
- Explain why byte-level tokens need not match Unicode boundaries.
- Explain why tokenizer artifacts and weights must be versioned together.
- `BpeTokenizerSuite` passes.

## Primary source

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Course reading map and critical summary](../09-papers/40-primary-reading-map.md)
