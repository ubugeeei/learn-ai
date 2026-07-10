# 27f — Paged KV cache and prefix reuse

## What you will build

Chapter 24's cache reserves `maximumContextLength` rows per request the
moment a session starts. That is correct and simple, and it means a
server's concurrency is dictated by the *worst-case* context length even
when the median request is fifty tokens. vLLM's PagedAttention insight is
that this is the same problem operating systems solved decades ago —
contiguous reservation wastes memory that paging reclaims — and the KV
cache has exactly the access pattern virtual memory wants: append-only
writes, position-indexed reads.

In this chapter you will:

- build a fixed page pool with per-page reference counts;
- give each request a page-table-indexed logical sequence;
- bound internal fragmentation to less than one page per sequence;
- implement prefix reuse via `fork` — sharing full pages, copying the
  partial one — and prove it safe without copy-on-write machinery;
- make exhaustion an explicit, recoverable error rather than an OOM.

## Prerequisites

You should understand Chapter 24 (what a KV cache stores, per-request
ownership, payload accounting) and Chapter 28c's cache-byte arithmetic.

## 1. The fragmentation argument

With full-context reservation, a request of length $n$ wastes
$(T - n) \cdot 2C \cdot 8$ bytes, unbounded by anything but $T$. With
pages of $s$ slots, the same request maps $\lceil n / s \rceil$ pages and
wastes at most $s - 1$ slots — the tail of its final page:

$$
\text{waste}_{\text{flat}} = (T - n) \cdot 2C \cdot 8
\qquad
\text{waste}_{\text{paged}} < s \cdot 2C \cdot 8
$$

The suite asserts the paged bound as a property while appending across
several page boundaries. The price is indirection: logical position $p$
lives at page `table(p / s)`, slot `p % s`, and a paged attention kernel
performs that lookup per score. This course pays the lookup in plain
Scala; production kernels pay it in GPU address arithmetic. The structure
is identical.

## 2. Ownership and the immutability invariant

The pool owns all storage plus a reference count per page. Sequences own
page *tables*. Two invariants make prefix sharing safe with nothing
fancier than refcounts:

1. appends only write into a sequence's final, partially filled page;
2. `fork` shares only *full* pages and copies the partial one.

Together they imply: any page with reference count above one is full,
and full pages are never written again. Shared pages are immutable by
construction, so there is no copy-on-write, no torn reads, and no
divergence hazard — parent and child each keep appending into pages they
exclusively own. This is the same reasoning that made Chapter 24's
"cache does not retain queries" argument work: identify what can never
change, then share exactly that.

Forking is how a serving stack reuses a common system-prompt prefix
across requests: render the prefix once, fork per request, and the
prefix's full pages are stored once regardless of fan-out.

## 3. Exhaustion and eviction

A fixed pool *will* run out. Allocation returns `Either`, so exhaustion
is a value the caller handles, not an exception mid-decode: the failed
sequence is untouched and retryable after memory is reclaimed. `fork` is
atomic under exhaustion — if the partial-page copy cannot be allocated,
every retained reference is released and the parent is unchanged.

`release` is the reclamation *mechanism*: it returns pages (decrementing
shared counts) and permanently retires the sequence. Which victim to
release — the eviction *policy* — belongs to the serving scheduler
(Chapter 27e, planned); building mechanism and policy separately is the
point of doing this chapter first.

Byte accounting deliberately lives at the pool. With sharing, summing
per-sequence footprints double-counts shared pages; the pool's allocated
page count is the ground truth a scheduler budgets against, which is why
`allocatedPayloadBytes` has no per-sequence counterpart.

## 4. Run the experiment

```console
$ nix develop -c sbt 'runMain learnai.testing.AllTests'
```

Then compute the concurrency win: with Chapter 29a's
`kvCachePayloadBytes`, compare how many 50-token requests fit in a fixed
budget under full-context reservation versus paging with $s = 16$.

## 5. Implementation walkthrough

`KvPagePool` allocates two flat arrays of `pageCount * pageSize *
channels` doubles (overflow-checked), a reference-count array, and a
free list holding every page. `allocate` pops a page and sets its count
to one; `retain` and `release` adjust counts, and a count reaching zero
pushes the page back. All raw row access is `private[serving]` and
bounds-checked at the offset computation, so a sequence cannot address
another sequence's rows except through its own page table.

`PagedKvSequence.append` computes the slot as `length % pageSize`; slot
zero means a page boundary, and only then does it allocate. The write
always targets `pageTable.last` — invariant one is visible as a single
line. `keyAt`/`valueAt` translate positions through the table and refuse
released sequences and out-of-range arguments.

`fork` retains the `length / pageSize` full pages, then handles the
partial page: none (share everything, allocate nothing — the boundary
case has its own test) or copy `length % pageSize` rows into a fresh
page. The error path releases everything it retained before returning,
which the atomicity test verifies by checking pool counts and parent
reads after a refused fork.

`release` walks the table, releases every page, clears state, and flips
the sequence inactive; every subsequent operation fails with a
"released" message.

## 6. Reading the tests

- the interleaved-sequences oracle appends to two sequences alternately
  — forcing their page tables to interleave pool pages — and compares
  every read against a trivially correct per-sequence list of appended
  rows;
- boundary tests pin page counts and wasted slots at 1, 4, and 5
  appends, plus the sub-page fragmentation bound as a rolling property;
- exhaustion tests show the explicit error, the untouched failed
  sequence, and reuse after release;
- fork tests count pool pages before and after (3 → 4 with a partial
  page; unchanged at a boundary), verify child reads, prove divergence
  isolation, and check that releasing the parent keeps shared pages
  alive until the child releases;
- the atomic-failure test forks from a full pool and verifies no
  reference leaks;
- accounting and rejection tests cover the byte formulas, released-use,
  wrong widths, and constructor validation.

## 7. Debugging checklist

1. Check reference counts first: pool pages allocated must equal the
   union of live page tables, counted once.
2. If two requests corrupt each other, log their page tables; a shared
   page with count one means a fork forgot to retain.
3. If a fork leaks pages on failure, audit the error path — atomicity
   bugs hide exclusively there.
4. Verify the immutability invariant directly in tests: any page with
   count above one must be full.
5. When translating this to a kernel, test the position-to-page
   arithmetic at slot zero and slot `pageSize - 1`; off-by-ones live at
   the boundaries.

## 8. Failure modes to test

- appends that write into a shared (full) page;
- forks that share the partial page instead of copying it;
- reference counts leaked on fork failure or double-counted on release;
- reads through a stale table after release;
- per-sequence byte sums presented as pool usage under sharing;
- exhaustion surfacing as an exception that kills an unrelated request.

## Exercises

1. Wire a paged sequence into Chapter 24's cached attention: implement
   `forwardCached` reading through `keyAt`/`valueAt` and prove
   equivalence against the flat cache on identical appends.
2. Add a high-water-mark statistic to the pool and drive it with a
   simulated arrival process; relate the peak to request length
   distribution.
3. Implement hash-based prefix matching: detect that a new request's
   prompt prefix equals an existing sequence's full pages and fork
   automatically.
4. Sweep the page size: measure fragmentation (small waste) against
   table length and lookup count (indirection cost) and argue a default.

## Completion criteria

You are done when you can:

- state both fragmentation formulas and the paged bound's proof;
- explain why full-page sharing needs no copy-on-write, from the two
  invariants;
- trace a fork, a divergent append, and a two-stage release through
  reference counts;
- explain why exhaustion must be a value and eviction a separate policy;
- say where byte accounting must live under sharing, and why.

## Primary sources

- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [SGLang: Efficient Execution of Structured LM Programs (RadixAttention)](https://arxiv.org/abs/2312.07104)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
