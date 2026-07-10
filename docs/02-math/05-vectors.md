# 05 — Vectors

## What you will build

An immutable `VectorD` containing finite `Double` values, with addition,
subtraction, scaling, Hadamard product, dot product, and norm. Source:
`src/main/scala/learnai/math/VectorD.scala`.

## From scalars to vectors

A scalar is one number. A vector is an ordered collection of scalars:

\[
\boldsymbol{x}=
\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}
\in\mathbb{R}^n
\]

`VectorD(2.0, -1.0, 3.0)` has shape `[3]`. Mathematical notation commonly
indexes from 1, while Scala indexes from 0.

| Mathematics | Scala | Value |
| --- | --- | --- |
| \(x_1\) | `x(0)` | `2.0` |
| \(x_2\) | `x(1)` | `-1.0` |
| \(n\) | `x.size` | `3` |

An LLM represents a token with hundreds or thousands of values: its embedding
vector.

## Element-wise operations

Equal-size vectors add corresponding elements:

\[
\boldsymbol{x}+\boldsymbol{y}=
[x_1+y_1,\ldots,x_n+y_n]
\]

```scala
VectorD(1.0, 2.0) + VectorD(3.0, 4.0)
// VectorD(4.0, 6.0)
```

Different sizes provide no valid correspondence, so `VectorD` rejects the
operation instead of truncating or padding.

Scalar multiplication and Hadamard multiplication are:

\[
a\boldsymbol{x}=[ax_1,\ldots,ax_n]
\]

\[
\boldsymbol{x}\odot\boldsymbol{y}=[x_1y_1,\ldots,x_ny_n]
\]

The Hadamard result remains a vector. A dot product does not.

## Dot product

\[
\boldsymbol{x}\cdot\boldsymbol{y}=\sum_{i=1}^{n}x_i y_i
\]

```scala
VectorD(1.0, 2.0, 3.0).dot(VectorD(4.0, 5.0, 6.0))
// 1*4 + 2*5 + 3*6 = 32
```

Shape: `[n] dot [n] -> scalar`. The result measures directional alignment with
magnitude included. Attention later uses query-key dot products as relevance
scores.

## Norm

The Euclidean or L2 norm is vector length:

\[
\lVert\boldsymbol{x}\rVert_2
=\sqrt{\sum_i x_i^2}
=\sqrt{\boldsymbol{x}\cdot\boldsymbol{x}}
\]

`VectorD(3.0, 4.0).norm` is `5.0`. Cosine similarity removes magnitude:

\[
\cos\theta=
\frac{\boldsymbol{x}\cdot\boldsymbol{y}}
{\lVert\boldsymbol{x}\rVert_2\lVert\boldsymbol{y}\rVert_2}
\]

The retrieval chapter uses normalized dot products for vector search.

## Why hide the backing array?

JVM `Array[Double]` provides contiguous primitive storage but is mutable. If a
caller shares it, a vector can change without the vector API observing it.

`VectorD` protects this representation invariant:

- construction validates every value is finite;
- owned arrays are not exposed;
- updates and operations return new vectors;
- shape checks occur before arithmetic.

Once constructed, a `VectorD` can be treated as an immutable finite value.

## Why use `while` inside kernels?

Numerical inner loops use explicit indices to avoid per-element objects and to
make operation count visible. The public API remains functional.

- addition: time \(O(n)\), output memory \(O(n)\);
- dot product: time \(O(n)\), result memory \(O(1)\);
- index access: time \(O(1)\).

## Verification

```console
$ nix develop -c sbt test
```

The suite checks values, immutability, shape failures, non-finite input, and
empty reductions.

## Implementation walkthrough

`VectorD` owns a private `Array[Double]` for efficient indexed loops but exposes
value semantics. Public construction copies incoming values into owned storage
and validates each element with `requireFinite`. `toVector` returns another
copy, so a caller cannot mutate the backing array after construction.

Element-wise operations use `zipMap`. Before allocating output, it calls
`requireSameSize`; this places the shape error at the vector boundary rather
than allowing an array index failure halfway through a loop. The output array is
fresh, so `a + b` cannot change `a` or `b`.

Trace a dot product by hand:

```text
a = [1, 2, 3]
b = [4, 5, 6]
products = [4, 10, 18]
sum = 32
```

The implementation generates the products by index and passes them to
`Numerics.compensatedSum`. Dot product is therefore one scalar reduction, not a
new public vector allocation. `squaredNorm` reuses `dot(this)` and `norm` takes
the square root, keeping the mathematical definition visible in code.

`updated` is a useful immutability example: clone the backing array, change the
clone, validate through the constructor path, and return a new `VectorD`.
`mean`, `max`, and `argmax` return `Either` on an empty vector because no result
exists. In contrast, `sum` of an empty vector is the additive identity `0.0`.

## Reading the tests

The addition test also asserts that inputs retain their original values; it
catches aliasing, not only arithmetic error. Size-mismatch tests ensure failure
occurs before partial work. The finite-value constructor test protects every
later operation's invariant. Empty-reduction tests distinguish operations with
valid identities from undefined selections. Equality/hash tests are implicit
in many `Assert.equal` calls and rely on array-content comparison, not object
identity.

## Debugging checklist

1. Print both sizes before investigating arithmetic.
2. If an input changes, inspect constructor and result-array ownership.
3. If a dot product is wrong only at large dynamic range, compare naive and
   compensated sums.
4. If `argmax` disagrees on ties, document and test the intended first-index
   policy.
5. If a non-finite value appears later, verify construction did not use an
   unsafe internal path.

## Exercises

1. Compute a Hadamard product by hand.
2. Add the L1 norm \(\sum_i|x_i|\) and tests with negative values.
3. Implement cosine similarity for aligned, orthogonal, and opposite vectors.
4. Design the return type for cosine similarity with a zero vector.
5. Observe what happens when `map` returns positive infinity.

## Completion criteria

- Distinguish scalar, vector, and shape.
- Explain the output-shape difference between Hadamard and dot products.
- Relate norm to dot product.
- Explain why the immutable wrapper is safer than a raw array.
- `VectorDSuite` passes.
