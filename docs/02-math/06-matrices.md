# 06 — Matrices and shapes

## What you will build

An immutable row-major `MatrixD` with transpose, matrix-vector multiplication,
and matrix multiplication. Source: `src/main/scala/learnai/math/MatrixD.scala`.

## A matrix has two axes

An $m$-by-$n$ matrix is:

$$
\boldsymbol{A}=
\begin{bmatrix}
a_{11}&\cdots&a_{1n}\\
\vdots&\ddots&\vdots\\
a_{m1}&\cdots&a_{mn}
\end{bmatrix}
\in\mathbb{R}^{m\times n}
$$

Its shape is `[m,n]`. In Scala, those dimensions are `matrix.rows` and
`matrix.columns`.

```scala
val matrix = MatrixD.fromRows(
  Vector(
    VectorD(1.0, 2.0, 3.0),
    VectorD(4.0, 5.0, 6.0)
  )
)
```

Shape is `[2,3]`; `matrix(1,2)` returns `6.0`. LLMs store embedding tables as
`[vocabulary,channels]` and dense weights as
`[inputChannels,outputChannels]` or its documented transpose.

## Row-major layout

```text
matrix:  [[a,b,c],
          [d,e,f]]
storage: [a,b,c,d,e,f]
offset(row,column) = row * columns + column
```

One-dimensional contiguous storage uses fewer objects and makes sequential
rows cache-friendly. `MatrixD` keeps dimensions so the flat storage retains its
two-dimensional meaning.

## Transpose

Transpose exchanges rows and columns:

$$
(\boldsymbol{A}^{\mathsf T})_{ij}=A_{ji}
$$

Shape changes `[m,n] -> [n,m]`, and transposing twice returns the original:

$$
(\boldsymbol{A}^{\mathsf T})^{\mathsf T}=\boldsymbol{A}
$$

Such always-true relationships are useful property tests.

## Matrix-vector multiplication

For `A [m,n]` and `x [n]`:

$$
\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x},\qquad
y_i=\sum_{j=1}^{n}A_{ij}x_j
$$

```text
[m,n] x [n] -> [m]
```

Every output is a dot product between one matrix row and the input vector. The
inner dimension must match. A neural-network row can be interpreted as one
output neuron's weights.

## Matrix multiplication

For `A [m,k]` and `B [k,n]`:

$$
\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B},\qquad
C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}
$$

```text
[m,k] x [k,n] -> [m,n]
       ^   ^
       inner dimensions match
```

Each output element is a dot product between a row of the left matrix and a
column of the right. The simple algorithm costs $O(mkn)$ time and $O(mn)$
output memory. Optimized libraries use blocking, vector instructions, threads,
and GPUs but compute the same equation.

## Runtime shapes

`MatrixD` has the same Scala type for `[2,3]` and `[4,8]`. Runtime dimensions
therefore require runtime checks. Error messages include both shapes.

Some shapes can be encoded in types, but batch and sequence lengths are often
known only at runtime. Static and dynamic validation are complementary.

## Empty dimensions

`[0,3]` and `[2,0]` both have zero elements but different meanings. Explicit
dimensions preserve that distinction. Some reductions, such as maximum or
mean, remain undefined and must return an error value.

## Implementation walkthrough

`MatrixD` stores `rows`, `columns`, and one row-major array. The flat offset is

```scala
row * columns + column
```

so adjacent columns of one row are contiguous. Every accessor validates row and
column before computing the offset. Construction validates that
`values.length == rows * columns`; shape and storage cannot disagree afterward.

Trace matrix multiplication with

```text
A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]
```

The result entry `C(1,0)` is row 1 of `A` dotted with column 0 of `B`:

```text
3*5 + 4*7 = 43
```

The implementation's loop nesting is `output row`, `output column`, `inner
index`. The inner loop changes the shared dimension and accumulates exactly the
products in that dot product. Output storage has `rows * other.columns`
elements. Compatibility is checked once with `columns == other.rows` before
allocation.

Transpose allocates a new array and maps `(row,column)` to `(column,row)`. A
view with strides would avoid copying, but it would complicate this first value
type and every later operation. `updated` follows the vector pattern: clone,
modify, return. Zero-sized dimensions are retained even when flat storage is
empty, because `[0,3]` and `[0,5]` carry different compatibility information.

## Reading the tests

Indexing tests enumerate every entry so a column-major bug cannot hide behind a
symmetric matrix. Multiplication uses small integer matrices with hand-derived
answers. The incompatible-shape test checks the public error rather than
waiting for an index exception. Double transpose is a useful property test.
Zero-dimension tests catch implementations that infer shape only from values.

## Debugging checklist

1. Write both operand shapes as `[m,k] x [k,n]` before reading values.
2. Check one incorrect output cell by expanding its exact dot product.
3. Print flat offsets for the first two rows to find row/column confusion.
4. If transpose works only for square matrices, test `[2,3]` immediately.
5. If empty matrices lose shape, inspect constructors that infer dimensions
   from nested collections.

## Exercises

1. Give the output shape and multiplications per element for `[2,3] x [3,4]`.
2. Add matrix Hadamard multiplication.
3. Implement an identity-matrix factory and test `A.matmul(I) == A`.
4. Return column sums and state the output shape first.
5. Explain why the simple loop reads the right matrix non-contiguously.

## Completion criteria

- Map matrix coordinates to row-major offsets.
- Map matvec and matmul equations to loop indices.
- Explain why inner matrix dimensions must match.
- Recall `[m,k] x [k,n] -> [m,n]` without reference.
- `MatrixDSuite` passes.
