# 12 — Tensor reverse-mode autodiff

## What you will build

A dense row-major Tensor with arbitrary rank, shape/index conversion,
element-wise operations, reductions, reshape, transpose, matrix multiplication,
and backward rules.

- shape: `src/main/scala/learnai/tensor/Shape.scala`
- tensor: `src/main/scala/learnai/tensor/Tensor.scala`

## Why scalar graphs do not scale

`Value` creates a JVM object for every number and operation. A `[1024,1024]`
output alone contains about a million scalars, with many more multiplication
and addition nodes.

A Tensor node groups an entire operation:

```text
scalar graph: millions of scalar nodes
Tensor graph: MatMul -> Add -> Tanh
```

The calculus is unchanged. Loops move inside a backward rule. Production
systems execute those rules as CPU or GPU kernels.

## Rank, shape, and size

- rank 0: scalar, shape `[]`, size 1;
- rank 1: vector, shape `[n]`;
- rank 2: matrix, shape `[m,n]`;
- rank 3: often `[batch,time,channels]`.

$$
\operatorname{size}([d_0,\ldots,d_{r-1}])=\prod_i d_i
$$

The scalar shape has no dimensions, but the empty product is one.

## Strides and offsets

Shape `[2,3,4]` has row-major strides `[12,4,1]`:

$$
\operatorname{offset}([i,j,k])=12i+4j+k
$$

Coordinate `[1,2,3]` maps to offset `23`. The final axis is contiguous in
memory. `Shape` computes dimensions, strides, and size once for every Tensor.

## No implicit broadcasting yet

Broadcasting repeats singleton or missing axes, but backward must then reduce
gradient along exactly those axes. This first engine requires equal shapes:

```text
[2,3] + [2,3] -> [2,3]
[2,3] + [3]   -> error
```

Later operations such as `addRowVector` name their broadcast and reduction
semantics explicitly.

## Element-wise backward

For $C=A\odot B$:

$$
\frac{\partial L}{\partial A_i}
=\frac{\partial L}{\partial C_i}B_i,\qquad
\frac{\partial L}{\partial B_i}
=\frac{\partial L}{\partial C_i}A_i
$$

This is scalar multiplication's rule applied at every index. Shared Tensor
paths still require gradient accumulation.

## Reduction backward

For $s=\sum_i x_i$:

$$
\frac{\partial s}{\partial x_i}=1
$$

The scalar upstream gradient is copied to every input element. Mean composes sum
with division by element count.

## Reshape and transpose

Reshape changes interpretation without changing flat order and requires equal
element count. Its backward keeps flat indices unchanged. Transpose reorders
elements, so backward applies the inverse index mapping.

## Matrix multiplication backward

For `C = A B`:

```text
A [m,k] x B [k,n] -> C [m,n]
```

$$
\frac{\partial L}{\partial A}
=\frac{\partial L}{\partial C}B^{\mathsf T}
$$

$$
\frac{\partial L}{\partial B}
=A^{\mathsf T}\frac{\partial L}{\partial C}
$$

The implementation distributes each output gradient to every inner-index
product that contributed during forward.

## Gradient checking

Tests perturb individual matrix elements by $+h$ and $-h$, then compare
finite differences with `matmul -> pow -> mean` backward. For larger tensors,
check all elements on tiny shapes and sampled elements on realistic shapes.

## Memory estimate

`Double` uses 8 bytes, so `[B,T,C]` data alone costs `8*B*T*C` bytes. Training
also needs gradients, intermediate activations, and optimizer state.

This educational engine eagerly allocates a same-size gradient for every node.
Production systems avoid gradients in inference, reuse buffers, and use
activation checkpointing.

## Implementation walkthrough

`Shape` is constructed before `Tensor`. It validates non-negative dimensions,
computes total size with checked multiplication, and precomputes row-major
strides. Both `offset` and `coordinates` validate their input; they are inverse
operations over valid coordinates.

`Tensor` then owns a shape, data array, equally sized gradient array, operation
label, parent vector, and backward closure. Public constructors distinguish
constants from trainable parameters. Operation constructors remain private so
callers cannot forge a graph node whose backward rule is missing.

Trace `[1,2] + [3,4]`. Addition validates equal shapes, allocates `[4,6]`, and
creates a node with both inputs as parents. During backward, each upstream
element is added to the corresponding gradient in both parents. Hadamard
multiplication uses the same flat loop but multiplies upstream by the opposite
operand value.

Matmul makes indexing more explicit. For output `(i,j)`, the forward inner loop
reads `A(i,k)` and `B(k,j)`. During backward the same output gradient `g(i,j)`
contributes:

```text
A.grad(i,k) += g(i,j) * B(k,j)
B.grad(k,j) += g(i,j) * A(i,k)
```

Looping over output cells and the shared index is a literal implementation of
the matrix-gradient equations. `reshape` copies flat order and sends gradients
back by the same flat index. `transpose2D` must invert its row/column mapping.

Backward again uses identity-based topological ordering. This engine eagerly
allocates gradients even for inference constants; the later cached path
deliberately detaches cached arrays because it is forward-only.

## Reading the tests

Shape tests use hand-derived offsets. Shared tensor paths catch missing
accumulation. Matmul backward has both exact small examples and finite-
difference element checks. Reshape/transpose tests verify gradient routing, not
only values. Explicit broadcasting rejection catches accidental acceptance of
ambiguous gradient-reduction rules. Cross-entropy and RMSNorm receive their own
numerical checks because they contain fused formulas.

## Debugging checklist

1. Write every operand and result shape before inspecting values.
2. Check a single flat index and its coordinates in both directions.
3. For backward errors, isolate one output element and one upstream gradient.
4. Confirm shared parents accumulate rather than overwrite.
5. If a finite-difference check fails, reduce the tensor to the smallest shape
   that still exercises the operation.
6. Distinguish a value-layout bug from a gradient-layout bug by testing forward
   first.

## Exercises

1. Compute strides and final offset for `[2,3,4,5]`.
2. Add sigmoid and check its gradient numerically.
3. Test `transpose2D.transpose2D` for values and gradients.
4. Derive matmul backward with indexed notation.
5. Compute data and gradient bytes for `[32,128,768]`.
6. Design `addRowVector` and its reduction axis.

## Completion criteria

- Convert among rank, shape, size, stride, and offset.
- Explain scalar-graph versus Tensor-graph granularity.
- Explain gradient movement through sum, reshape, and transpose.
- State both matmul gradient shapes and equations.
- Explain why broadcasting is initially explicit.
- `TensorSuite` passes.
