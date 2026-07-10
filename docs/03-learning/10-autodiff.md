# 10 — Reverse-mode automatic differentiation

## What you will build

A scalar computation graph that computes every parameter gradient with one
reverse traversal. Source: `src/main/scala/learnai/autodiff/Value.scala`.

## Treat an expression as a graph

Consider:

$$
x=2,\quad y=-3,\quad z=xy+x^2
$$

Normal evaluation may keep only results. Automatic differentiation records
operations as nodes and dependencies as edges:

```text
x ----(*)----+
 \   /       |
  \ y        (+)---- z
   \         |
    --pow(2)-+
```

Each `Value` node stores:

- forward `data`;
- gradient of the final output with respect to this node;
- the operation and parent nodes;
- a local backward rule.

## Forward mode and reverse mode

- forward mode propagates derivatives from one input toward all outputs;
- reverse mode propagates from one output toward all inputs.

Neural networks have many parameters and one scalar loss. Reverse mode obtains
all parameter gradients in one reverse traversal.

## Local derivative times upstream gradient

For $c=f(a,b)$, if $\partial L/\partial c$ arrives from above:

$$
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial c}\frac{\partial c}{\partial a}
$$

For multiplication $c=ab$, local derivatives are $b$ and $a$:

```scala
output.backwardRule = () =>
  accumulateGradient(other.data * output.gradient)
  other.accumulateGradient(data * output.gradient)
```

`output.gradient` is upstream information; the operand value is the local
derivative.

## Accumulate rather than overwrite

In $z=x^2+x$, `x` reaches the output through two paths:

$$
\frac{dz}{dx}=2x+1
$$

One path contributes `2x`, the other contributes `1`. Overwriting loses a path,
so gradients must accumulate. Parameter sharing, residual connections, and
repeated embedding lookup all depend on this rule.

## Topological order

A node's backward rule must run after all downstream paths have contributed.
Build a topological order, then traverse it in reverse:

```text
forward order:  leaves -> intermediates -> loss
backward order: loss   -> intermediates -> leaves
```

Shared nodes are tracked by object identity. The root gradient is seeded with:

$$
\frac{\partial L}{\partial L}=1
$$

## Local rules

| Forward | Local derivative |
| --- | --- |
| $a+b$ | $1,1$ |
| $ab$ | $b,a$ |
| $a^r$ | $ra^{r-1}$ |
| $e^a$ | $e^a$ |
| $\log a$ | $1/a$ |
| $\tanh a$ | $1-\tanh^2a$ |
| $\operatorname{ReLU}(a)$ | `1` for `a>0`, else `0` |

Subtraction and division are composed from existing operations:

$$
a-b=a+(-1)b,\qquad a/b=ab^{-1}
$$

## Parameter updates and graph lifetime

Operation data is a forward-time snapshot. A correct training step is:

1. build a fresh graph from current parameters;
2. call `backward()` on scalar loss;
3. update parameter leaves;
4. discard the old graph;
5. rebuild forward for the next step.

Updating a leaf does not retroactively recompute old intermediates.

## Gradient checks

Implement the same function with raw `Double` and `Value`, then compare
automatic and central-difference gradients:

```scala
val numeric = Calculus.derivative(rawFunction, at = x.data)
output.backward()
Assert.close(x.gradient, numeric, tolerance = 1e-8)
```

Hand derivation, finite differences, and autodiff are three independent views.

## Simplifications

- one JVM object per scalar node;
- recursive graph ordering;
- no higher-order derivatives;
- no in-place or parallel backward;
- no graph profiler or visualization.

The Tensor engine groups many scalars into one operation node.

## Implementation walkthrough

`Value` separates two kinds of nodes. `Value.parameter` creates a trainable
leaf whose data may change during optimization. `Value.constant` creates a
non-trainable leaf. Every arithmetic method computes forward data immediately,
creates an operation node with parent references, and installs a closure that
knows how to send an upstream gradient to those parents.

Trace `z = x * y + x.pow(2)` with `x=2`, `y=-3`:

```text
m = x*y   = -6
p = x^2   = 4
z = m+p   = -2
```

Backward begins by discovering nodes with a depth-first traversal. An
`IdentityHashMap` is used because two distinct nodes may have equal data and
labels but must remain separate graph vertices. The topological order contains
each object once. Gradients are cleared, `z.gradient` is seeded to one, and
closures run in reverse order.

The addition closure sends `1 * upstream` to both `m` and `p`. Multiplication
sends `y * upstream` to `x` and `x * upstream` to `y`. Power sends
`2*x * upstream` to `x`. The two contributions to `x` accumulate:

```text
from x*y: y = -3
from x^2: 2*x = 4
total dz/dx = 1
dz/dy = x = 2
```

`applyGradient` is restricted to trainable leaves. Updating operation nodes
would break the relationship between their stored data and parents. Graph data
is a forward-time snapshot, so training builds a new graph after every update.

## Reading the tests

The product/sum test checks local rules with hand arithmetic. The reused-node
test is the minimal detector for assignment instead of accumulation. A
composite function is checked with central differences, an implementation
independent of the graph. A second `backward()` call verifies old gradients are
cleared. Domain tests for `log` catch invalid forward values before backward.

## Debugging checklist

1. Draw the graph and mark shared object identities.
2. Print topological order and confirm every node appears once.
3. Seed only the scalar root with gradient one.
4. If a shared-node gradient is too small, search for assignment instead of
   accumulation.
5. If finite differences disagree, compare one local rule at a time.
6. Rebuild the graph after parameter updates.

## Exercises

1. Differentiate $(x+2)^3$ by hand and with `Value`.
2. Compose sigmoid from `exp` and arithmetic.
3. Add `sin` and check it numerically.
4. Replace gradient accumulation with assignment and observe the shared-node
   failure.
5. Design a DOT graph exporter.

## Completion criteria

- Map expression nodes and edges to code.
- Explain why reverse mode fits many inputs and one output.
- Separate upstream gradient from local derivative.
- Give an expression requiring gradient accumulation.
- Explain why the graph is rebuilt after updates.
- `ValueSuite` passes.
