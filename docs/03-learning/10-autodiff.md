# 10 — Reverse-mode automatic differentiation

## What you will build

A scalar computation graph that computes every parameter gradient with one
reverse traversal. Source: `src/main/scala/learnai/autodiff/Value.scala`.

## Treat an expression as a graph

Consider:

\[
x=2,\quad y=-3,\quad z=xy+x^2
\]

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

For \(c=f(a,b)\), if \(\partial L/\partial c\) arrives from above:

\[
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial c}\frac{\partial c}{\partial a}
\]

For multiplication \(c=ab\), local derivatives are \(b\) and \(a\):

```scala
output.backwardRule = () =>
  accumulateGradient(other.data * output.gradient)
  other.accumulateGradient(data * output.gradient)
```

`output.gradient` is upstream information; the operand value is the local
derivative.

## Accumulate rather than overwrite

In \(z=x^2+x\), `x` reaches the output through two paths:

\[
\frac{dz}{dx}=2x+1
\]

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

\[
\frac{\partial L}{\partial L}=1
\]

## Local rules

| Forward | Local derivative |
| --- | --- |
| \(a+b\) | \(1,1\) |
| \(ab\) | \(b,a\) |
| \(a^r\) | \(ra^{r-1}\) |
| \(e^a\) | \(e^a\) |
| \(\log a\) | \(1/a\) |
| \(\tanh a\) | \(1-\tanh^2a\) |
| \(\operatorname{ReLU}(a)\) | `1` for `a>0`, else `0` |

Subtraction and division are composed from existing operations:

\[
a-b=a+(-1)b,\qquad a/b=ab^{-1}
\]

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

## Exercises

1. Differentiate \((x+2)^3\) by hand and with `Value`.
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
