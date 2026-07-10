# 11 — Train neurons, layers, and an MLP

## What you will build

Use scalar `Value` autodiff to implement neurons, layers, a multilayer
perceptron, MSE loss, and SGD, then train XOR.

- network: `src/main/scala/learnai/nn/ScalarNetwork.scala`
- experiment: `src/main/scala/learnai/nn/Xor.scala`

## One neuron

For input \(\boldsymbol{x}\), weights \(\boldsymbol{w}\), and bias \(b\):

\[
z=\boldsymbol{w}\cdot\boldsymbol{x}+b
=\sum_i w_i x_i+b
\]

Then apply activation \(\phi\):

\[
y=\phi(z)
\]

```scala
val weightedInputs = weights.zip(inputs).map { case (weight, input) =>
  weight * input
}
activation(Value.sum(weightedInputs) + bias)
```

Weights and bias are trainable leaves. Inputs and intermediates are not.

## Why activation is necessary

Composing linear functions remains linear:

\[
\boldsymbol{W}_2(\boldsymbol{W}_1\boldsymbol{x}+\boldsymbol{b}_1)+\boldsymbol{b}_2
=(\boldsymbol{W}_2\boldsymbol{W}_1)\boldsymbol{x}
+(\boldsymbol{W}_2\boldsymbol{b}_1+\boldsymbol{b}_2)
\]

Without a nonlinear activation, depth adds no nonlinear decision boundary.

| Activation | Equation | Property |
| --- | --- | --- |
| Linear | \(x\) | output scores and regression |
| Tanh | \(\tanh x\) | smooth range `[-1,1]` |
| ReLU | \(\max(0,x)\) | simple, constant positive-side derivative |

Modern Transformer feed-forward layers often use GELU or SwiGLU, but the need
for nonlinearity is the same.

## Layers and MLPs

A layer evaluates multiple neurons in parallel:

\[
\boldsymbol{y}=\phi(\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b})
\]

For input width \(n\) and output width \(m\), weights have shape `[m,n]` and
bias has shape `[m]`.

The XOR model is:

```text
input [2]
  -> hidden [4], tanh
  -> output [1], tanh
```

Parameter count:

- hidden: `4*2 + 4 = 12`;
- output: `1*4 + 1 = 5`;
- total: `17`.

Counting parameters is the first step toward memory and compute estimates.

## XOR requires nonlinearity

The target is positive when input signs differ and negative when they match.
No single line separates the two classes. Hidden neurons create several
intermediate boundaries, nonlinear activation bends them, and the output layer
combines them.

## Mean squared error

\[
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

The language model later uses cross entropy, but the training-loop structure is
unchanged.

## Trace one step

```scala
val predictions = dataset.map(example => model(example.inputs).head)
val loss = Loss.meanSquaredError(predictions, targets)
loss.backward()
Sgd.step(model.parameters, learningRate)
```

1. forward all four examples;
2. reduce errors to one scalar;
3. backpropagate to all 17 parameters;
4. update parameters;
5. rebuild a new graph next step.

This is full-batch training. Large datasets use sampled mini-batches.

## Initialization

If all weights begin at zero, neurons receive equal gradients and remain
symmetric. Seeded random initialization gives them distinct roles.

Xavier uniform uses:

\[
w\sim U\left(
-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}
\right)
\]

It aims to keep activation scale from vanishing or exploding across layers.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
```

The fixed seed makes the initialization and learning curve reproducible. Tests
require low loss and the correct sign for all four examples.

## Exercises

1. Recompute the 17 parameters.
2. Compare hidden widths `1`, `2`, and `8` over several seeds.
3. Replace tanh with ReLU and compare loss curves.
4. Find the first step below loss `0.02` for several learning rates.
5. Use a linear output and inspect its range.
6. Train an AND dataset and test whether a hidden layer is needed.

## Completion criteria

- Decompose a neuron into weights, input, bias, and activation.
- Explain why stacked linear layers collapse to one.
- Compute layer shapes and parameter counts.
- Identify forward, loss, backward, and update in code.
- Explain why seeds are fixed for comparisons.
- `ScalarNetworkSuite` passes.
