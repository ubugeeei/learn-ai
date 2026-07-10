# 08 — Derivatives, gradients, and the chain rule

## What you will build

Central finite-difference derivatives for scalar and multivariable functions.
Source: `src/main/scala/learnai/math/Calculus.scala`.

The method is too slow for large-model training, but it provides an independent
reference for checking automatic differentiation.

## Functions map inputs to outputs

\[
f:\mathbb{R}\to\mathbb{R},\qquad y=f(x)
\]

For \(f(x)=x^2\), input `3` gives output `9`. Optimization also needs to know
how output changes when input changes slightly.

## From average change to instantaneous slope

Average change from \(x\) to \(x+h\) is:

\[
\frac{f(x+h)-f(x)}{h}
\]

The derivative is its limit as \(h\) approaches zero:

\[
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\]

It says how many output units change per small input unit. Units matter: a loss
derivative with respect to a weight has units `loss / weight`.

## Basic rules

| Function | Derivative |
| --- | --- |
| \(c\) | \(0\) |
| \(x\) | \(1\) |
| \(x^n\) | \(nx^{n-1}\) |
| \(f+g\) | \(f'+g'\) |
| \(fg\) | \(f'g+fg'\) |
| \(e^x\) | \(e^x\) |
| \(\log x\) | \(1/x\) |
| \(\tanh x\) | \(1-\tanh^2x\) |

For \(f(x)=x^2\), \(f'(x)=2x\), so slope at `3` is `6`.

## Numerical differentiation

A computer cannot directly evaluate the limit. Central difference uses a small
finite \(h\):

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

```scala
val above = function(at + step)
val below = function(at - step)
(above - below) / (2.0 * step)
```

Large \(h\) measures distant points; extremely small \(h\) amplifies
floating-point cancellation. The default `1e-5` is a practical starting point,
not a universal constant.

## Partial derivatives and gradients

A neural-network loss is a scalar function of all parameters:

\[
L:\mathbb{R}^n\to\mathbb{R}
\]

A partial derivative changes one input while holding the others fixed:

\[
\frac{\partial L}{\partial x_i}
\]

Collecting all partial derivatives produces the gradient:

\[
\nabla L(\boldsymbol{x})=
\begin{bmatrix}
\frac{\partial L}{\partial x_1}\\
\vdots\\
\frac{\partial L}{\partial x_n}
\end{bmatrix}
\]

The loss is scalar, but its parameter gradient has shape `[n]`.

## Gradient direction

For unit direction \(\boldsymbol{u}\), the directional derivative is:

\[
D_{\boldsymbol{u}}L=\nabla L\cdot\boldsymbol{u}
\]

The gradient points toward greatest local increase. Its negative points toward
greatest local decrease, motivating gradient descent.

## Chain rule

For a composition \(y=f(g(x))\):

\[
\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}
\]

If \(g(x)=x^2\) and \(f(g)=3g+1\), then:

\[
\frac{dy}{dx}=3\times2x=6x
\]

A neural network is a deep composition. Backpropagation applies the chain rule
from output to input while reusing shared intermediate calculations.

## Complexity of finite differences

For \(n\) parameters, central differences require `2n` forward evaluations.
That is infeasible for billions of parameters. Reverse-mode autodiff finds all
gradients for a scalar loss in roughly a small multiple of one forward cost.

Finite differences remain valuable as a small-input **gradient check** because
they use an independent mechanism.

## Run and verify

```console
$ nix develop -c sbt 'runMain learnai.math.runGradientLab'
$ nix develop -c sbt test
```

## Exercises

1. Differentiate \(3x^2+2x+1\) and compare at `x=2`.
2. Sweep step size from `1e-1` to `1e-12` and tabulate error.
3. Derive the gradient of \(x^2+y^2\).
4. Move a small distance opposite the gradient and observe loss.
5. Count the number of function calls made by `Calculus.gradient`.

## Completion criteria

- Explain a derivative as local output change per input change.
- State the shape of a gradient.
- Explain why moving opposite the gradient lowers loss locally.
- Apply the chain rule to two composed functions.
- Explain why finite differences are a test rather than the training engine.
- `CalculusSuite` passes.
