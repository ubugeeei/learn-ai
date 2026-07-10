# 09 — Build learning with gradient descent

## What you will build

Minimize a one-parameter function and record parameter, loss, and gradient at
every step. Source: `src/main/scala/learnai/learning/GradientDescent.scala`.

## Write learning as optimization

Let parameter $\theta$ control model behavior and loss $L(\theta)$ measure
error. Training searches for the parameter that minimizes loss:

$$
\theta^*=\operatorname*{arg\,min}_{\theta}L(\theta)
$$

`min` is the minimum value. `arg min` is the argument that produces it.

The first example moves a parameter toward `3`:

$$
L(\theta)=(\theta-3)^2
$$

This loss is non-negative and reaches zero only at $\theta=3$. Its derivative
is:

$$
\frac{dL}{d\theta}=2(\theta-3)
$$

## Update rule

The gradient points toward increasing loss, so subtract it:

$$
\theta_{t+1}=\theta_t-\eta\frac{dL}{d\theta_t}
$$

| Symbol | Meaning | Code |
| --- | --- | --- |
| $t$ | step | `step` |
| $\theta_t$ | current parameter | `parameter` |
| $L$ | loss function | `loss` |
| $dL/d\theta_t$ | local gradient | `gradient` |
| $\eta$ | learning rate | `learningRate` |

```scala
parameter -= learningRate * gradient
```

The order matters:

1. run forward with current parameters;
2. compute loss;
3. obtain gradients;
4. update parameters;
5. begin a new step with new parameters.

## Observe the trajectory

```console
$ nix develop -c sbt 'runMain learnai.learning.runGradientDescentLab'
```

Inspect more than the final value:

- Does loss consistently fall?
- Does gradient magnitude approach zero?
- Do parameters oscillate between points?
- Does any value become `NaN` or infinity?
- Has improvement become too small?

`DescentObservation` preserves the evidence needed to answer those questions.

## Learning-rate tradeoff

- too small: stable but slow;
- appropriate: rapid stable decrease;
- too large: overshoot, oscillation, or divergence.

For the quadratic example, distance from the target is multiplied by
$1-2\eta$ each step. At `η=0.1` it shrinks by `0.8`; at `η=1` it flips sign
without shrinking; above `1` it grows.

## Non-convex loss

The example is convex with one minimum. Neural-network loss surfaces are
high-dimensional and include saddle points, flat regions, and many paths.
Gradient descent sees only local slope and does not guarantee a global minimum.

Practical training combines initialization, mini-batch noise, optimizer state,
and large parameter spaces. Success is evaluated on unseen data, not only by
reaching a theoretical minimum.

## Train, validation, and test data

- training data updates parameters;
- validation data selects hyperparameters and stopping decisions;
- test data is reserved for final assessment.

Repeatedly adapting to test results leaks test information into development and
overstates generalization.

## Implementation walkthrough

`GradientDescent.minimize` receives an initial scalar, a loss function, its
derivative, a learning rate, and a step count. The derivative is injected rather
than hidden inside the optimizer, which keeps the update rule independent of
the objective.

At every iteration the method evaluates loss and gradient at the same current
parameter, records a `DescentObservation`, and only then updates:

```text
observe (step, parameter, loss, gradient)
parameter = parameter - learningRate * gradient
```

Recording before the update matters. If loss were recorded after updating but
gradient before, one row would describe two different points and the learning
curve would be misleading.

For (L(w)=(w-3)^2), (w_0=0), and learning rate `0.25`:

```text
gradient at 0 = 2(0-3) = -6
w1 = 0 - 0.25*(-6) = 1.5
loss falls from 9 to 2.25
```

The negative gradient points uphill at this point, so subtracting it moves
toward `3`. With learning rate `1`, the sequence oscillates between `0` and `6`;
with a rate larger than `1`, magnitude grows. This is why the learning rate is
part of the algorithm, not a cosmetic option.

The code validates step count, learning rate, initial value, and every computed
loss/gradient/update. A non-finite value stops at the first invalid iteration
instead of contaminating the remaining trajectory.

## Reading the tests

The convex quadratic gives a hand-known optimum and monotonic-loss expectation
for the chosen rate. A zero-step test verifies that only the initial observation
is returned; it catches off-by-one loop errors. Invalid hyperparameter tests
ensure negative steps/rates fail before invoking user functions.

## Debugging checklist

1. Print parameter, loss, and gradient from the same observation.
2. Check whether subtracting the gradient moves toward the hand-known optimum.
3. Reduce the problem to a quadratic before debugging a neural network.
4. If loss oscillates, compare learning rate with curvature; do not blame
   randomness in a deterministic objective.
5. Locate the first non-finite update and inspect its gradient magnitude.

## Exercises

1. Compare learning rates `0.01`, `0.5`, `1.0`, and `1.1`.
2. Start on both sides of `3` and explain gradient signs.
3. Minimize $(\theta+2)^2+1$ and predict its optimum.
4. Print history as CSV and plot it.
5. Replace the analytic derivative with numerical differentiation.

## Completion criteria

- Distinguish loss, parameter, gradient, learning rate, and step.
- Explain the minus sign in the update.
- Demonstrate too-small and too-large learning rates.
- Explain why training loss alone cannot prove generalization.
- `GradientDescentSuite` passes.
