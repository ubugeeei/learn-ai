# 04 — Numbers, functions, and floating-point error

## What you will build

Experiments that expose `Double` behavior, an approximate comparison, and
compensated summation. Source:
`src/main/scala/learnai/math/Numerics.scala`.

## Mathematical reals and computer numbers differ

The real numbers \(\mathbb{R}\) include integers, rational values,
\(\sqrt{2}\), and \(\pi\), with no gaps on the number line. A computer has
finite storage. Scala `Double` uses the 64-bit IEEE 754 binary64 format, which
encodes a sign, exponent, and finite significand.

Conceptually:

\[
(-1)^{\text{sign}}\times1.\text{fraction}\times2^{\text{exponent}}
\]

Values without a finite binary expansion are rounded. Decimal `0.1` is one of
them.

```console
$ nix develop -c sbt 'runMain learnai.math.runFloatingPointLab'
0.1 + 0.2            = 0.30000000000000004
exactly equals 0.3   = false
approximately equal = true
```

This is a consequence of finite representation, not a runtime defect. Neural
networks perform enormous numbers of additions and multiplications, so
rounding, overflow, and underflow are architectural concerns.

## Absolute and relative error

For approximation \(a\) and reference \(b\):

\[
E_{abs}=|a-b|
\]

An error of \(10^{-6}\) is huge near \(10^{-8}\) but tiny near \(10^{12}\).
Relative error accounts for scale:

\[
E_{rel}=\frac{|a-b|}{\max(|a|,|b|)}
\]

`approximatelyEqual` accepts the larger of an absolute and scale-relative
tolerance:

```scala
val difference = math.abs(left - right)
val scale = math.max(math.abs(left), math.abs(right))
difference <= math.max(absoluteTolerance, relativeTolerance * scale)
```

Absolute tolerance dominates near zero; relative tolerance dominates at large
magnitudes.

## Special values

`Double` also represents:

- positive and negative infinity;
- `NaN`, produced by undefined numeric operations such as `0.0 / 0.0`.

`NaN` is not equal to itself. Once introduced during training, it usually
propagates through later operations. Validate finiteness at value-producing
boundaries so failure occurs near its cause.

## Addition order changes a finite-precision result

Real addition is associative:

\[
(a+b)+c=a+(b+c)
\]

Finite arithmetic may lose a small addend beside a much larger number. Parallel
reductions with different operation orders therefore may not be bit-identical.

Kahan compensated summation tracks an estimate of lost low-order bits:

```scala
val corrected = next - compensation
val updated = sum + corrected
compensation = (updated - sum) - corrected
sum = updated
```

It does not make arithmetic exact, but often reduces error for mixed
magnitudes. `VectorD.sum` uses this implementation.

## Numerical stability

Mathematically equivalent equations can have very different finite behavior.
For example, direct `exp(logit)` may overflow in softmax. Subtracting the
largest logit changes no probability but keeps every exponential at most one.

For every numerical expression ask:

1. What input range is possible?
2. Can intermediate values overflow or underflow?
3. Does subtraction cancel significant digits?
4. How does error grow with operation count?
5. Is there a more stable equivalent form?

## Exercises

1. Predict and run `1e16 + 1.0 == 1e16`.
2. Reorder `Vector(1e16, 1.0, 1.0, -1e16)` and compare both sums.
3. Observe `Double.MaxValue * 2.0` and `Double.MinPositiveValue / 2.0`.
4. Construct a defect hidden by an excessively large relative tolerance.

## Completion criteria

- Explain the difference between a real number and `Double`.
- Distinguish values suitable for exact and approximate comparison.
- Explain why `NaN` is dangerous during training.
- `NumericsSuite` passes.
