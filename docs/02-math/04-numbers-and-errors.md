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

## Implementation walkthrough

`Numerics.approximatelyEqual` combines absolute and relative reasoning. An
absolute-only rule fails for large values; a relative-only rule behaves poorly
near zero. The implementation computes a scale from the larger magnitude and
accepts a difference bounded by the larger of the absolute tolerance and the
scaled relative tolerance.

Work one example. Suppose `left = 1_000_000.0`, `right = 1_000_000.01`,
absolute tolerance `1e-9`, and relative tolerance `1e-8`:

```text
difference     = 0.01
relative bound = 1_000_000.01 * 1e-8 = 0.0100000001
chosen bound   = max(1e-9, 0.0100000001)
result         = true
```

For values near zero, the absolute floor prevents division-by-a-tiny-scale
logic from becoming meaningless. `NaN` is rejected before subtraction because
every comparison with `NaN` is false. Equal infinities are handled explicitly;
opposite infinities are not approximately equal.

`compensatedSum` implements Kahan-style error compensation. At each step,
`value - compensation` restores low-order information lost by the previous
rounded addition. The new compensation measures the rounding error introduced
when adding the corrected value. This does not create exact real arithmetic,
but it substantially improves sums mixing very large and very small terms.

`requireFinite` centralizes the invariant that Tensor data, parameters, and
updates cannot contain `NaN` or infinity. Returning the validated value makes
it convenient inside assignments without duplicating a separate check.

## Reading the tests

The classic `0.1 + 0.2` example explains why exact binary equality is the wrong
oracle. A large-magnitude test prevents an absolute-only implementation. `NaN`
and infinity cases prevent invalid comparisons from passing accidentally. The
compensated-sum test orders large and small contributions specifically so naive
summation loses information; the expected total is derived from the chosen
terms, not from the compensated implementation.

## Debugging checklist

1. Locate the first non-finite value, not the final operation reporting it.
2. Print magnitudes before choosing a tolerance.
3. Check whether subtraction overflowed or involved infinity.
4. For unstable sums, compare forward, reverse, sorted, and compensated order.
5. Do not round intermediate ML values for display and feed them back into
   computation.

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
