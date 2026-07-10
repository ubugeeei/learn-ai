# 03 — Dependency-free tests and debugging

## What you will build

A minimal test runner that executes functions, compares values, reports
failures, and makes the build fail. Source lives under
`src/test/scala/learnai/testing`.

Production Scala projects normally use a test library. We build the first
runner ourselves to show that a test is an ordinary program with input,
execution, comparison, and reporting.

## Arrange, Act, Assert

```scala
test("mean averages non-empty values") {
  // Arrange
  val values = Vector(2.0, 1.0, 0.5)

  // Act
  val result = ScalaTour.mean(values)

  // Assert
  Assert.close(Assert.right(result), 7.0 / 6.0)
}
```

Each test checks one behavior. Its name should identify the broken property
without requiring the reader to inspect the body.

## Store a test body as a function

```scala
final case class TestCase(name: String, run: () => Unit)
```

`() => Unit` is a no-argument function that may perform an effect. The helper's
`body: => Unit` is a by-name argument, so constructing a `TestCase` does not run
the test. The runner decides when to invoke it.

The runner can therefore:

1. select a test;
2. call `run()`;
3. record pass on normal return;
4. record failure on an exception;
5. summarize all results.

## Do not compare floating-point results blindly

Many real values have no finite binary representation.

```scala
0.1 + 0.2 == 0.3 // may be false
```

Compare an absolute difference with a justified tolerance:

\[
|\text{actual}-\text{expected}|\leq\text{tolerance}
\]

```scala
Assert.close(actual, expected, tolerance = 1e-9)
```

A tolerance that is too large hides defects. Choose it from the value scale,
operation count, and expected error source. Later chapters combine absolute and
relative tolerances.

## Test failure paths

```scala
test("mean rejects an empty collection") {
  val result = ScalaTour.mean(Vector.empty)
  Assert.equal(Assert.left(result), "mean requires at least one value")
}
```

Numerical chapters intentionally test shape mismatches, invalid indices, empty
tensors, and non-finite values. Agent chapters test malformed tool arguments,
timeouts, exhausted budgets, and corrupt model responses.

## Explicit suite registration

`AllTests` contains the complete suite list:

```scala
private val suites: Vector[TestSuite] = Vector(
  ScalaTourSuite
)
```

There is no reflection-based discovery. New suites must be registered
explicitly, making the execution set visible in one file.

## Run tests

```console
$ nix develop -c sbt test
[pass] ScalaTour / square multiplies a value by itself
...
5 passed, 0 failed, 5 total
```

For a clean compile and all tests:

```console
$ nix develop -c sbt check
```

## Observe an intentional failure

Temporarily change the expected square from `9.0` to `8.0`, run the suite, and
find:

- the suite and test name;
- expected and actual values;
- the non-zero sbt result.

Restore `9.0` afterward. Seeing a test fail clarifies what its passing state
guarantees.

## Exercises

1. Add one test using `Assert.isTrue`.
2. Test `normalizeLabel` with an empty string.
3. Remove a suite from the registry and observe the execution count.
4. Explain why `Assert.close(Double.NaN, Double.NaN)` fails.

## Completion criteria

- Identify Arrange, Act, and Assert in an existing test.
- Explain why floating-point comparisons need tolerance.
- Add one normal and one failure-path test.
- `sbt check` succeeds.
