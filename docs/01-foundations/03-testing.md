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

## Implementation walkthrough

`TestCase` stores a name and a zero-argument function:

```scala
final case class TestCase(name: String, run: () => Unit)
```

The `test` helper accepts its body by name. That means the assertion body is not
executed while the suite object is initialized; it is wrapped in `() => body`
and runs later under the runner's error handling. Eager execution here would
throw before `AllTests` could report the suite and case name.

`Assert.equal` is generic because equality is useful for numbers, vectors,
enums, and domain values. `Assert.close` is deliberately specific to `Double`.
It checks the tolerance contract, rejects `NaN`, distinguishes unequal
infinities, and prints the observed difference. This is stricter than
`math.abs(a-b) <= tolerance`, which accidentally treats some invalid values as
successful comparisons.

`Assert.throws[E]` obtains the runtime class from `ClassTag`. Its three paths
are important:

1. no exception: fail because the invalid input was accepted;
2. expected exception type: return it so the test can inspect its message;
3. different exception type: fail and preserve diagnostic type/message.

`AllTests` takes the Cartesian traversal of suites and cases, calls `runOne`,
and counts Boolean results. A failed case is printed immediately, but remaining
cases still run. Only after the complete report does the runner throw, giving
sbt a non-zero process status.

## Reading and writing a regression test

Start from a bug statement, not a method name:

> A reused computation-graph node loses one gradient contribution.

Then create the smallest input where correct and buggy implementations differ,
calculate the answer independently, and name the behavior:

```scala
test("a reused node accumulates gradient from every path") { ... }
```

A useful test fails for one specific reason. If it initializes a large model,
trains randomly, serializes it, and checks one Boolean, the failure gives little
localization. Keep a focused unit test and add integration separately.

## Debugging checklist

1. Confirm the suite appears in `AllTests` output.
2. Run the smallest failing case mentally with hand-computable values.
3. If `Assert.close` fails, print scale, expected error source, and tolerance;
   do not widen it blindly.
4. If `Assert.throws` fails with a different exception, move validation closer
   to the public boundary.
5. If tests affect one another, search for shared mutation or reused RNG state.
6. If a timeout test hangs, ensure worker threads are interrupted and executors
   are shut down in `finally`.

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
