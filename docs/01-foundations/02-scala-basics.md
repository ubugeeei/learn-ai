# 02 — The Scala 3 needed for LLM implementation

## What you will build

A small command-line program that records loss observations and computes their
mean. It introduces the values, types, functions, collections, branches, and
error representation used throughout the course.

The sources sit together in one directory.

```text
src/main/scala/learnai/foundations/
├── ScalaTour.scala       # implementation grown by this chapter
└── ScalaTourSuite.scala  # executable specification paired with it
```

Later chapters use the same shape. Opening a package directory reveals the
implementation, tests, and runnable lesson together.

## Programs transform values

Treat a program as a transformation from typed input to typed output.

```scala
def square(value: Double): Double = value * value
```

| Part | Meaning |
| --- | --- |
| `def` | define a function |
| `square` | function name |
| `value` | input name |
| first `Double` | input type |
| final `Double` | output type |
| `value * value` | output expression |

`square(3.0)` returns `9.0`. A function that returns the same output for the
same input and does not mutate external state is a **pure function**. Pure
functions are central to numerical code because they are easy to test.

## `val` gives a value one stable name

```scala
val losses = Vector(2.0, 1.0, 0.5)
```

A `val` cannot be reassigned. The compiler infers `Vector[Double]`, though the
type may also be written explicitly:

```scala
val losses: Vector[Double] = Vector(2.0, 1.0, 0.5)
```

A type describes a set of values and the operations permitted on them. It lets
the compiler reject many meaning mismatches before execution.

## Combine related fields

```scala
final case class Observation(label: String, value: Double)
```

`Observation` groups a label and numeric measurement. A case class provides a
constructor, field access, value equality, and readable rendering. `final`
states that this learning type is not designed for subclassing.

## Transform a collection

`Vector` is an ordered immutable collection.

```scala
val losses = observations.map(observation => observation.value)
```

`map` applies `Observation => Double` to every element and returns a new
`Vector[Double]` with the same length.

```text
Vector[Observation] -- map(Observation => Double) --> Vector[Double]
```

The same container transformation idea later applies to batches and tensors.

## Represent an undefined mean

For $n$ values:

$$
\operatorname{mean}(x)=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

An empty collection has $n=0$, so its mean is undefined. Returning only a
`Double` cannot represent that failure. Use `Either`:

```scala
def mean(values: Vector[Double]): Either[String, Double] =
  if values.isEmpty then Left("mean requires at least one value")
  else Right(values.sum / values.size.toDouble)
```

- `Left(problem)` contains a failure reason.
- `Right(value)` contains the result.

The caller handles both cases:

```scala
mean(losses) match
  case Right(average) => println(average)
  case Left(problem)  => println(problem)
```

The agent chapters use the same idea for JSON parsing, model calls, and tool
execution.

## Keep I/O at the boundary

The JVM entrypoint is the `main` method on `learnai.Main`. Each chapter keeps an
ordinary runnable function, and `Main` selects one from the command name:

```scala
def runScalaTour(): Unit =
  println("visible side effect")
```

`println` changes the console, so it is a side effect. `Unit` signals that the
effect, rather than a returned data value, is the purpose. Keep calculations
pure and concentrate I/O at entry and exit boundaries.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.Main foundations'
mean loss: 1.167
the last loss is positive
```

## Implementation walkthrough

Read `ScalaTour.scala` in declaration order. `object ScalaTour` creates one
namespace; its methods are pure and therefore easy to call from both tests and
the shared CLI.

`square` shows the simplest typed function:

```scala
def square(value: Double): Double = value * value
```

The parameter and return annotations form the public contract. The expression
after `=` is the return value—there is no required `return` statement.

`mean` introduces an error channel:

```scala
def mean(values: Vector[Double]): Either[String, Double] =
  if values.isEmpty then Left("mean requires a non-empty collection")
  else Right(values.sum / values.size.toDouble)
```

An empty mean is mathematically undefined. Returning `0.0` would manufacture a
valid-looking number; throwing would make expected invalid input hard to
compose. `Either[String, Double]` forces the caller to handle `Left` or `Right`.
`toDouble` is explicit because `size` is an `Int`.

`describeSign` is a three-branch expression. Exactly one string is produced,
so the compiler can infer the branch result type. `normalizeLabel` demonstrates
method chaining on an immutable `String`: `trim` and `toLowerCase` each return a
new value.

Finally, `Main.main` maps the `foundations` argument to `runScalaTour()`. It calls
the same functions as the tests; the demo does not contain a second
implementation. The entrypoint stays singular while every lab remains an
ordinary reusable function. When printing `Either`, deliberately observe both `Right` and
`Left` so the error channel is not abstract.

## Reading the tests

`ScalaTourSuite` separates cases by behavior. The square test uses a value whose
answer is obvious by hand. The mean tests cover both a normal vector and the
empty boundary. The sign test enumerates all three branches instead of assuming
one positive example covers the conditional. The label test uses both outer
space and mixed case so removing either operation breaks it.

When adding a function, first write one example for every branch and one input
at each invalid boundary. Registering the suite in `AllTests` is a separate
step; an unregistered green file is not being executed.

## Debugging checklist

1. For a type error, read the required and found types before adding casts.
2. For an unhandled `Either`, pattern-match on both `Left` and `Right`.
3. For an indentation error, align sibling expressions and verify where the
   enclosing `object` or method ends.
4. For a main-class error, use the fully qualified name printed by sbt.
5. For an unexpected mutable result, search for `var` or a mutable collection;
   the tour functions need neither.

## Exercises

1. Explain why `square(-3.0)` is positive.
2. Call `mean(Vector.empty)` and print its `Left` message.
3. Add a step number to `Observation` and update its constructors.
4. Implement `minimum` as `Either[String, Double]`.
5. Implement a pure `isImproving` function for two losses.

## Completion criteria

- Point to a value, function, argument, return value, and type in the source.
- Explain how `Vector.map` relates input and output elements.
- Explain why the empty mean returns `Either`.
- Distinguish a pure calculation from I/O.
