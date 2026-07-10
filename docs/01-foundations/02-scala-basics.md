# 02 — The Scala 3 needed for LLM implementation

## What you will build

A small command-line program that records loss observations and computes their
mean. It introduces the values, types, functions, collections, branches, and
error representation used throughout the course.

Source: `src/main/scala/learnai/foundations/ScalaTour.scala`.

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

For \(n\) values:

\[
\operatorname{mean}(x)=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

An empty collection has \(n=0\), so its mean is undefined. Returning only a
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

An `@main` function is a program entrypoint:

```scala
@main def runScalaTour(): Unit =
  println("visible side effect")
```

`println` changes the console, so it is a side effect. `Unit` signals that the
effect, rather than a returned data value, is the purpose. Keep calculations
pure and concentrate I/O at entry and exit boundaries.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.foundations.runScalaTour'
mean loss: 1.167
the last loss is positive
```

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
