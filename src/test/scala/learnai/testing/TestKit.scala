package learnai.testing

import scala.reflect.ClassTag

final case class TestCase(name: String, run: () => Unit)

trait TestSuite:
  def name: String
  def tests: Vector[TestCase]

  protected final def test(testName: String)(body: => Unit): TestCase =
    TestCase(testName, () => body)

object Assert:
  def equal[A](actual: A, expected: A): Unit =
    if actual != expected then
      fail(s"expected <$expected> but got <$actual>")

  def close(actual: Double, expected: Double, tolerance: Double = 1e-9): Unit =
    require(tolerance >= 0.0, s"tolerance must be non-negative: $tolerance")
    val hasInvalidNumber = actual.isNaN || expected.isNaN
    val hasDifferentInfinity =
      (actual.isInfinite || expected.isInfinite) && actual != expected
    val difference = math.abs(actual - expected)
    if hasInvalidNumber || hasDifferentInfinity || difference > tolerance then
      fail(
        s"expected <$expected> +/- <$tolerance> but got <$actual> " +
          s"(difference: $difference)"
      )

  def isTrue(condition: Boolean, clue: => String = "condition was false"): Unit =
    if !condition then fail(clue)

  def left[A, B](result: Either[A, B]): A =
    result match
      case Left(value)  => value
      case Right(value) => fail(s"expected Left but got Right($value)")

  def right[A, B](result: Either[A, B]): B =
    result match
      case Right(value) => value
      case Left(value)  => fail(s"expected Right but got Left($value)")

  def throws[E <: Throwable: ClassTag](body: => Any): E =
    val expectedClass = summon[ClassTag[E]].runtimeClass
    try
      body
      fail(s"expected ${expectedClass.getSimpleName} to be thrown")
    catch
      case error: Throwable if expectedClass.isInstance(error) =>
        error.asInstanceOf[E]
      case error: Throwable =>
        fail(
          s"expected ${expectedClass.getSimpleName} but got " +
            s"${error.getClass.getSimpleName}: ${error.getMessage}"
        )

  private def fail(message: String): Nothing =
    throw new AssertionError(message)
