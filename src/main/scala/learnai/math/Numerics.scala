package learnai.math

object Numerics:
  val DefaultAbsoluteTolerance: Double = 1e-12
  val DefaultRelativeTolerance: Double = 1e-9

  def naiveSum(values: IterableOnce[Double]): Double = values.iterator.foldLeft(0.0)(_ + _)

  /**
   * Kahan compensated summation keeps a running estimate of bits lost to rounding. It does not make
   * floating-point arithmetic exact, but it often reduces error when magnitudes differ.
   */
  def compensatedSum(values: IterableOnce[Double]): Double =
    val iterator     = values.iterator
    var sum          = 0.0
    var compensation = 0.0
    while iterator.hasNext do
      val corrected = iterator.next() - compensation
      val updated   = sum + corrected
      compensation = (updated - sum) - corrected
      sum = updated
    sum

  def approximatelyEqual(
      left: Double,
      right: Double,
      absoluteTolerance: Double = DefaultAbsoluteTolerance,
      relativeTolerance: Double = DefaultRelativeTolerance
  ): Boolean =
    require(absoluteTolerance >= 0.0, "absolute tolerance must be non-negative")
    require(relativeTolerance >= 0.0, "relative tolerance must be non-negative")

    if left.isNaN || right.isNaN then false
    else if left == right then true
    else if left.isInfinite || right.isInfinite then false
    else
      val difference = math.abs(left - right)
      val scale      = math.max(math.abs(left), math.abs(right))
      difference <= math.max(absoluteTolerance, relativeTolerance * scale)

  def requireFinite(value: Double, label: String): Double =
    require(value.isFinite, s"$label must be finite, got $value")
    value

def runFloatingPointLab(): Unit =
  val decimalSurprise = 0.1 + 0.2
  println(f"0.1 + 0.2            = $decimalSurprise%.17f")
  println(s"exactly equals 0.3   = ${decimalSurprise == 0.3}")
  println(s"approximately equal = ${Numerics.approximatelyEqual(decimalSurprise, 0.3)}")

  val cancellationExample = Vector(1e16, 1.0, -1e16)
  println(s"naive sum            = ${Numerics.naiveSum(cancellationExample)}")
  println(s"compensated sum      = ${Numerics.compensatedSum(cancellationExample)}")
