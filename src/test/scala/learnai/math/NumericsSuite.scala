package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object NumericsSuite extends TestSuite:
  override val name: String = "Numerics"

  override val tests: Vector[TestCase] = Vector(
    test("approximate equality handles rounding error") {
      Assert.isTrue(Numerics.approximatelyEqual(0.1 + 0.2, 0.3))
    },
    test("approximate equality scales for large values") {
      Assert.isTrue(Numerics.approximatelyEqual(1e12 + 1.0, 1e12, relativeTolerance = 1e-11))
    },
    test("approximate equality rejects NaN and different infinities") {
      Assert.isTrue(!Numerics.approximatelyEqual(Double.NaN, Double.NaN))
      Assert.isTrue(!Numerics.approximatelyEqual(Double.PositiveInfinity, Double.NegativeInfinity))
      Assert.isTrue(Numerics.approximatelyEqual(Double.PositiveInfinity, Double.PositiveInfinity))
    },
    test("compensated sum recovers small contributions") {
      val values = Vector(1e16, 1.0, 1.0, -1e16)
      Assert.equal(Numerics.naiveSum(values), 0.0)
      Assert.equal(Numerics.compensatedSum(values), 2.0)
    }
  )
