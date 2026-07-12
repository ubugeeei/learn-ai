package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object CalculusSuite extends TestSuite:
  override val name: String = "Calculus"

  override val tests: Vector[TestCase] = specify(
    test("central difference approximates the derivative of a square") {
      Assert.close(Calculus.derivative(x => x * x, at = 3.0), 6.0, tolerance = 1e-8)
    },
    test("central difference approximates the derivative of sine") {
      Assert.close(Calculus.derivative(math.sin, at = 0.0), 1.0, tolerance = 1e-8)
    },
    test("gradient computes one partial derivative per dimension") {
      val function = (point: VectorD) =>
        val x = point(0)
        val y = point(1)
        x * x + 3.0 * x * y + y * y

      val gradient = Calculus.gradient(function, VectorD(2.0, -1.0))
      Assert.close(gradient(0), 1.0, tolerance = 1e-8)
      Assert.close(gradient(1), 4.0, tolerance = 1e-8)
    },
    test("directional derivative agrees with gradient dot unit direction") {
      val function = (point: VectorD) => point(0) * point(0) + point(1) * point(1)
      val point = VectorD(3.0, 4.0)
      val direction = VectorD(1.0, 0.0)
      val directional = Assert.right(
        Calculus.directionalDerivative(function, point, direction)
      )
      Assert.close(directional, 6.0, tolerance = 1e-8)
    },
    test("finite differences reject invalid steps") {
      val zero = Assert.throws[IllegalArgumentException] {
        Calculus.derivative(identity, 1.0, step = 0.0)
      }
      val nonFinite = Assert.throws[IllegalArgumentException] {
        Calculus.derivative(identity, 1.0, step = Double.NaN)
      }
      Assert.isTrue(zero.getMessage.contains("must be positive"))
      Assert.isTrue(nonFinite.getMessage.contains("must be finite"))
    }
  )
