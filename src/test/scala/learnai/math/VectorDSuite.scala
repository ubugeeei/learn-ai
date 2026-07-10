package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object VectorDSuite extends TestSuite:
  override val name: String = "VectorD"

  override val tests: Vector[TestCase] = Vector(
    test("addition is element-wise and leaves inputs unchanged") {
      val left = VectorD(1.0, 2.0, 3.0)
      val right = VectorD(4.0, 5.0, 6.0)

      Assert.equal(left + right, VectorD(5.0, 7.0, 9.0))
      Assert.equal(left, VectorD(1.0, 2.0, 3.0))
      Assert.equal(right, VectorD(4.0, 5.0, 6.0))
    },
    test("dot multiplies corresponding elements and sums them") {
      Assert.close(VectorD(1.0, 2.0, 3.0).dot(VectorD(4.0, 5.0, 6.0)), 32.0)
    },
    test("norm is the square root of the dot product with itself") {
      Assert.close(VectorD(3.0, 4.0).norm, 5.0)
    },
    test("operations reject mismatched sizes") {
      val error = Assert.throws[IllegalArgumentException] {
        VectorD(1.0, 2.0) + VectorD(3.0)
      }
      Assert.isTrue(error.getMessage.contains("left=2, right=1"))
    },
    test("constructor rejects non-finite values") {
      val nanError = Assert.throws[IllegalArgumentException](VectorD(1.0, Double.NaN))
      val infinityError =
        Assert.throws[IllegalArgumentException](VectorD(Double.PositiveInfinity))
      Assert.isTrue(nanError.getMessage.contains("must be finite"))
      Assert.isTrue(infinityError.getMessage.contains("must be finite"))
    },
    test("updated returns a new vector") {
      val original = VectorD(1.0, 2.0)
      val updated = original.updated(0, 9.0)
      Assert.equal(original, VectorD(1.0, 2.0))
      Assert.equal(updated, VectorD(9.0, 2.0))
    },
    test("empty reductions report undefined results") {
      Assert.equal(VectorD.empty.mean, Left("mean requires a non-empty vector"))
      Assert.equal(VectorD.empty.argmax, Left("argmax requires a non-empty vector"))
    }
  )
