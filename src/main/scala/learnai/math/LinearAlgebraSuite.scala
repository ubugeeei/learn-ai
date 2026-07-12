package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object LinearAlgebraSuite extends TestSuite:
  override val name: String = "LinearAlgebra"

  override val tests: Vector[TestCase] = specify(
    test("rank distinguishes independent duplicate and zero rows") {
      Assert.equal(LinearAlgebra.rank(MatrixD.fromRows(Vector(VectorD(1, 0), VectorD(0, 1)))), 2)
      Assert.equal(LinearAlgebra.rank(MatrixD.fromRows(Vector(VectorD(1, 2), VectorD(2, 4)))), 1)
      Assert.equal(LinearAlgebra.rank(MatrixD.zeros(3, 2)), 0)
    },
    test("power iteration finds a dominant eigenpair with a small residual") {
      val matrix   = MatrixD.fromRows(Vector(VectorD(3, 1), VectorD(1, 3)))
      val estimate = LinearAlgebra.dominantEigenpair(matrix, VectorD(1, 0))
      Assert.close(estimate.value, 4.0, 1e-8)
      Assert.isTrue(estimate.residualNorm < 1e-8)
      Assert.close(estimate.vector.norm, 1.0)
    },
    test("2x2 singular values match a diagonal reference") {
      val matrix = MatrixD.fromRows(Vector(VectorD(4, 0), VectorD(0, 2)))
      Assert.equal(LinearAlgebra.singularValues2x2(matrix), VectorD(4, 2))
      Assert.close(Assert.right(LinearAlgebra.conditionNumber2x2(matrix)), 2.0)
    },
    test("condition number rejects a rank-deficient matrix") {
      val matrix = MatrixD.fromRows(Vector(VectorD(1, 2), VectorD(2, 4)))
      Assert.isTrue(LinearAlgebra.conditionNumber2x2(matrix).isLeft)
    },
    test("outer product has rank one and expected values") {
      val result = LinearAlgebra.outer(VectorD(1, 2), VectorD(3, 4, 5))
      Assert.equal(result.toRows, Vector(Vector(3, 4, 5), Vector(6, 8, 10)))
      Assert.equal(LinearAlgebra.rank(result), 1)
    }
  )
