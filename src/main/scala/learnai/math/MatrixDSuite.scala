package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object MatrixDSuite extends TestSuite:
  private val matrix = MatrixD.fromRows(
    Vector(
      VectorD(1.0, 2.0, 3.0),
      VectorD(4.0, 5.0, 6.0)
    )
  )

  override val name: String = "MatrixD"

  override val tests: Vector[TestCase] = specify(
    test("row-major indexing retrieves each element") {
      Assert.close(matrix(0, 0), 1.0)
      Assert.close(matrix(0, 2), 3.0)
      Assert.close(matrix(1, 0), 4.0)
      Assert.close(matrix(1, 2), 6.0)
    },
    test("transpose swaps rows and columns") {
      val expected = MatrixD.fromRows(
        Vector(
          VectorD(1.0, 4.0),
          VectorD(2.0, 5.0),
          VectorD(3.0, 6.0)
        )
      )
      Assert.equal(matrix.transpose, expected)
      Assert.equal(matrix.transpose.transpose, matrix)
    },
    test("matrix-vector product takes a dot product per row") {
      Assert.equal(matrix.matvec(VectorD(1.0, 0.0, -1.0)), VectorD(-2.0, -2.0))
    },
    test("matrix multiplication produces the expected shape and values") {
      val right = MatrixD.fromRows(
        Vector(
          VectorD(7.0, 8.0),
          VectorD(9.0, 10.0),
          VectorD(11.0, 12.0)
        )
      )
      val expected = MatrixD.fromRows(
        Vector(
          VectorD(58.0, 64.0),
          VectorD(139.0, 154.0)
        )
      )
      Assert.equal(matrix.matmul(right), expected)
    },
    test("matrix multiplication rejects incompatible inner dimensions") {
      val error = Assert.throws[IllegalArgumentException] {
        matrix.matmul(MatrixD.zeros(2, 2))
      }
      Assert.isTrue(error.getMessage.contains("left=[2,3], right=[2,2]"))
    },
    test("updated leaves the original matrix unchanged") {
      val updated = matrix.updated(0, 1, 99.0)
      Assert.close(matrix(0, 1), 2.0)
      Assert.close(updated(0, 1), 99.0)
    },
    test("empty dimensions retain their declared shape") {
      val emptyRows = MatrixD.zeros(0, 3)
      val emptyColumns = MatrixD.zeros(2, 0)
      Assert.equal((emptyRows.rows, emptyRows.columns, emptyRows.size), (0, 3, 0))
      Assert.equal((emptyColumns.rows, emptyColumns.columns, emptyColumns.size), (2, 0, 0))
    }
  )
