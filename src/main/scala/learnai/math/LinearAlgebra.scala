package learnai.math

/** Dominant eigenpair approximation and convergence evidence from power iteration. */
final case class EigenEstimate(
    value: Double,
    vector: VectorD,
    iterations: Int,
    residualNorm: Double
)

/** Educational dense linear-algebra algorithms with explicit numerical tolerances. */
object LinearAlgebra:
  /** Numerical rank computed by Gaussian elimination with partial pivoting. */
  def rank(matrix: MatrixD, tolerance: Double = 1e-10): Int =
    require(tolerance >= 0.0 && tolerance.isFinite, s"invalid tolerance: $tolerance")
    val values   = Array.tabulate(matrix.rows, matrix.columns)(matrix.apply)
    var pivotRow = 0
    var column   = 0
    while pivotRow < matrix.rows && column < matrix.columns do
      var bestRow   = pivotRow
      var candidate = pivotRow + 1
      while candidate < matrix.rows do
        if math.abs(values(candidate)(column)) > math.abs(values(bestRow)(column)) then
          bestRow = candidate
        candidate += 1
      if math.abs(values(bestRow)(column)) > tolerance then
        val temporary = values(pivotRow)
        values(pivotRow) = values(bestRow)
        values(bestRow) = temporary
        candidate = pivotRow + 1
        while candidate < matrix.rows do
          val factor = values(candidate)(column) / values(pivotRow)(column)
          var inner  = column
          while inner < matrix.columns do
            values(candidate)(inner) -= factor * values(pivotRow)(inner)
            inner += 1
          candidate += 1
        pivotRow += 1
      column += 1
    pivotRow

  /** Approximates the largest-magnitude eigenpair of a non-empty square matrix. */
  def dominantEigenpair(
      matrix: MatrixD,
      initial: VectorD,
      maximumIterations: Int = 1000,
      tolerance: Double = 1e-10
  ): EigenEstimate =
    require(
      matrix.rows == matrix.columns && matrix.rows > 0,
      "power iteration requires a non-empty square matrix"
    )
    require(initial.size == matrix.columns, "initial eigenvector has the wrong size")
    require(initial.norm > 0.0, "initial eigenvector cannot be zero")
    require(maximumIterations > 0, "maximum iterations must be positive")
    require(tolerance > 0.0 && tolerance.isFinite, "tolerance must be finite and positive")

    var vector     = initial.scale(1.0 / initial.norm)
    var iteration  = 0
    var residual   = Double.PositiveInfinity
    var eigenvalue = 0.0
    while iteration < maximumIterations && residual > tolerance do
      val product   = matrix.matvec(vector)
      val norm      = product.norm
      require(norm > 0.0, "power iteration reached a zero vector")
      vector = product.scale(1.0 / norm)
      val projected = matrix.matvec(vector)
      eigenvalue = vector.dot(projected)
      residual = (projected - vector.scale(eigenvalue)).norm
      iteration += 1
    EigenEstimate(eigenvalue, vector, iteration, residual)

  /** Exact singular values for a real 2x2 matrix, ordered largest first. */
  def singularValues2x2(matrix: MatrixD): VectorD =
    require(matrix.rows == 2 && matrix.columns == 2, "singularValues2x2 requires shape [2,2]")
    val gram         = matrix.transpose.matmul(matrix)
    val trace        = gram(0, 0) + gram(1, 1)
    val determinant  = gram(0, 0) * gram(1, 1) - gram(0, 1) * gram(1, 0)
    val discriminant = math.sqrt(math.max(0.0, trace * trace - 4.0 * determinant))
    val largest      = math.sqrt(math.max(0.0, (trace + discriminant) / 2.0))
    val smallest     = math.sqrt(math.max(0.0, (trace - discriminant) / 2.0))
    VectorD(largest, smallest)

  /** Spectral condition number for a non-singular 2x2 matrix. */
  def conditionNumber2x2(matrix: MatrixD, tolerance: Double = 1e-12): Either[String, Double] =
    val singularValues = singularValues2x2(matrix)
    if singularValues(1) <= tolerance then Left("matrix is singular or numerically rank deficient")
    else Right(singularValues(0) / singularValues(1))

  /** Rank-one outer product `left * right^T`. */
  def outer(left: VectorD, right: VectorD): MatrixD = MatrixD
    .tabulate(left.size, right.size)((row, column) => left(row) * right(column))

/** Shows rank, conditioning, and eigen residuals on hand-checkable matrices. */
def runLinearAlgebraLab(): Unit =
  val matrix = MatrixD.fromRows(Vector(VectorD(3.0, 1.0), VectorD(1.0, 3.0)))
  val eigen  = LinearAlgebra.dominantEigenpair(matrix, VectorD(1.0, 0.0))
  println(s"rank:             ${LinearAlgebra.rank(matrix)}")
  println(f"dominant value:   ${eigen.value}%.9f")
  println(f"residual norm:    ${eigen.residualNorm}%.3e")
  println(s"singular values:  ${LinearAlgebra.singularValues2x2(matrix)}")
  println(s"condition number: ${LinearAlgebra.conditionNumber2x2(matrix)}")
