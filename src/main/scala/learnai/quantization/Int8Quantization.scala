package learnai.quantization

import learnai.math.MatrixD
import learnai.math.Numerics
import learnai.math.VectorD

/** Error statistics comparing an original and reconstructed numeric array. */
final case class QuantizationError(
    maximumAbsoluteError: Double,
    meanAbsoluteError: Double,
    rootMeanSquaredError: Double
)

/**
 * A symmetric per-row int8 matrix.
 *
 * Each row stores signed integers in `[-127,127]` and one positive `Double` scale. Reconstructed
 * values are `quantized * rowScale`. The unused `-128` code keeps positive and negative ranges
 * symmetric.
 */
final class QuantizedInt8Matrix private (
    val rows: Int,
    val columns: Int,
    private val quantizedValues: Array[Byte],
    val rowScales: Vector[Double]
):
  require(quantizedValues.length == rows * columns, "quantized data size does not match shape")
  require(rowScales.size == rows, "one quantization scale is required per row")
  require(
    rowScales.forall(scale => scale > 0.0 && scale.isFinite),
    "scales must be finite and positive"
  )

  /** Reads one signed int8 code as an `Int`. */
  def quantized(row: Int, column: Int): Int =
    require(row >= 0 && row < rows, s"row $row outside [0,$rows)")
    require(column >= 0 && column < columns, s"column $column outside [0,$columns)")
    quantizedValues(row * columns + column).toInt

  /** Reconstructs one approximate `Double` value. */
  def dequantized(row: Int, column: Int): Double = quantized(row, column).toDouble * rowScales(row)

  /** Materializes the full approximate matrix. */
  def dequantize: MatrixD = MatrixD.tabulate(rows, columns)(dequantized)

  /** Computes a matrix-vector product without materializing Double weights. */
  def matvec(input: VectorD): VectorD =
    require(
      input.size == columns,
      s"quantized matvec shape mismatch: matrix=[$rows,$columns], vector=[${input.size}]"
    )
    VectorD.tabulate(rows) { row =>
      var integerWeightedSum = 0.0
      var column             = 0
      while column < columns do
        integerWeightedSum += quantized(row, column).toDouble * input(column)
        column += 1
      Numerics.requireFinite(integerWeightedSum * rowScales(row), "quantized matvec output")
    }

  /** Payload bytes excluding object/collection headers. */
  def storageBytes: Long = quantizedValues.length.toLong + rowScales.size.toLong * 8L

  def doubleStorageBytes: Long = rows.toLong * columns.toLong * 8L

object QuantizedInt8Matrix:
  /**
   * Quantizes every row with `scale = max(abs(row)) / 127`.
   *
   * All-zero rows use scale `1.0`; every quantized code is still zero, and a positive scale keeps
   * the representation invariant simple.
   */
  def quantize(matrix: MatrixD): QuantizedInt8Matrix =
    val scales = Vector.tabulate(matrix.rows) { row =>
      var maximum = 0.0
      var column  = 0
      while column < matrix.columns do
        maximum = math.max(maximum, math.abs(matrix(row, column)))
        column += 1
      if maximum == 0.0 then 1.0 else maximum / 127.0
    }

    val values = new Array[Byte](matrix.size)
    var row    = 0
    while row < matrix.rows do
      var column = 0
      while column < matrix.columns do
        val rounded = math.round(matrix(row, column) / scales(row)).toInt
        val clamped = math.max(-127, math.min(127, rounded))
        values(row * matrix.columns + column) = clamped.toByte
        column += 1
      row += 1
    new QuantizedInt8Matrix(matrix.rows, matrix.columns, values, scales)

object QuantizationMetrics:
  /** Computes absolute and squared reconstruction errors for equal shapes. */
  def compare(original: MatrixD, reconstructed: MatrixD): QuantizationError =
    require(
      original.rows == reconstructed.rows && original.columns == reconstructed.columns,
      s"quantization metric shape mismatch: [${original.rows},${original.columns}] != " +
        s"[${reconstructed.rows},${reconstructed.columns}]"
    )
    require(original.size > 0, "quantization metrics require a non-empty matrix")
    var maximum     = 0.0
    var absoluteSum = 0.0
    var squareSum   = 0.0
    var row         = 0
    while row < original.rows do
      var column = 0
      while column < original.columns do
        val error = math.abs(original(row, column) - reconstructed(row, column))
        maximum = math.max(maximum, error)
        absoluteSum += error
        squareSum += error * error
        column += 1
      row += 1
    QuantizationError(
      maximum,
      absoluteSum / original.size.toDouble,
      math.sqrt(squareSum / original.size.toDouble)
    )

def runInt8QuantizationLab(): Unit =
  val matrix    = MatrixD
    .tabulate(4, 8)((row, column) => math.sin(row * 8.0 + column) * math.pow(10.0, row.toDouble))
  val quantized = QuantizedInt8Matrix.quantize(matrix)
  val error     = QuantizationMetrics.compare(matrix, quantized.dequantize)
  println(s"Double payload bytes: ${quantized.doubleStorageBytes}")
  println(s"int8 payload bytes:   ${quantized.storageBytes}")
  println(f"maximum abs error:    ${error.maximumAbsoluteError}%.8f")
  println(f"RMSE:                 ${error.rootMeanSquaredError}%.8f")
