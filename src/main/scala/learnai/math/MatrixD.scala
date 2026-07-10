package learnai.math

import java.util.Arrays

/** An immutable row-major matrix of finite `Double` values. */
final class MatrixD private (
    val rows: Int,
    val columns: Int,
    private val values: Array[Double]
):
  val size: Int = values.length

  def apply(row: Int, column: Int): Double =
    values(offset(row, column))

  def row(index: Int): VectorD =
    requireRow(index)
    VectorD.tabulate(columns)(column => values(index * columns + column))

  def column(index: Int): VectorD =
    requireColumn(index)
    VectorD.tabulate(rows)(row => values(row * columns + index))

  def updated(row: Int, column: Int, value: Double): MatrixD =
    Numerics.requireFinite(value, "matrix element")
    val result = values.clone()
    result(offset(row, column)) = value
    MatrixD.unsafeFromOwnedArray(rows, columns, result)

  def map(function: Double => Double): MatrixD =
    val result = new Array[Double](size)
    var index = 0
    while index < size do
      result(index) = function(values(index))
      index += 1
    MatrixD.fromOwnedArray(rows, columns, result)

  def zipMap(other: MatrixD)(function: (Double, Double) => Double): MatrixD =
    requireSameShape(other)
    val result = new Array[Double](size)
    var index = 0
    while index < size do
      result(index) = function(values(index), other.values(index))
      index += 1
    MatrixD.fromOwnedArray(rows, columns, result)

  def +(other: MatrixD): MatrixD = zipMap(other)(_ + _)

  def -(other: MatrixD): MatrixD = zipMap(other)(_ - _)

  def scale(scalar: Double): MatrixD =
    Numerics.requireFinite(scalar, "scale")
    map(_ * scalar)

  def transpose: MatrixD =
    MatrixD.tabulate(columns, rows)((row, column) => apply(column, row))

  /** Matrix-vector product: [rows, columns] x [columns] -> [rows]. */
  def matvec(vector: VectorD): VectorD =
    require(
      columns == vector.size,
      s"matvec shape mismatch: matrix=[$rows,$columns], vector=[${vector.size}]"
    )
    VectorD.tabulate(rows) { rowIndex =>
      dotCell(rowIndex, vector)
    }

  /** Matrix multiplication: [m, k] x [k, n] -> [m, n]. */
  def matmul(other: MatrixD): MatrixD =
    require(
      columns == other.rows,
      s"matmul shape mismatch: left=[$rows,$columns], " +
        s"right=[${other.rows},${other.columns}]"
    )
    MatrixD.tabulate(rows, other.columns) { (row, column) =>
      var sum = 0.0
      var compensation = 0.0
      var inner = 0
      while inner < columns do
        val product = apply(row, inner) * other(inner, column)
        val corrected = product - compensation
        val updated = sum + corrected
        compensation = (updated - sum) - corrected
        sum = updated
        inner += 1
      sum
    }

  def toRows: Vector[Vector[Double]] =
    Vector.tabulate(rows)(row(_).toVector)

  override def equals(other: Any): Boolean =
    other match
      case that: MatrixD =>
        rows == that.rows && columns == that.columns && Arrays.equals(values, that.values)
      case _ => false

  override def hashCode(): Int =
    31 * (31 * rows + columns) + Arrays.hashCode(values)

  override def toString: String =
    toRows.map(_.mkString("[", ", ", "]")).mkString("MatrixD(", ", ", ")")

  private def dotCell(rowIndex: Int, vector: VectorD): Double =
    var sum = 0.0
    var compensation = 0.0
    var columnIndex = 0
    while columnIndex < columns do
      val product = apply(rowIndex, columnIndex) * vector(columnIndex)
      val corrected = product - compensation
      val updated = sum + corrected
      compensation = (updated - sum) - corrected
      sum = updated
      columnIndex += 1
    Numerics.requireFinite(sum, "matrix-vector product")

  private def offset(row: Int, column: Int): Int =
    requireRow(row)
    requireColumn(column)
    row * columns + column

  private def requireRow(row: Int): Unit =
    require(row >= 0 && row < rows, s"row index $row outside [0, $rows)")

  private def requireColumn(column: Int): Unit =
    require(column >= 0 && column < columns, s"column index $column outside [0, $columns)")

  private def requireSameShape(other: MatrixD): Unit =
    require(
      rows == other.rows && columns == other.columns,
      s"matrix shape mismatch: left=[$rows,$columns], " +
        s"right=[${other.rows},${other.columns}]"
    )

object MatrixD:
  def fromRows(rows: Vector[VectorD]): MatrixD =
    if rows.isEmpty then zeros(0, 0)
    else
      val columnCount = rows.head.size
      require(
        rows.forall(_.size == columnCount),
        s"all rows must have the same size: ${rows.map(_.size).mkString(", ")}"
      )
      tabulate(rows.size, columnCount)((row, column) => rows(row)(column))

  def fill(rows: Int, columns: Int)(value: => Double): MatrixD =
    requireValidShape(rows, columns)
    fromOwnedArray(rows, columns, Array.fill(Math.multiplyExact(rows, columns))(value))

  def tabulate(rows: Int, columns: Int)(function: (Int, Int) => Double): MatrixD =
    requireValidShape(rows, columns)
    val values = new Array[Double](Math.multiplyExact(rows, columns))
    var row = 0
    while row < rows do
      var column = 0
      while column < columns do
        values(row * columns + column) = function(row, column)
        column += 1
      row += 1
    fromOwnedArray(rows, columns, values)

  def zeros(rows: Int, columns: Int): MatrixD = fill(rows, columns)(0.0)

  private def requireValidShape(rows: Int, columns: Int): Unit =
    require(rows >= 0, s"row count must be non-negative: $rows")
    require(columns >= 0, s"column count must be non-negative: $columns")
    val _ = Math.multiplyExact(rows, columns)

  private def fromOwnedArray(
      rows: Int,
      columns: Int,
      values: Array[Double]
  ): MatrixD =
    var index = 0
    while index < values.length do
      Numerics.requireFinite(values(index), s"matrix element at flat index $index")
      index += 1
    unsafeFromOwnedArray(rows, columns, values)

  private[math] def unsafeFromOwnedArray(
      rows: Int,
      columns: Int,
      values: Array[Double]
  ): MatrixD =
    new MatrixD(rows, columns, values)
