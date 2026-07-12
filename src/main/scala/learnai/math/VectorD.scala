package learnai.math

import java.util.Arrays

/**
 * An immutable, one-dimensional collection of finite `Double` values.
 *
 * The backing array is never exposed. Operations validate shape before doing arithmetic and
 * construct a fresh vector, so callers can reason about a `VectorD` as a value.
 */
final class VectorD private (private val values: Array[Double]):
  val size: Int = values.length

  def apply(index: Int): Double = values(index)

  def updated(index: Int, value: Double): VectorD =
    Numerics.requireFinite(value, "vector element")
    val result = values.clone()
    result(index) = value
    VectorD.unsafeFromOwnedArray(result)

  def map(function: Double => Double): VectorD =
    val result = new Array[Double](size)
    var index  = 0
    while index < size do
      result(index) = function(values(index))
      index += 1
    VectorD.fromOwnedArray(result)

  def zipMap(other: VectorD)(function: (Double, Double) => Double): VectorD =
    requireSameSize(other)
    val result = new Array[Double](size)
    var index  = 0
    while index < size do
      result(index) = function(values(index), other.values(index))
      index += 1
    VectorD.fromOwnedArray(result)

  def +(other: VectorD): VectorD = zipMap(other)(_ + _)

  def -(other: VectorD): VectorD = zipMap(other)(_ - _)

  def unary_- : VectorD = map(-_)

  def scale(scalar: Double): VectorD =
    Numerics.requireFinite(scalar, "scale")
    map(_ * scalar)

  def hadamard(other: VectorD): VectorD = zipMap(other)(_ * _)

  def dot(other: VectorD): Double =
    requireSameSize(other)
    val products = values.indices.iterator.map(index => values(index) * other.values(index))
    Numerics.requireFinite(Numerics.compensatedSum(products), "dot product")

  def sum: Double = Numerics.requireFinite(Numerics.compensatedSum(values), "vector sum")

  def squaredNorm: Double = dot(this)

  def norm: Double = math.sqrt(squaredNorm)

  def mean: Either[String, Double] =
    if size == 0 then Left("mean requires a non-empty vector") else Right(sum / size.toDouble)

  def max: Either[String, Double] =
    if size == 0 then Left("max requires a non-empty vector") else Right(values.max)

  def argmax: Either[String, Int] =
    if size == 0 then Left("argmax requires a non-empty vector")
    else
      var bestIndex = 0
      var index     = 1
      while index < size do
        if values(index) > values(bestIndex) then bestIndex = index
        index += 1
      Right(bestIndex)

  def toVector: Vector[Double] = values.toVector

  override def equals(other: Any): Boolean = other match
    case that: VectorD => Arrays.equals(values, that.values)
    case _             => false

  override def hashCode(): Int = Arrays.hashCode(values)

  override def toString: String = values.mkString("VectorD(", ", ", ")")

  private def requireSameSize(other: VectorD): Unit =
    require(size == other.size, s"vector size mismatch: left=$size, right=${other.size}")

object VectorD:
  val empty: VectorD = unsafeFromOwnedArray(Array.emptyDoubleArray)

  def apply(values: Double*): VectorD = fromOwnedArray(values.toArray)

  def from(values: IterableOnce[Double]): VectorD = fromOwnedArray(values.iterator.toArray)

  def fill(size: Int)(value: => Double): VectorD =
    require(size >= 0, s"vector size must be non-negative: $size")
    fromOwnedArray(Array.fill(size)(value))

  def tabulate(size: Int)(function: Int => Double): VectorD =
    require(size >= 0, s"vector size must be non-negative: $size")
    fromOwnedArray(Array.tabulate(size)(function))

  def zeros(size: Int): VectorD = fill(size)(0.0)

  private def fromOwnedArray(values: Array[Double]): VectorD =
    var index = 0
    while index < values.length do
      Numerics.requireFinite(values(index), s"vector element at index $index")
      index += 1
    unsafeFromOwnedArray(values)

  private[math] def unsafeFromOwnedArray(values: Array[Double]): VectorD = new VectorD(values)
