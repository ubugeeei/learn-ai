package learnai.tensor

/**
 * Immutable tensor dimensions with checked row-major coordinate conversion.
 *
 * Zero-sized dimensions are valid. Construction rejects negative dimensions and integer overflow
 * while computing total size and strides.
 */
final class Shape private (
    val dimensions: Vector[Int],
    private val strides: Vector[Int],
    val size: Int
):
  val rank: Int = dimensions.size

  def apply(axis: Int): Int = dimensions(axis)

  def offset(indices: Seq[Int]): Int =
    require(indices.size == rank, s"expected $rank indices for shape $this, got ${indices.size}")
    var result = 0
    var axis   = 0
    while axis < rank do
      val index     = indices(axis)
      val dimension = dimensions(axis)
      require(
        index >= 0 && index < dimension,
        s"index $index outside [0, $dimension) at axis $axis for shape $this"
      )
      result += index * strides(axis)
      axis += 1
    result

  def coordinates(flatIndex: Int): Vector[Int] =
    require(flatIndex >= 0 && flatIndex < size, s"flat index $flatIndex outside [0, $size)")
    val result    = new Array[Int](rank)
    var remainder = flatIndex
    var axis      = 0
    while axis < rank do
      result(axis) = remainder / strides(axis)
      remainder %= strides(axis)
      axis += 1
    result.toVector

  override def equals(other: Any): Boolean = other match
    case that: Shape => dimensions == that.dimensions
    case _           => false

  override def hashCode(): Int = dimensions.hashCode()

  override def toString: String = dimensions.mkString("[", ",", "]")

object Shape:
  val scalar: Shape = Shape()

  def apply(dimensions: Int*): Shape =
    val vector = dimensions.toVector
    vector.zipWithIndex.foreach { case (dimension, axis) =>
      require(dimension >= 0, s"dimension at axis $axis must be non-negative: $dimension")
    }

    val size    = vector.foldLeft(1)(Math.multiplyExact)
    val strides = new Array[Int](vector.size)
    var stride  = 1
    var axis    = vector.size - 1
    while axis >= 0 do
      strides(axis) = stride
      stride = Math.multiplyExact(stride, vector(axis))
      axis -= 1
    new Shape(vector, strides.toVector, size)
