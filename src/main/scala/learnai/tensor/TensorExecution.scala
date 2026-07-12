package learnai.tensor

/** A validated right-aligned broadcasting map from two input shapes to one output shape. */
final case class BroadcastPlan private (
    left: Shape,
    right: Shape,
    output: Shape,
    private val leftOffsets: Vector[Int],
    private val rightOffsets: Vector[Int]
):
  /** Applies an element-wise operation without materializing expanded input tensors. */
  def map(leftValues: IndexedSeq[Double], rightValues: IndexedSeq[Double])(
      operation: (Double, Double) => Double
  ): Vector[Double] =
    require(leftValues.size == left.size, s"left payload ${leftValues.size} != $left")
    require(rightValues.size == right.size, s"right payload ${rightValues.size} != $right")
    Vector.tabulate(output.size)(index => operation(leftValues(leftOffsets(index)), rightValues(rightOffsets(index))))

  /** Sums an output gradient over every replicated axis of the selected input. */
  def reduceGradient(outputGradient: IndexedSeq[Double], forLeft: Boolean): Vector[Double] =
    require(outputGradient.size == output.size, s"gradient ${outputGradient.size} != $output")
    val offsets = if forLeft then leftOffsets else rightOffsets
    val reduced = Array.fill(if forLeft then left.size else right.size)(0.0)
    outputGradient.indices.foreach(index => reduced(offsets(index)) += outputGradient(index))
    reduced.toVector

object BroadcastPlan:
  /** Constructs NumPy-style right-aligned broadcasting, rejecting incompatible dimensions. */
  def between(left: Shape, right: Shape): Either[String, BroadcastPlan] =
    val rank = math.max(left.rank, right.rank)
    val leftDimensions = Vector.fill(rank - left.rank)(1) ++ left.dimensions
    val rightDimensions = Vector.fill(rank - right.rank)(1) ++ right.dimensions
    val dimensions = leftDimensions.zip(rightDimensions).zipWithIndex.foldLeft[Either[String, Vector[Int]]](Right(Vector.empty)) {
      case (result, ((leftDimension, rightDimension), axis)) =>
        if leftDimension == rightDimension || leftDimension == 1 || rightDimension == 1 then
          result.map(_ :+ math.max(leftDimension, rightDimension))
        else Left(s"broadcast axis $axis is incompatible: $leftDimension != $rightDimension")
    }
    dimensions.map { outputDimensions =>
      val output = Shape(outputDimensions*)
      val leftStrides = rowMajorStrides(leftDimensions)
      val rightStrides = rowMajorStrides(rightDimensions)
      val outputStrides = rowMajorStrides(outputDimensions)
      val offsets = Vector.tabulate(output.size) { flat =>
        var remainder = flat
        var leftOffset = 0
        var rightOffset = 0
        outputDimensions.indices.foreach { axis =>
          val coordinate = if outputDimensions(axis) == 0 then 0 else remainder / outputStrides(axis)
          remainder = if outputDimensions(axis) == 0 then 0 else remainder % outputStrides(axis)
          if leftDimensions(axis) != 1 then leftOffset += coordinate * leftStrides(axis)
          if rightDimensions(axis) != 1 then rightOffset += coordinate * rightStrides(axis)
        }
        leftOffset -> rightOffset
      }
      BroadcastPlan(left, right, output, offsets.map(_._1), offsets.map(_._2))
    }

  private def rowMajorStrides(dimensions: Vector[Int]): Vector[Int] =
    dimensions.indices.map(axis => dimensions.drop(axis + 1).product).toVector

/** Immutable strided view sharing one backing payload; materialization is explicit. */
final case class StridedView private (
    backing: Vector[Double],
    shape: Shape,
    offset: Int,
    strides: Vector[Int]
):
  /** Reads through offset and strides without copying the backing payload. */
  def apply(indices: Int*): Double =
    require(indices.size == shape.rank, s"${indices.size} indices for rank ${shape.rank}")
    val location = indices.zipWithIndex.foldLeft(offset) { case (current, (index, axis)) =>
      require(index >= 0 && index < shape(axis), s"index $index outside axis $axis")
      current + index * strides(axis)
    }
    backing(location)

  /** Swaps two axes in metadata only. */
  def transpose(first: Int, second: Int): StridedView =
    require(first >= 0 && first < shape.rank && second >= 0 && second < shape.rank, "invalid axes")
    StridedView(backing, Shape(shape.dimensions.updated(first, shape(second)).updated(second, shape(first))*), offset, strides.updated(first, strides(second)).updated(second, strides(first)))

  /** Materializes logical row-major order, making the copy boundary observable. */
  def materialize: Vector[Double] =
    Vector.tabulate(shape.size) { flat =>
      var remainder = flat
      val indices = shape.dimensions.indices.map { axis =>
        val stride = shape.dimensions.drop(axis + 1).product
        val index = if stride == 0 then 0 else remainder / stride
        remainder = if stride == 0 then 0 else remainder % stride
        index
      }
      apply(indices*)
    }

object StridedView:
  /** Creates a contiguous row-major view over an existing immutable payload. */
  def contiguous(backing: Vector[Double], shape: Shape): StridedView =
    require(backing.size == shape.size, s"${backing.size} values do not fill $shape")
    val strides = shape.dimensions.indices.map(axis => shape.dimensions.drop(axis + 1).product).toVector
    StridedView(backing, shape, 0, strides)

/** Dense reference batched matrix multiplication for `[batch, rows, inner]` inputs. */
object BatchedMatmul:
  def apply(left: Tensor, right: Tensor): Tensor =
    require(left.rank == 3 && right.rank == 3, "batched matmul requires rank-3 tensors")
    val batch = left.shape(0)
    val rows = left.shape(1)
    val inner = left.shape(2)
    require(right.shape(0) == batch, "batch dimensions differ")
    require(right.shape(1) == inner, "inner dimensions differ")
    val columns = right.shape(2)
    val values = Vector.tabulate(batch * rows * columns) { flat =>
      val batchIndex = flat / (rows * columns)
      val within = flat % (rows * columns)
      val row = within / columns
      val column = within % columns
      (0 until inner).map(index => left(batchIndex, row, index) * right(batchIndex, index, column)).sum
    }
    Tensor.constant(Shape(batch, rows, columns), values, "batchedMatmul")

/** Explicit activation-lifetime accounting for forward/backward graph ownership. */
final class GraphLifetime private (val retainedBytes: Long):
  private var released = false
  def isReleased: Boolean = released
  def bytesInUse: Long = if released then 0L else retainedBytes
  /** Releases saved activations exactly once after backward or inference completion. */
  def release(): Unit =
    require(!released, "graph activations were already released")
    released = true

object GraphLifetime:
  def retain(shapes: Vector[Shape], bytesPerElement: Int): GraphLifetime =
    require(bytesPerElement > 0, s"bytes per element must be positive: $bytesPerElement")
    val bytes = shapes.foldLeft(0L)((total, shape) => Math.addExact(total, Math.multiplyExact(shape.size.toLong, bytesPerElement.toLong)))
    new GraphLifetime(bytes)

/** Runs the tensor execution lab with observable shape, view, batch, and lifetime facts. */
def runTensorExecutionLab(): Unit =
  val plan = BroadcastPlan.between(Shape(2, 1), Shape(1, 3)).toOption.get
  println(s"broadcast: ${plan.left} + ${plan.right} -> ${plan.output}")
  val view = StridedView.contiguous(Vector(1, 2, 3, 4, 5, 6).map(_.toDouble), Shape(2, 3))
  println(s"transpose view: ${view.transpose(0, 1).materialize.mkString(",")}")
  val lifetime = GraphLifetime.retain(Vector(Shape(2, 3), Shape(3, 4)), 4)
  println(s"retained activation bytes: ${lifetime.bytesInUse}")
