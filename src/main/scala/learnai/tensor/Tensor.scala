package learnai.tensor

import java.util.IdentityHashMap

import scala.collection.mutable.ArrayBuffer

import learnai.math.Numerics

/** A dense row-major tensor node with reverse-mode automatic differentiation.
  *
  * This first implementation deliberately has no broadcasting. Binary
  * element-wise operations require exactly equal shapes.
  */
final class Tensor private (
    val shape: Shape,
    private val currentData: Array[Double],
    val label: String,
    val operation: String,
    private val previous: Vector[Tensor],
    val isTrainable: Boolean
):
  private val currentGradient = Array.fill(shape.size)(0.0)
  private var backwardRule: () => Unit = () => ()

  def rank: Int = shape.rank

  def size: Int = shape.size

  def apply(indices: Int*): Double = currentData(shape.offset(indices))

  def valueAtFlat(index: Int): Double =
    requireFlatIndex(index)
    currentData(index)

  def gradientAt(indices: Int*): Double = currentGradient(shape.offset(indices))

  def gradientAtFlat(index: Int): Double =
    requireFlatIndex(index)
    currentGradient(index)

  def values: Vector[Double] = currentData.toVector

  def gradients: Vector[Double] = currentGradient.toVector

  def +(other: Tensor): Tensor =
    requireSameShape(other, "addition")
    val outputData = Array.tabulate(size)(index => currentData(index) + other.currentData(index))
    val output = Tensor.operation(shape, outputData, "+", Vector(this, other))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        val gradient = output.currentGradient(index)
        accumulateGradient(index, gradient)
        other.accumulateGradient(index, gradient)
        index += 1
    output

  def -(other: Tensor): Tensor = this + other.scale(-1.0)

  def hadamard(other: Tensor): Tensor =
    requireSameShape(other, "Hadamard product")
    val outputData = Array.tabulate(size)(index => currentData(index) * other.currentData(index))
    val output = Tensor.operation(shape, outputData, "hadamard", Vector(this, other))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        val upstream = output.currentGradient(index)
        accumulateGradient(index, other.currentData(index) * upstream)
        other.accumulateGradient(index, currentData(index) * upstream)
        index += 1
    output

  def scale(scalar: Double): Tensor =
    Numerics.requireFinite(scalar, "tensor scale")
    val outputData = currentData.map(_ * scalar)
    val output = Tensor.operation(shape, outputData, s"scale($scalar)", Vector(this))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        accumulateGradient(index, scalar * output.currentGradient(index))
        index += 1
    output

  def pow(exponent: Double): Tensor =
    Numerics.requireFinite(exponent, "tensor power exponent")
    val outputData = currentData.map(math.pow(_, exponent))
    val output = Tensor.operation(shape, outputData, s"pow($exponent)", Vector(this))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        val local = exponent * math.pow(currentData(index), exponent - 1.0)
        accumulateGradient(index, local * output.currentGradient(index))
        index += 1
    output

  def tanh: Tensor =
    unary("tanh", math.tanh, (_, output) => 1.0 - output * output)

  def relu: Tensor =
    unary(
      "relu",
      value => math.max(0.0, value),
      (input, _) => if input > 0.0 then 1.0 else 0.0
    )

  def exp: Tensor = unary("exp", math.exp, (_, output) => output)

  def log: Tensor =
    val invalid = currentData.indexWhere(_ <= 0.0)
    require(invalid < 0, s"log input at flat index $invalid must be positive")
    unary("log", math.log, (input, _) => 1.0 / input)

  def sum: Tensor =
    val output = Tensor.operation(
      Shape.scalar,
      Array(Numerics.compensatedSum(currentData)),
      "sum",
      Vector(this)
    )
    output.backwardRule = () =>
      val upstream = output.currentGradient(0)
      var index = 0
      while index < size do
        accumulateGradient(index, upstream)
        index += 1
    output

  def mean: Tensor =
    require(size > 0, "mean requires a non-empty tensor")
    sum.scale(1.0 / size.toDouble)

  def reshape(newShape: Shape): Tensor =
    require(
      size == newShape.size,
      s"reshape size mismatch: $shape has $size values, $newShape has ${newShape.size}"
    )
    val output = Tensor.operation(newShape, currentData.clone(), "reshape", Vector(this))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        accumulateGradient(index, output.currentGradient(index))
        index += 1
    output

  def transpose2D: Tensor =
    require(rank == 2, s"transpose2D requires rank 2, got shape $shape")
    val rows = shape(0)
    val columns = shape(1)
    val outputData = new Array[Double](size)
    var row = 0
    while row < rows do
      var column = 0
      while column < columns do
        outputData(column * rows + row) = currentData(row * columns + column)
        column += 1
      row += 1
    val output = Tensor.operation(Shape(columns, rows), outputData, "transpose2D", Vector(this))
    output.backwardRule = () =>
      var inputRow = 0
      while inputRow < rows do
        var inputColumn = 0
        while inputColumn < columns do
          val outputIndex = inputColumn * rows + inputRow
          accumulateGradient(inputRow * columns + inputColumn, output.currentGradient(outputIndex))
          inputColumn += 1
        inputRow += 1
    output

  /** Rank-2 matrix multiplication: [m,k] x [k,n] -> [m,n]. */
  def matmul(other: Tensor): Tensor =
    require(rank == 2 && other.rank == 2, s"matmul requires rank 2: $shape x ${other.shape}")
    val rows = shape(0)
    val inner = shape(1)
    val otherInner = other.shape(0)
    val columns = other.shape(1)
    require(
      inner == otherInner,
      s"matmul shape mismatch: $shape x ${other.shape}"
    )

    val outputData = new Array[Double](Math.multiplyExact(rows, columns))
    var row = 0
    while row < rows do
      var column = 0
      while column < columns do
        var sum = 0.0
        var index = 0
        while index < inner do
          sum += currentData(row * inner + index) * other.currentData(index * columns + column)
          index += 1
        outputData(row * columns + column) = sum
        column += 1
      row += 1

    val output = Tensor.operation(Shape(rows, columns), outputData, "matmul", Vector(this, other))
    output.backwardRule = () =>
      var outputRow = 0
      while outputRow < rows do
        var outputColumn = 0
        while outputColumn < columns do
          val upstream = output.currentGradient(outputRow * columns + outputColumn)
          var index = 0
          while index < inner do
            accumulateGradient(
              outputRow * inner + index,
              upstream * other.currentData(index * columns + outputColumn)
            )
            other.accumulateGradient(
              index * columns + outputColumn,
              upstream * currentData(outputRow * inner + index)
            )
            index += 1
          outputColumn += 1
        outputRow += 1
    output

  def backward(): Unit =
    require(shape == Shape.scalar, s"backward requires a scalar output, got $shape")
    val order = topologicalOrder()
    order.foreach(_.clearGradients())
    currentGradient(0) = 1.0
    order.reverseIterator.foreach(_.backwardRule())

  def clearGradients(): Unit =
    java.util.Arrays.fill(currentGradient, 0.0)

  def applyGradient(learningRate: Double): Unit =
    require(isTrainable, "only trainable leaf tensors may be updated")
    Numerics.requireFinite(learningRate, "learning rate")
    require(learningRate >= 0.0, s"learning rate must be non-negative: $learningRate")
    var index = 0
    while index < size do
      currentData(index) = Numerics.requireFinite(
        currentData(index) - learningRate * currentGradient(index),
        s"updated parameter '$label' at flat index $index"
      )
      index += 1

  /** Applies an optimizer-defined update to every trainable element.
    *
    * The callback receives `(flatIndex, data, gradient)` and returns the new
    * data value. This is the only mutation boundary used by tensor optimizers.
    */
  def updateParameter(function: (Int, Double, Double) => Double): Unit =
    require(isTrainable, "only trainable leaf tensors may be updated")
    var index = 0
    while index < size do
      currentData(index) = Numerics.requireFinite(
        function(index, currentData(index), currentGradient(index)),
        s"updated parameter '$label' at flat index $index"
      )
      index += 1

  override def toString: String =
    s"Tensor(shape=$shape, values=${currentData.mkString("[", ", ", "]")}, " +
      s"operation=$operation, label=$label)"

  private def unary(
      name: String,
      forward: Double => Double,
      derivative: (Double, Double) => Double
  ): Tensor =
    val outputData = currentData.map(forward)
    val output = Tensor.operation(shape, outputData, name, Vector(this))
    output.backwardRule = () =>
      var index = 0
      while index < size do
        val local = derivative(currentData(index), output.currentData(index))
        accumulateGradient(index, local * output.currentGradient(index))
        index += 1
    output

  private def accumulateGradient(index: Int, amount: Double): Unit =
    currentGradient(index) = Numerics.requireFinite(
      currentGradient(index) + amount,
      s"gradient for '$label' at flat index $index"
    )

  private def requireSameShape(other: Tensor, operation: String): Unit =
    require(shape == other.shape, s"$operation requires equal shapes: $shape != ${other.shape}")

  private def requireFlatIndex(index: Int): Unit =
    require(index >= 0 && index < size, s"flat index $index outside [0, $size)")

  private def topologicalOrder(): Vector[Tensor] =
    val visited = new IdentityHashMap[Tensor, java.lang.Boolean]()
    val order = ArrayBuffer.empty[Tensor]

    def visit(tensor: Tensor): Unit =
      if !visited.containsKey(tensor) then
        visited.put(tensor, java.lang.Boolean.TRUE)
        tensor.previous.foreach(visit)
        order += tensor

    visit(this)
    order.toVector

object Tensor:
  def constant(shape: Shape, values: IterableOnce[Double], label: String = ""): Tensor =
    create(shape, values.iterator.toArray, label, "constant", Vector.empty, isTrainable = false)

  def scalar(value: Double, label: String = ""): Tensor =
    constant(Shape.scalar, Vector(value), label)

  def parameter(shape: Shape, values: IterableOnce[Double], label: String): Tensor =
    require(label.nonEmpty, "a trainable tensor requires a label")
    create(shape, values.iterator.toArray, label, "parameter", Vector.empty, isTrainable = true)

  def fill(shape: Shape, value: Double): Tensor =
    constant(shape, Array.fill(shape.size)(value))

  def tabulate(shape: Shape)(function: Int => Double): Tensor =
    constant(shape, Array.tabulate(shape.size)(function))

  private def operation(
      shape: Shape,
      values: Array[Double],
      operation: String,
      previous: Vector[Tensor]
  ): Tensor =
    create(shape, values, label = "", operation, previous, isTrainable = false)

  private def create(
      shape: Shape,
      values: Array[Double],
      label: String,
      operation: String,
      previous: Vector[Tensor],
      isTrainable: Boolean
  ): Tensor =
    require(
      values.length == shape.size,
      s"data size ${values.length} does not match shape $shape with size ${shape.size}"
    )
    var index = 0
    while index < values.length do
      Numerics.requireFinite(values(index), s"tensor value at flat index $index")
      index += 1
    new Tensor(shape, values, label, operation, previous, isTrainable)
