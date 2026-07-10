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

  /** Selects rows from a rank-2 table: `[rows, columns] -> [N, columns]`.
    *
    * Repeated indices are allowed. During backward their gradient
    * contributions are accumulated into the same source row. This operation
    * is the basis of embedding lookup.
    */
  def gatherRows(rowIndices: Vector[Int]): Tensor =
    require(rank == 2, s"gatherRows requires rank 2, got shape $shape")
    val sourceRows = shape(0)
    val columns = shape(1)
    rowIndices.zipWithIndex.foreach { case (row, outputRow) =>
      require(
        row >= 0 && row < sourceRows,
        s"row index $row at output row $outputRow outside [0, $sourceRows)"
      )
    }
    val outputData = new Array[Double](Math.multiplyExact(rowIndices.size, columns))
    var outputRow = 0
    while outputRow < rowIndices.size do
      val sourceOffset = rowIndices(outputRow) * columns
      val outputOffset = outputRow * columns
      System.arraycopy(currentData, sourceOffset, outputData, outputOffset, columns)
      outputRow += 1

    val output = Tensor.operation(
      Shape(rowIndices.size, columns),
      outputData,
      "gatherRows",
      Vector(this)
    )
    output.backwardRule = () =>
      var selectedRow = 0
      while selectedRow < rowIndices.size do
        val sourceOffset = rowIndices(selectedRow) * columns
        val outputOffset = selectedRow * columns
        var column = 0
        while column < columns do
          accumulateGradient(
            sourceOffset + column,
            output.currentGradient(outputOffset + column)
          )
          column += 1
        selectedRow += 1
    output

  /** Mean cross entropy over rank-2 logits `[examples, classes]`.
    *
    * The forward pass uses log-sum-exp stabilization. The backward pass uses
    * the exact derivative `(softmax - oneHot) / examples`, avoiding a separate
    * one-hot Tensor and softmax graph.
    */
  def crossEntropy(targetIndices: Vector[Int]): Tensor =
    require(rank == 2, s"crossEntropy requires rank 2 logits, got shape $shape")
    val examples = shape(0)
    val classes = shape(1)
    require(examples > 0, "crossEntropy requires at least one example")
    require(classes > 0, "crossEntropy requires at least one class")
    require(
      targetIndices.size == examples,
      s"target count ${targetIndices.size} does not match example count $examples"
    )
    targetIndices.zipWithIndex.foreach { case (target, example) =>
      require(
        target >= 0 && target < classes,
        s"target $target for example $example outside [0, $classes)"
      )
    }

    val probabilities = new Array[Double](size)
    var totalLoss = 0.0
    var example = 0
    while example < examples do
      val rowOffset = example * classes
      var maximum = currentData(rowOffset)
      var column = 1
      while column < classes do
        maximum = math.max(maximum, currentData(rowOffset + column))
        column += 1

      var exponentialSum = 0.0
      column = 0
      while column < classes do
        val exponential = math.exp(currentData(rowOffset + column) - maximum)
        probabilities(rowOffset + column) = exponential
        exponentialSum += exponential
        column += 1

      column = 0
      while column < classes do
        probabilities(rowOffset + column) /= exponentialSum
        column += 1
      val target = targetIndices(example)
      totalLoss += maximum + math.log(exponentialSum) - currentData(rowOffset + target)
      example += 1

    val output = Tensor.operation(
      Shape.scalar,
      Array(totalLoss / examples.toDouble),
      "crossEntropy",
      Vector(this)
    )
    output.backwardRule = () =>
      val upstreamScale = output.currentGradient(0) / examples.toDouble
      var exampleIndex = 0
      while exampleIndex < examples do
        val rowOffset = exampleIndex * classes
        var classIndex = 0
        while classIndex < classes do
          val targetAdjustment =
            if classIndex == targetIndices(exampleIndex) then 1.0 else 0.0
          accumulateGradient(
            rowOffset + classIndex,
            (probabilities(rowOffset + classIndex) - targetAdjustment) * upstreamScale
          )
          classIndex += 1
        exampleIndex += 1
    output

  /** Mean cross entropy over only the unmasked rows of rank-2 logits.
    *
    * `targetMask(i)` set to false removes example `i` from the loss: it
    * contributes nothing to the forward mean and receives exactly zero
    * gradient. This is how packed datasets exclude padding and
    * cross-document targets from training while keeping fixed-shape
    * batches. The mean is taken over the *unmasked* count, so masked rows
    * do not dilute the loss scale.
    *
    * All target indices are validated against the class range, masked or
    * not, so an indexing bug fails loudly instead of hiding behind a mask.
    * At least one row must remain unmasked; an all-masked example set has
    * no defined mean.
    */
  def crossEntropyMasked(
      targetIndices: Vector[Int],
      targetMask: Vector[Boolean]
  ): Tensor =
    require(rank == 2, s"crossEntropyMasked requires rank 2 logits, got shape $shape")
    val examples = shape(0)
    val classes = shape(1)
    require(
      targetIndices.size == examples,
      s"target count ${targetIndices.size} does not match example count $examples"
    )
    require(
      targetMask.size == examples,
      s"mask count ${targetMask.size} does not match example count $examples"
    )
    targetIndices.zipWithIndex.foreach { case (target, example) =>
      require(
        target >= 0 && target < classes,
        s"target $target for example $example outside [0, $classes)"
      )
    }
    val unmaskedCount = targetMask.count(identity)
    require(unmaskedCount > 0, "crossEntropyMasked requires at least one unmasked target")

    val probabilities = new Array[Double](size)
    var totalLoss = 0.0
    var example = 0
    while example < examples do
      if targetMask(example) then
        val rowOffset = example * classes
        var maximum = currentData(rowOffset)
        var column = 1
        while column < classes do
          maximum = math.max(maximum, currentData(rowOffset + column))
          column += 1

        var exponentialSum = 0.0
        column = 0
        while column < classes do
          val exponential = math.exp(currentData(rowOffset + column) - maximum)
          probabilities(rowOffset + column) = exponential
          exponentialSum += exponential
          column += 1

        column = 0
        while column < classes do
          probabilities(rowOffset + column) /= exponentialSum
          column += 1
        totalLoss += maximum + math.log(exponentialSum) -
          currentData(rowOffset + targetIndices(example))
      example += 1

    val output = Tensor.operation(
      Shape.scalar,
      Array(totalLoss / unmaskedCount.toDouble),
      "crossEntropyMasked",
      Vector(this)
    )
    output.backwardRule = () =>
      val upstreamScale = output.currentGradient(0) / unmaskedCount.toDouble
      var exampleIndex = 0
      while exampleIndex < examples do
        if targetMask(exampleIndex) then
          val rowOffset = exampleIndex * classes
          var classIndex = 0
          while classIndex < classes do
            val targetAdjustment =
              if classIndex == targetIndices(exampleIndex) then 1.0 else 0.0
            accumulateGradient(
              rowOffset + classIndex,
              (probabilities(rowOffset + classIndex) - targetAdjustment) * upstreamScale
            )
            classIndex += 1
        exampleIndex += 1
    output

  /** Returns a defensive copy of one row from a rank-2 Tensor. */
  def rowValues(row: Int): Vector[Double] =
    require(rank == 2, s"rowValues requires rank 2, got shape $shape")
    val rows = shape(0)
    val columns = shape(1)
    require(row >= 0 && row < rows, s"row $row outside [0, $rows)")
    currentData.slice(row * columns, (row + 1) * columns).toVector

  /** Adds a rank-1 vector to every row of a rank-2 Tensor.
    *
    * Forward shape: `[rows, columns] + [columns] -> [rows, columns]`.
    * The row-vector gradient is reduced by summing over all rows.
    */
  def addRowVector(rowVector: Tensor): Tensor =
    require(rank == 2, s"addRowVector requires rank-2 input, got $shape")
    require(rowVector.rank == 1, s"addRowVector requires rank-1 vector, got ${rowVector.shape}")
    val rows = shape(0)
    val columns = shape(1)
    require(
      rowVector.shape(0) == columns,
      s"row vector shape ${rowVector.shape} does not match columns $columns"
    )
    val outputData = Array.tabulate(size) { index =>
      currentData(index) + rowVector.currentData(index % columns)
    }
    val output = Tensor.operation(shape, outputData, "addRowVector", Vector(this, rowVector))
    output.backwardRule = () =>
      var row = 0
      while row < rows do
        var column = 0
        while column < columns do
          val index = row * columns + column
          val upstream = output.currentGradient(index)
          accumulateGradient(index, upstream)
          rowVector.accumulateGradient(column, upstream)
          column += 1
        row += 1
    output

  /** Applies RMS normalization independently to every row and a learned scale.
    *
    * Forward shapes: input `[rows, channels]`, scale `[channels]`, output
    * `[rows, channels]`. No mean subtraction is performed.
    */
  def rmsNormRows(scale: Tensor, epsilon: Double): Tensor =
    require(rank == 2, s"rmsNormRows requires rank-2 input, got $shape")
    require(scale.rank == 1, s"RMS scale must have rank 1, got ${scale.shape}")
    require(epsilon > 0.0 && epsilon.isFinite, s"epsilon must be finite and positive: $epsilon")
    val rows = shape(0)
    val channels = shape(1)
    require(
      scale.shape(0) == channels,
      s"RMS scale shape ${scale.shape} does not match $channels channels"
    )

    val inverseRms = new Array[Double](rows)
    val outputData = new Array[Double](size)
    var row = 0
    while row < rows do
      val rowOffset = row * channels
      var sumSquares = 0.0
      var channel = 0
      while channel < channels do
        val value = currentData(rowOffset + channel)
        sumSquares += value * value
        channel += 1
      inverseRms(row) = 1.0 / math.sqrt(sumSquares / channels.toDouble + epsilon)
      channel = 0
      while channel < channels do
        outputData(rowOffset + channel) =
          currentData(rowOffset + channel) * inverseRms(row) * scale.currentData(channel)
        channel += 1
      row += 1

    val output = Tensor.operation(shape, outputData, "rmsNormRows", Vector(this, scale))
    output.backwardRule = () =>
      var rowIndex = 0
      while rowIndex < rows do
        val rowOffset = rowIndex * channels
        val inverse = inverseRms(rowIndex)
        var inputGradientDotInput = 0.0
        var channelIndex = 0
        while channelIndex < channels do
          val index = rowOffset + channelIndex
          val gradientBeforeNorm =
            output.currentGradient(index) * scale.currentData(channelIndex)
          inputGradientDotInput += gradientBeforeNorm * currentData(index)
          channelIndex += 1

        channelIndex = 0
        while channelIndex < channels do
          val index = rowOffset + channelIndex
          val upstream = output.currentGradient(index)
          val input = currentData(index)
          val gradientBeforeNorm = upstream * scale.currentData(channelIndex)
          val inputGradient =
            gradientBeforeNorm * inverse -
              input * inputGradientDotInput * math.pow(inverse, 3.0) / channels.toDouble
          accumulateGradient(index, inputGradient)
          scale.accumulateGradient(channelIndex, upstream * input * inverse)
          channelIndex += 1
        rowIndex += 1
    output

  /** Applies a numerically stable softmax independently to every rank-2 row. */
  def softmaxRows: Tensor =
    require(rank == 2, s"softmaxRows requires rank 2, got $shape")
    val rows = shape(0)
    val columns = shape(1)
    require(columns > 0, "softmaxRows requires at least one column")
    val outputData = new Array[Double](size)
    var row = 0
    while row < rows do
      val offset = row * columns
      var maximum = currentData(offset)
      var column = 1
      while column < columns do
        maximum = math.max(maximum, currentData(offset + column))
        column += 1
      var total = 0.0
      column = 0
      while column < columns do
        val exponential = math.exp(currentData(offset + column) - maximum)
        outputData(offset + column) = exponential
        total += exponential
        column += 1
      column = 0
      while column < columns do
        outputData(offset + column) /= total
        column += 1
      row += 1

    val output = Tensor.operation(shape, outputData, "softmaxRows", Vector(this))
    output.backwardRule = () =>
      var rowIndex = 0
      while rowIndex < rows do
        val offset = rowIndex * columns
        var weightedGradient = 0.0
        var columnIndex = 0
        while columnIndex < columns do
          weightedGradient +=
            output.currentGradient(offset + columnIndex) * output.currentData(offset + columnIndex)
          columnIndex += 1
        columnIndex = 0
        while columnIndex < columns do
          val index = offset + columnIndex
          val gradient = output.currentData(index) *
            (output.currentGradient(index) - weightedGradient)
          accumulateGradient(index, gradient)
          columnIndex += 1
        rowIndex += 1
    output

  /** Replaces the strict upper triangle of a square rank-2 Tensor.
    *
    * Backward passes gradients only through the diagonal and lower triangle,
    * which prevents causal attention from depending on future positions.
    */
  def causalMask(maskedValue: Double = -1e9): Tensor =
    require(rank == 2, s"causalMask requires rank 2, got $shape")
    require(shape(0) == shape(1), s"causalMask requires a square matrix, got $shape")
    Numerics.requireFinite(maskedValue, "causal mask value")
    val length = shape(0)
    val outputData = currentData.clone()
    var row = 0
    while row < length do
      var column = row + 1
      while column < length do
        outputData(row * length + column) = maskedValue
        column += 1
      row += 1
    val output = Tensor.operation(shape, outputData, "causalMask", Vector(this))
    output.backwardRule = () =>
      var rowIndex = 0
      while rowIndex < length do
        var columnIndex = 0
        while columnIndex <= rowIndex do
          val index = rowIndex * length + columnIndex
          accumulateGradient(index, output.currentGradient(index))
          columnIndex += 1
        rowIndex += 1
    output

  /** Copies a half-open range of columns from a rank-2 Tensor. */
  def sliceColumns(fromInclusive: Int, untilExclusive: Int): Tensor =
    require(rank == 2, s"sliceColumns requires rank 2, got $shape")
    val rows = shape(0)
    val columns = shape(1)
    require(
      fromInclusive >= 0 && fromInclusive < untilExclusive && untilExclusive <= columns,
      s"invalid column range [$fromInclusive,$untilExclusive) for $columns columns"
    )
    val outputColumns = untilExclusive - fromInclusive
    val outputData = new Array[Double](Math.multiplyExact(rows, outputColumns))
    var row = 0
    while row < rows do
      System.arraycopy(
        currentData,
        row * columns + fromInclusive,
        outputData,
        row * outputColumns,
        outputColumns
      )
      row += 1
    val output = Tensor.operation(
      Shape(rows, outputColumns),
      outputData,
      s"sliceColumns($fromInclusive,$untilExclusive)",
      Vector(this)
    )
    output.backwardRule = () =>
      var rowIndex = 0
      while rowIndex < rows do
        var outputColumn = 0
        while outputColumn < outputColumns do
          accumulateGradient(
            rowIndex * columns + fromInclusive + outputColumn,
            output.currentGradient(rowIndex * outputColumns + outputColumn)
          )
          outputColumn += 1
        rowIndex += 1
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
    backwardInternal(accumulateTrainableGradients = false)

  /** Runs reverse mode while preserving gradients already stored in trainable leaves.
    *
    * Intermediate graph-node gradients are always cleared. This operation is
    * intended for gradient accumulation across separately constructed
    * microbatch graphs. The caller owns the accumulation boundary and must
    * clear parameter gradients before the first microbatch and after the
    * optimizer step.
    */
  def backwardAccumulating(): Unit =
    backwardInternal(accumulateTrainableGradients = true)

  private def backwardInternal(accumulateTrainableGradients: Boolean): Unit =
    require(shape == Shape.scalar, s"backward requires a scalar output, got $shape")
    val order = topologicalOrder()
    order.foreach { tensor =>
      if !accumulateTrainableGradients || !tensor.isTrainable then tensor.clearGradients()
    }
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

  /** Replaces all values of a trainable leaf from a defensive input sequence.
    *
    * This operation is reserved for checkpoint loading. Shape/element count
    * and finite-value invariants are checked before any element is changed, so
    * a rejected assignment leaves the parameter untouched.
    */
  def assignParameterValues(values: IndexedSeq[Double]): Unit =
    require(isTrainable, "only trainable leaf tensors may be assigned")
    require(
      values.size == size,
      s"assigned value count ${values.size} does not match shape $shape with size $size"
    )
    values.zipWithIndex.foreach { case (value, index) =>
      Numerics.requireFinite(value, s"assigned parameter '$label' at flat index $index")
    }
    var index = 0
    while index < size do
      currentData(index) = values(index)
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

  /** Concatenates rank-2 Tensors along their column axis.
    *
    * Every input must have the same row count. Backward slices the upstream
    * gradient back into the corresponding input column range.
    */
  def concatenateColumns(tensors: Vector[Tensor]): Tensor =
    require(tensors.nonEmpty, "concatenateColumns requires at least one tensor")
    require(tensors.forall(_.rank == 2), s"all tensors must have rank 2: ${tensors.map(_.shape)}")
    val rows = tensors.head.shape(0)
    require(
      tensors.forall(_.shape(0) == rows),
      s"all tensors must have $rows rows: ${tensors.map(_.shape)}"
    )
    val totalColumns = tensors.iterator.map(_.shape(1)).sum
    val outputData = new Array[Double](Math.multiplyExact(rows, totalColumns))
    var row = 0
    while row < rows do
      var outputColumn = 0
      tensors.foreach { tensor =>
        val columns = tensor.shape(1)
        System.arraycopy(
          tensor.currentData,
          row * columns,
          outputData,
          row * totalColumns + outputColumn,
          columns
        )
        outputColumn += columns
      }
      row += 1

    val output = operation(
      Shape(rows, totalColumns),
      outputData,
      "concatenateColumns",
      tensors
    )
    output.backwardRule = () =>
      var inputOffset = 0
      tensors.foreach { tensor =>
        val columns = tensor.shape(1)
        var inputRow = 0
        while inputRow < rows do
          var column = 0
          while column < columns do
            tensor.accumulateGradient(
              inputRow * columns + column,
              output.currentGradient(inputRow * totalColumns + inputOffset + column)
            )
            column += 1
          inputRow += 1
        inputOffset += columns
      }
    output

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
