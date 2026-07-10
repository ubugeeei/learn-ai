package learnai.autodiff

import java.util.IdentityHashMap

import scala.collection.mutable.ArrayBuffer

import learnai.math.Numerics

/** A scalar value and one node in a reverse-mode automatic differentiation graph.
  *
  * Forward data is captured when an operation is created. Calling `backward`
  * fills gradients on the reachable graph. Trainable leaf values may be updated
  * only after backward; callers must build a fresh forward graph afterwards.
  */
final class Value private (
    private var currentData: Double,
    val label: String,
    val operation: String,
    private val previous: Vector[Value],
    val isTrainable: Boolean
):
  private var currentGradient = 0.0
  private var backwardRule: () => Unit = () => ()

  def data: Double = currentData

  def gradient: Double = currentGradient

  def +(other: Value): Value =
    val output = Value.operation(data + other.data, "+", Vector(this, other))
    output.backwardRule = () =>
      accumulateGradient(output.gradient)
      other.accumulateGradient(output.gradient)
    output

  def +(other: Double): Value = this + Value.constant(other)

  def -(other: Value): Value = this + (-other)

  def -(other: Double): Value = this - Value.constant(other)

  def unary_- : Value = this * -1.0

  def *(other: Value): Value =
    val output = Value.operation(data * other.data, "*", Vector(this, other))
    output.backwardRule = () =>
      accumulateGradient(other.data * output.gradient)
      other.accumulateGradient(data * output.gradient)
    output

  def *(other: Double): Value = this * Value.constant(other)

  def /(other: Value): Value = this * other.pow(-1.0)

  def /(other: Double): Value = this / Value.constant(other)

  def pow(exponent: Double): Value =
    Numerics.requireFinite(exponent, "power exponent")
    val output = Value.operation(math.pow(data, exponent), s"pow($exponent)", Vector(this))
    output.backwardRule = () =>
      val localDerivative = exponent * math.pow(data, exponent - 1.0)
      accumulateGradient(localDerivative * output.gradient)
    output

  def exp: Value =
    val output = Value.operation(math.exp(data), "exp", Vector(this))
    output.backwardRule = () => accumulateGradient(output.data * output.gradient)
    output

  def log: Value =
    require(data > 0.0, s"log input must be positive, got $data")
    val output = Value.operation(math.log(data), "log", Vector(this))
    output.backwardRule = () => accumulateGradient(output.gradient / data)
    output

  def tanh: Value =
    val output = Value.operation(math.tanh(data), "tanh", Vector(this))
    output.backwardRule = () =>
      accumulateGradient((1.0 - output.data * output.data) * output.gradient)
    output

  def relu: Value =
    val output = Value.operation(math.max(0.0, data), "relu", Vector(this))
    output.backwardRule = () =>
      val localDerivative = if data > 0.0 then 1.0 else 0.0
      accumulateGradient(localDerivative * output.gradient)
    output

  /** Reverse-mode differentiation from this scalar output to all ancestors. */
  def backward(): Unit =
    val order = topologicalOrder()
    order.foreach(_.clearGradient())
    currentGradient = 1.0
    order.reverseIterator.foreach(_.backwardRule())

  def clearGradient(): Unit =
    currentGradient = 0.0

  def applyGradient(learningRate: Double): Unit =
    require(isTrainable, "only trainable leaf values may be updated")
    Numerics.requireFinite(learningRate, "learning rate")
    require(learningRate >= 0.0, s"learning rate must be non-negative: $learningRate")
    currentData = Numerics.requireFinite(
      currentData - learningRate * currentGradient,
      s"updated parameter '$label'"
    )

  override def toString: String =
    s"Value(data=$data, gradient=$gradient, operation=$operation, label=$label)"

  private def accumulateGradient(amount: Double): Unit =
    currentGradient = Numerics.requireFinite(
      currentGradient + amount,
      s"gradient for '$label'"
    )

  private def topologicalOrder(): Vector[Value] =
    val visited = new IdentityHashMap[Value, java.lang.Boolean]()
    val order = ArrayBuffer.empty[Value]

    def visit(value: Value): Unit =
      if !visited.containsKey(value) then
        visited.put(value, java.lang.Boolean.TRUE)
        value.previous.foreach(visit)
        order += value

    visit(this)
    order.toVector

object Value:
  def constant(data: Double, label: String = ""): Value =
    create(data, label, operation = "constant", previous = Vector.empty, isTrainable = false)

  def parameter(data: Double, label: String): Value =
    require(label.nonEmpty, "a trainable parameter requires a label")
    create(data, label, operation = "parameter", previous = Vector.empty, isTrainable = true)

  def sum(values: IterableOnce[Value]): Value =
    values.iterator.foldLeft(constant(0.0))(_ + _)

  private def operation(data: Double, operation: String, previous: Vector[Value]): Value =
    create(data, label = "", operation, previous, isTrainable = false)

  private def create(
      data: Double,
      label: String,
      operation: String,
      previous: Vector[Value],
      isTrainable: Boolean
  ): Value =
    Numerics.requireFinite(data, s"value produced by $operation")
    new Value(data, label, operation, previous, isTrainable)
