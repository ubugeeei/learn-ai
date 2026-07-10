package learnai.nn

import java.util.SplittableRandom

import learnai.autodiff.Value

enum Activation:
  case Linear
  case Tanh
  case Relu

  def apply(value: Value): Value =
    this match
      case Linear => value
      case Tanh   => value.tanh
      case Relu   => value.relu

final class Neuron private[nn] (
    val weights: Vector[Value],
    val bias: Value,
    val activation: Activation
):
  val inputSize: Int = weights.size

  def apply(inputs: Vector[Value]): Value =
    require(
      inputs.size == inputSize,
      s"neuron input size mismatch: expected $inputSize, got ${inputs.size}"
    )
    val weightedInputs = weights.zip(inputs).map { case (weight, input) =>
      weight * input
    }
    activation(Value.sum(weightedInputs) + bias)

  def parameters: Vector[Value] = weights :+ bias

final class Layer private (
    val inputSize: Int,
    val outputSize: Int,
    val neurons: Vector[Neuron]
):
  def apply(inputs: Vector[Value]): Vector[Value] =
    require(
      inputs.size == inputSize,
      s"layer input size mismatch: expected $inputSize, got ${inputs.size}"
    )
    neurons.map(neuron => neuron(inputs))

  def parameters: Vector[Value] = neurons.flatMap(_.parameters)

object Layer:
  def random(
      inputSize: Int,
      outputSize: Int,
      activation: Activation,
      random: SplittableRandom,
      labelPrefix: String
  ): Layer =
    require(inputSize > 0, s"layer input size must be positive: $inputSize")
    require(outputSize > 0, s"layer output size must be positive: $outputSize")
    require(labelPrefix.nonEmpty, "layer label prefix cannot be empty")

    val bound = math.sqrt(6.0 / (inputSize.toDouble + outputSize.toDouble))
    val neurons = Vector.tabulate(outputSize) { outputIndex =>
      val weights = Vector.tabulate(inputSize) { inputIndex =>
        val initial = random.nextDouble(-bound, bound)
        Value.parameter(initial, s"$labelPrefix.neuron$outputIndex.weight$inputIndex")
      }
      val bias = Value.parameter(0.0, s"$labelPrefix.neuron$outputIndex.bias")
      new Neuron(weights, bias, activation)
    }
    new Layer(inputSize, outputSize, neurons)

final class MultiLayerPerceptron private (val layers: Vector[Layer]):
  val inputSize: Int = layers.head.inputSize
  val outputSize: Int = layers.last.outputSize

  def apply(rawInputs: Vector[Double]): Vector[Value] =
    require(
      rawInputs.size == inputSize,
      s"network input size mismatch: expected $inputSize, got ${rawInputs.size}"
    )
    val inputs = rawInputs.map(Value.constant(_))
    layers.foldLeft(inputs) { (current, layer) => layer(current) }

  def parameters: Vector[Value] = layers.flatMap(_.parameters)

object MultiLayerPerceptron:
  def random(
      inputSize: Int,
      layerSizes: Vector[Int],
      hiddenActivation: Activation,
      outputActivation: Activation,
      seed: Long
  ): MultiLayerPerceptron =
    require(inputSize > 0, s"network input size must be positive: $inputSize")
    require(layerSizes.nonEmpty, "a network requires at least one layer")
    require(layerSizes.forall(_ > 0), s"all layer sizes must be positive: $layerSizes")

    val random = new SplittableRandom(seed)
    val dimensions = inputSize +: layerSizes
    val lastLayerIndex = layerSizes.size - 1
    val layers = layerSizes.indices.map { index =>
      val activation =
        if index == lastLayerIndex then outputActivation else hiddenActivation
      Layer.random(
        inputSize = dimensions(index),
        outputSize = dimensions(index + 1),
        activation = activation,
        random = random,
        labelPrefix = s"layer$index"
      )
    }.toVector
    new MultiLayerPerceptron(layers)

object Loss:
  def meanSquaredError(
      predictions: Vector[Value],
      targets: Vector[Double]
  ): Value =
    require(predictions.nonEmpty, "mean squared error requires predictions")
    require(
      predictions.size == targets.size,
      s"prediction and target sizes differ: ${predictions.size} != ${targets.size}"
    )
    val squaredErrors = predictions.zip(targets).map { case (prediction, target) =>
      (prediction - target).pow(2.0)
    }
    Value.sum(squaredErrors) / squaredErrors.size.toDouble

object Sgd:
  def step(parameters: Vector[Value], learningRate: Double): Unit =
    require(parameters.forall(_.isTrainable), "SGD accepts only trainable values")
    parameters.foreach(_.applyGradient(learningRate))
