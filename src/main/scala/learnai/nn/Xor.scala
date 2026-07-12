package learnai.nn

final case class LabeledPoint(inputs: Vector[Double], target: Double)

final case class XorTrainingResult(model: MultiLayerPerceptron, lossHistory: Vector[Double])

object Xor:
  val dataset: Vector[LabeledPoint] = Vector(
    LabeledPoint(Vector(-1.0, -1.0), -1.0),
    LabeledPoint(Vector(-1.0, 1.0), 1.0),
    LabeledPoint(Vector(1.0, -1.0), 1.0),
    LabeledPoint(Vector(1.0, 1.0), -1.0)
  )

  def train(seed: Long = 42L, steps: Int = 500, learningRate: Double = 0.05): XorTrainingResult =
    require(steps >= 0, s"training steps must be non-negative: $steps")
    val model = MultiLayerPerceptron.random(
      inputSize = 2,
      layerSizes = Vector(4, 1),
      hiddenActivation = Activation.Tanh,
      outputActivation = Activation.Tanh,
      seed = seed
    )

    val history = Vector.newBuilder[Double]
    var step    = 0
    while step < steps do
      val predictions = dataset.map(example => model(example.inputs).head)
      val targets     = dataset.map(_.target)
      val loss        = Loss.meanSquaredError(predictions, targets)
      history += loss.data
      loss.backward()
      Sgd.step(model.parameters, learningRate)
      step += 1

    XorTrainingResult(model, history.result())

  def predict(model: MultiLayerPerceptron): Vector[(LabeledPoint, Double)] = dataset
    .map(example => example -> model(example.inputs).head.data)

def trainXor(): Unit =
  val result      = Xor.train()
  val initialLoss = result.lossHistory.headOption.getOrElse(Double.NaN)
  val finalLoss   = result.lossHistory.lastOption.getOrElse(Double.NaN)
  println(f"initial loss: $initialLoss%.6f")
  println(f"final loss:   $finalLoss%.6f")
  Xor.predict(result.model).foreach { case (example, prediction) =>
    println(f"input=${example.inputs.mkString("[", ", ", "]")} " + f"target=${example
        .target}%4.1f prediction=$prediction% .6f")
  }
