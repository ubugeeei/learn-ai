package learnai.learning

import learnai.math.Numerics

/** One immutable point in an optimization trace, recorded before its update. */
final case class DescentObservation(step: Int, parameter: Double, loss: Double, gradient: Double)

/** Reference scalar gradient descent with explicit validation and a complete trace. */
object GradientDescent:
  def minimize(
      loss: Double => Double,
      derivative: Double => Double,
      initialParameter: Double,
      learningRate: Double,
      steps: Int
  ): Vector[DescentObservation] =
    Numerics.requireFinite(initialParameter, "initial parameter")
    Numerics.requireFinite(learningRate, "learning rate")
    require(learningRate > 0.0, s"learning rate must be positive: $learningRate")
    require(steps >= 0, s"steps must be non-negative: $steps")

    val history   = Vector.newBuilder[DescentObservation]
    var parameter = initialParameter
    var step      = 0
    while step <= steps do
      val currentLoss = Numerics.requireFinite(loss(parameter), s"loss at step $step")
      val gradient    = Numerics.requireFinite(derivative(parameter), s"gradient at step $step")
      history += DescentObservation(step, parameter, currentLoss, gradient)
      if step < steps then
        parameter -= learningRate * gradient
        parameter = Numerics.requireFinite(parameter, s"parameter after step $step")
      step += 1
    history.result()

def runGradientDescentLab(): Unit =
  val target     = 3.0
  val loss       = (parameter: Double) => math.pow(parameter - target, 2.0)
  val derivative = (parameter: Double) => 2.0 * (parameter - target)
  val history    = GradientDescent
    .minimize(loss, derivative, initialParameter = -4.0, learningRate = 0.1, steps = 20)

  println("step | parameter | loss       | gradient")
  history.foreach { observation =>
    println(f"${observation.step}%4d | ${observation.parameter}%9.5f | " + f"${observation
        .loss}%10.6f | ${observation.gradient}%9.5f")
  }
