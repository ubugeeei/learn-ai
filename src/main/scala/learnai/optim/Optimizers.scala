package learnai.optim

import java.util.IdentityHashMap
import java.util.SplittableRandom

import learnai.math.Numerics
import learnai.tensor.Shape
import learnai.tensor.Tensor

final case class OptimizerStats(
    step: Long,
    gradientNorm: Double,
    gradientScale: Double,
    learningRate: Double
)

/** Complete AdamW state for one ordered parameter list.
  *
  * Moments are stored by parameter *position*, so a snapshot is only valid
  * for the same parameter ordering it was captured from. Every value is
  * validated on restore: first moments must be finite and second moments
  * finite and non-negative, because a corrupted checkpoint must fail loudly
  * rather than silently poison later updates.
  */
final case class AdamWSnapshot(
    step: Long,
    firstMoments: Vector[Vector[Double]],
    secondMoments: Vector[Vector[Double]]
):
  require(step >= 0L, s"optimizer step must be non-negative: $step")
  require(
    firstMoments.size == secondMoments.size,
    s"moment list sizes differ: ${firstMoments.size} != ${secondMoments.size}"
  )

object AdamWSnapshot:
  /** The state of an optimizer that has not performed any update. */
  def zero(parameters: Vector[Tensor]): AdamWSnapshot =
    AdamWSnapshot(
      step = 0L,
      firstMoments = parameters.map(parameter => Vector.fill(parameter.size)(0.0)),
      secondMoments = parameters.map(parameter => Vector.fill(parameter.size)(0.0))
    )

object GradientNorm:
  def global(parameters: Vector[Tensor]): Double =
    require(parameters.forall(_.isTrainable), "gradient norm accepts only trainable tensors")
    val squared = parameters.iterator.flatMap(_.gradients.iterator).map { gradient =>
      gradient * gradient
    }
    Numerics.requireFinite(math.sqrt(Numerics.compensatedSum(squared)), "global gradient norm")

  def clippingScale(globalNorm: Double, maximumNorm: Option[Double]): Double =
    maximumNorm match
      case None => 1.0
      case Some(maximum) =>
        Numerics.requireFinite(maximum, "maximum gradient norm")
        require(maximum > 0.0, s"maximum gradient norm must be positive: $maximum")
        if globalNorm > maximum then maximum / globalNorm else 1.0

final class TensorSgd(
    val learningRate: Double,
    val weightDecay: Double = 0.0,
    val maximumGradientNorm: Option[Double] = None
):
  validateHyperparameters()
  private var currentStep = 0L

  def step(parameters: Vector[Tensor]): OptimizerStats =
    stepAtLearningRate(parameters, learningRate)

  /** Applies one update using a schedule-provided learning rate. */
  def stepAtLearningRate(
      parameters: Vector[Tensor],
      effectiveLearningRate: Double
  ): OptimizerStats =
    require(parameters.nonEmpty, "SGD requires at least one parameter tensor")
    require(parameters.forall(_.isTrainable), "SGD accepts only trainable tensors")
    Numerics.requireFinite(effectiveLearningRate, "effective learning rate")
    require(
      effectiveLearningRate >= 0.0,
      s"effective learning rate must be non-negative: $effectiveLearningRate"
    )
    val norm = GradientNorm.global(parameters)
    val scale = GradientNorm.clippingScale(norm, maximumGradientNorm)

    parameters.foreach { parameter =>
      parameter.updateParameter { (_, data, gradient) =>
        val decayed = data * (1.0 - effectiveLearningRate * weightDecay)
        decayed - effectiveLearningRate * scale * gradient
      }
    }
    currentStep += 1
    OptimizerStats(currentStep, norm, scale, effectiveLearningRate)

  private def validateHyperparameters(): Unit =
    Numerics.requireFinite(learningRate, "learning rate")
    Numerics.requireFinite(weightDecay, "weight decay")
    require(learningRate >= 0.0, s"learning rate must be non-negative: $learningRate")
    require(weightDecay >= 0.0, s"weight decay must be non-negative: $weightDecay")
    maximumGradientNorm.foreach { maximum =>
      Numerics.requireFinite(maximum, "maximum gradient norm")
      require(maximum > 0.0, s"maximum gradient norm must be positive: $maximum")
    }

final class AdamW(
    val learningRate: Double,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-8,
    val weightDecay: Double = 0.01,
    val maximumGradientNorm: Option[Double] = None
):
  validateHyperparameters()

  private final case class State(firstMoment: Array[Double], secondMoment: Array[Double])

  private val states = new IdentityHashMap[Tensor, State]()
  private var currentStep = 0L

  def step(parameters: Vector[Tensor]): OptimizerStats =
    stepAtLearningRate(parameters, learningRate)

  /** Applies one AdamW update using a schedule-provided learning rate. */
  def stepAtLearningRate(
      parameters: Vector[Tensor],
      effectiveLearningRate: Double
  ): OptimizerStats =
    require(parameters.nonEmpty, "AdamW requires at least one parameter tensor")
    require(parameters.forall(_.isTrainable), "AdamW accepts only trainable tensors")
    Numerics.requireFinite(effectiveLearningRate, "effective learning rate")
    require(
      effectiveLearningRate >= 0.0,
      s"effective learning rate must be non-negative: $effectiveLearningRate"
    )
    currentStep += 1
    val norm = GradientNorm.global(parameters)
    val scale = GradientNorm.clippingScale(norm, maximumGradientNorm)
    val firstBiasCorrection = 1.0 - math.pow(beta1, currentStep.toDouble)
    val secondBiasCorrection = 1.0 - math.pow(beta2, currentStep.toDouble)

    parameters.foreach { parameter =>
      val state = states.computeIfAbsent(
        parameter,
        ignored => State(Array.fill(parameter.size)(0.0), Array.fill(parameter.size)(0.0))
      )
      parameter.updateParameter { (index, data, rawGradient) =>
        val gradient = rawGradient * scale
        state.firstMoment(index) =
          beta1 * state.firstMoment(index) + (1.0 - beta1) * gradient
        state.secondMoment(index) =
          beta2 * state.secondMoment(index) + (1.0 - beta2) * gradient * gradient

        val correctedFirst = state.firstMoment(index) / firstBiasCorrection
        val correctedSecond = state.secondMoment(index) / secondBiasCorrection
        val adaptiveUpdate = correctedFirst / (math.sqrt(correctedSecond) + epsilon)
        val decayed = data * (1.0 - effectiveLearningRate * weightDecay)
        decayed - effectiveLearningRate * adaptiveUpdate
      }
    }
    OptimizerStats(currentStep, norm, scale, effectiveLearningRate)

  /** Captures step count and both moment arrays in parameter order.
    *
    * A parameter that has never been stepped yields zero moments, matching
    * the state `stepAtLearningRate` would lazily create for it.
    */
  def snapshot(parameters: Vector[Tensor]): AdamWSnapshot =
    require(parameters.nonEmpty, "snapshot requires at least one parameter tensor")
    require(parameters.forall(_.isTrainable), "snapshot accepts only trainable tensors")
    val moments = parameters.map { parameter =>
      Option(states.get(parameter)) match
        case Some(state) => (state.firstMoment.toVector, state.secondMoment.toVector)
        case None =>
          (Vector.fill(parameter.size)(0.0), Vector.fill(parameter.size)(0.0))
    }
    AdamWSnapshot(currentStep, moments.map(_._1), moments.map(_._2))

  /** Replaces all optimizer state from a snapshot captured for the same
    * parameter ordering.
    *
    * Every moment vector is validated for size, finiteness, and (for second
    * moments) non-negativity before any state is changed, so a rejected
    * snapshot leaves the optimizer untouched.
    */
  def restore(parameters: Vector[Tensor], snapshot: AdamWSnapshot): Unit =
    require(parameters.nonEmpty, "restore requires at least one parameter tensor")
    require(parameters.forall(_.isTrainable), "restore accepts only trainable tensors")
    require(
      snapshot.firstMoments.size == parameters.size,
      s"snapshot covers ${snapshot.firstMoments.size} tensors, expected ${parameters.size}"
    )
    parameters.zipWithIndex.foreach { case (parameter, index) =>
      val first = snapshot.firstMoments(index)
      val second = snapshot.secondMoments(index)
      require(
        first.size == parameter.size && second.size == parameter.size,
        s"snapshot moments for tensor $index have sizes ${first.size}/${second.size}, " +
          s"expected ${parameter.size}"
      )
      first.foreach(value => Numerics.requireFinite(value, s"first moment of tensor $index"))
      second.foreach { value =>
        Numerics.requireFinite(value, s"second moment of tensor $index")
        require(value >= 0.0, s"second moment of tensor $index must be non-negative: $value")
      }
    }
    states.clear()
    currentStep = snapshot.step
    parameters.zipWithIndex.foreach { case (parameter, index) =>
      val _ = states.put(
        parameter,
        State(snapshot.firstMoments(index).toArray, snapshot.secondMoments(index).toArray)
      )
    }

  private def validateHyperparameters(): Unit =
    Vector(
      "learning rate" -> learningRate,
      "beta1" -> beta1,
      "beta2" -> beta2,
      "epsilon" -> epsilon,
      "weight decay" -> weightDecay
    ).foreach { case (name, value) => Numerics.requireFinite(value, name) }
    require(learningRate >= 0.0, s"learning rate must be non-negative: $learningRate")
    require(beta1 >= 0.0 && beta1 < 1.0, s"beta1 must be in [0,1): $beta1")
    require(beta2 >= 0.0 && beta2 < 1.0, s"beta2 must be in [0,1): $beta2")
    require(epsilon > 0.0, s"epsilon must be positive: $epsilon")
    require(weightDecay >= 0.0, s"weight decay must be non-negative: $weightDecay")
    maximumGradientNorm.foreach { maximum =>
      Numerics.requireFinite(maximum, "maximum gradient norm")
      require(maximum > 0.0, s"maximum gradient norm must be positive: $maximum")
    }

object Initialization:
  def xavierUniform(
      shape: Shape,
      fanIn: Int,
      fanOut: Int,
      random: SplittableRandom,
      label: String
  ): Tensor =
    require(fanIn > 0, s"fan-in must be positive: $fanIn")
    require(fanOut > 0, s"fan-out must be positive: $fanOut")
    val bound = math.sqrt(6.0 / (fanIn.toDouble + fanOut.toDouble))
    uniform(shape, -bound, bound, random, label)

  def heUniform(
      shape: Shape,
      fanIn: Int,
      random: SplittableRandom,
      label: String
  ): Tensor =
    require(fanIn > 0, s"fan-in must be positive: $fanIn")
    val bound = math.sqrt(6.0 / fanIn.toDouble)
    uniform(shape, -bound, bound, random, label)

  def zeros(shape: Shape, label: String): Tensor =
    Tensor.parameter(shape, Array.fill(shape.size)(0.0), label)

  private def uniform(
      shape: Shape,
      minimum: Double,
      maximum: Double,
      random: SplittableRandom,
      label: String
  ): Tensor =
    Tensor.parameter(
      shape,
      Array.fill(shape.size)(random.nextDouble(minimum, maximum)),
      label
    )
