package learnai.diagnostics

import learnai.tensor.Shape
import learnai.tensor.Tensor

/**
 * Numerical-gradient configuration for deterministic scalar losses.
 *
 * A probe passes when `absoluteError <= absoluteTolerance + relativeTolerance * gradientScale`,
 * where `gradientScale` is the larger analytical/numerical magnitude. The absolute term protects
 * gradients near zero; the relative term scales with larger gradients.
 */
final case class GradientCheckConfig(
    epsilon: Double = 1e-5,
    absoluteTolerance: Double = 1e-6,
    relativeTolerance: Double = 1e-4,
    maximumCoordinatesPerParameter: Int = 4
):
  require(epsilon > 0.0 && epsilon.isFinite, s"epsilon must be finite and positive: $epsilon")
  require(
    absoluteTolerance >= 0.0 && absoluteTolerance.isFinite,
    s"absolute tolerance must be finite and non-negative: $absoluteTolerance"
  )
  require(
    relativeTolerance >= 0.0 && relativeTolerance.isFinite,
    s"relative tolerance must be finite and non-negative: $relativeTolerance"
  )
  require(
    maximumCoordinatesPerParameter > 0,
    s"maximum coordinates per parameter must be positive: $maximumCoordinatesPerParameter"
  )

/** One analytical-versus-central-difference comparison. */
final case class GradientProbe(
    parameterLabel: String,
    shape: Shape,
    flatIndex: Int,
    coordinates: Vector[Int],
    analytical: Double,
    numerical: Double,
    absoluteError: Double,
    relativeError: Double,
    allowedError: Double
):
  val passed: Boolean    = absoluteError <= allowedError
  val errorRatio: Double =
    if allowedError == 0.0 then if absoluteError == 0.0 then 0.0 else Double.PositiveInfinity
    else absoluteError / allowedError

/** Complete gradient-check result retaining every inspected coordinate. */
final case class GradientCheckReport(loss: Double, probes: Vector[GradientProbe]):
  require(probes.nonEmpty, "gradient-check report requires at least one probe")
  val passed: Boolean                     = probes.forall(_.passed)
  val failedProbes: Vector[GradientProbe] = probes.filterNot(_.passed)
  val worstProbe: GradientProbe           = probes.maxBy(_.errorRatio)
  val maximumAbsoluteError: Double        = probes.iterator.map(_.absoluteError).max
  val maximumRelativeError: Double        = probes.iterator.map(_.relativeError).max
  val maximumErrorRatio: Double           = probes.iterator.map(_.errorRatio).max

object GradientChecker:
  /**
   * Compares reverse-mode gradients with central differences and restores parameters.
   *
   * `lossFactory` must build a fresh deterministic scalar graph connected to the supplied trainable
   * parameters. The method first performs one backward pass, then evaluates `f(theta + epsilon)`
   * and `f(theta - epsilon)` for a deterministic subset of each parameter. Every parameter value is
   * restored in a `finally` block, including when loss construction fails. Diagnostic gradients are
   * cleared before returning; run the checker between training steps because pre-existing gradients
   * are intentionally not preserved.
   *
   * Cost is one forward/backward pass plus two forward passes per probe. This is a correctness
   * diagnostic for small models, not a training algorithm.
   */
  def check(
      parameters: Vector[Tensor],
      lossFactory: () => Tensor,
      config: GradientCheckConfig = GradientCheckConfig()
  ): GradientCheckReport =
    require(parameters.nonEmpty, "gradient checking requires at least one parameter")
    require(parameters.forall(_.isTrainable), "gradient checking accepts only trainable leaves")
    require(
      parameters.distinct.size == parameters.size,
      "gradient-check parameter references must be unique"
    )
    require(
      parameters.map(_.label).distinct.size == parameters.size,
      "gradient-check parameter labels must be unique"
    )

    val originals = parameters.map(parameter => parameter -> parameter.values)
    try
      parameters.foreach(_.clearGradients())
      val analyticalLoss = lossFactory()
      require(
        analyticalLoss.shape == Shape.scalar,
        s"gradient checking requires scalar loss, got ${analyticalLoss.shape}"
      )
      val lossValue      = analyticalLoss.valueAtFlat(0)
      analyticalLoss.backward()

      val probes = Vector.newBuilder[GradientProbe]
      originals.foreach { case (parameter, originalValues) =>
        selectedIndices(parameter.size, config.maximumCoordinatesPerParameter).foreach {
          flatIndex =>
            val analytical    = parameter.gradientAtFlat(flatIndex)
            val plus          = evaluatePerturbation(
              parameter,
              originalValues,
              flatIndex,
              config.epsilon,
              lossFactory
            )
            val minus         = evaluatePerturbation(
              parameter,
              originalValues,
              flatIndex,
              -config.epsilon,
              lossFactory
            )
            val numerical     = (plus - minus) / (2.0 * config.epsilon)
            val absoluteError = math.abs(analytical - numerical)
            val scale         = math.max(math.abs(analytical), math.abs(numerical))
            val relativeError = absoluteError / math.max(scale, 1e-30)
            val allowedError  = config.absoluteTolerance + config.relativeTolerance * scale
            probes += GradientProbe(
              parameter.label,
              parameter.shape,
              flatIndex,
              parameter.shape.coordinates(flatIndex),
              analytical,
              numerical,
              absoluteError,
              relativeError,
              allowedError
            )
        }
      }
      GradientCheckReport(lossValue, probes.result())
    finally
      originals.foreach { case (parameter, values) => parameter.assignParameterValues(values) }
      parameters.foreach(_.clearGradients())

  private def evaluatePerturbation(
      parameter: Tensor,
      originalValues: Vector[Double],
      flatIndex: Int,
      delta: Double,
      lossFactory: () => Tensor
  ): Double =
    val perturbed = originalValues.updated(flatIndex, originalValues(flatIndex) + delta)
    parameter.assignParameterValues(perturbed)
    try
      val loss = lossFactory()
      require(
        loss.shape == Shape.scalar,
        s"gradient checking requires scalar loss, got ${loss.shape}"
      )
      loss.valueAtFlat(0)
    finally parameter.assignParameterValues(originalValues)

  private def selectedIndices(size: Int, maximumCoordinates: Int): Vector[Int] =
    require(size > 0, "cannot gradient-check an empty parameter")
    if size <= maximumCoordinates then Vector.range(0, size)
    else if maximumCoordinates == 1 then Vector(0)
    else
      Vector.tabulate(maximumCoordinates) { probe =>
        (probe.toLong * (size - 1).toLong / (maximumCoordinates - 1).toLong).toInt
      }

/** One parameter's logical identity, shape, and dense payload estimate. */
final case class ParameterEntry(label: String, shape: Shape, elements: Int, payloadBytes: Long)

/** Deterministic inventory of uniquely owned trainable parameters. */
final case class ParameterInventory(entries: Vector[ParameterEntry], bytesPerElement: Int):
  require(entries.nonEmpty, "parameter inventory requires at least one entry")
  val totalElements: Long     = entries.iterator.map(_.elements.toLong).sum
  val totalPayloadBytes: Long = entries.iterator.map(_.payloadBytes).sum

object ParameterInventory:
  /**
   * Describes dense parameter payload only; gradients, optimizer state, and activations are
   * separate.
   */
  def from(parameters: Vector[Tensor], bytesPerElement: Int = 8): ParameterInventory =
    require(parameters.nonEmpty, "parameter inventory requires at least one parameter")
    require(bytesPerElement > 0, s"bytes per element must be positive: $bytesPerElement")
    require(parameters.distinct.size == parameters.size, "parameter references must be unique")
    require(
      parameters.map(_.label).distinct.size == parameters.size,
      "parameter labels must be unique"
    )
    ParameterInventory(
      parameters.map { parameter =>
        ParameterEntry(
          parameter.label,
          parameter.shape,
          parameter.size,
          Math.multiplyExact(parameter.size.toLong, bytesPerElement.toLong)
        )
      },
      bytesPerElement
    )
