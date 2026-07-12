package learnai.math

/** Point estimate with standard error and a normal-approximation interval. */
final case class MeanEstimate(
    mean: Double,
    standardError: Double,
    lower: Double,
    upper: Double,
    samples: Int
)

/** One binary probability forecast and its observed zero-or-one outcome. */
final case class ProbabilityForecast(probability: Double, outcome: Int):
  require(
    probability >= 0.0 && probability <= 1.0 && probability.isFinite,
    s"invalid probability: $probability"
  )
  require(outcome == 0 || outcome == 1, s"outcome must be zero or one: $outcome")

/** One populated reliability bin used to diagnose probability calibration. */
final case class CalibrationBin(
    lower: Double,
    upper: Double,
    count: Int,
    meanPrediction: Double,
    eventRate: Double
)

/** Small statistical estimators with explicit sample and uncertainty semantics. */
object Statistics:
  def sampleMean(values: Vector[Double]): Either[String, Double] =
    if values.isEmpty then Left("sample mean requires at least one observation")
    else Right(Numerics.compensatedSum(values) / values.size)

  /** Unbiased sample variance with denominator `n - 1`. */
  def sampleVariance(values: Vector[Double]): Either[String, Double] =
    if values.size < 2 then Left("sample variance requires at least two observations")
    else
      val mean       = AssertFinite(sampleMean(values))
      val sumSquares = Numerics
        .compensatedSum(values.iterator.map(value => math.pow(value - mean, 2)))
      Right(sumSquares / (values.size - 1))

  /** Two-sided normal-approximation confidence interval for a sample mean. */
  def meanConfidenceInterval(
      values: Vector[Double],
      z: Double = 1.959963984540054
  ): Either[String, MeanEstimate] =
    if z <= 0.0 || !z.isFinite then Left(s"z score must be finite and positive: $z")
    else
      sampleVariance(values).flatMap { variance =>
        sampleMean(values).map { mean =>
          val standardError = math.sqrt(variance / values.size)
          MeanEstimate(
            mean,
            standardError,
            mean - z * standardError,
            mean + z * standardError,
            values.size
          )
        }
      }

  /** Reliability bins; empty bins are omitted rather than assigned invented rates. */
  def reliability(forecasts: Vector[ProbabilityForecast], binCount: Int): Vector[CalibrationBin] =
    require(binCount > 0, s"bin count must be positive: $binCount")
    Vector.tabulate(binCount) { index =>
      val lower   = index.toDouble / binCount
      val upper   = (index + 1).toDouble / binCount
      val members = forecasts.filter { forecast =>
        val assigned = math.min((forecast.probability * binCount).toInt, binCount - 1)
        assigned == index
      }
      if members.isEmpty then None
      else
        Some(CalibrationBin(
          lower,
          upper,
          members.size,
          members.map(_.probability).sum / members.size,
          members.map(_.outcome).sum.toDouble / members.size
        ))
    }.flatten

  /** Expected calibration error weighted by populated-bin frequency. */
  def expectedCalibrationError(
      forecasts: Vector[ProbabilityForecast],
      binCount: Int
  ): Either[String, Double] =
    if forecasts.isEmpty then Left("calibration error requires at least one forecast")
    else
      val bins = reliability(forecasts, binCount)
      Right(
        bins.map(bin =>
          bin.count.toDouble / forecasts.size * math.abs(bin.meanPrediction - bin.eventRate)
        ).sum
      )

  private def AssertFinite(result: Either[String, Double]): Double = result
    .fold(message => throw new IllegalArgumentException(message), identity)

/** Prints an estimate, uncertainty interval, and a small calibration report. */
def runStatisticsLab(): Unit =
  val estimate  = Statistics.meanConfidenceInterval(Vector(1.0, 2.0, 3.0, 4.0, 5.0))
  println(s"mean estimate: $estimate")
  val forecasts = Vector(
    ProbabilityForecast(0.1, 0),
    ProbabilityForecast(0.2, 0),
    ProbabilityForecast(0.8, 1),
    ProbabilityForecast(0.9, 1)
  )
  println(s"reliability:   ${Statistics.reliability(forecasts, 2)}")
  println(s"calibration:   ${Statistics.expectedCalibrationError(forecasts, 2)}")
