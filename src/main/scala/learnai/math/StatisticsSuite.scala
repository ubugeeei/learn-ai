package learnai.math

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object StatisticsSuite extends TestSuite:
  override val name: String = "Statistics"

  override val tests: Vector[TestCase] = specify(
    test("sample mean and unbiased variance match a hand calculation") {
      val values = Vector(1.0, 2.0, 3.0)
      Assert.close(Assert.right(Statistics.sampleMean(values)), 2.0)
      Assert.close(Assert.right(Statistics.sampleVariance(values)), 1.0)
    },
    test("mean interval exposes standard error and symmetric bounds") {
      val estimate = Assert.right(Statistics.meanConfidenceInterval(Vector(1, 2, 3), z = 2.0))
      Assert.close(estimate.standardError, math.sqrt(1.0 / 3.0))
      Assert.close(estimate.mean - estimate.lower, estimate.upper - estimate.mean)
      Assert.equal(estimate.samples, 3)
    },
    test("undefined estimates return descriptive errors") {
      Assert.isTrue(Statistics.sampleMean(Vector.empty).isLeft)
      Assert.isTrue(Statistics.sampleVariance(Vector(1)).isLeft)
      Assert.isTrue(Statistics.meanConfidenceInterval(Vector(1), z = 0).isLeft)
    },
    test("perfectly calibrated deterministic forecasts have zero error") {
      val forecasts = Vector(ProbabilityForecast(0, 0), ProbabilityForecast(1, 1))
      Assert.close(Assert.right(Statistics.expectedCalibrationError(forecasts, 2)), 0.0)
    },
    test("reliability omits empty bins and weights populated bins") {
      val forecasts = Vector(ProbabilityForecast(0.2, 0), ProbabilityForecast(0.8, 0))
      val bins      = Statistics.reliability(forecasts, 4)
      Assert.equal(bins.size, 2)
      Assert.close(Assert.right(Statistics.expectedCalibrationError(forecasts, 4)), 0.5)
      Assert.isTrue(Statistics.expectedCalibrationError(Vector.empty, 4).isLeft)
    }
  )
