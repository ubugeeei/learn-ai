package learnai.testing

import learnai.foundations.ScalaTourSuite
import learnai.math.NumericsSuite
import learnai.math.VectorDSuite
import learnai.math.MatrixDSuite
import learnai.math.ProbabilitySuite
import learnai.math.CalculusSuite
import learnai.learning.GradientDescentSuite
import learnai.autodiff.ValueSuite
import learnai.nn.ScalarNetworkSuite
import learnai.tensor.TensorSuite
import learnai.optim.OptimizersSuite

object AllTests:
  private val suites: Vector[TestSuite] = Vector(
    ScalaTourSuite,
    NumericsSuite,
    VectorDSuite,
    MatrixDSuite,
    ProbabilitySuite,
    CalculusSuite,
    GradientDescentSuite,
    ValueSuite,
    ScalarNetworkSuite,
    TensorSuite,
    OptimizersSuite
  )

  def main(arguments: Array[String]): Unit =
    val results = for
      suite <- suites
      testCase <- suite.tests
    yield runOne(suite.name, testCase)

    val failures = results.count(result => !result)
    val passed = results.size - failures
    println(s"\n$passed passed, $failures failed, ${results.size} total")

    if failures > 0 then
      throw new AssertionError(s"$failures test(s) failed")

  private def runOne(suiteName: String, testCase: TestCase): Boolean =
    val fullName = s"$suiteName / ${testCase.name}"
    try
      testCase.run()
      println(s"[pass] $fullName")
      true
    catch
      case error: Throwable =>
        println(s"[fail] $fullName")
        println(s"       ${error.getClass.getSimpleName}: ${error.getMessage}")
        false
