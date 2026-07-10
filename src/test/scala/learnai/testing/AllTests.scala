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
import learnai.text.Utf8Suite
import learnai.text.BpeTokenizerSuite
import learnai.data.CausalDatasetSuite
import learnai.lm.BigramLanguageModelSuite
import learnai.transformer.LayersSuite
import learnai.transformer.AttentionSuite
import learnai.transformer.TransformerBlockSuite
import learnai.transformer.MiniGptSuite
import learnai.lm.SamplingSuite

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
    OptimizersSuite,
    Utf8Suite,
    BpeTokenizerSuite,
    CausalDatasetSuite,
    BigramLanguageModelSuite,
    LayersSuite,
    AttentionSuite,
    TransformerBlockSuite,
    MiniGptSuite,
    SamplingSuite
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
