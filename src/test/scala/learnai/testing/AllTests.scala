package learnai.testing

import learnai.accounting.ModelAccountingSuite
import learnai.foundations.ScalaTourSuite
import learnai.foundations.ComplexitySuite
import learnai.foundations.JvmSystemsSuite
import learnai.math.NumericsSuite
import learnai.math.VectorDSuite
import learnai.math.MatrixDSuite
import learnai.math.LinearAlgebraSuite
import learnai.math.StatisticsSuite
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
import learnai.data.PackingSuite
import learnai.data.ShuffleSuite
import learnai.diagnostics.BenchmarkSuite
import learnai.distributed.DataParallelSuite
import learnai.diagnostics.GradientCheckSuite
import learnai.experiment.ExperimentManifestSuite
import learnai.finetune.LoraSuite
import learnai.finetune.SftSuite
import learnai.lm.BigramLanguageModelSuite
import learnai.transformer.LayersSuite
import learnai.transformer.AttentionSuite
import learnai.transformer.GroupedQueryAttentionSuite
import learnai.transformer.GptLineageSuite
import learnai.transformer.RopeSuite
import learnai.transformer.SwiGluSuite
import learnai.transformer.TiledAttentionSuite
import learnai.transformer.TransformerBlockSuite
import learnai.training.MiniGptTrainingSuite
import learnai.training.ResumableTrainingSuite
import learnai.training.TrainingWorkflowSuite
import learnai.transformer.MiniGptSuite
import learnai.lm.SamplingSuite
import learnai.io.MiniGptCheckpointSuite
import learnai.io.TrainingBundleSuite
import learnai.quantization.Int8QuantizationSuite
import learnai.random.SplitMix64Suite
import learnai.json.JsonSuite
import learnai.agent.AgentRuntimeSuite
import learnai.agent.PlanningSuite
import learnai.agent.EvaluationSuite
import learnai.retrieval.RetrievalSuite
import learnai.serving.PagedKvCacheSuite
import learnai.serving.SpeculativeDecodingSuite

object AllTests:
  private val suites: Vector[TestSuite] = Vector(
    DocumentationLanguageSuite,
    DocumentationStructureSuite,
    RepositoryCoverageSuite,
    BenchmarkSuite,
    GradientCheckSuite,
    ExperimentManifestSuite,
    ScalaTourSuite,
    ComplexitySuite,
    JvmSystemsSuite,
    NumericsSuite,
    VectorDSuite,
    MatrixDSuite,
    LinearAlgebraSuite,
    StatisticsSuite,
    ProbabilitySuite,
    CalculusSuite,
    GradientDescentSuite,
    ValueSuite,
    ScalarNetworkSuite,
    TensorSuite,
    OptimizersSuite,
    SplitMix64Suite,
    Utf8Suite,
    BpeTokenizerSuite,
    CausalDatasetSuite,
    ShuffleSuite,
    PackingSuite,
    BigramLanguageModelSuite,
    LayersSuite,
    AttentionSuite,
    TransformerBlockSuite,
    RopeSuite,
    SwiGluSuite,
    GroupedQueryAttentionSuite,
    GptLineageSuite,
    TiledAttentionSuite,
    ModelAccountingSuite,
    MiniGptSuite,
    MiniGptTrainingSuite,
    ResumableTrainingSuite,
    TrainingWorkflowSuite,
    DataParallelSuite,
    SftSuite,
    LoraSuite,
    SamplingSuite,
    MiniGptCheckpointSuite,
    TrainingBundleSuite,
    Int8QuantizationSuite,
    PagedKvCacheSuite,
    SpeculativeDecodingSuite,
    JsonSuite,
    AgentRuntimeSuite,
    PlanningSuite,
    EvaluationSuite,
    RetrievalSuite
  )

  def main(arguments: Array[String]): Unit =
    val results =
      for
        suite    <- suites
        testCase <- suite.tests
      yield runOne(suite.name, testCase)

    val failures = results.count(result => !result)
    val passed   = results.size - failures
    println(s"\n$passed passed, $failures failed, ${results.size} total")

    if failures > 0 then throw new AssertionError(s"$failures test(s) failed")

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
