package learnai.transformer

import learnai.diagnostics.Benchmark
import learnai.diagnostics.BenchmarkConfig
import learnai.diagnostics.GradientCheckConfig
import learnai.diagnostics.GradientChecker
import learnai.diagnostics.ParameterInventory
import learnai.text.TokenId

/** Runs parameter, gradient, and forward-timing diagnostics on one tiny MiniGPT. */
def runMiniGptDiagnostics(): Unit =
  val config  = MiniGptConfig(
    vocabularySize = 7,
    maximumContextLength = 4,
    channels = 4,
    headCount = 2,
    hiddenChannels = 8,
    layerCount = 1
  )
  val model   = MiniGpt.random(config, seed = 42L)
  val inputs  = Vector(0, 1, 2, 3).map(TokenId(_))
  val targets = Vector(1, 2, 3, 4).map(TokenId(_))

  val inventory = ParameterInventory.from(model.parameters)
  println(s"parameter tensors: ${inventory.entries.size}")
  println(s"parameter elements: ${inventory.totalElements}")
  println(s"dense payload:      ${inventory.totalPayloadBytes} bytes")
  inventory.entries.foreach { entry =>
    println(f"  ${entry.label}%-42s ${entry.shape.toString}%-10s ${entry.elements}%6d elements")
  }

  val gradientReport = GradientChecker.check(
    model.parameters,
    () => model.loss(inputs, targets),
    GradientCheckConfig(maximumCoordinatesPerParameter = 2)
  )
  println(s"gradient probes:    ${gradientReport.probes.size}")
  println(s"gradient passed:    ${gradientReport.passed}")
  println(f"maximum abs error:  ${gradientReport.maximumAbsoluteError}%.3e")
  println(f"maximum error ratio:${gradientReport.maximumErrorRatio}%10.3e")
  val worst          = gradientReport.worstProbe
  println(f"worst probe:        ${worst.parameterLabel}[${worst.flatIndex}] " + f"analytical=${worst
      .analytical}%.6e numerical=${worst.numerical}%.6e " + f"ratio=${worst.errorRatio}%.3f")

  val benchmark  = Benchmark.measure(
    "MiniGPT forward",
    BenchmarkConfig(warmupIterations = 10, measurementIterations = 30, operationsPerMeasurement = 3)
  ) {
    val logits = model.logits(inputs)
    java.lang.Double.doubleToRawLongBits(logits.values.sum)
  }
  val statistics = benchmark.statistics
  println(s"benchmark:          ${benchmark.name}")
  println(f"median:             ${statistics.medianNanoseconds / 1_000_000.0}%.3f ms/op")
  println(f"p95:                ${statistics.percentile95Nanoseconds / 1_000_000.0}%.3f ms/op")
  println(f"variation:          ${statistics.coefficientOfVariation * 100.0}%.2f%%")
  println(s"checksum:           ${benchmark.checksum}")
  println(s"runtime:            ${benchmark.runtime.javaVmName} ${benchmark.runtime
      .javaRuntimeVersion}, " + s"${benchmark.runtime.operatingSystem}, ${benchmark.runtime
      .architecture}, " + s"${benchmark.runtime.availableProcessors} processors")
