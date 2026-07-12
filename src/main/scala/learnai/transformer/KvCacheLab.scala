package learnai.transformer

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.SplittableRandom

import scala.jdk.CollectionConverters.*

import learnai.lm.SamplingConfig
import learnai.text.TokenId

/** Dependency-free benchmark utilities specific to the KV-cache experiment. */
private object KvCacheLab:
  final case class Timing(samplesMillis: Vector[Double]):
    require(samplesMillis.nonEmpty, "timing requires at least one sample")
    private val ordered       = samplesMillis.sorted
    val minimumMillis: Double = ordered.head
    val medianMillis: Double  = ordered(ordered.size / 2)
    val maximumMillis: Double = ordered.last

  def measure(iterations: Int)(operation: => Long): (Timing, Long) =
    require(iterations > 0, s"measurement iterations must be positive: $iterations")
    val samples   = Vector.newBuilder[Double]
    var checksum  = 0L
    var iteration = 0
    while iteration < iterations do
      val started = System.nanoTime()
      checksum ^= operation
      val elapsed = System.nanoTime() - started
      samples += elapsed.toDouble / 1_000_000.0
      iteration += 1
    Timing(samples.result()) -> checksum

  def cpuDescription(): String = linuxCpuDescription().orElse(macCpuDescription())
    .orElse(Option(System.getenv("PROCESSOR_IDENTIFIER")).filter(_.nonEmpty))
    .getOrElse(System.getProperty("os.arch", "unknown architecture"))

  private def linuxCpuDescription(): Option[String] =
    val cpuInfo = Path.of("/proc/cpuinfo")
    if !Files.isRegularFile(cpuInfo) then None
    else
      Files.readAllLines(cpuInfo, StandardCharsets.UTF_8).asScala.find(_.startsWith("model name"))
        .flatMap(_.split(":", 2).lift(1)).map(_.trim).filter(_.nonEmpty)

  private def macCpuDescription(): Option[String] =
    if System.getProperty("os.name", "").toLowerCase.contains("mac") then
      try
        val process  = new ProcessBuilder("sysctl", "-n", "machdep.cpu.brand_string")
          .redirectErrorStream(true).start()
        val output   = new String(process.getInputStream.readAllBytes(), StandardCharsets.UTF_8).trim
        val exitCode = process.waitFor()
        Option.when(exitCode == 0 && output.nonEmpty)(output)
      catch case _: Exception => None
    else None

/**
 * Compares reference and cached generation after JIT warmup.
 *
 * This is an educational microbenchmark rather than a substitute for JMH. It records the
 * environment, exact model configuration, warmup count, repeated samples, variation, output
 * checksum, and equivalence check so that its numbers cannot be mistaken for an unqualified
 * performance claim.
 */
def runKvCacheLab(): Unit =
  val config                = MiniGptConfig(
    vocabularySize = 32,
    maximumContextLength = 32,
    channels = 16,
    headCount = 4,
    hiddenChannels = 32,
    layerCount = 2
  )
  val modelSeed             = 42L
  val samplingSeed          = 73L
  val prompt                = Vector.tabulate(8)(index => TokenId(index % config.vocabularySize))
  val generatedTokens       = 16
  val warmupIterations      = 5
  val measurementIterations = 15
  val sampling              = SamplingConfig(temperature = 0.8)
  val model                 = MiniGpt.random(config, modelSeed)

  def reference(): Vector[TokenId] = model
    .generate(prompt, generatedTokens, sampling.temperature, new SplittableRandom(samplingSeed))
    .fold(message => throw new IllegalStateException(message), identity)

  def cached(): CachedGenerationResult = model
    .generateCached(prompt, generatedTokens, sampling, new SplittableRandom(samplingSeed))
    .fold(message => throw new IllegalStateException(message), identity)

  val referenceOutput = reference()
  val cachedOutput    = cached()
  require(referenceOutput == cachedOutput.tokens, "cached generation differs from reference")

  var warmup         = 0
  var warmupChecksum = 0L
  while warmup < warmupIterations do
    warmupChecksum ^= reference().iterator.map(_.value.toLong).sum
    warmupChecksum ^= cached().tokens.iterator.map(_.value.toLong).sum
    warmup += 1

  val (referenceTiming, referenceChecksum) = KvCacheLab
    .measure(measurementIterations)(reference().iterator.map(_.value.toLong).sum)
  val (cachedTiming, cachedChecksum)       = KvCacheLab
    .measure(measurementIterations)(cached().tokens.iterator.map(_.value.toLong).sum)

  def render(timing: KvCacheLab.Timing): String =
    f"min=${timing.minimumMillis}%.3f ms, median=${timing.medianMillis}%.3f ms, max=${timing
        .maximumMillis}%.3f ms"

  println(s"JVM:              ${System.getProperty("java.runtime.version")}")
  println(s"OS:               ${System.getProperty("os.name")} ${System.getProperty("os.version")}")
  println(s"CPU:              ${KvCacheLab.cpuDescription()}")
  println(s"model:            $config")
  println(s"seeds:            model=$modelSeed, sampling=$samplingSeed")
  println(s"workload:         prompt=${prompt.size}, generated=$generatedTokens")
  println(s"iterations:       warmup=$warmupIterations, measured=$measurementIterations")
  println(s"reference timing: ${render(referenceTiming)}")
  println(s"cached timing:    ${render(cachedTiming)}")
  println(s"token work:       ${cachedOutput.statistics.referenceTokenEvaluations} -> ${cachedOutput
      .statistics.tokenEvaluations}")
  println(s"cache payload:    ${cachedOutput.statistics.allocatedCachePayloadBytes} bytes")
  println(s"equivalent:       ${referenceOutput == cachedOutput.tokens}")
  println(s"checksum:         ${warmupChecksum ^ referenceChecksum ^ cachedChecksum}")
