package learnai.diagnostics

/** Monotonic clock boundary used to make benchmark orchestration testable.
  *
  * Production measurements use `System.nanoTime`. Tests may supply a scripted
  * clock to verify batching and statistics without depending on wall time.
  */
trait NanoClock:
  def nanoTime(): Long

object NanoClock:
  val system: NanoClock = new NanoClock:
    override def nanoTime(): Long = System.nanoTime()

/** Controls warmup, sampling, and batching for one microbenchmark.
  *
  * `operationsPerMeasurement` repeats the operation inside one timed region.
  * Batching reduces timer-resolution noise for very small operations. The
  * reported samples are normalized back to nanoseconds per operation.
  */
final case class BenchmarkConfig(
    warmupIterations: Int = 10,
    measurementIterations: Int = 30,
    operationsPerMeasurement: Int = 1
):
  require(warmupIterations >= 0, s"warmup iterations must be non-negative: $warmupIterations")
  require(
    measurementIterations > 0,
    s"measurement iterations must be positive: $measurementIterations"
  )
  require(
    operationsPerMeasurement > 0,
    s"operations per measurement must be positive: $operationsPerMeasurement"
  )

/** Descriptive statistics over normalized nanoseconds-per-operation samples.
  *
  * Median averages the two central samples for an even count. Percentile 95
  * uses the nearest-rank definition. Standard deviation is the population
  * standard deviation of this observed sample vector; it is descriptive, not
  * a confidence interval for a larger population.
  */
final case class BenchmarkStatistics private (
    samplesNanoseconds: Vector[Double],
    minimumNanoseconds: Double,
    medianNanoseconds: Double,
    percentile95Nanoseconds: Double,
    maximumNanoseconds: Double,
    meanNanoseconds: Double,
    standardDeviationNanoseconds: Double
):
  val operationsPerSecond: Double = 1_000_000_000.0 / meanNanoseconds
  val coefficientOfVariation: Double = standardDeviationNanoseconds / meanNanoseconds

object BenchmarkStatistics:
  /** Calculates deterministic descriptive statistics from positive samples. */
  def from(samplesNanoseconds: Vector[Double]): BenchmarkStatistics =
    require(samplesNanoseconds.nonEmpty, "benchmark statistics require at least one sample")
    samplesNanoseconds.zipWithIndex.foreach { case (sample, index) =>
      require(
        sample > 0.0 && sample.isFinite,
        s"benchmark sample $index must be finite and positive: $sample"
      )
    }
    val sorted = samplesNanoseconds.sorted
    val median =
      if sorted.size % 2 == 1 then sorted(sorted.size / 2)
      else
        val upper = sorted.size / 2
        (sorted(upper - 1) + sorted(upper)) / 2.0
    val percentile95Index = math.ceil(0.95 * sorted.size.toDouble).toInt - 1
    val mean = samplesNanoseconds.sum / samplesNanoseconds.size.toDouble
    val variance = samplesNanoseconds.iterator
      .map(sample => math.pow(sample - mean, 2.0))
      .sum / samplesNanoseconds.size.toDouble
    BenchmarkStatistics(
      samplesNanoseconds,
      sorted.head,
      median,
      sorted(percentile95Index),
      sorted.last,
      mean,
      math.sqrt(variance)
    )

/** Runtime identity recorded beside a local timing observation. */
final case class RuntimeFingerprint(
    javaRuntimeVersion: String,
    javaVmName: String,
    javaVmVersion: String,
    operatingSystem: String,
    architecture: String,
    availableProcessors: Int
)

object RuntimeFingerprint:
  def current: RuntimeFingerprint = RuntimeFingerprint(
    System.getProperty("java.runtime.version", "unknown"),
    System.getProperty("java.vm.name", "unknown"),
    System.getProperty("java.vm.version", "unknown"),
    s"${System.getProperty("os.name", "unknown")} ${System.getProperty("os.version", "unknown")}",
    System.getProperty("os.arch", "unknown"),
    Runtime.getRuntime.availableProcessors()
  )

/** One benchmark result with raw samples, environment, and anti-elision checksum. */
final case class BenchmarkResult(
    name: String,
    config: BenchmarkConfig,
    statistics: BenchmarkStatistics,
    checksum: Long,
    runtime: RuntimeFingerprint
):
  require(name.nonEmpty, "benchmark name cannot be empty")

object Benchmark:
  /** Measures a side-effect-free operation that returns an observable checksum value.
    *
    * The operation must be deterministic for a fixed setup and must perform the
    * work under study. Its returned `Long` is mixed into the result so the JVM
    * cannot trivially discard an unused calculation. Warmup runs execute the
    * same batch shape but are excluded from timing statistics.
    *
    * This is an educational harness, not a replacement for JMH. It does not
    * control compiler forks, CPU frequency, process affinity, garbage
    * collection, or hardware performance counters.
    */
  def measure(
      name: String,
      config: BenchmarkConfig,
      clock: NanoClock = NanoClock.system,
      runtime: RuntimeFingerprint = RuntimeFingerprint.current
  )(operation: => Long): BenchmarkResult =
    require(name.nonEmpty, "benchmark name cannot be empty")
    var checksum = 0x6a09e667f3bcc909L

    def consume(value: Long): Unit =
      checksum = java.lang.Long.rotateLeft(checksum ^ value, 17) * 0x9e3779b97f4a7c15L

    def runBatch(): Unit =
      var operationIndex = 0
      while operationIndex < config.operationsPerMeasurement do
        consume(operation)
        operationIndex += 1

    var warmup = 0
    while warmup < config.warmupIterations do
      runBatch()
      warmup += 1

    val samples = Vector.newBuilder[Double]
    var measurement = 0
    while measurement < config.measurementIterations do
      val started = clock.nanoTime()
      runBatch()
      val elapsed = clock.nanoTime() - started
      require(
        elapsed > 0L,
        s"measurement $measurement elapsed $elapsed ns; increase operationsPerMeasurement"
      )
      samples += elapsed.toDouble / config.operationsPerMeasurement.toDouble
      measurement += 1

    BenchmarkResult(name, config, BenchmarkStatistics.from(samples.result()), checksum, runtime)
