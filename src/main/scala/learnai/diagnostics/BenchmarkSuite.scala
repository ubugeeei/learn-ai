package learnai.diagnostics

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object BenchmarkSuite extends TestSuite:
  override val name: String = "Benchmark"

  override val tests: Vector[TestCase] = specify(
    test("statistics use documented median percentile mean and variation") {
      val statistics = BenchmarkStatistics.from(Vector(4.0, 1.0, 3.0, 2.0))
      Assert.close(statistics.minimumNanoseconds, 1.0)
      Assert.close(statistics.medianNanoseconds, 2.5)
      Assert.close(statistics.percentile95Nanoseconds, 4.0)
      Assert.close(statistics.maximumNanoseconds, 4.0)
      Assert.close(statistics.meanNanoseconds, 2.5)
      Assert.close(statistics.standardDeviationNanoseconds, math.sqrt(1.25))
      Assert.close(statistics.operationsPerSecond, 400_000_000.0)
    },
    test("runner excludes warmup and normalizes batched measurements per operation") {
      val clock = new ScriptedNanoClock(Vector(100L, 160L, 200L, 280L))
      var operations = 0
      val runtime = RuntimeFingerprint("test-java", "test-vm", "1", "test-os", "test-arch", 1)
      val result = Benchmark.measure(
        "scripted",
        BenchmarkConfig(
          warmupIterations = 1,
          measurementIterations = 2,
          operationsPerMeasurement = 2
        ),
        clock,
        runtime
      ) {
        operations += 1
        operations.toLong
      }

      Assert.equal(operations, 6)
      Assert.equal(result.statistics.samplesNanoseconds, Vector(30.0, 40.0))
      Assert.close(result.statistics.medianNanoseconds, 35.0)
      Assert.equal(result.runtime, runtime)
      Assert.isTrue(result.checksum != 0L)
    },
    test("invalid configuration and non-increasing clocks fail near the boundary") {
      val warmupError = Assert.throws[IllegalArgumentException] {
        BenchmarkConfig(warmupIterations = -1)
      }
      val measurementError = Assert.throws[IllegalArgumentException] {
        BenchmarkConfig(measurementIterations = 0)
      }
      val clockError = Assert.throws[IllegalArgumentException] {
        Benchmark.measure(
          "zero-time",
          BenchmarkConfig(warmupIterations = 0, measurementIterations = 1),
          new ScriptedNanoClock(Vector(5L, 5L))
        )(1L)
      }
      Assert.isTrue(warmupError.getMessage.contains("non-negative"))
      Assert.isTrue(measurementError.getMessage.contains("positive"))
      Assert.isTrue(clockError.getMessage.contains("increase operationsPerMeasurement"))
    }
  )

  private final class ScriptedNanoClock(values: Vector[Long]) extends NanoClock:
    private var index = 0

    override def nanoTime(): Long =
      require(index < values.size, "scripted clock exhausted")
      val value = values(index)
      index += 1
      value
