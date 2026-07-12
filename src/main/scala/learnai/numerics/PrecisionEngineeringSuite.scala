package learnai.numerics

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object PrecisionEngineeringSuite extends TestSuite:
  override val name: String = "PrecisionEngineering"

  override val tests: Vector[TestCase] = specify(
    test("formats expose storage width and round-trip representative values") {
      Assert.equal(FloatFormat.Float32.bits, 32)
      Assert.equal(FloatFormat.Float16.bits, 16)
      Assert.equal(FloatFormat.BFloat16.bits, 16)
      Vector(0.0, -1.5, 42.25).foreach(value =>
        FloatFormat.values.foreach(format => Assert.close(format.round(value), value, 0.0))
      )
    },
    test("float16 has more local precision while bfloat16 has more range") {
      val nearOne   = 1.001
      val fp16Error = math.abs(FloatFormat.Float16.round(nearOne) - nearOne)
      val bf16Error = math.abs(FloatFormat.BFloat16.round(nearOne) - nearOne)
      Assert.isTrue(fp16Error < bf16Error)
      Assert.isTrue(FloatFormat.Float16.round(100000.0).isInfinite)
      Assert.isTrue(FloatFormat.BFloat16.round(100000.0).isFinite)
    },
    test("bfloat16 uses round-to-nearest-even at discarded-bit ties") {
      val even = java.lang.Float.intBitsToFloat(0x3f808000)
      val odd  = java.lang.Float.intBitsToFloat(0x3f818000)
      Assert.equal(BFloat16Codec.bits(even) & 0xffff, 0x3f80)
      Assert.equal(BFloat16Codec.bits(odd) & 0xffff, 0x3f82)
    },
    test("higher-precision accumulation retains a small residual") {
      val values = Vector(100000000.0, 1.0, -100000000.0)
      Assert.equal(Accumulation.naiveFloat32(values), 0.0)
      Assert.equal(Accumulation.float64(values), 1.0)
    },
    test("finite scaled gradients unscale before the optimizer and eventually grow") {
      val first  = DynamicLossScaler(1024, growthInterval = 2).unscale(Vector(1024, -2048))
      Assert.equal(first.gradients, Vector(1.0, -2.0))
      Assert.isTrue(!first.skipped)
      Assert.equal(first.next.scale, 1024.0)
      val second = first.next.unscale(Vector(512))
      Assert.equal(second.next.scale, 2048.0)
      Assert.equal(second.next.finiteSteps, 0)
    },
    test("non-finite gradients skip the update and reduce the scale") {
      val result = DynamicLossScaler(1024).unscale(Vector(1.0, Double.PositiveInfinity))
      Assert.isTrue(result.skipped)
      Assert.equal(result.gradients, Vector.empty)
      Assert.equal(result.next.scale, 512.0)
    },
    test("invalid scaler policies fail at construction") {
      Assert.throws[IllegalArgumentException](DynamicLossScaler(0))
      Assert.throws[IllegalArgumentException](DynamicLossScaler(1, growthInterval = 0))
      Assert.throws[IllegalArgumentException](DynamicLossScaler(1, backoffFactor = 1.0))
      ()
    }
  )
