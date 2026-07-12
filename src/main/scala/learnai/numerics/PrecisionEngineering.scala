package learnai.numerics

/** Observable storage formats used to explain precision/range trade-offs. */
enum FloatFormat:
  case Float32, Float16, BFloat16

  def bits: Int = this match
    case Float32  => 32
    case Float16  => 16
    case BFloat16 => 16

  /** Rounds a Double through the selected finite-width storage representation. */
  def round(value: Double): Double = this match
    case Float32  => value.toFloat.toDouble
    case Float16  => java.lang.Float.float16ToFloat(java.lang.Float.floatToFloat16(value.toFloat)).toDouble
    case BFloat16 => BFloat16Codec.round(value.toFloat).toDouble

/** IEEE-like bfloat16 conversion using round-to-nearest-even on the discarded low 16 bits. */
object BFloat16Codec:
  def bits(value: Float): Short =
    val raw = java.lang.Float.floatToRawIntBits(value)
    val exponent = raw & 0x7f800000
    if exponent == 0x7f800000 then (raw >>> 16).toShort
    else
      val roundingBias = 0x7fff + ((raw >>> 16) & 1)
      ((raw + roundingBias) >>> 16).toShort

  def fromBits(value: Short): Float =
    java.lang.Float.intBitsToFloat((value.toInt & 0xffff) << 16)

  def round(value: Float): Float = fromBits(bits(value))

/** Reference accumulation policies that separate operand storage from reduction precision. */
object Accumulation:
  def naiveFloat32(values: IterableOnce[Double]): Double =
    var total = 0.0f
    values.iterator.foreach(value => total = total + value.toFloat)
    total.toDouble

  def float64(values: IterableOnce[Double]): Double = values.iterator.sum

  /** Pairwise reduction reduces error growth from a long left-associated sum. */
  def pairwiseFloat32(values: IndexedSeq[Double]): Double =
    def sum(from: Int, until: Int): Float =
      if until <= from then 0.0f
      else if until - from == 1 then values(from).toFloat
      else
        val middle = from + (until - from) / 2
        sum(from, middle) + sum(middle, until)
    sum(0, values.size).toDouble

/** State and policy for dynamic loss scaling in mixed-precision training. */
final case class DynamicLossScaler(
    scale: Double,
    growthFactor: Double = 2.0,
    backoffFactor: Double = 0.5,
    growthInterval: Int = 2000,
    finiteSteps: Int = 0
):
  require(scale > 0.0 && scale.isFinite, s"invalid loss scale: $scale")
  require(growthFactor > 1.0 && growthFactor.isFinite, s"invalid growth factor: $growthFactor")
  require(backoffFactor > 0.0 && backoffFactor < 1.0, s"invalid backoff: $backoffFactor")
  require(growthInterval > 0, s"invalid growth interval: $growthInterval")
  require(finiteSteps >= 0 && finiteSteps < growthInterval, s"invalid finite steps: $finiteSteps")

  def scaleLoss(loss: Double): Double = loss * scale

  /** Returns unscaled gradients and the next scaler, or skips the update on NaN/Infinity. */
  def unscale(gradients: Vector[Double]): LossScaleResult =
    if gradients.exists(!_.isFinite) then
      LossScaleResult(Vector.empty, copy(scale = math.max(scale * backoffFactor, 1.0), finiteSteps = 0), skipped = true)
    else
      val nextFiniteSteps = finiteSteps + 1
      val grow = nextFiniteSteps == growthInterval
      val next = copy(
        scale = if grow then scale * growthFactor else scale,
        finiteSteps = if grow then 0 else nextFiniteSteps
      )
      LossScaleResult(gradients.map(_ / scale), next, skipped = false)

final case class LossScaleResult(gradients: Vector[Double], next: DynamicLossScaler, skipped: Boolean)

/** Prints storage, accumulation, and loss-scaling differences on hand-sized values. */
def runPrecisionLab(): Unit =
  val value = 1.0001
  FloatFormat.values.foreach(format => println(f"$format%-8s ${format.round(value)}%.8f"))
  val cancellation = Vector(100000000.0, 1.0, -100000000.0)
  println(s"naive float32 sum: ${Accumulation.naiveFloat32(cancellation)}")
  println(s"float64 sum:       ${Accumulation.float64(cancellation)}")
  val result = DynamicLossScaler(scale = 1024, growthInterval = 2).unscale(Vector(1024, 2048))
  println(s"unscaled gradients: ${result.gradients.mkString(", ")}")
