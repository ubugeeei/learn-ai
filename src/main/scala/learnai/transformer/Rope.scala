package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor

/**
 * Rotary position encoding (RoPE) over one attention head's channels.
 *
 * RoPE encodes position by rotating channel pairs instead of adding a learned position vector. Pair
 * `j` of a row at absolute position `m` is rotated by the angle `m * theta_j`, where
 * `theta_j = base^(-2j / headChannels)`. Because a rotation is orthogonal, the dot product between
 * a query rotated at position `m` and a key rotated at position `n` depends only on the relative
 * offset `m - n`. Attention scores therefore become translation-invariant: shifting an entire
 * window by a constant offset does not change any score.
 *
 * This implementation uses the half-split pair layout: pair `j` is the channel pair
 * `(j, j + headChannels / 2)`. The original paper interleaves pairs as `(2j, 2j + 1)`; the two
 * layouts differ only by a fixed channel permutation, so they are equivalent as long as queries and
 * keys agree.
 *
 * The rotation is composed entirely from existing differentiable Tensor operations (`sliceColumns`,
 * `hadamard`, addition, and column concatenation) with cosine/sine tables as constants, so
 * gradients flow to the rotated input without any new backward rule. RoPE adds zero trainable
 * parameters.
 */
final class RotaryPositionEncoding private (val headChannels: Int, val base: Double):
  require(headChannels > 0, s"head channels must be positive: $headChannels")
  require(headChannels % 2 == 0, s"head channels must be even for pairing: $headChannels")
  require(base > 0.0 && base.isFinite, s"frequency base must be finite and positive: $base")

  /** Number of independently rotated channel pairs. */
  val pairCount: Int = headChannels / 2

  /** Per-pair rotation frequency `theta_j = base^(-2j / headChannels)`. */
  val frequencies: Vector[Double] = Vector
    .tabulate(pairCount)(pair => math.pow(base, -2.0 * pair.toDouble / headChannels.toDouble))

  /** Rotation angle applied to one pair at one absolute position. */
  def angle(position: Int, pair: Int): Double =
    require(position >= 0, s"position must be non-negative: $position")
    require(pair >= 0 && pair < pairCount, s"pair $pair outside [0, $pairCount)")
    position.toDouble * frequencies(pair)

  /**
   * Rotates every row of a `[time, headChannels]` tensor by its position.
   *
   * Row `t` is treated as absolute position `startPosition + t`, so a cached decoder can rotate one
   * appended row without recomputing the past. The output keeps the input shape, preserves each
   * pair's Euclidean norm, and participates in reverse-mode differentiation.
   */
  def rotate(input: Tensor, startPosition: Int = 0): Tensor =
    require(
      input.rank == 2 && input.shape(1) == headChannels,
      s"RoPE expected [time,$headChannels], got ${input.shape}"
    )
    require(startPosition >= 0, s"start position must be non-negative: $startPosition")
    val time = input.shape(0)
    require(time > 0, "RoPE requires at least one time position")
    val _    = Math.addExact(startPosition, time - 1)

    val cosTable      = table(time, startPosition, math.cos)
    val sinTable      = table(time, startPosition, math.sin)
    val firstHalf     = input.sliceColumns(0, pairCount)
    val secondHalf    = input.sliceColumns(pairCount, headChannels)
    val rotatedFirst  = firstHalf.hadamard(cosTable) - secondHalf.hadamard(sinTable)
    val rotatedSecond = firstHalf.hadamard(sinTable) + secondHalf.hadamard(cosTable)
    Tensor.concatenateColumns(Vector(rotatedFirst, rotatedSecond))

  /** Builds a `[time, pairCount]` constant trigonometric table. */
  private def table(time: Int, startPosition: Int, function: Double => Double): Tensor = Tensor
    .tabulate(Shape(time, pairCount)) { flatIndex =>
      val row  = flatIndex / pairCount
      val pair = flatIndex % pairCount
      function(angle(startPosition + row, pair))
    }

object RotaryPositionEncoding:
  /** Conventional frequency base from the RoFormer paper. */
  val DefaultBase: Double = 10000.0

  def create(headChannels: Int, base: Double = DefaultBase): RotaryPositionEncoding =
    new RotaryPositionEncoding(headChannels, base)

/**
 * Multi-head causal self-attention with rotary position encoding.
 *
 * The layer mirrors [[CausalSelfAttention]] but replaces learned absolute position embeddings with
 * RoPE: each head's query and key slices are rotated by their absolute position before scores are
 * computed. Values are never rotated, because position must influence *where* attention looks, not
 * *what* it retrieves.
 *
 * Consequences verified by the test suite:
 *   - the layer has exactly the same trainable parameters as standard attention (RoPE is
 *     parameter-free);
 *   - the full output is invariant under a common position offset, which is the property that lets
 *     a sliding KV cache avoid the absolute-position rebuild required in Chapter 24.
 */
final class RotaryCausalSelfAttention private (
    val channels: Int,
    val headCount: Int,
    val rope: RotaryPositionEncoding,
    val queryProjection: Linear,
    val keyProjection: Linear,
    val valueProjection: Linear,
    val outputProjection: Linear
):
  require(channels > 0, s"attention channels must be positive: $channels")
  require(headCount > 0, s"head count must be positive: $headCount")
  require(
    channels % headCount == 0,
    s"channels $channels must be divisible by head count $headCount"
  )

  val headChannels: Int = channels / headCount
  require(
    rope.headChannels == headChannels,
    s"RoPE head channels ${rope.headChannels} do not match attention head channels $headChannels"
  )

  /** Runs rotary causal attention and returns only the hidden states. */
  def apply(input: Tensor, startPosition: Int = 0): Tensor =
    forwardWithWeights(input, startPosition).output

  /**
   * Runs attention while retaining per-head weights for inspection and tests.
   *
   * Row `t` of the input is treated as absolute position `startPosition + t`. Scores use rotated
   * queries and keys; the value aggregation and output projection are identical to standard
   * attention.
   */
  def forwardWithWeights(input: Tensor, startPosition: Int = 0): AttentionResult =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"rotary attention expected [time,$channels], got ${input.shape}"
    )
    require(input.shape(0) > 0, "rotary attention requires at least one time position")
    require(startPosition >= 0, s"start position must be non-negative: $startPosition")
    val query = queryProjection(input)
    val key   = keyProjection(input)
    val value = valueProjection(input)
    val scale = 1.0 / math.sqrt(headChannels.toDouble)

    val headResults  = Vector.tabulate(headCount) { head =>
      val from      = head * headChannels
      val until     = from + headChannels
      val headQuery = rope.rotate(query.sliceColumns(from, until), startPosition)
      val headKey   = rope.rotate(key.sliceColumns(from, until), startPosition)
      val headValue = value.sliceColumns(from, until)
      val scores    = headQuery.matmul(headKey.transpose2D).scale(scale)
      val weights   = scores.causalMask().softmaxRows
      weights -> weights.matmul(headValue)
    }
    val weights      = headResults.map(_._1)
    val concatenated = Tensor.concatenateColumns(headResults.map(_._2))
    AttentionResult(outputProjection(concatenated), weights)

  def parameters: Vector[Tensor] = queryProjection.parameters ++ keyProjection.parameters ++
    valueProjection.parameters ++ outputProjection.parameters

object RotaryCausalSelfAttention:
  /** Creates four reproducibly initialized projections plus a parameter-free RoPE. */
  def random(
      channels: Int,
      headCount: Int,
      random: SplittableRandom,
      label: String,
      base: Double = RotaryPositionEncoding.DefaultBase
  ): RotaryCausalSelfAttention =
    require(headCount > 0, s"head count must be positive: $headCount")
    require(channels % headCount == 0, s"channels $channels not divisible by $headCount heads")
    new RotaryCausalSelfAttention(
      channels,
      headCount,
      RotaryPositionEncoding.create(channels / headCount, base),
      Linear.random(channels, channels, random, s"$label.query"),
      Linear.random(channels, channels, random, s"$label.key"),
      Linear.random(channels, channels, random, s"$label.value"),
      Linear.random(channels, channels, random, s"$label.output")
    )
