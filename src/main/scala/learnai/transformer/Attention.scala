package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor

/** Attention output plus the normalized causal weights of every head. */
final case class AttentionResult(output: Tensor, weightsByHead: Vector[Tensor])

/** Multi-head causal self-attention for one sequence.
  *
  * Input and output shapes are `[time, channels]`. Channels are split evenly
  * across heads. Each head owns a different channel slice of the shared Q/K/V
  * projections, then all head outputs are concatenated and projected.
  */
final class CausalSelfAttention private (
    val channels: Int,
    val headCount: Int,
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

  /** Runs causal attention and returns only the projected hidden states. */
  def apply(input: Tensor): Tensor = forwardWithWeights(input).output

  /** Runs attention while retaining per-head weights for inspection and tests. */
  def forwardWithWeights(input: Tensor): AttentionResult =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"attention expected [time,$channels], got ${input.shape}"
    )
    require(input.shape(0) > 0, "attention requires at least one time position")
    val query = queryProjection(input)
    val key = keyProjection(input)
    val value = valueProjection(input)
    val scale = 1.0 / math.sqrt(headChannels.toDouble)

    val headResults = Vector.tabulate(headCount) { head =>
      val from = head * headChannels
      val until = from + headChannels
      val headQuery = query.sliceColumns(from, until)
      val headKey = key.sliceColumns(from, until)
      val headValue = value.sliceColumns(from, until)
      val scores = headQuery.matmul(headKey.transpose2D).scale(scale)
      val weights = scores.causalMask().softmaxRows
      weights -> weights.matmul(headValue)
    }
    val weights = headResults.map(_._1)
    val concatenated = Tensor.concatenateColumns(headResults.map(_._2))
    AttentionResult(outputProjection(concatenated), weights)

  def parameters: Vector[Tensor] =
    queryProjection.parameters ++
      keyProjection.parameters ++
      valueProjection.parameters ++
      outputProjection.parameters

object CausalSelfAttention:
  /** Creates four reproducibly initialized dense projections. */
  def random(
      channels: Int,
      headCount: Int,
      random: SplittableRandom,
      label: String
  ): CausalSelfAttention =
    require(headCount > 0, s"head count must be positive: $headCount")
    require(channels % headCount == 0, s"channels $channels not divisible by $headCount heads")
    new CausalSelfAttention(
      channels,
      headCount,
      Linear.random(channels, channels, random, s"$label.query"),
      Linear.random(channels, channels, random, s"$label.key"),
      Linear.random(channels, channels, random, s"$label.value"),
      Linear.random(channels, channels, random, s"$label.output")
    )
