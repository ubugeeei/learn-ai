package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor

/** Position-wise two-layer feed-forward network.
  *
  * Every time row is transformed independently with shared parameters:
  * `[time, channels] -> [time, hiddenChannels] -> [time, channels]`.
  */
final class FeedForward private (
    val channels: Int,
    val hiddenChannels: Int,
    val expansion: Linear,
    val projection: Linear
):
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"feed-forward expected [time,$channels], got ${input.shape}"
    )
    projection(expansion(input).relu)

  def parameters: Vector[Tensor] = expansion.parameters ++ projection.parameters

object FeedForward:
  /** Creates Xavier-initialized expansion and projection layers. */
  def random(
      channels: Int,
      hiddenChannels: Int,
      random: SplittableRandom,
      label: String
  ): FeedForward =
    require(channels > 0, s"channels must be positive: $channels")
    require(hiddenChannels > 0, s"hidden channels must be positive: $hiddenChannels")
    new FeedForward(
      channels,
      hiddenChannels,
      Linear.random(channels, hiddenChannels, random, s"$label.expansion"),
      Linear.random(hiddenChannels, channels, random, s"$label.projection")
    )

/** One pre-normalized decoder-only Transformer block.
  *
  * The block preserves `[time, channels]` shape and contains two residual
  * branches: causal self-attention and a position-wise feed-forward network.
  */
final class TransformerBlock private (
    val channels: Int,
    val attentionNorm: RmsNorm,
    val attention: CausalSelfAttention,
    val feedForwardNorm: RmsNorm,
    val feedForward: FeedForward
):
  /** Applies pre-norm attention and feed-forward residual updates. */
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"Transformer block expected [time,$channels], got ${input.shape}"
    )
    val afterAttention = input + attention(attentionNorm(input))
    afterAttention + feedForward(feedForwardNorm(afterAttention))

  /** Returns every trainable Tensor exactly once in forward ownership order. */
  def parameters: Vector[Tensor] =
    attentionNorm.parameters ++
      attention.parameters ++
      feedForwardNorm.parameters ++
      feedForward.parameters

object TransformerBlock:
  /** Creates one reproducible pre-norm block. */
  def random(
      channels: Int,
      headCount: Int,
      hiddenChannels: Int,
      epsilon: Double,
      random: SplittableRandom,
      label: String
  ): TransformerBlock =
    require(channels > 0, s"channels must be positive: $channels")
    require(hiddenChannels > 0, s"hidden channels must be positive: $hiddenChannels")
    new TransformerBlock(
      channels,
      RmsNorm.create(channels, epsilon, s"$label.attentionNorm"),
      CausalSelfAttention.random(channels, headCount, random, s"$label.attention"),
      RmsNorm.create(channels, epsilon, s"$label.feedForwardNorm"),
      FeedForward.random(channels, hiddenChannels, random, s"$label.feedForward")
    )
