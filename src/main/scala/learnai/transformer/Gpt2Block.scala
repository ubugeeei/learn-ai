package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor

/** GPT-2 position-wise four-times expansion with approximate GELU. */
final class Gpt2FeedForward private (
    val channels: Int,
    val expansion: Linear,
    val projection: Linear
):
  def apply(input: Tensor): Tensor = projection(expansion(input).geluApprox)

  def parameters: Vector[Tensor] = expansion.parameters ++ projection.parameters

object Gpt2FeedForward:
  def random(channels: Int, random: SplittableRandom, label: String): Gpt2FeedForward =
    require(channels > 0, s"channels must be positive: $channels")
    new Gpt2FeedForward(
      channels,
      Linear.random(channels, Math.multiplyExact(channels, 4), random, s"$label.c_fc"),
      Linear.random(Math.multiplyExact(channels, 4), channels, random, s"$label.c_proj")
    )

  /** Builds the GPT-2 MLP from checkpoint-compatible affine projections. */
  def fromProjections(channels: Int, expansion: Linear, projection: Linear): Gpt2FeedForward =
    require(
      expansion.inputChannels == channels && expansion.outputChannels == 4 * channels,
      s"expansion must be [$channels,${4 * channels}]"
    )
    require(
      projection.inputChannels == 4 * channels && projection.outputChannels == channels,
      s"projection must be [${4 * channels},$channels]"
    )
    new Gpt2FeedForward(channels, expansion, projection)

/** One GPT-2 pre-LayerNorm causal decoder block with residual branches. */
final class Gpt2Block private (
    val channels: Int,
    val attentionNorm: LayerNorm,
    val attention: CausalSelfAttention,
    val feedForwardNorm: LayerNorm,
    val feedForward: Gpt2FeedForward
):
  def apply(input: Tensor): Tensor =
    val afterAttention = input + attention(attentionNorm(input))
    afterAttention + feedForward(feedForwardNorm(afterAttention))

  def parameters: Vector[Tensor] = attentionNorm.parameters ++ attention.parameters ++
    feedForwardNorm.parameters ++ feedForward.parameters

object Gpt2Block:
  def random(
      channels: Int,
      headCount: Int,
      epsilon: Double,
      random: SplittableRandom,
      label: String
  ): Gpt2Block = new Gpt2Block(
    channels,
    LayerNorm.create(channels, epsilon, s"$label.ln_1"),
    CausalSelfAttention.random(channels, headCount, random, s"$label.attn"),
    LayerNorm.create(channels, epsilon, s"$label.ln_2"),
    Gpt2FeedForward.random(channels, random, s"$label.mlp")
  )

  /** Assembles one block from validated checkpoint-owned components. */
  def fromComponents(
      channels: Int,
      attentionNorm: LayerNorm,
      attention: CausalSelfAttention,
      feedForwardNorm: LayerNorm,
      feedForward: Gpt2FeedForward
  ): Gpt2Block =
    require(attentionNorm.channels == channels, "attention LayerNorm channel mismatch")
    require(attention.channels == channels, "attention channel mismatch")
    require(feedForwardNorm.channels == channels, "MLP LayerNorm channel mismatch")
    require(feedForward.channels == channels, "MLP channel mismatch")
    new Gpt2Block(channels, attentionNorm, attention, feedForwardNorm, feedForward)
