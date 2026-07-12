package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor

/**
 * Position-wise SwiGLU feed-forward network.
 *
 * The Chapter 20 feed-forward network computes `projection(relu(expansion(x)))`. SwiGLU replaces
 * the single expansion with two parallel projections — a *gate* passed through SiLU and an *up*
 * path kept linear — multiplied elementwise before the down projection:
 *
 * `down( silu(gate(x)) ⊙ up(x) )`
 *
 * The gate lets the network modulate each hidden channel multiplicatively instead of only clipping
 * it at zero, which is the mechanism the GLU -variants paper credits for its quality gain at equal
 * parameter count.
 *
 * Because SwiGLU owns three weight matrices where the ReLU network owns two, a fair ablation must
 * shrink the hidden width. Use [[SwiGluFeedForward.parameterMatchedHiddenChannels]] to compute the
 * width whose total parameter count best matches a given ReLU baseline; the usual "two-thirds" rule
 * of thumb falls out of that formula.
 *
 * SiLU itself is composed from existing verified Tensor operations through the identity
 * `x * sigmoid(x) = x/2 * (1 + tanh(x/2))`, so no new backward rule is introduced and gradient
 * checks cover the whole composition.
 */
final class SwiGluFeedForward private (
    val channels: Int,
    val hiddenChannels: Int,
    val gateProjection: Linear,
    val upProjection: Linear,
    val downProjection: Linear
):
  require(channels > 0, s"channels must be positive: $channels")
  require(hiddenChannels > 0, s"hidden channels must be positive: $hiddenChannels")

  /** Transforms every `[time, channels]` row independently. */
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"SwiGLU expected [time,$channels], got ${input.shape}"
    )
    val gated = SwiGluFeedForward.silu(gateProjection(input))
    downProjection(gated.hadamard(upProjection(input)))

  def parameters: Vector[Tensor] = gateProjection.parameters ++ upProjection.parameters ++
    downProjection.parameters

  /** Total trainable element count, used by the parameter-matched ablation. */
  def parameterCount: Int = parameters.map(_.size).sum

object SwiGluFeedForward:
  /**
   * SiLU (swish) activation: `x * sigmoid(x)`.
   *
   * Implemented as `x/2 + x/2 * tanh(x/2)` so it reuses the tested `tanh` backward rule and avoids
   * `exp` overflow for large negative inputs.
   */
  def silu(input: Tensor): Tensor =
    val half = input.scale(0.5)
    half + half.hadamard(half.tanh)

  /** Creates three Xavier-initialized projections from a caller-owned RNG. */
  def random(
      channels: Int,
      hiddenChannels: Int,
      random: SplittableRandom,
      label: String
  ): SwiGluFeedForward =
    require(channels > 0, s"channels must be positive: $channels")
    require(hiddenChannels > 0, s"hidden channels must be positive: $hiddenChannels")
    new SwiGluFeedForward(
      channels,
      hiddenChannels,
      Linear.random(channels, hiddenChannels, random, s"$label.gate"),
      Linear.random(channels, hiddenChannels, random, s"$label.up"),
      Linear.random(hiddenChannels, channels, random, s"$label.down")
    )

  /** Builds a network from explicit parameter values for deterministic tests. */
  def fromValues(
      channels: Int,
      hiddenChannels: Int,
      gateWeights: Vector[Double],
      gateBiases: Vector[Double],
      upWeights: Vector[Double],
      upBiases: Vector[Double],
      downWeights: Vector[Double],
      downBiases: Vector[Double],
      label: String
  ): SwiGluFeedForward = new SwiGluFeedForward(
    channels,
    hiddenChannels,
    Linear.fromValues(channels, hiddenChannels, gateWeights, gateBiases, s"$label.gate"),
    Linear.fromValues(channels, hiddenChannels, upWeights, upBiases, s"$label.up"),
    Linear.fromValues(hiddenChannels, channels, downWeights, downBiases, s"$label.down")
  )

  /**
   * Hidden width whose SwiGLU parameter count best matches a ReLU baseline.
   *
   * A ReLU feed-forward with hidden width `H` owns `2CH + H + C` trainable elements; SwiGLU with
   * hidden width `S` owns `3CS + 2S + C`. Equating the two gives `S = H (2C + 1) / (3C + 2)`, which
   * approaches `2H/3` as the channel count grows. The result is rounded to the nearest integer and
   * floored at one.
   */
  def parameterMatchedHiddenChannels(channels: Int, reluHiddenChannels: Int): Int =
    require(channels > 0, s"channels must be positive: $channels")
    require(reluHiddenChannels > 0, s"ReLU hidden channels must be positive: $reluHiddenChannels")
    val matched = reluHiddenChannels.toDouble * (2.0 * channels + 1.0) / (3.0 * channels + 2.0)
    math.max(1, math.round(matched).toInt)
