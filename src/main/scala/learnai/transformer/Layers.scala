package learnai.transformer

import java.util.SplittableRandom

import learnai.optim.Initialization
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.text.TokenId
import learnai.text.TokenId.*

/**
 * A trainable lookup table with shape `[entries, channels]`.
 *
 * Lookup output has shape `[indices.size, channels]`. Repeated indices share one parameter row and
 * therefore accumulate gradients.
 */
final class Embedding private (val entries: Int, val channels: Int, val weight: Tensor):
  require(entries > 0, s"embedding entries must be positive: $entries")
  require(channels > 0, s"embedding channels must be positive: $channels")
  require(
    weight.shape == Shape(entries, channels),
    s"invalid embedding weight shape ${weight.shape}"
  )

  /** Looks up integer row indices after validating their vocabulary range. */
  def apply(indices: Vector[Int]): Tensor =
    indices.zipWithIndex.foreach { case (index, position) =>
      require(
        index >= 0 && index < entries,
        s"embedding index $index at position $position outside [0, $entries)"
      )
    }
    weight.gatherRows(indices)

  def parameters: Vector[Tensor] = Vector(weight)

object Embedding:
  /** Xavier-initializes an embedding table using a caller-owned RNG. */
  def random(entries: Int, channels: Int, random: SplittableRandom, label: String): Embedding =
    val weight = Initialization
      .xavierUniform(Shape(entries, channels), fanIn = entries, fanOut = channels, random, label)
    new Embedding(entries, channels, weight)

  /** Builds an embedding from explicit row-major values for tests or loading. */
  def fromValues(entries: Int, channels: Int, values: Vector[Double], label: String): Embedding =
    new Embedding(entries, channels, Tensor.parameter(Shape(entries, channels), values, label))

/** Combines learned token and absolute-position embeddings by addition. */
final class TokenPositionEmbedding(val tokens: Embedding, val positions: Embedding):
  require(
    tokens.channels == positions.channels,
    s"token and position channels differ: ${tokens.channels} != ${positions.channels}"
  )

  val channels: Int             = tokens.channels
  val maximumContextLength: Int = positions.entries

  /** Returns `[time, channels]` embeddings for one token sequence. */
  def apply(tokenIds: Vector[TokenId]): Tensor =
    require(
      tokenIds.size <= maximumContextLength,
      s"sequence length ${tokenIds.size} exceeds maximum $maximumContextLength"
    )
    val tokenValues    = tokenIds.map(_.value)
    val positionValues = tokenIds.indices.toVector
    tokens(tokenValues) + positions(positionValues)

  /** Embeds one token at an explicit absolute position for cached decoding. */
  def at(tokenId: TokenId, position: Int): Tensor =
    require(
      position >= 0 && position < maximumContextLength,
      s"position $position outside [0, $maximumContextLength)"
    )
    tokens(Vector(tokenId.value)) + positions(Vector(position))

  def parameters: Vector[Tensor] = tokens.parameters ++ positions.parameters

/** A dense affine transform over the final channel axis of a rank-2 input. */
final class Linear private (
    val inputChannels: Int,
    val outputChannels: Int,
    val weight: Tensor,
    val bias: Tensor
):
  require(
    weight.shape == Shape(inputChannels, outputChannels),
    s"invalid weight shape ${weight.shape}"
  )
  require(bias.shape == Shape(outputChannels), s"invalid bias shape ${bias.shape}")

  /** Applies `[rows,inputChannels] x [inputChannels,outputChannels] + bias`. */
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == inputChannels,
      s"linear expected [rows,$inputChannels], got ${input.shape}"
    )
    input.matmul(weight).addRowVector(bias)

  def parameters: Vector[Tensor] = Vector(weight, bias)

object Linear:
  /** Creates a Xavier-initialized weight and zero bias. */
  def random(
      inputChannels: Int,
      outputChannels: Int,
      random: SplittableRandom,
      label: String
  ): Linear =
    require(inputChannels > 0, s"input channels must be positive: $inputChannels")
    require(outputChannels > 0, s"output channels must be positive: $outputChannels")
    val weight = Initialization.xavierUniform(
      Shape(inputChannels, outputChannels),
      inputChannels,
      outputChannels,
      random,
      s"$label.weight"
    )
    val bias   = Initialization.zeros(Shape(outputChannels), s"$label.bias")
    new Linear(inputChannels, outputChannels, weight, bias)

  /** Builds a layer from explicit parameter values for deterministic tests. */
  def fromValues(
      inputChannels: Int,
      outputChannels: Int,
      weights: Vector[Double],
      biases: Vector[Double],
      label: String
  ): Linear = new Linear(
    inputChannels,
    outputChannels,
    Tensor.parameter(Shape(inputChannels, outputChannels), weights, s"$label.weight"),
    Tensor.parameter(Shape(outputChannels), biases, s"$label.bias")
  )

/** Root-mean-square normalization with one learned scale per channel. */
final class RmsNorm private (val channels: Int, val epsilon: Double, val scale: Tensor):
  require(channels > 0, s"RMSNorm channels must be positive: $channels")
  require(scale.shape == Shape(channels), s"invalid RMSNorm scale shape ${scale.shape}")

  /** Normalizes each `[channels]` row independently. */
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"RMSNorm expected [rows,$channels], got ${input.shape}"
    )
    input.rmsNormRows(scale, epsilon)

  def parameters: Vector[Tensor] = Vector(scale)

object RmsNorm:
  /** Initializes every learned scale to one, preserving initial magnitude. */
  def create(channels: Int, epsilon: Double, label: String): RmsNorm =
    val scale = Tensor.parameter(Shape(channels), Vector.fill(channels)(1.0), s"$label.scale")
    new RmsNorm(channels, epsilon, scale)

/** GPT-2 LayerNorm with learned scale and bias for every channel. */
final class LayerNorm private (
    val channels: Int,
    val epsilon: Double,
    val scale: Tensor,
    val bias: Tensor
):
  /** Normalizes each input row across channels. */
  def apply(input: Tensor): Tensor = input.layerNormRows(scale, bias, epsilon)

  def parameters: Vector[Tensor] = Vector(scale, bias)

object LayerNorm:
  /** Creates identity-affine LayerNorm parameters. */
  def create(channels: Int, epsilon: Double, label: String): LayerNorm =
    require(channels > 0, s"LayerNorm channels must be positive: $channels")
    require(epsilon > 0.0 && epsilon.isFinite, s"invalid LayerNorm epsilon: $epsilon")
    new LayerNorm(
      channels,
      epsilon,
      Tensor.parameter(Shape(channels), Vector.fill(channels)(1.0), s"$label.weight"),
      Tensor.parameter(Shape(channels), Vector.fill(channels)(0.0), s"$label.bias")
    )

  /** Builds LayerNorm from checkpoint values after validating both channel vectors. */
  def fromValues(
      channels: Int,
      epsilon: Double,
      weights: Vector[Double],
      biases: Vector[Double],
      label: String
  ): LayerNorm =
    require(weights.size == channels, s"LayerNorm weight has ${weights.size}, expected $channels")
    require(biases.size == channels, s"LayerNorm bias has ${biases.size}, expected $channels")
    require(epsilon > 0.0 && epsilon.isFinite, s"invalid LayerNorm epsilon: $epsilon")
    new LayerNorm(
      channels,
      epsilon,
      Tensor.parameter(Shape(channels), weights, s"$label.weight"),
      Tensor.parameter(Shape(channels), biases, s"$label.bias")
    )
