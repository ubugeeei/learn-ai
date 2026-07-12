package learnai.transformer

/**
 * Decoder-only GPT architecture values needed to reproduce parameter inventory.
 *
 * This contract follows GPT-2/nanoGPT naming: learned token and position embeddings, pre-LayerNorm
 * blocks, biased QKV/output projections, a 4x GELU MLP, final LayerNorm, and an output head tied to
 * token embeddings.
 */
final case class Gpt2Config(
    vocabularySize: Int,
    contextLength: Int,
    channels: Int,
    headCount: Int,
    layerCount: Int
):
  require(vocabularySize > 0, s"vocabulary size must be positive: $vocabularySize")
  require(contextLength > 0, s"context length must be positive: $contextLength")
  require(channels > 0, s"channels must be positive: $channels")
  require(
    headCount > 0 && channels % headCount == 0,
    s"$headCount heads must divide $channels channels"
  )
  require(layerCount > 0, s"layer count must be positive: $layerCount")

  val hiddenChannels: Int = Math.multiplyExact(channels, 4)

/** Closed-form GPT-2-compatible parameter groups, retaining tied-weight ownership. */
final case class Gpt2ParameterInventory(
    tokenEmbedding: Long,
    positionEmbedding: Long,
    attentionPerLayer: Long,
    feedForwardPerLayer: Long,
    normsPerLayer: Long,
    finalNorm: Long,
    layers: Int
):
  val perLayer: Long = Math
    .addExact(Math.addExact(attentionPerLayer, feedForwardPerLayer), normsPerLayer)
  val total: Long    = Math.addExact(
    Math.addExact(tokenEmbedding, positionEmbedding),
    Math.addExact(Math.multiplyExact(perLayer, layers.toLong), finalNorm)
  )

/** Published GPT-family reference configurations and compatibility accounting. */
object GptLineage:
  val Gpt2Small: Gpt2Config  = Gpt2Config(50257, 1024, 768, 12, 12)
  val Gpt2Medium: Gpt2Config = Gpt2Config(50257, 1024, 1024, 16, 24)
  val Gpt2Large: Gpt2Config  = Gpt2Config(50257, 1024, 1280, 20, 36)
  val Gpt2Xl: Gpt2Config     = Gpt2Config(50257, 1024, 1600, 25, 48)

  /** Counts GPT-2 tensors exactly, including biases and counting tied embeddings once. */
  def parameterInventory(config: Gpt2Config): Gpt2ParameterInventory =
    val channels          = config.channels.toLong
    val hidden            = config.hiddenChannels.toLong
    val tokenEmbedding    = Math.multiplyExact(config.vocabularySize.toLong, channels)
    val positionEmbedding = Math.multiplyExact(config.contextLength.toLong, channels)
    val qkv               = Math.addExact(Math.multiplyExact(channels, 3L * channels), 3L * channels)
    val attentionOutput   = Math.addExact(Math.multiplyExact(channels, channels), channels)
    val expansion         = Math.addExact(Math.multiplyExact(channels, hidden), hidden)
    val projection        = Math.addExact(Math.multiplyExact(hidden, channels), channels)
    Gpt2ParameterInventory(
      tokenEmbedding,
      positionEmbedding,
      Math.addExact(qkv, attentionOutput),
      Math.addExact(expansion, projection),
      4L * channels,
      2L * channels,
      config.layerCount
    )

  /** Architectural differences that prevent a MiniGPT checkpoint from being a GPT-2 checkpoint. */
  def miniGptCompatibilityDifferences: Vector[String] = Vector(
    "MiniGPT uses RMSNorm; GPT-2 uses LayerNorm with learned scale and bias",
    "MiniGPT feed-forward uses ReLU; GPT-2 uses approximate GELU",
    "MiniGPT initializes with Xavier; GPT-2 uses its published normal/residual scaling recipe",
    "MiniGPT tokenizer/checkpoint formats are not GPT-2 byte-level BPE and TensorFlow/PyTorch formats",
    "MiniGPT does not implement GPT-2 dropout or pretrained weight-name import"
  )

/** Prints reference GPT-2 sizes and the exact reasons MiniGPT is not checkpoint compatible. */
def runGptLineageLab(): Unit =
  Vector(
    "gpt2"        -> GptLineage.Gpt2Small,
    "gpt2-medium" -> GptLineage.Gpt2Medium,
    "gpt2-large"  -> GptLineage.Gpt2Large,
    "gpt2-xl"     -> GptLineage.Gpt2Xl
  ).foreach { case (name, config) =>
    println(f"$name%-12s ${GptLineage.parameterInventory(config).total}%,d parameters")
  }
  println("\nWhy MiniGPT is not GPT-2 checkpoint compatible:")
  GptLineage.miniGptCompatibilityDifferences.foreach(value => println(s"- $value"))
