package learnai.transformer

import java.util.SplittableRandom
import java.util.random.RandomGenerator

import learnai.math.Categorical
import learnai.math.Probability
import learnai.math.VectorD
import learnai.lm.Sampling
import learnai.lm.SamplingConfig
import learnai.optim.AdamW
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.text.BpeTrainer
import learnai.text.TokenId
import learnai.text.TokenId.*

/** Architectural hyperparameters that determine every MiniGPT Tensor shape. */
final case class MiniGptConfig(
    vocabularySize: Int,
    maximumContextLength: Int,
    channels: Int,
    headCount: Int,
    hiddenChannels: Int,
    layerCount: Int,
    normalizationEpsilon: Double = 1e-5
):
  require(vocabularySize > 0, s"vocabulary size must be positive: $vocabularySize")
  require(maximumContextLength > 0, s"maximum context must be positive: $maximumContextLength")
  require(channels > 0, s"channels must be positive: $channels")
  require(headCount > 0, s"head count must be positive: $headCount")
  require(channels % headCount == 0, s"channels $channels not divisible by $headCount heads")
  require(hiddenChannels > 0, s"hidden channels must be positive: $hiddenChannels")
  require(layerCount > 0, s"layer count must be positive: $layerCount")
  require(
    normalizationEpsilon > 0.0 && normalizationEpsilon.isFinite,
    s"normalization epsilon must be finite and positive: $normalizationEpsilon"
  )

/** A decoder-only Transformer language model for one sequence at a time.
  *
  * The model uses learned absolute positions, pre-norm blocks, causal
  * multi-head attention, and weight tying between token embeddings and the
  * output classifier. It is intentionally small and CPU-oriented but performs
  * real reverse-mode training through every component.
  */
final class MiniGpt private (
    val config: MiniGptConfig,
    val embeddings: TokenPositionEmbedding,
    val blocks: Vector[TransformerBlock],
    val finalNorm: RmsNorm
):
  /** Computes next-token logits with shape `[time, vocabularySize]`. */
  def logits(tokenIds: Vector[TokenId]): Tensor =
    require(tokenIds.nonEmpty, "MiniGPT requires at least one input token")
    val embedded = embeddings(tokenIds)
    val hidden = blocks.foldLeft(embedded) { (current, block) => block(current) }
    val normalized = finalNorm(hidden)
    normalized.matmul(embeddings.tokens.weight.transpose2D)

  /** Computes mean causal cross entropy over one aligned sequence pair. */
  def loss(inputs: Vector[TokenId], targets: Vector[TokenId]): Tensor =
    require(inputs.nonEmpty, "MiniGPT loss requires a non-empty sequence")
    require(
      inputs.size == targets.size,
      s"input and target lengths differ: ${inputs.size} != ${targets.size}"
    )
    val targetValues = targets.zipWithIndex.map { case (tokenId, index) =>
      val value = tokenId.value
      require(
        value >= 0 && value < config.vocabularySize,
        s"target token $value at index $index outside [0, ${config.vocabularySize})"
      )
      value
    }
    logits(inputs).crossEntropy(targetValues)

  /** Computes mean cross entropy over only the unmasked target positions.
    *
    * This is the training loss for packed sequences (Chapter 17b): padding
    * targets and cross-document targets carry `false` in the mask and
    * receive no loss and no gradient. At least one position must remain
    * unmasked.
    */
  def lossMasked(
      inputs: Vector[TokenId],
      targets: Vector[TokenId],
      targetMask: Vector[Boolean]
  ): Tensor =
    require(inputs.nonEmpty, "MiniGPT loss requires a non-empty sequence")
    require(
      inputs.size == targets.size && inputs.size == targetMask.size,
      s"input/target/mask lengths differ: ${inputs.size}/${targets.size}/${targetMask.size}"
    )
    val targetValues = targets.zipWithIndex.map { case (tokenId, index) =>
      val value = tokenId.value
      require(
        value >= 0 && value < config.vocabularySize,
        s"target token $value at index $index outside [0, ${config.vocabularySize})"
      )
      value
    }
    logits(inputs).crossEntropyMasked(targetValues, targetMask)

  /** Returns the distribution after the final token of a non-empty context.
    *
    * Contexts longer than the configured maximum are cropped from the left.
    * Position IDs restart at zero for the retained sliding window.
    */
  def nextDistribution(
      context: Vector[TokenId],
      temperature: Double = 1.0
  ): Either[String, Categorical] =
    require(context.nonEmpty, "next-token prediction requires a non-empty context")
    require(
      temperature > 0.0 && temperature.isFinite,
      s"temperature must be finite and positive: $temperature"
    )
    val retained = context.takeRight(config.maximumContextLength)
    val output = logits(retained)
    val finalRow = VectorD.from(output.rowValues(retained.size - 1)).scale(1.0 / temperature)
    Probability.softmax(finalRow)

  /** Autoregressively samples tokens and returns the prompt plus generated suffix. */
  def generate(
      prompt: Vector[TokenId],
      newTokenCount: Int,
      temperature: Double,
      random: RandomGenerator
  ): Either[String, Vector[TokenId]] =
    require(prompt.nonEmpty, "generation requires a non-empty prompt")
    require(newTokenCount >= 0, s"new token count must be non-negative: $newTokenCount")
    var generated = prompt
    var remaining = newTokenCount
    var error: Option[String] = None
    while remaining > 0 && error.isEmpty do
      nextDistribution(generated, temperature) match
        case Left(message) => error = Some(message)
        case Right(distribution) =>
          generated = generated :+ TokenId(distribution.sample(random))
          remaining -= 1
    error match
      case Some(message) => Left(message)
      case None          => Right(generated)

  /** Samples with one KV cache per block and returns deterministic work metrics.
    *
    * Within the context capacity, each generated token adds one model token
    * evaluation instead of recomputing the full prefix. If a learned absolute-
    * position window becomes full, the retained window is rebuilt with
    * positions starting at zero so results remain equivalent to
    * `nextDistribution` rather than silently changing positional semantics.
    */
  def generateCached(
      prompt: Vector[TokenId],
      newTokenCount: Int,
      sampling: SamplingConfig,
      random: RandomGenerator
  ): Either[String, CachedGenerationResult] =
    require(prompt.nonEmpty, "cached generation requires a non-empty prompt")
    require(newTokenCount >= 0, s"new token count must be non-negative: $newTokenCount")

    val session = new MiniGptInferenceSession(this)
    if newTokenCount == 0 then
      Right(
        CachedGenerationResult(
          prompt,
          CachedGenerationStatistics(0L, 0L, 0, 0, session.allocatedCachePayloadBytes)
        )
      )
    else
      var generated = prompt
      var activeContext = prompt.takeRight(config.maximumContextLength)
      var nextLogits = session.prefill(activeContext)
      var remaining = newTokenCount
      var referenceTokenEvaluations = 0L
      var cacheRebuilds = 0
      var peakCachedTokens = session.length
      var error: Option[String] = None

      while remaining > 0 && error.isEmpty do
        referenceTokenEvaluations += math.min(generated.size, config.maximumContextLength).toLong
        Sampling.sample(nextLogits, sampling, random) match
          case Left(message) => error = Some(message)
          case Right(nextToken) =>
            generated :+= nextToken
            remaining -= 1
            if remaining > 0 then
              if session.length < config.maximumContextLength then
                activeContext :+= nextToken
                nextLogits = session.append(nextToken)
              else
                activeContext = (activeContext :+ nextToken).takeRight(config.maximumContextLength)
                nextLogits = session.rebuild(activeContext)
                cacheRebuilds += 1
              peakCachedTokens = math.max(peakCachedTokens, session.length)

      error match
        case Some(message) => Left(message)
        case None =>
          Right(
            CachedGenerationResult(
              generated,
              CachedGenerationStatistics(
                session.evaluatedTokens,
                referenceTokenEvaluations,
                cacheRebuilds,
                peakCachedTokens,
                session.allocatedCachePayloadBytes
              )
            )
          )

  /** Every owned trainable Tensor exactly once; the tied logit head adds none. */
  def parameters: Vector[Tensor] =
    embeddings.parameters ++ blocks.flatMap(_.parameters) ++ finalNorm.parameters

  def parameterCount: Int = parameters.map(_.size).sum

object MiniGpt:
  /** Initializes all parameter Tensors from one deterministic random stream. */
  def random(config: MiniGptConfig, seed: Long): MiniGpt =
    val random = new SplittableRandom(seed)
    val tokenEmbedding = Embedding.random(
      config.vocabularySize,
      config.channels,
      random,
      "embeddings.token"
    )
    val positionEmbedding = Embedding.random(
      config.maximumContextLength,
      config.channels,
      random,
      "embeddings.position"
    )
    val blocks = Vector.tabulate(config.layerCount) { index =>
      TransformerBlock.random(
        config.channels,
        config.headCount,
        config.hiddenChannels,
        config.normalizationEpsilon,
        random,
        s"blocks.$index"
      )
    }
    new MiniGpt(
      config,
      new TokenPositionEmbedding(tokenEmbedding, positionEmbedding),
      blocks,
      RmsNorm.create(config.channels, config.normalizationEpsilon, "finalNorm")
    )

object MiniGptTrainer:
  /** Trains one fixed sequence with full-sequence AdamW updates.
    *
    * This helper is deliberately small; dataset sampling and validation loops
    * remain explicit exercises. Returned loss values are recorded before each
    * update.
    */
  def trainSequence(
      model: MiniGpt,
      inputs: Vector[TokenId],
      targets: Vector[TokenId],
      steps: Int,
      learningRate: Double
  ): Vector[Double] =
    require(steps >= 0, s"training steps must be non-negative: $steps")
    val optimizer = new AdamW(
      learningRate = learningRate,
      weightDecay = 0.0,
      maximumGradientNorm = Some(1.0)
    )
    val history = Vector.newBuilder[Double]
    var step = 0
    while step < steps do
      val loss = model.loss(inputs, targets)
      history += loss.valueAtFlat(0)
      loss.backward()
      val _ = optimizer.step(model.parameters)
      step += 1
    history.result()

@main def trainMiniGpt(): Unit =
  val corpus = Vector.fill(20)("to be or not to be. ").mkString
  val tokenizer = BpeTrainer.train(Vector(corpus), targetVocabularySize = 264)
  val tokens = tokenizer.encode(corpus)
  val contextLength = 12
  val window = tokens.take(contextLength + 1)
  val config = MiniGptConfig(
    vocabularySize = tokenizer.vocabularySize,
    maximumContextLength = contextLength,
    channels = 12,
    headCount = 3,
    hiddenChannels = 24,
    layerCount = 1
  )
  val model = MiniGpt.random(config, seed = 42L)
  val history = MiniGptTrainer.trainSequence(
    model,
    window.dropRight(1),
    window.drop(1),
    steps = 120,
    learningRate = 0.03
  )
  println(s"parameters:   ${model.parameterCount}")
  println(f"initial loss: ${history.head}%.6f")
  println(f"final loss:   ${history.last}%.6f")

  val generated = model.generate(
    prompt = window.take(2),
    newTokenCount = 30,
    temperature = 0.5,
    random = new SplittableRandom(7L)
  )
  generated.flatMap(tokenizer.decode) match
    case Right(text)   => println(s"generated: $text")
    case Left(problem) => println(s"decode failed: $problem")
