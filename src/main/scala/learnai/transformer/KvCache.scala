package learnai.transformer

import learnai.math.VectorD
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.text.TokenId

/**
 * Fixed-capacity key/value storage for one self-attention layer.
 *
 * Keys and values use row-major `[capacity, channels]` arrays. Appending one token writes exactly
 * one row to each array; previous rows are never projected again. The cache is mutable because it
 * belongs to one inference session, not to the model or its trainable parameters.
 *
 * `allocatedPayloadBytes` and `usedPayloadBytes` count only `Double` array payloads. JVM object and
 * array headers are deliberately excluded.
 */
final class AttentionKeyValueCache private (val channels: Int, val capacity: Int):
  private val keys          = new Array[Double](Math.multiplyExact(channels, capacity))
  private val values        = new Array[Double](Math.multiplyExact(channels, capacity))
  private var currentLength = 0

  /** Number of token rows currently available to attention. */
  def length: Int = currentLength

  def remainingCapacity: Int = capacity - currentLength

  def allocatedPayloadBytes: Long = 2L * channels.toLong * capacity.toLong *
    java.lang.Double.BYTES.toLong

  def usedPayloadBytes: Long = 2L * channels.toLong * currentLength.toLong *
    java.lang.Double.BYTES.toLong

  /** Removes logical contents without reallocating the fixed backing arrays. */
  def clear(): Unit = currentLength = 0

  private[transformer] def append(key: Tensor, value: Tensor): Unit =
    require(
      key.shape == Shape(1, channels),
      s"cached key must have shape [1,$channels], got ${key.shape}"
    )
    require(
      value.shape == Shape(1, channels),
      s"cached value must have shape [1,$channels], got ${value.shape}"
    )
    require(currentLength < capacity, s"KV cache capacity $capacity is exhausted")
    val offset      = currentLength * channels
    val keyValues   = key.values
    val valueValues = value.values
    var channel     = 0
    while channel < channels do
      keys(offset + channel) = keyValues(channel)
      values(offset + channel) = valueValues(channel)
      channel += 1
    currentLength += 1

  private[transformer] def keyAt(position: Int, channel: Int): Double =
    requirePositionAndChannel(position, channel)
    keys(position * channels + channel)

  private[transformer] def valueAt(position: Int, channel: Int): Double =
    requirePositionAndChannel(position, channel)
    values(position * channels + channel)

  private def requirePositionAndChannel(position: Int, channel: Int): Unit =
    require(
      position >= 0 && position < currentLength,
      s"cache position $position outside [0, $currentLength)"
    )
    require(channel >= 0 && channel < channels, s"cache channel $channel outside [0, $channels)")

object AttentionKeyValueCache:
  /** Allocates empty key and value arrays for one attention layer. */
  def create(channels: Int, capacity: Int): AttentionKeyValueCache =
    require(channels > 0, s"cache channels must be positive: $channels")
    require(capacity > 0, s"cache capacity must be positive: $capacity")
    new AttentionKeyValueCache(channels, capacity)

/**
 * Mutable inference state for one MiniGPT request.
 *
 * A session evaluates one token at a time and owns one KV cache per Transformer block. It is
 * intentionally not thread-safe: concurrent requests must use independent sessions so that cache
 * ownership is explicit.
 */
final class MiniGptInferenceSession(val model: MiniGpt):
  private val caches                        = model.blocks.map { _ =>
    AttentionKeyValueCache.create(model.config.channels, model.config.maximumContextLength)
  }
  private var currentLength                 = 0
  private var totalEvaluatedTokens          = 0L
  private var latestLogits: Option[VectorD] = None

  def length: Int = currentLength

  /** Total token forward passes, including any explicit context rebuilds. */
  def evaluatedTokens: Long = totalEvaluatedTokens

  def allocatedCachePayloadBytes: Long = caches.iterator.map(_.allocatedPayloadBytes).sum

  def usedCachePayloadBytes: Long = caches.iterator.map(_.usedPayloadBytes).sum

  def lastLogits: Option[VectorD] = latestLogits

  /** Clears both context and cumulative request statistics. */
  def reset(): Unit =
    clearContext()
    totalEvaluatedTokens = 0L

  /** Replaces the session context and returns logits after its final token. */
  def prefill(tokenIds: Vector[TokenId]): VectorD =
    require(tokenIds.nonEmpty, "KV-cache prefill requires at least one token")
    require(
      tokenIds.size <= model.config.maximumContextLength,
      s"prefill length ${tokenIds.size} exceeds maximum ${model.config.maximumContextLength}"
    )
    reset()
    tokenIds.foldLeft(VectorD.empty)((_, tokenId) => append(tokenId))

  /** Evaluates one new token and appends its projected keys and values. */
  def append(tokenId: TokenId): VectorD =
    require(
      currentLength < model.config.maximumContextLength,
      s"inference context ${model.config.maximumContextLength} is full"
    )
    val embedded   = model.embeddings.at(tokenId, currentLength)
    val hidden     = model.blocks.zip(caches)
      .foldLeft(embedded) { case (input, (block, cache)) => block.forwardCached(input, cache) }
    val normalized = model.finalNorm(hidden)
    val output     = normalized.matmul(model.embeddings.tokens.weight.transpose2D)
    val result     = VectorD.from(output.rowValues(0))
    currentLength += 1
    totalEvaluatedTokens += 1L
    latestLogits = Some(result)
    result

  /** Rebuilds a shifted absolute-position window while retaining work counts. */
  private[transformer] def rebuild(tokenIds: Vector[TokenId]): VectorD =
    require(tokenIds.nonEmpty, "KV-cache rebuild requires at least one token")
    require(
      tokenIds.size <= model.config.maximumContextLength,
      s"rebuild length ${tokenIds.size} exceeds maximum ${model.config.maximumContextLength}"
    )
    clearContext()
    tokenIds.foldLeft(VectorD.empty)((_, tokenId) => append(tokenId))

  private def clearContext(): Unit =
    caches.foreach(_.clear())
    currentLength = 0
    latestLogits = None

/**
 * Accounting returned with cached generation.
 *
 * `referenceTokenEvaluations` is the number of token rows the uncached sliding-window
 * implementation would evaluate for the same decisions. It is a deterministic work count, not a
 * wall-clock speed claim.
 */
final case class CachedGenerationStatistics(
    tokenEvaluations: Long,
    referenceTokenEvaluations: Long,
    cacheRebuilds: Int,
    peakCachedTokens: Int,
    allocatedCachePayloadBytes: Long
):
  require(tokenEvaluations >= 0L, "token evaluations cannot be negative")
  require(referenceTokenEvaluations >= 0L, "reference token evaluations cannot be negative")
  require(cacheRebuilds >= 0, "cache rebuilds cannot be negative")
  require(peakCachedTokens >= 0, "peak cached tokens cannot be negative")
  require(allocatedCachePayloadBytes >= 0L, "allocated cache bytes cannot be negative")

final case class CachedGenerationResult(
    tokens: Vector[TokenId],
    statistics: CachedGenerationStatistics
)
