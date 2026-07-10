package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor

/** Causal self-attention with fewer key/value heads than query heads.
  *
  * Standard multi-head attention (Chapter 19) gives every query head its own
  * key and value projections. During cached inference (Chapter 24) the KV
  * cache — not the weights — dominates memory, and its size is proportional
  * to the number of key/value heads. Grouped-query attention (GQA) keeps all
  * query heads but shares each key/value head across a *group* of
  * consecutive query heads:
  *
  * ```text
  * query heads:      q0 q1 q2 q3 q4 q5   (queryHeadCount = 6)
  * key/value heads:  k0    k1    k2      (keyValueHeadCount = 3, groupSize = 2)
  * ownership:        q0,q1 -> k0   q2,q3 -> k1   q4,q5 -> k2
  * ```
  *
  * The two boundary settings recover named architectures:
  *   - `keyValueHeadCount == queryHeadCount` is exactly multi-head attention
  *     (the test suite proves output equality against Chapter 19);
  *   - `keyValueHeadCount == 1` is multi-query attention (MQA).
  *
  * Cached KV payload per token shrinks from `2 * channels` values to
  * `2 * keyValueHeadCount * headChannels`, a factor of
  * `queryHeadCount / keyValueHeadCount`; [[keyValuePayloadBytesPerToken]]
  * exposes the exact accounting so serving-memory claims can be computed
  * rather than asserted.
  */
final class GroupedQueryAttention private (
    val channels: Int,
    val queryHeadCount: Int,
    val keyValueHeadCount: Int,
    val queryProjection: Linear,
    val keyProjection: Linear,
    val valueProjection: Linear,
    val outputProjection: Linear
):
  require(channels > 0, s"attention channels must be positive: $channels")
  require(queryHeadCount > 0, s"query head count must be positive: $queryHeadCount")
  require(keyValueHeadCount > 0, s"key/value head count must be positive: $keyValueHeadCount")
  require(
    channels % queryHeadCount == 0,
    s"channels $channels must be divisible by query head count $queryHeadCount"
  )
  require(
    queryHeadCount % keyValueHeadCount == 0,
    s"query head count $queryHeadCount must be divisible by key/value head count $keyValueHeadCount"
  )

  /** Channel width of every head, query and key/value alike. */
  val headChannels: Int = channels / queryHeadCount

  /** Number of consecutive query heads sharing one key/value head. */
  val groupSize: Int = queryHeadCount / keyValueHeadCount

  /** Total key (or value) projection width: `keyValueHeadCount * headChannels`. */
  val keyValueChannels: Int = keyValueHeadCount * headChannels

  require(
    queryProjection.inputChannels == channels && queryProjection.outputChannels == channels,
    s"query projection must map $channels -> $channels"
  )
  require(
    keyProjection.inputChannels == channels && keyProjection.outputChannels == keyValueChannels,
    s"key projection must map $channels -> $keyValueChannels"
  )
  require(
    valueProjection.inputChannels == channels && valueProjection.outputChannels == keyValueChannels,
    s"value projection must map $channels -> $keyValueChannels"
  )
  require(
    outputProjection.inputChannels == channels && outputProjection.outputChannels == channels,
    s"output projection must map $channels -> $channels"
  )

  /** Key/value head that owns one query head. */
  def keyValueHeadFor(queryHead: Int): Int =
    require(
      queryHead >= 0 && queryHead < queryHeadCount,
      s"query head $queryHead outside [0, $queryHeadCount)"
    )
    queryHead / groupSize

  /** Runs grouped causal attention and returns only the hidden states. */
  def apply(input: Tensor): Tensor = forwardWithWeights(input).output

  /** Runs attention while retaining per-query-head weights for inspection.
    *
    * Every query head computes its own scores and weighted values, but reads
    * keys and values from the channel slice of its owning key/value head.
    */
  def forwardWithWeights(input: Tensor): AttentionResult =
    require(
      input.rank == 2 && input.shape(1) == channels,
      s"grouped attention expected [time,$channels], got ${input.shape}"
    )
    require(input.shape(0) > 0, "grouped attention requires at least one time position")
    val query = queryProjection(input)
    val key = keyProjection(input)
    val value = valueProjection(input)
    val scale = 1.0 / math.sqrt(headChannels.toDouble)

    val headResults = Vector.tabulate(queryHeadCount) { head =>
      val queryFrom = head * headChannels
      val keyValueFrom = keyValueHeadFor(head) * headChannels
      val headQuery = query.sliceColumns(queryFrom, queryFrom + headChannels)
      val headKey = key.sliceColumns(keyValueFrom, keyValueFrom + headChannels)
      val headValue = value.sliceColumns(keyValueFrom, keyValueFrom + headChannels)
      val scores = headQuery.matmul(headKey.transpose2D).scale(scale)
      val weights = scores.causalMask().softmaxRows
      weights -> weights.matmul(headValue)
    }
    val weights = headResults.map(_._1)
    val concatenated = Tensor.concatenateColumns(headResults.map(_._2))
    AttentionResult(outputProjection(concatenated), weights)

  /** Cached key+value bytes per token position at `Double` precision.
    *
    * A Chapter 24 style cache stores one key row and one value row per
    * layer and token. Under GQA those rows are `keyValueChannels` wide, so
    * the per-token payload is `2 * keyValueChannels * 8` bytes instead of
    * multi-head attention's `2 * channels * 8`.
    */
  def keyValuePayloadBytesPerToken: Long =
    2L * keyValueChannels.toLong * java.lang.Double.BYTES.toLong

  /** Total cached payload for one request across all layers. */
  def cachePayloadBytes(contextLength: Int, layerCount: Int): Long =
    require(contextLength > 0, s"context length must be positive: $contextLength")
    require(layerCount > 0, s"layer count must be positive: $layerCount")
    Math.multiplyExact(
      keyValuePayloadBytesPerToken,
      Math.multiplyExact(contextLength.toLong, layerCount.toLong)
    )

  def parameters: Vector[Tensor] =
    queryProjection.parameters ++
      keyProjection.parameters ++
      valueProjection.parameters ++
      outputProjection.parameters

  /** Total trainable element count, used by ablation accounting. */
  def parameterCount: Int = parameters.map(_.size).sum

object GroupedQueryAttention:
  /** Creates reproducibly initialized projections with narrow key/value widths. */
  def random(
      channels: Int,
      queryHeadCount: Int,
      keyValueHeadCount: Int,
      random: SplittableRandom,
      label: String
  ): GroupedQueryAttention =
    require(queryHeadCount > 0, s"query head count must be positive: $queryHeadCount")
    require(keyValueHeadCount > 0, s"key/value head count must be positive: $keyValueHeadCount")
    require(
      channels % queryHeadCount == 0,
      s"channels $channels not divisible by $queryHeadCount query heads"
    )
    require(
      queryHeadCount % keyValueHeadCount == 0,
      s"query head count $queryHeadCount not divisible by $keyValueHeadCount key/value heads"
    )
    val keyValueChannels = channels / queryHeadCount * keyValueHeadCount
    new GroupedQueryAttention(
      channels,
      queryHeadCount,
      keyValueHeadCount,
      Linear.random(channels, channels, random, s"$label.query"),
      Linear.random(channels, keyValueChannels, random, s"$label.key"),
      Linear.random(channels, keyValueChannels, random, s"$label.value"),
      Linear.random(channels, channels, random, s"$label.output")
    )

  /** Assembles attention from caller-provided projections.
    *
    * This factory exists for equivalence oracles: a grouped layer built from
    * a Chapter 19 layer's projections with `keyValueHeadCount ==
    * queryHeadCount` must reproduce its output exactly. All shape invariants
    * are re-validated by the constructor.
    */
  def fromProjections(
      channels: Int,
      queryHeadCount: Int,
      keyValueHeadCount: Int,
      queryProjection: Linear,
      keyProjection: Linear,
      valueProjection: Linear,
      outputProjection: Linear
  ): GroupedQueryAttention =
    new GroupedQueryAttention(
      channels,
      queryHeadCount,
      keyValueHeadCount,
      queryProjection,
      keyProjection,
      valueProjection,
      outputProjection
    )
