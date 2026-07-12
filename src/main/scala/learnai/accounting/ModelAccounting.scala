package learnai.accounting

import learnai.transformer.MiniGptConfig

/**
 * Deterministic closed-form estimators for MiniGPT size, work, and memory.
 *
 * Every quantity here is derived from the architecture configuration alone — no measurement, no
 * sampling — so estimates are reproducible and disagreements with reality are bugs in the formula,
 * not noise. The test suite anchors the formulas to the real implementation: the parameter formula
 * must equal `MiniGpt.random(config).parameterCount` exactly, and the attention and cache formulas
 * are checked against the live tensors and caches they describe.
 *
 * Conventions, stated once and used consistently:
 *   - one fused multiply-add counts as 2 FLOPs;
 *   - FLOP counts include the matrix products (projections, attention scores, weighted values,
 *     feed-forward, logits) and exclude bias additions, normalization, softmax, and residual
 *     additions, whose cost is `O(C)` per token against the matrices' `O(C^2)`;
 *   - byte counts assume this course's `Double` (8-byte) values. Real systems divide by 2 or 4 for
 *     fp16/int8; the *structure* of the accounting is what transfers.
 */
object ModelAccounting:
  private val BytesPerValue = 8L

  /**
   * Exact trainable-parameter count of the Chapter 21 architecture.
   *
   * token embedding `V*C` + position embedding `T*C` + per layer (two norm scales `2C`, four
   * attention projections `4(C^2+C)`, feed-forward `CF+F+FC+C`) + final norm `C`. The tied logit
   * head adds nothing — it reuses the token embedding.
   */
  def parameterCount(config: MiniGptConfig): Long =
    val channels    = config.channels.toLong
    val hidden      = config.hiddenChannels.toLong
    val embeddings  = config.vocabularySize.toLong * channels +
      config.maximumContextLength.toLong * channels
    val attention   = 4L * (channels * channels + channels)
    val feedForward = channels * hidden + hidden + hidden * channels + channels
    val norms       = 2L * channels
    embeddings + config.layerCount.toLong * (attention + feedForward + norms) + channels

  /** Bytes to hold the parameters once. */
  def parameterBytes(config: MiniGptConfig): Long = Math
    .multiplyExact(parameterCount(config), BytesPerValue)

  /** Bytes for one gradient per parameter, as reverse mode requires. */
  def gradientBytes(config: MiniGptConfig): Long = parameterBytes(config)

  /** Bytes for AdamW's two moment arrays (Chapter 13). */
  def adamWStateBytes(config: MiniGptConfig): Long = Math.multiplyExact(2L, parameterBytes(config))

  /**
   * Parameters + gradients + optimizer moments: the resident training state that exists regardless
   * of batch size. Activations come on top and scale with tokens; this floor is why training a
   * model takes about four times the memory of serving it at equal precision.
   */
  def trainingResidentBytes(config: MiniGptConfig): Long = Math
    .addExact(Math.addExact(parameterBytes(config), gradientBytes(config)), adamWStateBytes(config))

  /**
   * FLOPs to evaluate ONE new token whose attention covers `attendedPositions`.
   *
   * Per layer: `8C^2` for the four projections, `4CF` for the feed-forward, and `4C * attended` for
   * scores plus weighted values. The tied logit head adds `2CV` once. This is exactly the work of a
   * Chapter 24 cached decode step at the given cache occupancy.
   */
  def decodeStepFlops(config: MiniGptConfig, attendedPositions: Int): Long =
    require(attendedPositions > 0, s"attended positions must be positive: $attendedPositions")
    require(
      attendedPositions <= config.maximumContextLength,
      s"attended positions $attendedPositions exceed context ${config.maximumContextLength}"
    )
    val channels    = config.channels.toLong
    val hidden      = config.hiddenChannels.toLong
    val projections = 8L * channels * channels
    val feedForward = 4L * channels * hidden
    val attention   = 4L * channels * attendedPositions.toLong
    val perLayer    = projections + feedForward + attention
    Math.addExact(
      Math.multiplyExact(config.layerCount.toLong, perLayer),
      2L * channels * config.vocabularySize.toLong
    )

  /**
   * FLOPs to prefill a prompt of `tokens`, summed over its decode steps.
   *
   * Token `i` (one-based) attends to `i` positions, so the quadratic attention term sums to
   * `4CL * t(t+1)/2` while every linear term scales with `t`. The implementation evaluates the
   * closed form; the suite cross-checks it against literally summing [[decodeStepFlops]].
   */
  def prefillFlops(config: MiniGptConfig, tokens: Int): Long =
    require(tokens > 0, s"token count must be positive: $tokens")
    require(
      tokens <= config.maximumContextLength,
      s"prompt of $tokens tokens exceeds context ${config.maximumContextLength}"
    )
    val channels       = config.channels.toLong
    val hidden         = config.hiddenChannels.toLong
    val layers         = config.layerCount.toLong
    val tokenCount     = tokens.toLong
    val linearPerToken = layers * (8L * channels * channels + 4L * channels * hidden) +
      2L * channels * config.vocabularySize.toLong
    val attentionSum   = 4L * channels * layers * (tokenCount * (tokenCount + 1L) / 2L)
    Math.addExact(Math.multiplyExact(linearPerToken, tokenCount), attentionSum)

  /**
   * Stored attention-weight values for a full forward at length `t`: `L * H * t^2`. This is the
   * quadratic memory term that motivates IO-aware attention; the estimator matches the live tensors
   * the Chapter 19 implementation actually retains.
   */
  def attentionWeightValues(config: MiniGptConfig, contextLength: Int): Long =
    require(contextLength > 0, s"context length must be positive: $contextLength")
    require(
      contextLength <= config.maximumContextLength,
      s"context $contextLength exceeds maximum ${config.maximumContextLength}"
    )
    Math.multiplyExact(
      Math.multiplyExact(config.layerCount.toLong, config.headCount.toLong),
      contextLength.toLong * contextLength.toLong
    )

  /**
   * KV cache payload for one request at full occupancy: `L * 2 * C * 8 * t`.
   *
   * Matches `AttentionKeyValueCache.allocatedPayloadBytes` per layer. With grouped-query attention
   * (Chapter 28c) replace `C` by `keyValueHeadCount * headChannels`.
   */
  def kvCachePayloadBytes(config: MiniGptConfig, contextLength: Int): Long =
    require(contextLength > 0, s"context length must be positive: $contextLength")
    require(
      contextLength <= config.maximumContextLength,
      s"context $contextLength exceeds maximum ${config.maximumContextLength}"
    )
    Math.multiplyExact(
      Math.multiplyExact(config.layerCount.toLong, 2L * config.channels.toLong),
      Math.multiplyExact(contextLength.toLong, BytesPerValue)
    )
