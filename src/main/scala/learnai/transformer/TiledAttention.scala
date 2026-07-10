package learnai.transformer

/** Streaming (online) softmax and tiled causal attention.
  *
  * Standard attention materializes a full `[time, time]` score matrix, and
  * Chapter 29a showed that this quadratic term dominates memory at long
  * contexts. FlashAttention's observation is that softmax can be computed
  * *incrementally*: scores can arrive in tiles, and a small running state —
  * maximum, denominator, and an unnormalized value accumulator — is enough
  * to produce the exact same output without ever holding a full score row.
  *
  * This module is a faithful CPU simulation of that algorithm, not a fused
  * kernel: it demonstrates and tests the mathematics (the rescaling
  * recurrence and its exactness) while the memory win on real hardware
  * comes from executing the same recurrence inside on-chip SRAM. It is
  * forward-only by design; the trainable path remains Chapter 19's, and
  * equality with it is this module's correctness oracle.
  *
  * The running state for one query row is:
  *
  * ```text
  * m  running maximum of all scores seen so far
  * d  running denominator: sum of exp(score - m)
  * a  running unnormalized output: sum of exp(score - m) * valueRow
  * ```
  *
  * When a tile with local maximum m' arrives, previous contributions are
  * rescaled by `exp(m - max(m, m'))`, which is always in (0, 1]; nothing is
  * ever exponentiated above zero, so extreme scores cannot overflow.
  */
object TiledAttention:

  /** Softmax state after streaming any prefix of a score row. */
  final case class OnlineSoftmaxState(maximum: Double, denominator: Double):
    require(!maximum.isNaN, "online softmax maximum cannot be NaN")
    require(
      denominator >= 0.0 && denominator.isFinite,
      s"online softmax denominator must be finite and non-negative: $denominator"
    )

  object OnlineSoftmaxState:
    /** State before any score: an empty maximum and a zero denominator. */
    val empty: OnlineSoftmaxState =
      OnlineSoftmaxState(Double.NegativeInfinity, 0.0)

  /** Folds one tile of scores into the running softmax state.
    *
    * This is the exact FlashAttention recurrence for the normalizer:
    * `m'' = max(m, tileMax)` and
    * `d'' = d * exp(m - m'') + sum(exp(tile - m''))`.
    */
  def foldTile(state: OnlineSoftmaxState, tile: Vector[Double]): OnlineSoftmaxState =
    require(tile.nonEmpty, "a softmax tile requires at least one score")
    tile.foreach(score => require(!score.isNaN, "softmax scores cannot be NaN"))
    val tileMaximum = tile.max
    val merged = math.max(state.maximum, tileMaximum)
    val rescaledPrevious =
      if state.denominator == 0.0 then 0.0
      else state.denominator * math.exp(state.maximum - merged)
    val tileSum = tile.iterator.map(score => math.exp(score - merged)).sum
    OnlineSoftmaxState(merged, rescaledPrevious + tileSum)

  /** Computes an exact softmax by streaming the scores in tiles.
    *
    * One pass builds the normalizer state; a second pass normalizes. A
    * fused kernel avoids the second pass over memory by rescaling its
    * output accumulator instead — that is [[attendRowTiled]].
    */
  def softmaxStreamed(scores: Vector[Double], tileSize: Int): Vector[Double] =
    require(scores.nonEmpty, "softmax requires at least one score")
    require(tileSize > 0, s"tile size must be positive: $tileSize")
    val state = scores
      .grouped(tileSize)
      .foldLeft(OnlineSoftmaxState.empty)((current, tile) => foldTile(current, tile.toVector))
    scores.map(score => math.exp(score - state.maximum) / state.denominator)

  /** Computes one query's causal attention output in tiles.
    *
    * `keys` and `values` hold one row per attendable position (the causal
    * prefix including the query's own position). The full score row is
    * never materialized: at most `tileSize` scores plus one
    * channel-width accumulator exist at any moment, which is the memory
    * argument [[materializedFloatsPerRow]] quantifies.
    */
  def attendRowTiled(
      query: Vector[Double],
      keys: Vector[Vector[Double]],
      values: Vector[Vector[Double]],
      scale: Double,
      tileSize: Int
  ): Vector[Double] =
    val channels = query.size
    require(channels > 0, "attention requires at least one channel")
    require(keys.nonEmpty, "attention requires at least one attendable position")
    require(keys.size == values.size, s"key count ${keys.size} != value count ${values.size}")
    require(tileSize > 0, s"tile size must be positive: $tileSize")
    require(scale > 0.0 && scale.isFinite, s"scale must be finite and positive: $scale")
    keys.zipWithIndex.foreach { case (key, index) =>
      require(key.size == channels, s"key $index width ${key.size} != $channels")
    }
    values.zipWithIndex.foreach { case (value, index) =>
      require(value.size == channels, s"value $index width ${value.size} != $channels")
    }

    var maximum = Double.NegativeInfinity
    var denominator = 0.0
    val accumulator = new Array[Double](channels)
    var tileStart = 0
    while tileStart < keys.size do
      val tileEnd = math.min(tileStart + tileSize, keys.size)
      // Tile scores are the only per-position floats this pass holds.
      val tileScores = new Array[Double](tileEnd - tileStart)
      var tileMaximum = Double.NegativeInfinity
      var position = tileStart
      while position < tileEnd do
        var dot = 0.0
        var channel = 0
        while channel < channels do
          dot += query(channel) * keys(position)(channel)
          channel += 1
        val score = dot * scale
        tileScores(position - tileStart) = score
        tileMaximum = math.max(tileMaximum, score)
        position += 1

      val merged = math.max(maximum, tileMaximum)
      val previousRescale = if denominator == 0.0 then 0.0 else math.exp(maximum - merged)
      var channel = 0
      while channel < channels do
        accumulator(channel) *= previousRescale
        channel += 1
      denominator *= previousRescale

      position = tileStart
      while position < tileEnd do
        val weight = math.exp(tileScores(position - tileStart) - merged)
        denominator += weight
        channel = 0
        while channel < channels do
          accumulator(channel) += weight * values(position)(channel)
          channel += 1
        position += 1
      maximum = merged
      tileStart = tileEnd

    Vector.tabulate(channels)(channel => accumulator(channel) / denominator)

  /** Full causal attention over `[time, channels]` rows, one tiled pass per
    * query row. Row `t` attends to positions `0..t`, so causality holds by
    * construction rather than by masking a materialized square matrix.
    */
  def causalAttentionTiled(
      queries: Vector[Vector[Double]],
      keys: Vector[Vector[Double]],
      values: Vector[Vector[Double]],
      scale: Double,
      tileSize: Int
  ): Vector[Vector[Double]] =
    require(queries.nonEmpty, "attention requires at least one query row")
    require(
      queries.size == keys.size && keys.size == values.size,
      s"row counts differ: ${queries.size}/${keys.size}/${values.size}"
    )
    Vector.tabulate(queries.size) { time =>
      attendRowTiled(
        queries(time),
        keys.take(time + 1),
        values.take(time + 1),
        scale,
        tileSize
      )
    }

  /** Peak per-position floats one tiled query row materializes:
    * `tileSize` scores plus the channel-wide accumulator and two scalars.
    * The untiled row holds all `contextLength` scores instead — the
    * quadratic term of Chapter 29a once multiplied by rows, heads, and
    * layers.
    */
  def materializedFloatsPerRow(channels: Int, tileSize: Int): Long =
    require(channels > 0, s"channels must be positive: $channels")
    require(tileSize > 0, s"tile size must be positive: $tileSize")
    tileSize.toLong + channels.toLong + 2L
