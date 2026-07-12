package learnai.data

import learnai.random.SplitMix64

/**
 * A restart position inside an endless epoch-shuffled traversal.
 *
 * The cursor is the *entire* iteration state: the epoch selects a permutation and the position
 * indexes into it. Storing these two numbers in a checkpoint is sufficient to continue a data sweep
 * exactly, which is the same resumability contract Chapter 22c establishes for training.
 */
final case class ShuffleCursor(epoch: Long, position: Int):
  require(epoch >= 0L, s"epoch must be non-negative: $epoch")
  require(position >= 0, s"position must be non-negative: $position")

object DeterministicShuffle:
  /**
   * Fisher–Yates permutation of `[0, size)` driven by a SplitMix64 stream.
   *
   * The permutation is a pure function of `(size, seed, epoch)`. Each epoch derives its own
   * generator by offsetting the seed counter with the golden gamma, the same stream-separation
   * device SplitMix64 itself uses, so consecutive epochs get well-mixed, mutually independent
   * orders without any O(epoch) skipping.
   */
  def permutation(size: Int, seed: Long, epoch: Long): Vector[Int] =
    require(size > 0, s"permutation size must be positive: $size")
    require(epoch >= 0L, s"epoch must be non-negative: $epoch")
    val generator = SplitMix64.seeded(seed + epoch * SplitMix64.GoldenGamma)
    val order     = Array.tabulate(size)(identity)
    var index     = size - 1
    while index > 0 do
      val swapWith = generator.nextInt(index + 1)
      val kept     = order(index)
      order(index) = order(swapWith)
      order(swapWith) = kept
      index -= 1
    order.toVector

/**
 * An endless, deterministic, restartable shuffled sweep over example indices.
 *
 * Every epoch visits each of the `size` indices exactly once in an order that changes per epoch.
 * The sweep itself is immutable and stateless; all iteration state lives in the caller-owned
 * [[ShuffleCursor]], so two consumers can traverse independently and a checkpointed cursor can be
 * resumed in another process.
 *
 * Permutations are recomputed per epoch rather than cached: the goal of this reference
 * implementation is an auditable contract, not throughput.
 */
final class ShuffledSweep(val size: Int, val seed: Long):
  require(size > 0, s"sweep size must be positive: $size")

  /** The example index the cursor currently points at. */
  def indexAt(cursor: ShuffleCursor): Int =
    requireValid(cursor)
    DeterministicShuffle.permutation(size, seed, cursor.epoch)(cursor.position)

  /** The cursor one step forward, rolling into the next epoch at the end. */
  def next(cursor: ShuffleCursor): ShuffleCursor =
    requireValid(cursor)
    if cursor.position + 1 < size then cursor.copy(position = cursor.position + 1)
    else ShuffleCursor(Math.addExact(cursor.epoch, 1L), 0)

  /**
   * Reads `count` consecutive indices and returns the cursor after them.
   *
   * The returned cursor makes reads composable: `take(c, a + b)` equals `take(c, a)` followed by
   * `take(after, b)`.
   */
  def take(cursor: ShuffleCursor, count: Int): (Vector[Int], ShuffleCursor) =
    require(count >= 0, s"count must be non-negative: $count")
    requireValid(cursor)
    val indices   = Vector.newBuilder[Int]
    var current   = cursor
    var remaining = count
    while remaining > 0 do
      indices += indexAt(current)
      current = next(current)
      remaining -= 1
    (indices.result(), current)

  private def requireValid(cursor: ShuffleCursor): Unit =
    require(cursor.position < size, s"cursor position ${cursor.position} outside [0, $size)")
