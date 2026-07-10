package learnai.random

import java.util.random.RandomGenerator

/** A SplitMix64 random generator whose complete state is one observable Long.
  *
  * `java.util.SplittableRandom` is deterministic for a fixed seed but hides
  * its internal state, so a training run using it can be *replayed* from
  * update zero yet never *resumed* from the middle. Exact resume (Chapter
  * 22c) requires a generator whose full state can be captured in a
  * checkpoint and restored bit-for-bit later, possibly in another process.
  *
  * SplitMix64 is the output mixer published by Steele, Lea, and Flood and
  * used as the seeding generator inside the JDK. Its entire state is a
  * 64-bit counter advanced by a fixed odd constant; each output applies an
  * avalanching bit-mix to the counter. Capturing [[state]] and calling
  * [[SplitMix64.fromState]] therefore reproduces the *future* of the stream
  * exactly, which is the property checkpointing needs.
  *
  * The class implements [[java.util.random.RandomGenerator]], so it drops
  * into every API in this codebase that samples batches or tokens.
  * `nextInt(bound)` and `nextDouble()` are overridden with explicitly
  * specified algorithms so streams do not depend on JDK default-method
  * details.
  *
  * Instances are mutable and single-owner, like every RNG in this project:
  * sharing one generator between concurrent consumers interleaves their
  * streams nondeterministically.
  */
final class SplitMix64 private (private var counter: Long) extends RandomGenerator:

  /** The complete generator state; store this in a checkpoint. */
  def state: Long = counter

  /** Returns 64 mixed bits and advances the counter by the golden gamma. */
  override def nextLong(): Long =
    counter += SplitMix64.GoldenGamma
    var mixed = counter
    mixed = (mixed ^ (mixed >>> 30)) * 0xbf58476d1ce4e5b9L
    mixed = (mixed ^ (mixed >>> 27)) * 0x94d049bb133111ebL
    mixed ^ (mixed >>> 31)

  /** Uniform value in `[0, bound)` using unbiased rejection sampling.
    *
    * Takes the top 31 bits of one `nextLong` output and rejects draws from
    * the incomplete final block, the same technique `java.util.Random`
    * documents. Rejection consumes additional outputs, and that consumption
    * is part of the deterministic stream contract.
    */
  override def nextInt(bound: Int): Int =
    require(bound > 0, s"bound must be positive: $bound")
    var result = 0
    var accepted = false
    while !accepted do
      val bits = (nextLong() >>> 33).toInt
      result = bits % bound
      accepted = bits - result + (bound - 1) >= 0
    result

  /** Uniform value in `[0, 1)` from the top 53 bits of one output. */
  override def nextDouble(): Double =
    (nextLong() >>> 11).toDouble * SplitMix64.DoubleUnit

object SplitMix64:
  /** Weyl-sequence increment: 2^64 divided by the golden ratio, forced odd. */
  val GoldenGamma: Long = 0x9e3779b97f4a7c15L

  private val DoubleUnit: Double = 1.0 / (1L << 53).toDouble

  /** Creates a generator whose first output is the mix of `seed + gamma`. */
  def seeded(seed: Long): SplitMix64 = new SplitMix64(seed)

  /** Recreates a generator from a state captured by [[SplitMix64#state]].
    *
    * `fromState(rng.state)` produces a generator whose future outputs equal
    * the original's, which is the exact-resume contract. The function is
    * identical to [[seeded]] because the state *is* the seed counter; it has
    * a separate name so call sites document intent.
    */
  def fromState(state: Long): SplitMix64 = new SplitMix64(state)
