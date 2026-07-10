package learnai.random

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object SplitMix64Suite extends TestSuite:
  override val name: String = "SplitMix64"

  /** Independent transcription of the published SplitMix64 reference code.
    *
    * Written directly from the algorithm description (advance by the golden
    * gamma, then two multiply-xorshift mixing rounds) rather than by calling
    * the production class, so a transcription error in either copy fails the
    * comparison test.
    */
  private def referenceStream(seed: Long, count: Int): Vector[Long] =
    var state = seed
    Vector.fill(count) {
      state = state + 0x9e3779b97f4a7c15L
      var z = state
      z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L
      z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL
      z ^ (z >>> 31)
    }

  override val tests: Vector[TestCase] = Vector(
    test("outputs match an independent transcription of the reference algorithm") {
      Vector(0L, 1L, 42L, -1L, Long.MinValue).foreach { seed =>
        val generator = SplitMix64.seeded(seed)
        val actual = Vector.fill(32)(generator.nextLong())
        Assert.equal(actual, referenceStream(seed, 32))
      }
    },
    test("a generator rebuilt from a captured state continues the exact stream") {
      val original = SplitMix64.seeded(2024L)
      val consumed = Vector.fill(17)(original.nextLong())
      Assert.equal(consumed.size, 17)
      val captured = original.state
      val expectedFuture = Vector.fill(50)(original.nextLong())

      val resumed = SplitMix64.fromState(captured)
      val actualFuture = Vector.fill(50)(resumed.nextLong())
      Assert.equal(actualFuture, expectedFuture)
      Assert.equal(resumed.state, original.state)
    },
    test("state is the seed before any draw and advances by the golden gamma") {
      val generator = SplitMix64.seeded(7L)
      Assert.equal(generator.state, 7L)
      val _ = generator.nextLong()
      Assert.equal(generator.state, 7L + SplitMix64.GoldenGamma)
    },
    test("bounded integers stay in range and reproduce across equal seeds") {
      val bounds = Vector(1, 2, 3, 7, 100, Int.MaxValue)
      bounds.foreach { bound =>
        val generator = SplitMix64.seeded(99L)
        (0 until 200).foreach { _ =>
          val value = generator.nextInt(bound)
          Assert.isTrue(value >= 0 && value < bound, s"$value outside [0, $bound)")
        }
      }
      val first = SplitMix64.seeded(5L)
      val second = SplitMix64.seeded(5L)
      Assert.equal(
        Vector.fill(100)(first.nextInt(13)),
        Vector.fill(100)(second.nextInt(13))
      )
    },
    test("bounded integers cover every residue of a small bound") {
      val generator = SplitMix64.seeded(3L)
      val observed = Vector.fill(300)(generator.nextInt(5)).toSet
      Assert.equal(observed, Set(0, 1, 2, 3, 4))
    },
    test("doubles are uniform unit-interval values built from the top 53 bits") {
      val generator = SplitMix64.seeded(11L)
      val values = Vector.fill(500)(generator.nextDouble())
      values.foreach { value =>
        Assert.isTrue(value >= 0.0 && value < 1.0, s"$value outside [0, 1)")
      }
      // Coarse uniformity: both halves of the interval must be visited often.
      val below = values.count(_ < 0.5)
      Assert.isTrue(below > 150 && below < 350, s"suspicious half-interval count: $below")
    },
    test("non-positive bounds are rejected") {
      val generator = SplitMix64.seeded(1L)
      val zero = Assert.throws[IllegalArgumentException](generator.nextInt(0))
      Assert.isTrue(zero.getMessage.contains("positive"))
      val negative = Assert.throws[IllegalArgumentException](generator.nextInt(-4))
      Assert.isTrue(negative.getMessage.contains("positive"))
    }
  )
