package learnai.data

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ShuffleSuite extends TestSuite:
  override val name: String = "DeterministicShuffle"

  override val tests: Vector[TestCase] = Vector(
    test("every epoch permutation is a bijection over the index range") {
      Vector(1, 2, 5, 16, 97).foreach { size =>
        (0L until 4L).foreach { epoch =>
          val order = DeterministicShuffle.permutation(size, seed = 11L, epoch)
          Assert.equal(order.sorted, Vector.range(0, size))
        }
      }
    },
    test("permutations are pure functions of size seed and epoch") {
      val first = DeterministicShuffle.permutation(32, seed = 5L, epoch = 3L)
      val second = DeterministicShuffle.permutation(32, seed = 5L, epoch = 3L)
      Assert.equal(first, second)
      val differentSeed = DeterministicShuffle.permutation(32, seed = 6L, epoch = 3L)
      Assert.isTrue(first != differentSeed, "different seeds should reorder 32 items")
      val differentEpoch = DeterministicShuffle.permutation(32, seed = 5L, epoch = 4L)
      Assert.isTrue(first != differentEpoch, "different epochs should reorder 32 items")
    },
    test("one epoch of a sweep visits every example exactly once") {
      val sweep = new ShuffledSweep(size = 10, seed = 21L)
      val (indices, after) = sweep.take(ShuffleCursor(epoch = 0L, position = 0), 10)
      Assert.equal(indices.sorted, Vector.range(0, 10))
      Assert.equal(after, ShuffleCursor(epoch = 1L, position = 0))
    },
    test("a sweep resumed from a checkpointed cursor continues exactly") {
      val sweep = new ShuffledSweep(size = 7, seed = 8L)
      val start = ShuffleCursor(epoch = 0L, position = 0)
      // 25 reads cross three epoch boundaries for size 7.
      val (straight, straightEnd) = sweep.take(start, 25)

      val (firstChunk, middle) = sweep.take(start, 11)
      val (secondChunk, resumedEnd) = sweep.take(middle, 14)
      Assert.equal(firstChunk ++ secondChunk, straight)
      Assert.equal(resumedEnd, straightEnd)
    },
    test("epoch boundaries reshuffle instead of repeating the previous order") {
      val sweep = new ShuffledSweep(size = 16, seed = 2L)
      val (firstEpoch, cursor) = sweep.take(ShuffleCursor(0L, 0), 16)
      val (secondEpoch, _) = sweep.take(cursor, 16)
      Assert.equal(firstEpoch.sorted, secondEpoch.sorted)
      Assert.isTrue(
        firstEpoch != secondEpoch,
        "consecutive epochs of 16 items should not share one order"
      )
    },
    test("invalid sizes cursors and counts are rejected") {
      val badSize = Assert.throws[IllegalArgumentException](new ShuffledSweep(0, seed = 1L))
      Assert.isTrue(badSize.getMessage.contains("positive"))
      val badPermutation = Assert.throws[IllegalArgumentException] {
        DeterministicShuffle.permutation(0, seed = 1L, epoch = 0L)
      }
      Assert.isTrue(badPermutation.getMessage.contains("positive"))

      val sweep = new ShuffledSweep(size = 3, seed = 1L)
      val outsideCursor = Assert.throws[IllegalArgumentException] {
        sweep.indexAt(ShuffleCursor(epoch = 0L, position = 3))
      }
      Assert.isTrue(outsideCursor.getMessage.contains("outside"))
      val negativeCount = Assert.throws[IllegalArgumentException] {
        sweep.take(ShuffleCursor(0L, 0), -1)
      }
      Assert.isTrue(negativeCount.getMessage.contains("non-negative"))
      val negativeEpoch = Assert.throws[IllegalArgumentException](ShuffleCursor(-1L, 0))
      Assert.isTrue(negativeEpoch.getMessage.contains("non-negative"))
    }
  )
