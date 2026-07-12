package learnai.foundations

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ComplexitySuite extends TestSuite:
  override val name: String = "Complexity"

  override val tests: Vector[TestCase] = specify(
    test("linear search comparison count follows the inspected prefix") {
      Assert.equal(Complexity.linearSearch(Vector(4, 7, 9), 4), CountedResult(Some(0), 1))
      Assert.equal(Complexity.linearSearch(Vector(4, 7, 9), 9), CountedResult(Some(2), 3))
      Assert.equal(Complexity.linearSearch(Vector(4, 7, 9), 1), CountedResult(None, 3))
    },
    test("binary search uses logarithmic comparisons on a power-of-two range") {
      val result = Complexity.binarySearch(Vector.range(0, 1024), -1)
      Assert.equal(result.value, None)
      Assert.isTrue(result.comparisons <= 11, s"comparison count was ${result.comparisons}")
    },
    test("row-major and column-first traversal visit equal cells in different orders") {
      val rowFirst = Complexity.traversalOffsets(2, 3, rowFirst = true)
      val columnFirst = Complexity.traversalOffsets(2, 3, rowFirst = false)
      Assert.equal(rowFirst, Vector(0, 1, 2, 3, 4, 5))
      Assert.equal(columnFirst, Vector(0, 3, 1, 4, 2, 5))
      Assert.equal(rowFirst.sorted, columnFirst.sorted)
    },
    test("doubling growth has linear total copies") {
      Vector.range(0, 1000).foreach { size =>
        Assert.isTrue(Complexity.doublingArrayCopies(size) < math.max(1, size * 2).toLong)
      }
    },
    test("payload accounting is exact and overflow checked") {
      Assert.equal(Complexity.doublePayloadBytes(3), 24L)
      Assert.throws[ArithmeticException](Complexity.doublePayloadBytes(Long.MaxValue))
      val _ = Assert.throws[IllegalArgumentException](Complexity.doublePayloadBytes(-1))
    }
  )
