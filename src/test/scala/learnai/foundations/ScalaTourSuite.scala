package learnai.foundations

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ScalaTourSuite extends TestSuite:
  override val name: String = "ScalaTour"

  override val tests: Vector[TestCase] = Vector(
    test("square multiplies a value by itself") {
      Assert.close(ScalaTour.square(-3.0), 9.0)
    },
    test("mean averages non-empty values") {
      val result = ScalaTour.mean(Vector(2.0, 1.0, 0.5))
      Assert.close(Assert.right(result), 7.0 / 6.0)
    },
    test("mean rejects an empty collection") {
      val result = ScalaTour.mean(Vector.empty)
      Assert.equal(Assert.left(result), "mean requires at least one value")
    },
    test("describeSign covers negative, zero, and positive numbers") {
      Assert.equal(ScalaTour.describeSign(-0.5), "negative")
      Assert.equal(ScalaTour.describeSign(0.0), "zero")
      Assert.equal(ScalaTour.describeSign(0.5), "positive")
    },
    test("normalizeLabel removes surrounding space and case differences") {
      Assert.equal(ScalaTour.normalizeLabel("  Training LOSS "), "training loss")
    }
  )
