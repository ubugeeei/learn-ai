package learnai.math

import java.util.SplittableRandom

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ProbabilitySuite extends TestSuite:
  override val name: String = "Probability"

  override val tests: Vector[TestCase] = specify(
    test("categorical distributions require non-negative values summing to one") {
      Assert.isTrue(Categorical.from(VectorD(0.25, 0.75)).isRight)
      Assert
        .equal(Categorical.from(VectorD(0.2, 0.2)), Left("probabilities must sum to 1.0, got 0.4"))
      Assert.isTrue(Assert.left(Categorical.from(VectorD(-0.1, 1.1))).contains("index 0"))
    },
    test("softmax returns normalized positive probabilities") {
      val distribution = Assert.right(Probability.softmax(VectorD(1.0, 2.0, 3.0)))
      Assert.close(distribution.probabilities.sum, 1.0)
      Assert.isTrue(distribution.probabilities.toVector.forall(_ > 0.0))
      Assert.equal(Assert.right(distribution.probabilities.argmax), 2)
    },
    test("softmax is stable for very large logits") {
      val distribution = Assert.right(Probability.softmax(VectorD(10000.0, 10001.0)))
      Assert.close(distribution.probabilities.sum, 1.0)
      Assert.isTrue(distribution.probability(1) > distribution.probability(0))
    },
    test("adding a constant to every logit leaves softmax unchanged") {
      val original = Assert.right(Probability.softmax(VectorD(-1.0, 0.0, 1.0)))
      val shifted  = Assert.right(Probability.softmax(VectorD(99.0, 100.0, 101.0)))
      original.probabilities.toVector.zip(shifted.probabilities.toVector)
        .foreach { case (left, right) => Assert.close(left, right) }
    },
    test("cross entropy is lower when the target logit is larger") {
      val lowTarget  = Assert.right(Probability.crossEntropy(VectorD(0.0, 2.0), 0))
      val highTarget = Assert.right(Probability.crossEntropy(VectorD(0.0, 2.0), 1))
      Assert.isTrue(highTarget < lowTarget)
    },
    test("fair coin entropy is log two") {
      val fairCoin = Assert.right(Categorical.from(VectorD(0.5, 0.5)))
      Assert.close(fairCoin.entropy, math.log(2.0))
    },
    test("sampling from a one-hot distribution always selects its nonzero event") {
      val certain = Assert.right(Categorical.from(VectorD(0.0, 0.0, 1.0)))
      val random  = new SplittableRandom(42L)
      val samples = Vector.fill(100)(certain.sample(random))
      Assert.isTrue(samples.forall(_ == 2))
    }
  )
