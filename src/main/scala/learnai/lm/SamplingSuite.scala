package learnai.lm

import java.util.SplittableRandom

import learnai.math.VectorD
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object SamplingSuite extends TestSuite:
  override val name: String = "Sampling"

  override val tests: Vector[TestCase] = specify(
    test("top-k one turns sampling into greedy decoding") {
      val logits = VectorD(-2.0, 4.0, 1.0)
      val distribution = Assert.right(
        Sampling.distribution(logits, SamplingConfig(topK = Some(1)))
      )
      Assert.equal(distribution.probabilities, VectorD(0.0, 1.0, 0.0))
      val samples = Vector.fill(100)(distribution.sample(new SplittableRandom(1L)))
      Assert.isTrue(samples.forall(_ == 1))
    },
    test("top-k clamps to vocabulary size and remains normalized") {
      val distribution = Assert.right(
        Sampling.distribution(VectorD(1.0, 2.0), SamplingConfig(topK = Some(99)))
      )
      Assert.close(distribution.probabilities.sum, 1.0)
      Assert.isTrue(distribution.probabilities.toVector.forall(_ > 0.0))
    },
    test("top-p retains the smallest high-probability nucleus") {
      val logits = VectorD(math.log(0.6), math.log(0.25), math.log(0.1), math.log(0.05))
      val distribution = Assert.right(
        Sampling.distribution(logits, SamplingConfig(topP = Some(0.8)))
      )
      Assert.isTrue(distribution.probability(0) > 0.0)
      Assert.isTrue(distribution.probability(1) > 0.0)
      Assert.close(distribution.probability(2), 0.0)
      Assert.close(distribution.probability(3), 0.0)
      Assert.close(distribution.probabilities.sum, 1.0)
    },
    test("lower temperature reduces entropy for unequal logits") {
      val logits = VectorD(0.0, 1.0, 2.0)
      val cold = Assert.right(
        Sampling.distribution(logits, SamplingConfig(temperature = 0.25))
      )
      val hot = Assert.right(
        Sampling.distribution(logits, SamplingConfig(temperature = 2.0))
      )
      Assert.isTrue(cold.entropy < hot.entropy)
    },
    test("equal-probability top-k ties prefer smaller token IDs") {
      val distribution = Assert.right(
        Sampling.distribution(VectorD(0.0, 0.0, 0.0), SamplingConfig(topK = Some(2)))
      )
      Assert.isTrue(distribution.probability(0) > 0.0)
      Assert.isTrue(distribution.probability(1) > 0.0)
      Assert.close(distribution.probability(2), 0.0)
    },
    test("sampling is reproducible for equal seeds") {
      val logits = VectorD(0.0, 1.0, 2.0)
      val config = SamplingConfig(temperature = 0.8, topK = Some(2), topP = Some(0.95))
      val firstRandom = new SplittableRandom(42L)
      val secondRandom = new SplittableRandom(42L)
      val first = Vector.fill(50)(Assert.right(Sampling.sample(logits, config, firstRandom)))
      val second = Vector.fill(50)(Assert.right(Sampling.sample(logits, config, secondRandom)))
      Assert.equal(first, second)
    },
    test("invalid policy values fail at construction") {
      val temperature = Assert.throws[IllegalArgumentException](SamplingConfig(temperature = 0.0))
      val topK = Assert.throws[IllegalArgumentException](SamplingConfig(topK = Some(0)))
      val topP = Assert.throws[IllegalArgumentException](SamplingConfig(topP = Some(1.1)))
      Assert.isTrue(temperature.getMessage.contains("temperature"))
      Assert.isTrue(topK.getMessage.contains("top-k"))
      Assert.isTrue(topP.getMessage.contains("top-p"))
    }
  )
