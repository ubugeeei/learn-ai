package learnai.lm

import java.util.SplittableRandom

import learnai.text.TokenId
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object BigramLanguageModelSuite extends TestSuite:
  override val name: String = "BigramLanguageModel"

  private val cycleInputs = Vector(0, 1, 2, 0, 1, 2).map(TokenId(_))
  private val cycleTargets = Vector(1, 2, 0, 1, 2, 0).map(TokenId(_))

  override val tests: Vector[TestCase] = Vector(
    test("logit lookup returns one transition row per input token") {
      val model = BigramLanguageModel.fromLogits(
        3,
        Vector(
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 9.0
        )
      )
      val logits = model(Vector(TokenId(2), TokenId(0)))
      Assert.equal(logits.shape, learnai.tensor.Shape(2, 3))
      Assert.equal(logits.values, Vector(7.0, 8.0, 9.0, 1.0, 2.0, 3.0))
    },
    test("training lowers loss and learns a deterministic three-token cycle") {
      val model = BigramLanguageModel.random(vocabularySize = 3, seed = 11L)
      val history = BigramTrainer.train(
        model,
        cycleInputs,
        cycleTargets,
        steps = 150,
        learningRate = 0.1
      )
      Assert.isTrue(history.last < history.head * 0.02)
      Vector(0 -> 1, 1 -> 2, 2 -> 0).foreach { case (current, expected) =>
        val distribution = Assert.right(model.nextDistribution(TokenId(current)))
        Assert.equal(Assert.right(distribution.probabilities.argmax), expected)
      }
    },
    test("generation includes its start token and is seed reproducible") {
      val model = BigramLanguageModel.fromLogits(
        2,
        Vector(0.0, 0.0, 0.0, 0.0)
      )
      val first = Assert.right(
        model.generate(TokenId(0), 20, 1.0, new SplittableRandom(99L))
      )
      val second = Assert.right(
        model.generate(TokenId(0), 20, 1.0, new SplittableRandom(99L))
      )
      Assert.equal(first, second)
      Assert.equal(first.size, 21)
      Assert.equal(first.head, TokenId(0))
    },
    test("zero new tokens returns only the validated start token") {
      val model = BigramLanguageModel.random(2, seed = 1L)
      Assert.equal(
        model.generate(TokenId(1), 0, 1.0, new SplittableRandom(2L)),
        Right(Vector(TokenId(1)))
      )
    },
    test("invalid token IDs and temperatures are rejected") {
      val model = BigramLanguageModel.random(2, seed = 1L)
      val tokenError = Assert.throws[IllegalArgumentException] {
        model.loss(Vector(TokenId(2)), Vector(TokenId(0)))
      }
      val temperatureError = Assert.throws[IllegalArgumentException] {
        model.nextDistribution(TokenId(0), temperature = 0.0)
      }
      Assert.isTrue(tokenError.getMessage.contains("outside"))
      Assert.isTrue(temperatureError.getMessage.contains("temperature"))
    }
  )
