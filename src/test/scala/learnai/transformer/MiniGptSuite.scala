package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.text.TokenId
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object MiniGptSuite extends TestSuite:
  override val name: String = "MiniGpt"

  private val config = MiniGptConfig(
    vocabularySize = 5,
    maximumContextLength = 4,
    channels = 4,
    headCount = 2,
    hiddenChannels = 8,
    layerCount = 1
  )

  override val tests: Vector[TestCase] = Vector(
    test("logits have one vocabulary row per input position") {
      val model = MiniGpt.random(config, seed = 1L)
      val logits = model.logits(Vector(TokenId(0), TokenId(1), TokenId(2)))
      Assert.equal(logits.shape, Shape(3, 5))
      Assert.isTrue(logits.values.forall(_.isFinite))
    },
    test("parameter count includes a tied token embedding only once") {
      val model = MiniGpt.random(config, seed = 2L)
      val embeddings = 5 * 4 + 4 * 4
      val block = 4 * (4 * 4 + 4) + 2 * 4 + (4 * 8 + 8) + (8 * 4 + 4)
      val finalNorm = 4
      Assert.equal(model.parameterCount, embeddings + block + finalNorm)
      Assert.equal(model.parameters.distinct.size, model.parameters.size)
      Assert.isTrue(!model.parameters.exists(_.label.contains("lmHead")))
    },
    test("training backpropagates through the full model and lowers loss") {
      val model = MiniGpt.random(config, seed = 3L)
      val inputs = Vector(0, 1, 0, 2).map(TokenId(_))
      val targets = Vector(1, 0, 2, 0).map(TokenId(_))
      val history = MiniGptTrainer.trainSequence(
        model,
        inputs,
        targets,
        steps = 120,
        learningRate = 0.03
      )
      Assert.isTrue(history.last < history.head * 0.1)
      Assert.isTrue(history.last < 0.2)
    },
    test("changing a future token leaves earlier logits unchanged") {
      val model = MiniGpt.random(config, seed = 4L)
      val original = model.logits(Vector(0, 1, 2, 3).map(TokenId(_))).rowValues(1)
      val changed = model.logits(Vector(0, 1, 4, 4).map(TokenId(_))).rowValues(1)
      original.zip(changed).foreach { case (left, right) => Assert.close(left, right) }
    },
    test("generation is reproducible and crops context beyond its maximum") {
      val model = MiniGpt.random(config, seed = 5L)
      val prompt = Vector(0, 1, 2, 3, 4, 0).map(TokenId(_))
      val first = Assert.right(model.generate(prompt, 5, 1.0, new SplittableRandom(9L)))
      val second = Assert.right(model.generate(prompt, 5, 1.0, new SplittableRandom(9L)))
      Assert.equal(first, second)
      Assert.equal(first.take(prompt.size), prompt)
      Assert.equal(first.size, prompt.size + 5)
    },
    test("invalid configuration sequence and target values fail near the boundary") {
      val configError = Assert.throws[IllegalArgumentException] {
        config.copy(channels = 5)
      }
      val model = MiniGpt.random(config, seed = 6L)
      val lengthError = Assert.throws[IllegalArgumentException] {
        model.logits(Vector.fill(5)(TokenId(0)))
      }
      val targetError = Assert.throws[IllegalArgumentException] {
        model.loss(Vector(TokenId(0)), Vector(TokenId(5)))
      }
      val promptError = Assert.throws[IllegalArgumentException] {
        model.generate(Vector.empty, 1, 1.0, new SplittableRandom(1L))
      }
      Assert.isTrue(configError.getMessage.contains("not divisible"))
      Assert.isTrue(lengthError.getMessage.contains("exceeds"))
      Assert.isTrue(targetError.getMessage.contains("outside"))
      Assert.isTrue(promptError.getMessage.contains("non-empty prompt"))
    }
  )
