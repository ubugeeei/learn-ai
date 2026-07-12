package learnai.transformer

import java.util.SplittableRandom

import learnai.lm.SamplingConfig
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

  override val tests: Vector[TestCase] = specify(
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
    test("cached inference matches full-prefix logits at every position") {
      val model = MiniGpt.random(config, seed = 6L)
      val session = new MiniGptInferenceSession(model)
      val tokens = Vector(0, 1, 2, 3).map(TokenId(_))
      tokens.indices.foreach { index =>
        val cached = session.append(tokens(index))
        val reference = model.logits(tokens.take(index + 1)).rowValues(index)
        cached.toVector.zip(reference).foreach { case (actual, expected) =>
          Assert.close(actual, expected, tolerance = 1e-11)
        }
      }
      Assert.equal(session.length, config.maximumContextLength)
      Assert.equal(session.evaluatedTokens, 4L)
      Assert.equal(
        session.usedCachePayloadBytes,
        config.layerCount.toLong * 2L * config.channels.toLong * 4L * 8L
      )
    },
    test("cached generation matches the reference across context rebuilds") {
      val model = MiniGpt.random(config, seed = 7L)
      val prompt = Vector(0, 1, 2, 3).map(TokenId(_))
      val reference = Assert.right(
        model.generate(prompt, 3, temperature = 0.9, new SplittableRandom(11L))
      )
      val cached = Assert.right(
        model.generateCached(
          prompt,
          newTokenCount = 3,
          SamplingConfig(temperature = 0.9),
          new SplittableRandom(11L)
        )
      )
      Assert.equal(cached.tokens, reference)
      Assert.equal(cached.statistics.cacheRebuilds, 2)
      Assert.equal(cached.statistics.tokenEvaluations, 12L)
      Assert.equal(cached.statistics.referenceTokenEvaluations, 12L)
    },
    test("KV caching reduces deterministic token work before capacity") {
      val model = MiniGpt.random(config, seed = 8L)
      val result = Assert.right(
        model.generateCached(
          Vector(0, 1).map(TokenId(_)),
          newTokenCount = 3,
          SamplingConfig(temperature = 1.0, topK = Some(1)),
          new SplittableRandom(12L)
        )
      )
      Assert.equal(result.statistics.tokenEvaluations, 4L)
      Assert.equal(result.statistics.referenceTokenEvaluations, 9L)
      Assert.equal(result.statistics.cacheRebuilds, 0)
      Assert.equal(result.statistics.peakCachedTokens, 4)
      Assert.equal(result.tokens.size, 5)
    },
    test("inference sessions make capacity and reset ownership explicit") {
      val model = MiniGpt.random(config, seed = 9L)
      val session = new MiniGptInferenceSession(model)
      val _ = session.prefill(Vector.fill(config.maximumContextLength)(TokenId(0)))
      val overflow = Assert.throws[IllegalArgumentException] {
        session.append(TokenId(0))
      }
      Assert.isTrue(overflow.getMessage.contains("full"))
      session.reset()
      Assert.equal(session.length, 0)
      Assert.equal(session.evaluatedTokens, 0L)
      Assert.isTrue(session.lastLogits.isEmpty)
    },
    test("invalid configuration sequence and target values fail near the boundary") {
      val configError = Assert.throws[IllegalArgumentException] {
        config.copy(channels = 5)
      }
      val model = MiniGpt.random(config, seed = 10L)
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
