package learnai.serving

import learnai.random.SplitMix64
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.text.TokenId.*
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object SpeculativeDecodingSuite extends TestSuite:
  override val name: String = "SpeculativeDecoding"

  private val adversarialPairs = Vector(
    (Vector(0.7, 0.2, 0.1), Vector(0.1, 0.2, 0.7)),
    (Vector(0.5, 0.5, 0.0), Vector(0.0, 0.5, 0.5)),
    (Vector(0.25, 0.25, 0.25, 0.25), Vector(0.97, 0.01, 0.01, 0.01)),
    (Vector(1.0, 0.0), Vector(0.5, 0.5)),
    (Vector(0.6, 0.3, 0.1), Vector(0.6, 0.3, 0.1))
  )

  override val tests: Vector[TestCase] = specify(
    test("the emitted distribution equals the target for adversarial pairs") {
      // Executable Theorem 1: min(p,q) + (1 - alpha) * residual == p,
      // computed from the implementation's own accept and residual rules.
      adversarialPairs.foreach { case (target, draft) =>
        val emitted = SpeculativeSampling.outputDistribution(target, draft)
        emitted.zip(target).foreach { case (actual, expected) =>
          Assert.close(actual, expected, tolerance = 1e-12)
        }
      }
    },
    test("acceptance probability is the overlap sum with its boundary cases") {
      Assert.close(
        SpeculativeSampling.acceptanceProbability(Vector(0.7, 0.2, 0.1), Vector(0.1, 0.2, 0.7)),
        0.1 + 0.2 + 0.1,
        tolerance = 1e-15
      )
      val identical = Vector(0.6, 0.3, 0.1)
      Assert.close(
        SpeculativeSampling.acceptanceProbability(identical, identical),
        1.0,
        tolerance = 1e-15
      )
      // Nearly disjoint supports accept almost nothing.
      Assert.close(
        SpeculativeSampling.acceptanceProbability(Vector(1.0, 0.0), Vector(0.0, 1.0)),
        0.0,
        tolerance = 1e-15
      )
    },
    test("the residual is supported exactly where the target exceeds the draft") {
      val target    = Vector(0.7, 0.2, 0.1)
      val draft     = Vector(0.1, 0.2, 0.7)
      val residual  = Assert.right(SpeculativeSampling.residualDistribution(target, draft))
      Assert.close(residual.sum, 1.0, tolerance = 1e-15)
      Assert.close(residual(0), 1.0, tolerance = 1e-15)
      Assert.equal(residual(1), 0.0)
      Assert.equal(residual(2), 0.0)
      val undefined = SpeculativeSampling.residualDistribution(target, target)
      Assert.isTrue(Assert.left(undefined).contains("undefined"))
    },
    test("an identical draft is always accepted") {
      val distribution = Vector(0.5, 0.3, 0.2)
      val random       = SplitMix64.seeded(7L)
      (0 until 500).foreach { _ =>
        val token        = sampleFrom(distribution, random)
        val verification = SpeculativeSampling
          .verifyDraftToken(distribution, distribution, token, random)
        Assert.isTrue(verification.accepted, "identical distributions must always accept")
        Assert.equal(verification.token, token)
      }
    },
    test("Monte Carlo output frequencies converge to the target distribution") {
      val target = Vector(0.6, 0.25, 0.1, 0.05)
      val draft  = Vector(0.1, 0.3, 0.35, 0.25)
      val random = SplitMix64.seeded(13L)
      val draws  = 40_000
      val counts = new Array[Int](target.size)
      var draw   = 0
      while draw < draws do
        val proposal     = sampleFrom(draft, random)
        val verification = SpeculativeSampling.verifyDraftToken(target, draft, proposal, random)
        counts(verification.token) += 1
        draw += 1
      counts.zipWithIndex.foreach { case (count, token) =>
        val frequency = count.toDouble / draws.toDouble
        Assert.close(frequency, target(token), tolerance = 0.01)
      }
    },
    test("a draft identical to the target never rejects end to end") {
      val config = tinyConfig
      val model  = MiniGpt.random(config, seed = 3L)
      val prompt = Vector(0, 1, 2).map(TokenId(_))
      val result = Assert.right(SpeculativeDecoding.generate(
        target = model,
        draft = model,
        prompt = prompt,
        newTokenCount = 12,
        lookahead = 3,
        temperature = 1.0,
        random = SplitMix64.seeded(5L)
      ))
      Assert.equal(result.tokens.size, prompt.size + 12)
      Assert.equal(result.statistics.rejectedDrafts, 0)
      Assert.close(result.statistics.acceptanceRate, 1.0, tolerance = 0.0)
      Assert.equal(result.statistics.emittedTokens, 12)
    },
    test("a different draft still emits the requested tokens with consistent accounting") {
      val config     = tinyConfig
      val target     = MiniGpt.random(config, seed = 3L)
      val draft      = MiniGpt.random(config, seed = 99L)
      val prompt     = Vector(0, 1).map(TokenId(_))
      val result     = Assert.right(SpeculativeDecoding.generate(
        target,
        draft,
        prompt,
        newTokenCount = 15,
        lookahead = 4,
        temperature = 1.0,
        random = SplitMix64.seeded(11L)
      ))
      val statistics = result.statistics
      Assert.equal(result.tokens.size, prompt.size + 15)
      Assert.equal(statistics.emittedTokens, 15)
      Assert.isTrue(statistics.acceptanceRate >= 0.0 && statistics.acceptanceRate <= 1.0)
      Assert.equal(statistics.targetVerificationPasses, statistics.rounds.toLong)
      Assert.isTrue(statistics.rounds <= 15, "each round must emit at least one token")
      result.tokens.foreach { token =>
        Assert.isTrue(
          token.value >= 0 && token.value < config.vocabularySize,
          s"token ${token.value} outside the vocabulary"
        )
      }
    },
    test("invalid distributions tokens and configurations are rejected") {
      val notNormalized   = Assert.throws[IllegalArgumentException] {
        SpeculativeSampling.acceptanceProbability(Vector(0.9, 0.3), Vector(0.5, 0.5))
      }
      Assert.isTrue(notNormalized.getMessage.contains("sums to"))
      val negative        = Assert.throws[IllegalArgumentException] {
        SpeculativeSampling.acceptanceProbability(Vector(1.2, -0.2), Vector(0.5, 0.5))
      }
      Assert.isTrue(negative.getMessage.contains("non-negative"))
      val sizeMismatch    = Assert.throws[IllegalArgumentException] {
        SpeculativeSampling.acceptanceProbability(Vector(1.0), Vector(0.5, 0.5))
      }
      Assert.isTrue(sizeMismatch.getMessage.contains("size"))
      val impossibleDraft = Assert.throws[IllegalArgumentException] {
        SpeculativeSampling.verifyDraftToken(
          Vector(0.5, 0.5),
          Vector(1.0, 0.0),
          draftToken = 1,
          SplitMix64.seeded(1L)
        )
      }
      Assert.isTrue(impossibleDraft.getMessage.contains("zero draft probability"))

      val config             = tinyConfig
      val model              = MiniGpt.random(config, seed = 1L)
      val other              = MiniGpt.random(config.copy(vocabularySize = 6), seed = 1L)
      val vocabularyMismatch = Assert.throws[IllegalArgumentException] {
        SpeculativeDecoding
          .generate(model, other, Vector(TokenId(0)), 4, 2, 1.0, SplitMix64.seeded(1L))
      }
      Assert.isTrue(vocabularyMismatch.getMessage.contains("vocabulary mismatch"))
      val hugeLookahead      = Assert.throws[IllegalArgumentException] {
        SpeculativeDecoding.generate(
          model,
          model,
          Vector(TokenId(0)),
          4,
          lookahead = config.maximumContextLength,
          temperature = 1.0,
          random = SplitMix64.seeded(1L)
        )
      }
      Assert.isTrue(hugeLookahead.getMessage.contains("context room"))
    }
  )

  private val tinyConfig = MiniGptConfig(
    vocabularySize = 5,
    maximumContextLength = 8,
    channels = 4,
    headCount = 2,
    hiddenChannels = 8,
    layerCount = 1
  )

  /** Inverse-CDF draft sampler independent of the code under test. */
  private def sampleFrom(distribution: Vector[Double], random: SplitMix64): Int =
    val threshold  = random.nextDouble()
    var cumulative = 0.0
    var index      = 0
    var chosen     = -1
    while index < distribution.size && chosen < 0 do
      cumulative += distribution(index)
      if cumulative > threshold then chosen = index
      index += 1
    if chosen >= 0 then chosen else distribution.size - 1
