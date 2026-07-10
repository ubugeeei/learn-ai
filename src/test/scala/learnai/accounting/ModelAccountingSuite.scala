package learnai.accounting

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.transformer.AttentionKeyValueCache
import learnai.transformer.CausalSelfAttention
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object ModelAccountingSuite extends TestSuite:
  override val name: String = "ModelAccounting"

  private val configurations = Vector(
    MiniGptConfig(
      vocabularySize = 5,
      maximumContextLength = 4,
      channels = 4,
      headCount = 2,
      hiddenChannels = 8,
      layerCount = 1
    ),
    MiniGptConfig(
      vocabularySize = 64,
      maximumContextLength = 16,
      channels = 12,
      headCount = 3,
      hiddenChannels = 24,
      layerCount = 2
    ),
    MiniGptConfig(
      vocabularySize = 300,
      maximumContextLength = 32,
      channels = 16,
      headCount = 4,
      hiddenChannels = 64,
      layerCount = 3
    )
  )

  override val tests: Vector[TestCase] = Vector(
    test("the parameter formula matches the real model for every configuration") {
      configurations.foreach { config =>
        val model = MiniGpt.random(config, seed = 1L)
        Assert.equal(ModelAccounting.parameterCount(config), model.parameterCount.toLong)
      }
    },
    test("a decode step's FLOPs match an independent hand calculation") {
      // C = 2, F = 4, L = 1, V = 3, attending to 2 positions:
      // projections 8 * 2 * 2 = 32; feed-forward 4 * 2 * 4 = 32;
      // attention 4 * 2 * 2 = 16; logits 2 * 2 * 3 = 12; total 92.
      val config = MiniGptConfig(
        vocabularySize = 3,
        maximumContextLength = 4,
        channels = 2,
        headCount = 1,
        hiddenChannels = 4,
        layerCount = 1
      )
      Assert.equal(ModelAccounting.decodeStepFlops(config, attendedPositions = 2), 92L)
      // One more attended position adds exactly the 4C attention term.
      Assert.equal(
        ModelAccounting.decodeStepFlops(config, 3) -
          ModelAccounting.decodeStepFlops(config, 2),
        4L * 2L
      )
    },
    test("prefill equals the sum of its decode steps for every configuration") {
      configurations.foreach { config =>
        val tokens = config.maximumContextLength
        val summed = (1 to tokens).map { position =>
          ModelAccounting.decodeStepFlops(config, position)
        }.sum
        Assert.equal(ModelAccounting.prefillFlops(config, tokens), summed)
      }
    },
    test("attention weight accounting matches the tensors the layer retains") {
      val config = configurations(1)
      val attention = CausalSelfAttention.random(
        config.channels,
        config.headCount,
        new SplittableRandom(2L),
        "attention"
      )
      val contextLength = 5
      val input = Tensor.constant(
        Shape(contextLength, config.channels),
        Vector.fill(contextLength * config.channels)(0.25)
      )
      val retainedValues = attention
        .forwardWithWeights(input)
        .weightsByHead
        .map(_.size.toLong)
        .sum
      val perLayerEstimate =
        ModelAccounting.attentionWeightValues(config, contextLength) / config.layerCount
      Assert.equal(perLayerEstimate, retainedValues)
    },
    test("KV cache accounting matches a real allocated cache") {
      val config = configurations(1)
      val contextLength = 7
      val cache = AttentionKeyValueCache.create(config.channels, capacity = contextLength)
      Assert.equal(
        ModelAccounting.kvCachePayloadBytes(config, contextLength),
        cache.allocatedPayloadBytes * config.layerCount
      )
    },
    test("training memory is four times the parameter bytes") {
      configurations.foreach { config =>
        val parameterBytes = ModelAccounting.parameterBytes(config)
        Assert.equal(parameterBytes, ModelAccounting.parameterCount(config) * 8L)
        Assert.equal(ModelAccounting.gradientBytes(config), parameterBytes)
        Assert.equal(ModelAccounting.adamWStateBytes(config), 2L * parameterBytes)
        Assert.equal(ModelAccounting.trainingResidentBytes(config), 4L * parameterBytes)
      }
    },
    test("invalid context and token arguments are rejected") {
      val config = configurations.head
      val zeroAttended = Assert.throws[IllegalArgumentException] {
        ModelAccounting.decodeStepFlops(config, 0)
      }
      Assert.isTrue(zeroAttended.getMessage.contains("positive"))
      val overContext = Assert.throws[IllegalArgumentException] {
        ModelAccounting.decodeStepFlops(config, config.maximumContextLength + 1)
      }
      Assert.isTrue(overContext.getMessage.contains("exceed"))
      val overPrefill = Assert.throws[IllegalArgumentException] {
        ModelAccounting.prefillFlops(config, config.maximumContextLength + 1)
      }
      Assert.isTrue(overPrefill.getMessage.contains("exceeds"))
      val zeroCache = Assert.throws[IllegalArgumentException] {
        ModelAccounting.kvCachePayloadBytes(config, 0)
      }
      Assert.isTrue(zeroCache.getMessage.contains("positive"))
    }
  )
