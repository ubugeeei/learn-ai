package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object GroupedQueryAttentionSuite extends TestSuite:
  override val name: String = "GroupedQueryAttention"

  override val tests: Vector[TestCase] = Vector(
    test("equal query and key/value head counts reproduce multi-head attention") {
      // Independent oracle: the Chapter 19 implementation. Built from the
      // same projections, the grouped layer must match it to rounding error.
      val standard = CausalSelfAttention.random(8, 4, new SplittableRandom(1L), "standard")
      val grouped = GroupedQueryAttention.fromProjections(
        channels = 8,
        queryHeadCount = 4,
        keyValueHeadCount = 4,
        standard.queryProjection,
        standard.keyProjection,
        standard.valueProjection,
        standard.outputProjection
      )
      val random = new SplittableRandom(2L)
      val input = Tensor.constant(Shape(3, 8), Vector.fill(24)(random.nextDouble(-1.0, 1.0)))
      val standardResult = standard.forwardWithWeights(input)
      val groupedResult = grouped.forwardWithWeights(input)
      standardResult.output.values.zip(groupedResult.output.values).foreach {
        case (expected, actual) => Assert.close(actual, expected, tolerance = 1e-12)
      }
      standardResult.weightsByHead.zip(groupedResult.weightsByHead).foreach {
        case (expected, actual) =>
          expected.values.zip(actual.values).foreach { case (left, right) =>
            Assert.close(right, left, tolerance = 1e-12)
          }
      }
    },
    test("grouped sharing equals full-width attention with duplicated key/value columns") {
      // Oracle: expand the narrow key/value projections by duplicating each
      // key/value head's column block once per owning query head, then run
      // the (already MHA-verified) ungrouped configuration. Both layers see
      // identical mathematics if and only if group ownership is correct.
      val channels = 8
      val queryHeads = 4
      val keyValueHeads = 2
      val headChannels = channels / queryHeads
      val groupSize = queryHeads / keyValueHeads
      val keyValueChannels = keyValueHeads * headChannels
      val random = new SplittableRandom(3L)

      def randomValues(count: Int): Vector[Double] =
        Vector.fill(count)(random.nextDouble(-0.5, 0.5))

      val queryProjection =
        Linear.fromValues(channels, channels, randomValues(64), randomValues(8), "query")
      val outputProjection =
        Linear.fromValues(channels, channels, randomValues(64), randomValues(8), "output")
      val keyWeights = randomValues(channels * keyValueChannels)
      val keyBiases = randomValues(keyValueChannels)
      val valueWeights = randomValues(channels * keyValueChannels)
      val valueBiases = randomValues(keyValueChannels)

      def expandRow(row: Vector[Double]): Vector[Double] =
        row.grouped(headChannels).toVector.flatMap { block =>
          Vector.fill(groupSize)(block).flatten
        }
      def expandWeights(weights: Vector[Double]): Vector[Double] =
        weights.grouped(keyValueChannels).toVector.flatMap(expandRow)

      val grouped = GroupedQueryAttention.fromProjections(
        channels,
        queryHeads,
        keyValueHeads,
        queryProjection,
        Linear.fromValues(channels, keyValueChannels, keyWeights, keyBiases, "key"),
        Linear.fromValues(channels, keyValueChannels, valueWeights, valueBiases, "value"),
        outputProjection
      )
      val expanded = GroupedQueryAttention.fromProjections(
        channels,
        queryHeads,
        queryHeads,
        queryProjection,
        Linear.fromValues(channels, channels, expandWeights(keyWeights), expandRow(keyBiases), "key"),
        Linear.fromValues(channels, channels, expandWeights(valueWeights), expandRow(valueBiases), "value"),
        outputProjection
      )
      val input = Tensor.constant(Shape(4, channels), randomValues(4 * channels))
      grouped(input).values.zip(expanded(input).values).foreach { case (actual, expected) =>
        Assert.close(actual, expected, tolerance = 1e-12)
      }
      Assert.equal(Vector(0, 1, 2, 3).map(grouped.keyValueHeadFor), Vector(0, 0, 1, 1))
    },
    test("multi-query attention shares a single key/value head across all queries") {
      val attention =
        GroupedQueryAttention.random(8, 4, 1, new SplittableRandom(4L), "mqa")
      Assert.equal(attention.groupSize, 4)
      Assert.equal(Vector(0, 1, 2, 3).map(attention.keyValueHeadFor), Vector(0, 0, 0, 0))
      Assert.equal(attention.keyValueChannels, 2)
      // Cached payload per token shrinks by exactly the head ratio.
      val fullWidth =
        GroupedQueryAttention.random(8, 4, 4, new SplittableRandom(4L), "mha")
      Assert.equal(
        fullWidth.keyValuePayloadBytesPerToken,
        attention.keyValuePayloadBytesPerToken * 4L
      )
    },
    test("causal masking holds for every query head") {
      val attention =
        GroupedQueryAttention.random(8, 4, 2, new SplittableRandom(5L), "gqa")
      val random = new SplittableRandom(6L)
      val prefix = Vector.fill(2 * 8)(random.nextDouble(-1.0, 1.0))
      val futureA = Vector.fill(8)(random.nextDouble(-1.0, 1.0))
      val futureB = Vector.fill(8)(random.nextDouble(-1.0, 1.0))
      val first = attention(Tensor.constant(Shape(3, 8), prefix ++ futureA))
      val second = attention(Tensor.constant(Shape(3, 8), prefix ++ futureB))
      (0 until 2).foreach { row =>
        first.rowValues(row).zip(second.rowValues(row)).foreach { case (left, right) =>
          Assert.close(left, right, tolerance = 1e-12)
        }
      }
      val result = attention.forwardWithWeights(Tensor.constant(Shape(3, 8), prefix ++ futureA))
      Assert.equal(result.weightsByHead.size, 4)
      result.weightsByHead.foreach { weights =>
        (0 until 3).foreach { row =>
          Assert.close(weights.rowValues(row).sum, 1.0)
          (row + 1 until 3).foreach(future => Assert.close(weights(row, future), 0.0))
        }
      }
    },
    test("future inputs receive no gradient from a prefix-only loss") {
      val attention =
        GroupedQueryAttention.random(4, 2, 1, new SplittableRandom(7L), "gqa")
      val input = Tensor.parameter(Shape(3, 4), Vector.tabulate(12)(_.toDouble / 10.0), "input")
      val firstPositionLoss = attention(input).gatherRows(Vector(0)).sum
      firstPositionLoss.backward()
      Assert.isTrue(input.gradients.slice(4, 12).forall(_ == 0.0))
      Assert.isTrue(input.gradients.take(4).exists(_ != 0.0))
    },
    test("parameter and cache-byte accounting follow the closed formulas") {
      val channels = 12
      val attention =
        GroupedQueryAttention.random(channels, 6, 2, new SplittableRandom(8L), "gqa")
      val keyValueChannels = 2 * (channels / 6)
      Assert.equal(attention.keyValueChannels, keyValueChannels)
      Assert.equal(
        attention.parameterCount,
        2 * (channels * channels + channels) +
          2 * (channels * keyValueChannels + keyValueChannels)
      )
      Assert.equal(
        attention.keyValuePayloadBytesPerToken,
        2L * keyValueChannels.toLong * 8L
      )
      Assert.equal(
        attention.cachePayloadBytes(contextLength = 128, layerCount = 4),
        attention.keyValuePayloadBytesPerToken * 128L * 4L
      )
      val fullWidth =
        GroupedQueryAttention.random(channels, 6, 6, new SplittableRandom(8L), "mha")
      Assert.isTrue(
        attention.parameterCount < fullWidth.parameterCount,
        "narrow key/value projections must reduce the parameter count"
      )
    },
    test("invalid configurations and inputs are rejected") {
      val indivisibleChannels = Assert.throws[IllegalArgumentException] {
        GroupedQueryAttention.random(10, 4, 2, new SplittableRandom(1L), "gqa")
      }
      Assert.isTrue(indivisibleChannels.getMessage.contains("not divisible"))
      val indivisibleHeads = Assert.throws[IllegalArgumentException] {
        GroupedQueryAttention.random(12, 4, 3, new SplittableRandom(1L), "gqa")
      }
      Assert.isTrue(indivisibleHeads.getMessage.contains("not divisible"))
      val moreKeyValueThanQuery = Assert.throws[IllegalArgumentException] {
        GroupedQueryAttention.random(8, 2, 4, new SplittableRandom(1L), "gqa")
      }
      Assert.isTrue(moreKeyValueThanQuery.getMessage.contains("not divisible"))

      val attention = GroupedQueryAttention.random(8, 4, 2, new SplittableRandom(1L), "gqa")
      val wrongWidth = Assert.throws[IllegalArgumentException] {
        attention(Tensor.constant(Shape(2, 4), Vector.fill(8)(1.0)))
      }
      Assert.isTrue(wrongWidth.getMessage.contains("expected"))
      val badContext = Assert.throws[IllegalArgumentException] {
        attention.cachePayloadBytes(contextLength = 0, layerCount = 1)
      }
      Assert.isTrue(badContext.getMessage.contains("positive"))

      val narrowKey = Linear.random(8, 2, new SplittableRandom(1L), "narrowKey")
      val mismatchedProjection = Assert.throws[IllegalArgumentException] {
        GroupedQueryAttention.fromProjections(
          channels = 8,
          queryHeadCount = 4,
          keyValueHeadCount = 4,
          Linear.random(8, 8, new SplittableRandom(1L), "query"),
          narrowKey,
          Linear.random(8, 8, new SplittableRandom(1L), "value"),
          Linear.random(8, 8, new SplittableRandom(1L), "output")
        )
      }
      Assert.isTrue(mismatchedProjection.getMessage.contains("key projection"))
    }
  )
