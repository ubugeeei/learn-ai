package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object AttentionSuite extends TestSuite:
  override val name: String = "CausalSelfAttention"

  override val tests: Vector[TestCase] = specify(
    test("each attention row is normalized and assigns zero weight to the future") {
      val attention = CausalSelfAttention.random(4, 2, new SplittableRandom(1L), "attention")
      val input     = Tensor
        .constant(Shape(3, 4), Vector(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
      val result    = attention.forwardWithWeights(input)
      Assert.equal(result.output.shape, Shape(3, 4))
      Assert.equal(result.weightsByHead.size, 2)
      result.weightsByHead.foreach { weights =>
        Vector(0, 1, 2).foreach { row =>
          Assert.close(weights.rowValues(row).sum, 1.0)
          (row + 1 until 3).foreach(future => Assert.close(weights(row, future), 0.0))
        }
      }
    },
    test("changing future tokens cannot change an earlier output") {
      val attention     = CausalSelfAttention.random(4, 2, new SplittableRandom(2L), "attention")
      val first         = Tensor.constant(
        Shape(3, 4),
        Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
      )
      val changedFuture = Tensor.constant(
        Shape(3, 4),
        Vector(1.0, 2.0, 3.0, 4.0, -50.0, 60.0, -70.0, 80.0, 90.0, -100.0, 110.0, -120.0)
      )
      val firstOutput   = attention(first).rowValues(0)
      val changedOutput = attention(changedFuture).rowValues(0)
      firstOutput.zip(changedOutput).foreach { case (left, right) => Assert.close(left, right) }
    },
    test("future inputs receive no gradient from a prefix-only loss") {
      val attention         = CausalSelfAttention.random(4, 2, new SplittableRandom(3L), "attention")
      val input             = Tensor.parameter(Shape(3, 4), Vector.tabulate(12)(_.toDouble / 10.0), "input")
      val firstPositionLoss = attention(input).sliceColumns(0, 4).reshape(Shape(3, 4))
        .gatherRows(Vector(0)).sum
      firstPositionLoss.backward()
      Assert.isTrue(input.gradients.slice(4, 12).forall(_ == 0.0))
      Assert.isTrue(input.gradients.take(4).exists(_ != 0.0))
    },
    test("multi-head split and concatenation preserve channel width") {
      val attention = CausalSelfAttention.random(6, 3, new SplittableRandom(4L), "attention")
      val output    = attention(Tensor.constant(Shape(2, 6), Vector.fill(12)(0.25)))
      Assert.equal(attention.headChannels, 2)
      Assert.equal(output.shape, Shape(2, 6))
      Assert.equal(attention.parameters.map(_.size).sum, 4 * (6 * 6 + 6))
    },
    test("cached one-token attention matches every full-prefix final row") {
      val attention = CausalSelfAttention.random(4, 2, new SplittableRandom(5L), "attention")
      val rows      = Vector(
        Vector(1.0, 2.0, 3.0, 4.0),
        Vector(-1.0, 0.5, 2.0, -0.25),
        Vector(3.0, -2.0, 1.0, 0.0)
      )
      val cache     = AttentionKeyValueCache.create(channels = 4, capacity = rows.size)

      rows.indices.foreach { index =>
        val cached    = attention.forwardCached(Tensor.constant(Shape(1, 4), rows(index)), cache)
        val prefix    = Tensor.constant(Shape(index + 1, 4), rows.take(index + 1).flatten)
        val reference = attention(prefix).rowValues(index)
        cached.rowValues(0).zip(reference).foreach { case (actual, expected) =>
          Assert.close(actual, expected, tolerance = 1e-12)
        }
      }

      Assert.equal(cache.length, 3)
      Assert.equal(cache.usedPayloadBytes, 3L * 4L * 2L * 8L)
      Assert.equal(cache.usedPayloadBytes, cache.allocatedPayloadBytes)
    },
    test("a fixed KV cache rejects overflow and can be reused after clear") {
      val attention = CausalSelfAttention.random(4, 2, new SplittableRandom(6L), "attention")
      val cache     = AttentionKeyValueCache.create(channels = 4, capacity = 1)
      val token     = Tensor.constant(Shape(1, 4), Vector.fill(4)(0.25))
      val _         = attention.forwardCached(token, cache)
      val overflow  = Assert.throws[IllegalArgumentException](attention.forwardCached(token, cache))
      Assert.isTrue(overflow.getMessage.contains("exhausted"))
      cache.clear()
      Assert.equal(cache.length, 0)
      Assert.equal(attention.forwardCached(token, cache).shape, Shape(1, 4))
    },
    test("head count must divide channels") {
      val error = Assert.throws[IllegalArgumentException] {
        CausalSelfAttention.random(5, 2, new SplittableRandom(1L), "attention")
      }
      Assert.isTrue(error.getMessage.contains("not divisible"))
    }
  )
