package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object RopeSuite extends TestSuite:
  override val name: String = "RotaryPositionEncoding"

  override val tests: Vector[TestCase] = specify(
    test("a single pair rotates by exactly position times theta") {
      // With headChannels = 2 there is one pair and theta_0 = base^0 = 1,
      // so the unit vector (1, 0) at position m must become (cos m, sin m).
      val rope = RotaryPositionEncoding.create(headChannels = 2)
      Vector(0, 1, 2, 7).foreach { position =>
        val rotated = rope.rotate(Tensor.constant(Shape(1, 2), Vector(1.0, 0.0)), position)
        Assert.close(rotated(0, 0), math.cos(position.toDouble), tolerance = 1e-12)
        Assert.close(rotated(0, 1), math.sin(position.toDouble), tolerance = 1e-12)
      }
    },
    test("pair frequencies follow the published geometric schedule") {
      val rope = RotaryPositionEncoding.create(headChannels = 4, base = 100.0)
      Assert.equal(rope.pairCount, 2)
      Assert.close(rope.frequencies(0), 1.0, tolerance = 1e-15)
      // theta_1 = 100^(-2/4) = 0.1
      Assert.close(rope.frequencies(1), 0.1, tolerance = 1e-12)
      Assert.close(rope.angle(position = 3, pair = 1), 0.3, tolerance = 1e-12)
    },
    test("position zero is the identity rotation") {
      val rope   = RotaryPositionEncoding.create(headChannels = 6)
      val values = Vector(0.5, -1.25, 2.0, 3.0, -0.75, 0.125)
      val input  = Tensor.constant(Shape(1, 6), values)
      rope.rotate(input, startPosition = 0).values.zip(values)
        .foreach { case (actual, expected) => Assert.close(actual, expected, tolerance = 1e-12) }
    },
    test("rotation preserves the Euclidean norm of every pair") {
      val rope    = RotaryPositionEncoding.create(headChannels = 8)
      val random  = new SplittableRandom(11L)
      val values  = Vector.fill(3 * 8)(random.nextDouble(-2.0, 2.0))
      val input   = Tensor.constant(Shape(3, 8), values)
      val rotated = rope.rotate(input, startPosition = 5)
      (0 until 3).foreach { row =>
        (0 until rope.pairCount).foreach { pair =>
          val first  = pair
          val second = pair + rope.pairCount
          val before = math.hypot(input(row, first), input(row, second))
          val after  = math.hypot(rotated(row, first), rotated(row, second))
          Assert.close(after, before, tolerance = 1e-12)
        }
      }
    },
    test("rotated dot products depend only on the relative offset") {
      val rope   = RotaryPositionEncoding.create(headChannels = 8)
      val random = new SplittableRandom(23L)
      val query  = Vector.fill(8)(random.nextDouble(-1.0, 1.0))
      val key    = Vector.fill(8)(random.nextDouble(-1.0, 1.0))

      def rotatedDot(queryPosition: Int, keyPosition: Int): Double =
        val rotatedQuery = rope.rotate(Tensor.constant(Shape(1, 8), query), queryPosition).values
        val rotatedKey   = rope.rotate(Tensor.constant(Shape(1, 8), key), keyPosition).values
        rotatedQuery.zip(rotatedKey).map(_ * _).sum

      val reference = rotatedDot(queryPosition = 9, keyPosition = 4)
      Vector(0, 1, 13, 40).foreach { shift =>
        Assert.close(
          rotatedDot(queryPosition = 9 + shift, keyPosition = 4 + shift),
          reference,
          tolerance = 1e-9
        )
      }
    },
    test("autodiff gradients through the rotation match finite differences") {
      val rope         = RotaryPositionEncoding.create(headChannels = 4)
      val random       = new SplittableRandom(31L)
      val inputValues  = Vector.fill(2 * 4)(random.nextDouble(-1.0, 1.0))
      val lossWeights  = Vector.fill(2 * 4)(random.nextDouble(-1.0, 1.0))
      val weightTensor = Tensor.constant(Shape(2, 4), lossWeights)

      def loss(values: Vector[Double]): Double = rope
        .rotate(Tensor.constant(Shape(2, 4), values), startPosition = 3).hadamard(weightTensor).sum
        .valueAtFlat(0)

      val parameter = Tensor.parameter(Shape(2, 4), inputValues, "ropeInput")
      rope.rotate(parameter, startPosition = 3).hadamard(weightTensor).sum.backward()

      val step = 1e-6
      inputValues.indices.foreach { index =>
        val bumpedUp   = inputValues.updated(index, inputValues(index) + step)
        val bumpedDown = inputValues.updated(index, inputValues(index) - step)
        val numerical  = (loss(bumpedUp) - loss(bumpedDown)) / (2.0 * step)
        Assert.close(parameter.gradientAtFlat(index), numerical, tolerance = 1e-6)
      }
    },
    test("rotary attention keeps causal masking intact") {
      val attention     = RotaryCausalSelfAttention.random(4, 2, new SplittableRandom(3L), "rotary")
      val base          = Tensor.constant(
        Shape(3, 4),
        Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
      )
      val changedFuture = Tensor.constant(
        Shape(3, 4),
        Vector(1.0, 2.0, 3.0, 4.0, -50.0, 60.0, -70.0, 80.0, 90.0, -100.0, 110.0, -120.0)
      )
      attention(base).rowValues(0).zip(attention(changedFuture).rowValues(0)).foreach {
        case (left, right) => Assert.close(left, right)
      }
      val result        = attention.forwardWithWeights(base)
      result.weightsByHead.foreach { weights =>
        (0 until 3).foreach { row =>
          Assert.close(weights.rowValues(row).sum, 1.0)
          (row + 1 until 3).foreach(future => Assert.close(weights(row, future), 0.0))
        }
      }
    },
    test("a common position offset does not change rotary attention output") {
      // This is the property that lets a sliding KV cache skip the
      // absolute-position rebuild required by learned position embeddings.
      val attention = RotaryCausalSelfAttention.random(8, 2, new SplittableRandom(7L), "rotary")
      val random    = new SplittableRandom(41L)
      val input     = Tensor.constant(Shape(4, 8), Vector.fill(32)(random.nextDouble(-1.0, 1.0)))
      val atZero    = attention(input, startPosition = 0)
      Vector(1, 5, 100).foreach { offset =>
        atZero.values.zip(attention(input, startPosition = offset).values).foreach {
          case (reference, shifted) => Assert.close(shifted, reference, tolerance = 1e-9)
        }
      }
    },
    test("rotary attention has identical parameter count to standard attention") {
      val rotary   = RotaryCausalSelfAttention.random(8, 2, new SplittableRandom(5L), "rotary")
      val standard = CausalSelfAttention.random(8, 2, new SplittableRandom(5L), "standard")
      Assert.equal(rotary.parameters.map(_.size).sum, standard.parameters.map(_.size).sum)
    },
    test("invalid construction and inputs are rejected") {
      val oddChannels = Assert
        .throws[IllegalArgumentException](RotaryPositionEncoding.create(headChannels = 3))
      Assert.isTrue(oddChannels.getMessage.contains("even"))
      val badBase     = Assert.throws[IllegalArgumentException] {
        RotaryPositionEncoding.create(headChannels = 4, base = 0.0)
      }
      Assert.isTrue(badBase.getMessage.contains("positive"))

      val rope             = RotaryPositionEncoding.create(headChannels = 4)
      val negativePosition = Assert.throws[IllegalArgumentException] {
        rope.rotate(Tensor.constant(Shape(1, 4), Vector.fill(4)(1.0)), startPosition = -1)
      }
      Assert.isTrue(negativePosition.getMessage.contains("non-negative"))
      val wrongShape       = Assert.throws[IllegalArgumentException] {
        rope.rotate(Tensor.constant(Shape(1, 6), Vector.fill(6)(1.0)))
      }
      Assert.isTrue(wrongShape.getMessage.contains("expected"))
      // channels = 6 with 2 heads gives an odd head width of 3, which RoPE
      // cannot pair; construction must fail before any projection is built.
      val oddHeadWidth     = Assert.throws[IllegalArgumentException] {
        RotaryCausalSelfAttention.random(6, 2, new SplittableRandom(1L), "rotary")
      }
      Assert.isTrue(oddHeadWidth.getMessage.contains("even"))
    }
  )
