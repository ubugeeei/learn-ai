package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object TransformerBlockSuite extends TestSuite:
  override val name: String = "TransformerBlock"

  private def create(seed: Long): TransformerBlock =
    TransformerBlock.random(
      channels = 4,
      headCount = 2,
      hiddenChannels = 8,
      epsilon = 1e-5,
      random = new SplittableRandom(seed),
      label = "block0"
    )

  override val tests: Vector[TestCase] = specify(
    test("block preserves time and channel shape") {
      val block = create(seed = 1L)
      val input = Tensor.constant(Shape(3, 4), Vector.tabulate(12)(_.toDouble / 10.0))
      Assert.equal(block(input).shape, Shape(3, 4))
    },
    test("parameter count includes norms attention and feed-forward layers") {
      val block = create(seed = 2L)
      val attentionParameters = 4 * (4 * 4 + 4)
      val normParameters = 2 * 4
      val feedForwardParameters = (4 * 8 + 8) + (8 * 4 + 4)
      Assert.equal(
        block.parameters.map(_.size).sum,
        attentionParameters + normParameters + feedForwardParameters
      )
      Assert.equal(block.parameters.distinct.size, block.parameters.size)
    },
    test("a scalar loss sends finite gradients to input and every parameter") {
      val block = create(seed = 3L)
      val input = Tensor.parameter(Shape(3, 4), Vector.tabulate(12)(index => index * 0.03 + 0.1), "input")
      block(input).pow(2.0).mean.backward()

      Assert.isTrue(input.gradients.forall(_.isFinite))
      Assert.isTrue(input.gradients.exists(_ != 0.0))
      block.parameters.foreach { parameter =>
        Assert.isTrue(parameter.gradients.forall(_.isFinite), s"non-finite gradient in ${parameter.label}")
      }
      Assert.isTrue(block.parameters.flatMap(_.gradients).exists(_ != 0.0))
    },
    test("the complete block preserves prefix causality") {
      val block = create(seed = 4L)
      val prefix = Vector(1.0, 2.0, 3.0, 4.0)
      val first = Tensor.constant(
        Shape(3, 4),
        prefix ++ Vector(5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
      )
      val changed = Tensor.constant(
        Shape(3, 4),
        prefix ++ Vector(-50.0, 60.0, -70.0, 80.0, 90.0, -100.0, 110.0, -120.0)
      )
      block(first).rowValues(0).zip(block(changed).rowValues(0)).foreach {
        case (left, right) => Assert.close(left, right)
      }
    },
    test("invalid feed-forward width is rejected") {
      val error = Assert.throws[IllegalArgumentException] {
        TransformerBlock.random(
          4,
          2,
          hiddenChannels = 0,
          epsilon = 1e-5,
          new SplittableRandom(1L),
          "block"
        )
      }
      Assert.isTrue(error.getMessage.contains("hidden channels"))
    }
  )
