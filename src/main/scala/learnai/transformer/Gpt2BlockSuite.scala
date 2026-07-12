package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Gpt2BlockSuite extends TestSuite:
  override val name: String = "Gpt2Block"

  override val tests: Vector[TestCase] = specify(
    test("LayerNorm gives each row zero mean and unit variance") {
      val norm = LayerNorm.create(3, 1e-12, "norm")
      val output = norm(Tensor.constant(Shape(2, 3), Vector(1, 2, 3, 10, 20, 30)))
      Vector(0, 1).foreach { row =>
        val values = Vector.tabulate(3)(column => output(row, column))
        val mean = values.sum / 3.0
        val variance = values.map(value => math.pow(value - mean, 2)).sum / 3.0
        Assert.close(mean, 0.0, 1e-10)
        Assert.close(variance, 1.0, 1e-9)
      }
    },
    test("LayerNorm input gradients agree with central differences") {
      val original = Vector(1.0, -2.0, 0.5)
      val probe = Tensor.constant(Shape(1, 3), Vector(0.2, -0.7, 1.1))
      val input = Tensor.parameter(Shape(1, 3), original, "input")
      val scale = Tensor.parameter(Shape(3), Vector(1.2, 0.8, -0.5), "scale")
      val bias = Tensor.parameter(Shape(3), Vector(0.1, -0.2, 0.3), "bias")
      input.layerNormRows(scale, bias, 1e-5).hadamard(probe).sum.backward()

      def loss(values: Vector[Double]): Double =
        Tensor.constant(Shape(1, 3), values)
          .layerNormRows(
            Tensor.constant(Shape(3), scale.values),
            Tensor.constant(Shape(3), bias.values),
            1e-5
          )
          .hadamard(probe)
          .sum
          .valueAtFlat(0)

      val step = 1e-6
      original.indices.foreach { index =>
        val plus = original.updated(index, original(index) + step)
        val minus = original.updated(index, original(index) - step)
        val numerical = (loss(plus) - loss(minus)) / (2.0 * step)
        Assert.close(input.gradientAtFlat(index), numerical, 1e-6)
      }
    },
    test("approximate GELU has the GPT-2 symmetry identity and finite gradients") {
      val input = Tensor.parameter(Shape(3), Vector(-2.0, 0.0, 2.0), "gelu.input")
      val output = input.geluApprox
      Assert.close(output.valueAtFlat(2) - output.valueAtFlat(0), 2.0, 1e-12)
      Assert.close(output.valueAtFlat(1), 0.0)
      output.sum.backward()
      input.gradients.foreach(gradient => Assert.isTrue(gradient.isFinite))
    },
    test("GPT-2 block preserves shape and matches inventory parameter count") {
      val channels = 4
      val block = Gpt2Block.random(channels, 2, 1e-5, new SplittableRandom(7), "h.0")
      val input = Tensor.constant(Shape(3, channels), Vector.tabulate(12)(_.toDouble / 10.0))
      Assert.equal(block(input).shape, input.shape)
      val parameterCount = block.parameters.map(_.size.toLong).sum
      val expected = GptLineage.parameterInventory(Gpt2Config(10, 8, channels, 2, 1)).perLayer
      Assert.equal(parameterCount, expected)
    },
    test("GPT-2 block sends gradients to input and every parameter") {
      val block = Gpt2Block.random(4, 2, 1e-5, new SplittableRandom(11), "h.0")
      val input = Tensor.parameter(Shape(2, 4), Vector.tabulate(8)(index => index * 0.1 - 0.3), "input")
      block(input).sum.backward()
      Assert.isTrue(input.gradients.exists(_ != 0.0))
      block.parameters.foreach(parameter =>
        Assert.isTrue(parameter.gradients.exists(_ != 0.0), s"no gradient for ${parameter.label}")
      )
    }
  )
