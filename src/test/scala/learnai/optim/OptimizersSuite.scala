package learnai.optim

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object OptimizersSuite extends TestSuite:
  override val name: String = "Optimizers"

  override val tests: Vector[TestCase] = Vector(
    test("SGD follows the negative gradient") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(2.0), "x")
      val loss = parameter.pow(2.0).sum
      loss.backward()
      val stats = new TensorSgd(learningRate = 0.1).step(Vector(parameter))
      Assert.close(parameter.valueAtFlat(0), 1.6)
      Assert.close(stats.gradientNorm, 4.0)
      Assert.equal(stats.step, 1L)
    },
    test("global norm clipping preserves gradient direction") {
      val parameter = Tensor.parameter(Shape(2), Vector(0.0, 0.0), "x")
      val coefficients = Tensor.constant(Shape(2), Vector(3.0, 4.0))
      parameter.hadamard(coefficients).sum.backward()
      val stats = new TensorSgd(
        learningRate = 1.0,
        maximumGradientNorm = Some(1.0)
      ).step(Vector(parameter))
      Assert.close(stats.gradientNorm, 5.0)
      Assert.close(stats.gradientScale, 0.2)
      Assert.close(parameter.valueAtFlat(0), -0.6)
      Assert.close(parameter.valueAtFlat(1), -0.8)
    },
    test("AdamW reduces a scalar quadratic") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(5.0), "x")
      val optimizer = new AdamW(learningRate = 0.1, weightDecay = 0.0)
      var step = 0
      while step < 100 do
        parameter.pow(2.0).sum.backward()
        val _ = optimizer.step(Vector(parameter))
        step += 1
      Assert.isTrue(math.abs(parameter.valueAtFlat(0)) < 0.05)
    },
    test("decoupled weight decay updates a zero-gradient parameter") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(2.0), "x")
      parameter.scale(0.0).sum.backward()
      val _ = new TensorSgd(learningRate = 0.1, weightDecay = 0.5).step(Vector(parameter))
      Assert.close(parameter.valueAtFlat(0), 1.9)
    },
    test("Xavier initialization is bounded and deterministic") {
      val first = Initialization.xavierUniform(
        Shape(3, 2),
        fanIn = 2,
        fanOut = 3,
        new SplittableRandom(9L),
        "weights"
      )
      val second = Initialization.xavierUniform(
        Shape(3, 2),
        fanIn = 2,
        fanOut = 3,
        new SplittableRandom(9L),
        "weights"
      )
      val bound = math.sqrt(6.0 / 5.0)
      Assert.equal(first.values, second.values)
      Assert.isTrue(first.values.forall(value => value >= -bound && value < bound))
    }
  )
