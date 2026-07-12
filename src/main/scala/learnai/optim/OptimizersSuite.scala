package learnai.optim

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object OptimizersSuite extends TestSuite:
  override val name: String = "Optimizers"

  override val tests: Vector[TestCase] = specify(
    test("SGD follows the negative gradient") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(2.0), "x")
      val loss      = parameter.pow(2.0).sum
      loss.backward()
      val stats     = new TensorSgd(learningRate = 0.1).step(Vector(parameter))
      Assert.close(parameter.valueAtFlat(0), 1.6)
      Assert.close(stats.gradientNorm, 4.0)
      Assert.equal(stats.step, 1L)
      Assert.close(stats.learningRate, 0.1)
    },
    test("global norm clipping preserves gradient direction") {
      val parameter    = Tensor.parameter(Shape(2), Vector(0.0, 0.0), "x")
      val coefficients = Tensor.constant(Shape(2), Vector(3.0, 4.0))
      parameter.hadamard(coefficients).sum.backward()
      val stats        = new TensorSgd(learningRate = 1.0, maximumGradientNorm = Some(1.0))
        .step(Vector(parameter))
      Assert.close(stats.gradientNorm, 5.0)
      Assert.close(stats.gradientScale, 0.2)
      Assert.close(parameter.valueAtFlat(0), -0.6)
      Assert.close(parameter.valueAtFlat(1), -0.8)
    },
    test("AdamW reduces a scalar quadratic") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(5.0), "x")
      val optimizer = new AdamW(learningRate = 0.1, weightDecay = 0.0)
      var step      = 0
      while step < 100 do
        parameter.pow(2.0).sum.backward()
        val _ = optimizer.step(Vector(parameter))
        step += 1
      Assert.isTrue(math.abs(parameter.valueAtFlat(0)) < 0.05)
    },
    test("decoupled weight decay updates a zero-gradient parameter") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(2.0), "x")
      parameter.scale(0.0).sum.backward()
      val _         = new TensorSgd(learningRate = 0.1, weightDecay = 0.5).step(Vector(parameter))
      Assert.close(parameter.valueAtFlat(0), 1.9)
    },
    test("a schedule-provided learning rate controls one update without replacing the default") {
      val parameter = Tensor.parameter(Shape.scalar, Vector(2.0), "x")
      val optimizer = new TensorSgd(learningRate = 0.1)
      parameter.pow(2.0).sum.backward()
      val scheduled = optimizer.stepAtLearningRate(Vector(parameter), effectiveLearningRate = 0.25)
      Assert.close(parameter.valueAtFlat(0), 1.0)
      Assert.close(scheduled.learningRate, 0.25)

      parameter.pow(2.0).sum.backward()
      val default = optimizer.step(Vector(parameter))
      Assert.close(parameter.valueAtFlat(0), 0.8)
      Assert.close(default.learningRate, 0.1)
    },
    test("a restored snapshot continues AdamW exactly like an uninterrupted run") {
      val initial = Vector(1.5, -2.0)

      def quadraticStep(optimizer: AdamW, parameter: Tensor): Unit =
        parameter.clearGradients()
        parameter.pow(2.0).sum.backward()
        val _ = optimizer.step(Vector(parameter))

      // Uninterrupted reference: five consecutive updates.
      val reference          = Tensor.parameter(Shape(2), initial, "x")
      val referenceOptimizer = new AdamW(learningRate = 0.05)
      (0 until 5).foreach(_ => quadraticStep(referenceOptimizer, reference))

      // Chunked run: three updates, snapshot, resume in a *new* optimizer
      // attached to a *new* tensor rebuilt from the captured values.
      val firstHalf      = Tensor.parameter(Shape(2), initial, "x")
      val firstOptimizer = new AdamW(learningRate = 0.05)
      (0 until 3).foreach(_ => quadraticStep(firstOptimizer, firstHalf))
      val captured       = firstOptimizer.snapshot(Vector(firstHalf))
      Assert.equal(captured.step, 3L)

      val resumed          = Tensor.parameter(Shape(2), firstHalf.values, "x")
      val resumedOptimizer = new AdamW(learningRate = 0.05)
      resumedOptimizer.restore(Vector(resumed), captured)
      (0 until 2).foreach(_ => quadraticStep(resumedOptimizer, resumed))

      Assert.equal(resumed.values, reference.values)
    },
    test("an unstepped optimizer snapshots to the zero state") {
      val parameter = Tensor.parameter(Shape(3), Vector(1.0, 2.0, 3.0), "x")
      val optimizer = new AdamW(learningRate = 0.1)
      Assert.equal(optimizer.snapshot(Vector(parameter)), AdamWSnapshot.zero(Vector(parameter)))
    },
    test("restore rejects mismatched or corrupted snapshots without mutating state") {
      val parameter = Tensor.parameter(Shape(2), Vector(1.0, 2.0), "x")
      val optimizer = new AdamW(learningRate = 0.1)

      val wrongCount = Assert.throws[IllegalArgumentException] {
        optimizer.restore(
          Vector(parameter),
          AdamWSnapshot(1L, Vector(Vector(0.0), Vector(0.0)), Vector(Vector(0.0), Vector(0.0)))
        )
      }
      Assert.isTrue(wrongCount.getMessage.contains("covers"))

      val wrongSize = Assert.throws[IllegalArgumentException] {
        optimizer
          .restore(Vector(parameter), AdamWSnapshot(1L, Vector(Vector(0.0)), Vector(Vector(0.0))))
      }
      Assert.isTrue(wrongSize.getMessage.contains("sizes"))

      val negativeSecondMoment = Assert.throws[IllegalArgumentException] {
        optimizer.restore(
          Vector(parameter),
          AdamWSnapshot(1L, Vector(Vector(0.0, 0.0)), Vector(Vector(0.1, -0.1)))
        )
      }
      Assert.isTrue(negativeSecondMoment.getMessage.contains("non-negative"))

      val negativeStep = Assert
        .throws[IllegalArgumentException](AdamWSnapshot(-1L, Vector.empty, Vector.empty))
      Assert.isTrue(negativeStep.getMessage.contains("non-negative"))

      // The rejected restores must not have poisoned the optimizer.
      parameter.pow(2.0).sum.backward()
      val stats = optimizer.step(Vector(parameter))
      Assert.equal(stats.step, 1L)
    },
    test("Xavier initialization is bounded and deterministic") {
      val first  = Initialization
        .xavierUniform(Shape(3, 2), fanIn = 2, fanOut = 3, new SplittableRandom(9L), "weights")
      val second = Initialization
        .xavierUniform(Shape(3, 2), fanIn = 2, fanOut = 3, new SplittableRandom(9L), "weights")
      val bound  = math.sqrt(6.0 / 5.0)
      Assert.equal(first.values, second.values)
      Assert.isTrue(first.values.forall(value => value >= -bound && value < bound))
    }
  )
