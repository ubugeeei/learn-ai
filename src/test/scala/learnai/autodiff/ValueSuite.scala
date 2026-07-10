package learnai.autodiff

import learnai.math.Calculus
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ValueSuite extends TestSuite:
  override val name: String = "ValueAutodiff"

  override val tests: Vector[TestCase] = Vector(
    test("backward applies product and sum rules") {
      val x = Value.parameter(2.0, "x")
      val y = Value.parameter(-3.0, "y")
      val output = x * y + x.pow(2.0)

      output.backward()

      Assert.close(output.data, -2.0)
      Assert.close(x.gradient, 1.0)
      Assert.close(y.gradient, 2.0)
    },
    test("a reused node accumulates gradient from every path") {
      val x = Value.parameter(3.0, "x")
      val output = x * x + x
      output.backward()
      Assert.close(x.gradient, 7.0)
    },
    test("autodiff agrees with finite differences on a composite function") {
      val rawFunction = (x: Double) => math.tanh(math.exp(x * x - 0.5))
      val x = Value.parameter(0.7, "x")
      val output = (x.pow(2.0) - 0.5).exp.tanh
      output.backward()

      val numeric = Calculus.derivative(rawFunction, at = x.data)
      Assert.close(x.gradient, numeric, tolerance = 1e-8)
    },
    test("backward clears gradients from a previous traversal") {
      val x = Value.parameter(2.0, "x")
      val output = x.pow(3.0)
      output.backward()
      val first = x.gradient
      output.backward()
      Assert.close(first, 12.0)
      Assert.close(x.gradient, 12.0)
    },
    test("gradient update is restricted to trainable leaves") {
      val parameter = Value.parameter(2.0, "weight")
      val loss = parameter.pow(2.0)
      loss.backward()
      parameter.applyGradient(learningRate = 0.1)
      Assert.close(parameter.data, 1.6)

      val constant = Value.constant(2.0)
      val error = Assert.throws[IllegalArgumentException] {
        constant.applyGradient(0.1)
      }
      Assert.isTrue(error.getMessage.contains("trainable"))
    },
    test("log rejects values outside its mathematical domain") {
      val error = Assert.throws[IllegalArgumentException](Value.constant(0.0).log)
      Assert.isTrue(error.getMessage.contains("must be positive"))
    }
  )
