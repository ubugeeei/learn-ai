package learnai.learning

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object GradientDescentSuite extends TestSuite:
  override val name: String = "GradientDescent"

  override val tests: Vector[TestCase] = Vector(
    test("gradient descent reduces a convex quadratic loss") {
      val target = 3.0
      val history = GradientDescent.minimize(
        parameter => math.pow(parameter - target, 2.0),
        parameter => 2.0 * (parameter - target),
        initialParameter = -4.0,
        learningRate = 0.1,
        steps = 100
      )
      Assert.isTrue(history.last.loss < history.head.loss)
      Assert.close(history.last.parameter, target, tolerance = 2e-9)
    },
    test("zero steps records only the initial state") {
      val history = GradientDescent.minimize(
        parameter => parameter * parameter,
        parameter => 2.0 * parameter,
        initialParameter = 2.0,
        learningRate = 0.1,
        steps = 0
      )
      Assert.equal(history.size, 1)
      Assert.equal(history.head, DescentObservation(0, 2.0, 4.0, 4.0))
    },
    test("invalid hyperparameters are rejected") {
      val rateError = Assert.throws[IllegalArgumentException] {
        GradientDescent.minimize(identity, identity, 1.0, learningRate = 0.0, steps = 1)
      }
      val stepError = Assert.throws[IllegalArgumentException] {
        GradientDescent.minimize(identity, identity, 1.0, learningRate = 0.1, steps = -1)
      }
      Assert.isTrue(rateError.getMessage.contains("learning rate"))
      Assert.isTrue(stepError.getMessage.contains("steps"))
    }
  )
