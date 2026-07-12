package learnai.diagnostics

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object GradientCheckSuite extends TestSuite:
  override val name: String = "GradientCheck"

  override val tests: Vector[TestCase] = specify(
    test("central differences match a hand-computable quadratic and restore values") {
      val parameter = Tensor.parameter(Shape(3), Vector(1.5, -2.0, 0.0), "quadratic.x")
      val original = parameter.values
      val report = GradientChecker.check(
        Vector(parameter),
        () => parameter.pow(2.0).sum,
        GradientCheckConfig(maximumCoordinatesPerParameter = 3)
      )

      Assert.isTrue(report.passed)
      Assert.close(report.loss, 6.25)
      Assert.equal(report.probes.map(_.coordinates), Vector(Vector(0), Vector(1), Vector(2)))
      report.probes.map(_.analytical).zip(Vector(3.0, -4.0, 0.0)).foreach {
        case (actual, expected) => Assert.close(actual, expected)
      }
      Assert.equal(parameter.values, original)
      Assert.equal(parameter.gradients, Vector(0.0, 0.0, 0.0))
    },
    test("deterministic coordinate sampling includes both ends of a parameter") {
      val parameter = Tensor.parameter(Shape(10), Vector.tabulate(10)(_.toDouble + 1.0), "wide.x")
      val report = GradientChecker.check(
        Vector(parameter),
        () => parameter.pow(2.0).sum,
        GradientCheckConfig(maximumCoordinatesPerParameter = 3)
      )
      Assert.equal(report.probes.map(_.flatIndex), Vector(0, 4, 9))
    },
    test("failed probes retain analytical numerical and tolerance evidence") {
      val parameter = Tensor.parameter(Shape(1), Vector(2.0), "cubic.x")
      val report = GradientChecker.check(
        Vector(parameter),
        () => parameter.pow(3.0).sum,
        GradientCheckConfig(
          epsilon = 0.1,
          absoluteTolerance = 0.0,
          relativeTolerance = 0.0,
          maximumCoordinatesPerParameter = 1
        )
      )
      Assert.isTrue(!report.passed)
      Assert.equal(report.failedProbes.size, 1)
      Assert.close(report.worstProbe.analytical, 12.0)
      Assert.close(report.worstProbe.numerical, 12.01, tolerance = 1e-10)
      Assert.close(report.worstProbe.absoluteError, 0.01, tolerance = 1e-10)
    },
    test("parameter values are restored when a perturbed loss construction fails") {
      val parameter = Tensor.parameter(Shape(1), Vector(3.0), "restore.x")
      val original = parameter.values
      var calls = 0
      val error = Assert.throws[IllegalStateException] {
        GradientChecker.check(
          Vector(parameter),
          () =>
            calls += 1
            if calls == 2 then throw new IllegalStateException("injected loss failure")
            parameter.pow(2.0).sum,
          GradientCheckConfig(maximumCoordinatesPerParameter = 1)
        )
      }
      Assert.isTrue(error.getMessage.contains("injected"))
      Assert.equal(parameter.values, original)
      Assert.equal(parameter.gradients, Vector(0.0))
    },
    test("MiniGPT sampled parameter gradients agree with finite differences") {
      val config = MiniGptConfig(
        vocabularySize = 5,
        maximumContextLength = 3,
        channels = 4,
        headCount = 2,
        hiddenChannels = 6,
        layerCount = 1
      )
      val model = MiniGpt.random(config, seed = 77L)
      val inputs = Vector(0, 1, 2).map(TokenId(_))
      val targets = Vector(1, 2, 3).map(TokenId(_))
      val report = GradientChecker.check(
        model.parameters,
        () => model.loss(inputs, targets),
        GradientCheckConfig(
          absoluteTolerance = 2e-5,
          relativeTolerance = 2e-3,
          maximumCoordinatesPerParameter = 1
        )
      )
      Assert.isTrue(
        report.passed,
        s"worst probe: ${report.worstProbe}"
      )
    },
    test("parameter inventory distinguishes dense payload from other training memory") {
      val first = Tensor.parameter(Shape(2, 3), Vector.fill(6)(0.0), "first")
      val second = Tensor.parameter(Shape(4), Vector.fill(4)(0.0), "second")
      val inventory = ParameterInventory.from(Vector(first, second), bytesPerElement = 8)
      Assert.equal(inventory.totalElements, 10L)
      Assert.equal(inventory.totalPayloadBytes, 80L)
      Assert.equal(inventory.entries.map(_.payloadBytes), Vector(48L, 32L))

      val duplicateError = Assert.throws[IllegalArgumentException] {
        ParameterInventory.from(Vector(first, first))
      }
      Assert.isTrue(duplicateError.getMessage.contains("unique"))
    }
  )
