package learnai.nn

import learnai.autodiff.Value
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object ScalarNetworkSuite extends TestSuite:
  override val name: String = "ScalarNetwork"

  override val tests: Vector[TestCase] = Vector(
    test("a neuron computes activation of weighted inputs plus bias") {
      val neuron = new Neuron(
        weights = Vector(Value.parameter(2.0, "w0"), Value.parameter(-1.0, "w1")),
        bias = Value.parameter(0.5, "b"),
        activation = Activation.Linear
      )
      val output = neuron(Vector(Value.constant(3.0), Value.constant(4.0)))
      Assert.close(output.data, 2.5)
    },
    test("MLP construction is deterministic for the same seed") {
      val first = MultiLayerPerceptron.random(
        2,
        Vector(3, 1),
        Activation.Tanh,
        Activation.Linear,
        seed = 7L
      )
      val second = MultiLayerPerceptron.random(
        2,
        Vector(3, 1),
        Activation.Tanh,
        Activation.Linear,
        seed = 7L
      )
      Assert.equal(first.parameters.map(_.data), second.parameters.map(_.data))
      Assert.equal(first.parameters.map(_.label), second.parameters.map(_.label))
    },
    test("XOR training reduces loss and learns every class") {
      val result = Xor.train(seed = 42L, steps = 500, learningRate = 0.05)
      Assert.isTrue(result.lossHistory.last < result.lossHistory.head * 0.02)
      Assert.isTrue(result.lossHistory.last < 0.02)

      Xor.predict(result.model).foreach { case (example, prediction) =>
        Assert.isTrue(
          math.signum(prediction) == math.signum(example.target),
          s"wrong XOR class for ${example.inputs}: target=${example.target}, prediction=$prediction"
        )
      }
    },
    test("network rejects an input with the wrong size") {
      val model = MultiLayerPerceptron.random(
        2,
        Vector(1),
        Activation.Tanh,
        Activation.Linear,
        seed = 1L
      )
      val error = Assert.throws[IllegalArgumentException](model(Vector(1.0)))
      Assert.isTrue(error.getMessage.contains("expected 2, got 1"))
    }
  )
