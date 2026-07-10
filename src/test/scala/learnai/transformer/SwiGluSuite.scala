package learnai.transformer

import java.util.SplittableRandom

import learnai.optim.TensorSgd
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object SwiGluSuite extends TestSuite:
  override val name: String = "SwiGluFeedForward"

  override val tests: Vector[TestCase] = Vector(
    test("the tanh-composed SiLU matches the sigmoid definition exactly") {
      // Independent oracle: silu(z) = z / (1 + exp(-z)) computed directly,
      // versus the layer's z/2 + z/2 * tanh(z/2) composition.
      val inputs = Vector(-3.0, -1.0, 0.0, 0.5, 2.0, 10.0)
      val activated = SwiGluFeedForward.silu(Tensor.constant(Shape(1, inputs.size), inputs))
      inputs.zipWithIndex.foreach { case (z, index) =>
        val expected = z / (1.0 + math.exp(-z))
        Assert.close(activated(0, index), expected, tolerance = 1e-12)
      }
    },
    test("a zero gate closes the network down to its output bias") {
      val network = SwiGluFeedForward.fromValues(
        channels = 2,
        hiddenChannels = 2,
        gateWeights = Vector.fill(4)(0.0),
        gateBiases = Vector.fill(2)(0.0),
        upWeights = Vector(1.0, 2.0, 3.0, 4.0),
        upBiases = Vector(0.5, -0.5),
        downWeights = Vector(1.0, -1.0, 2.0, -2.0),
        downBiases = Vector(0.7, -0.3),
        label = "closed"
      )
      val output = network(Tensor.constant(Shape(1, 2), Vector(5.0, -8.0)))
      Assert.close(output(0, 0), 0.7, tolerance = 1e-12)
      Assert.close(output(0, 1), -0.3, tolerance = 1e-12)
    },
    test("shape is preserved and the parameter count follows 3CS + 2S + C") {
      val network = SwiGluFeedForward.random(6, 10, new SplittableRandom(2L), "swiglu")
      val output = network(Tensor.constant(Shape(3, 6), Vector.fill(18)(0.25)))
      Assert.equal(output.shape, Shape(3, 6))
      Assert.equal(network.parameterCount, 3 * 6 * 10 + 2 * 10 + 6)
    },
    test("the matched hidden width is the closest to the ReLU baseline count") {
      val channels = 8
      val reluHidden = 32
      val matched = SwiGluFeedForward.parameterMatchedHiddenChannels(channels, reluHidden)
      Assert.equal(matched, 21)

      val random = new SplittableRandom(3L)
      val baseline = FeedForward.random(channels, reluHidden, random, "relu")
        .parameters.map(_.size).sum

      def swigluCount(hidden: Int): Int =
        SwiGluFeedForward.random(channels, hidden, random, "swiglu").parameterCount

      val differenceAtMatched = math.abs(swigluCount(matched) - baseline)
      Assert.isTrue(
        differenceAtMatched <= math.abs(swigluCount(matched - 1) - baseline),
        "one hidden unit fewer should not match the baseline better"
      )
      Assert.isTrue(
        differenceAtMatched <= math.abs(swigluCount(matched + 1) - baseline),
        "one hidden unit more should not match the baseline better"
      )
      // The exact formula approaches the published two-thirds rule of thumb.
      Assert.close(matched.toDouble / reluHidden.toDouble, 2.0 / 3.0, tolerance = 0.05)
    },
    test("autodiff gradients through the network match finite differences") {
      val network = SwiGluFeedForward.random(3, 4, new SplittableRandom(5L), "swiglu")
      val random = new SplittableRandom(7L)
      val inputValues = Vector.fill(2 * 3)(random.nextDouble(-1.0, 1.0))
      val lossWeights = Vector.fill(2 * 3)(random.nextDouble(-1.0, 1.0))
      val weightTensor = Tensor.constant(Shape(2, 3), lossWeights)

      def loss(values: Vector[Double]): Double =
        network(Tensor.constant(Shape(2, 3), values))
          .hadamard(weightTensor)
          .sum
          .valueAtFlat(0)

      val parameter = Tensor.parameter(Shape(2, 3), inputValues, "swigluInput")
      network(parameter).hadamard(weightTensor).sum.backward()

      val step = 1e-6
      inputValues.indices.foreach { index =>
        val bumpedUp = inputValues.updated(index, inputValues(index) + step)
        val bumpedDown = inputValues.updated(index, inputValues(index) - step)
        val numerical = (loss(bumpedUp) - loss(bumpedDown)) / (2.0 * step)
        Assert.close(parameter.gradientAtFlat(index), numerical, tolerance = 1e-6)
      }
    },
    test("the gated network learns a small nonlinear mapping end to end") {
      // XOR-patterned regression: linear layers alone cannot represent it,
      // so a large loss drop is evidence the gate path trains correctly.
      val network = SwiGluFeedForward.random(2, 8, new SplittableRandom(11L), "swiglu")
      val inputs = Tensor.constant(
        Shape(4, 2),
        Vector(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
      )
      val targets = Tensor.constant(
        Shape(4, 2),
        Vector(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0)
      )
      val optimizer = new TensorSgd(learningRate = 0.1)

      def currentLoss(): Tensor = (network(inputs) - targets).pow(2.0).mean

      val initialLoss = currentLoss().valueAtFlat(0)
      var iteration = 0
      while iteration < 300 do
        network.parameters.foreach(_.clearGradients())
        currentLoss().backward()
        val _ = optimizer.step(network.parameters)
        iteration += 1
      network.parameters.foreach(_.clearGradients())
      val finalLoss = currentLoss().valueAtFlat(0)

      Assert.isTrue(
        finalLoss < initialLoss / 10.0,
        s"expected at least a 10x loss reduction, got $initialLoss -> $finalLoss"
      )
    },
    test("invalid dimensions and input shapes are rejected") {
      val zeroChannels = Assert.throws[IllegalArgumentException] {
        SwiGluFeedForward.random(0, 4, new SplittableRandom(1L), "swiglu")
      }
      Assert.isTrue(zeroChannels.getMessage.contains("positive"))
      val zeroHidden = Assert.throws[IllegalArgumentException] {
        SwiGluFeedForward.random(4, 0, new SplittableRandom(1L), "swiglu")
      }
      Assert.isTrue(zeroHidden.getMessage.contains("positive"))
      val badBaseline = Assert.throws[IllegalArgumentException] {
        SwiGluFeedForward.parameterMatchedHiddenChannels(4, 0)
      }
      Assert.isTrue(badBaseline.getMessage.contains("positive"))

      val network = SwiGluFeedForward.random(4, 6, new SplittableRandom(1L), "swiglu")
      val wrongWidth = Assert.throws[IllegalArgumentException] {
        network(Tensor.constant(Shape(2, 3), Vector.fill(6)(1.0)))
      }
      Assert.isTrue(wrongWidth.getMessage.contains("expected"))
    }
  )
