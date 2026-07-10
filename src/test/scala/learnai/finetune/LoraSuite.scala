package learnai.finetune

import java.util.SplittableRandom

import learnai.optim.TensorSgd
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.transformer.Linear

object LoraSuite extends TestSuite:
  override val name: String = "LoraLinear"

  private def randomInput(rows: Int, columns: Int, seed: Long): Tensor =
    val random = new SplittableRandom(seed)
    Tensor.constant(
      Shape(rows, columns),
      Vector.fill(rows * columns)(random.nextDouble(-1.0, 1.0))
    )

  override val tests: Vector[TestCase] = Vector(
    test("a freshly wrapped layer computes exactly the base function") {
      // B starts at zero, so the low-rank update is exactly zero — not
      // approximately: the scaled product of anything with a zero matrix.
      val base = Linear.random(4, 3, new SplittableRandom(1L), "base")
      val adapted = LoraLinear.wrap(base, rank = 2, alpha = 8.0, new SplittableRandom(2L), "lora")
      val input = randomInput(rows = 5, columns = 4, seed = 3L)
      Assert.equal(adapted(input).values, base(input).values)
    },
    test("merging the adapter reproduces the adapted forward pass") {
      val base = Linear.random(4, 3, new SplittableRandom(4L), "base")
      val random = new SplittableRandom(5L)
      val adapted = LoraLinear.fromValues(
        base,
        rank = 2,
        alpha = 4.0,
        downValues = Vector.fill(4 * 2)(random.nextDouble(-0.5, 0.5)),
        upValues = Vector.fill(2 * 3)(random.nextDouble(-0.5, 0.5)),
        label = "lora"
      )
      val merged = adapted.merged("merged")
      val input = randomInput(rows = 6, columns = 4, seed = 6L)
      adapted(input).values.zip(merged(input).values).foreach { case (viaAdapter, viaMerge) =>
        Assert.close(viaMerge, viaAdapter, tolerance = 1e-12)
      }
      Assert.equal(merged.bias.values, base.bias.values)
    },
    test("training the adapter leaves the frozen base bitwise unchanged") {
      val base = Linear.random(3, 3, new SplittableRandom(7L), "base")
      val baseWeightsBefore = base.weight.values
      val baseBiasBefore = base.bias.values
      val adapted = LoraLinear.wrap(base, rank = 1, alpha = 2.0, new SplittableRandom(8L), "lora")
      val optimizer = new TensorSgd(learningRate = 0.1)
      val input = randomInput(rows = 4, columns = 3, seed = 9L)
      val target = randomInput(rows = 4, columns = 3, seed = 10L)

      (0 until 25).foreach { _ =>
        adapted.allParameters.foreach(_.clearGradients())
        (adapted(input) - target).pow(2.0).mean.backward()
        val _ = optimizer.step(adapted.trainableParameters)
      }

      Assert.equal(base.weight.values, baseWeightsBefore)
      Assert.equal(base.bias.values, baseBiasBefore)
      Assert.isTrue(
        adapted.adapterUp.values.exists(_ != 0.0),
        "the zero-initialized up projection should have moved during training"
      )
    },
    test("a rank-limited adapter can still fit a reachable target") {
      // The target is the base layer plus a rank-1 perturbation, so a rank-1
      // adapter can represent it exactly; training should approach it.
      val base = Linear.random(3, 3, new SplittableRandom(11L), "base")
      val random = new SplittableRandom(12L)
      val targetLayer = LoraLinear.fromValues(
        base,
        rank = 1,
        alpha = 1.0,
        downValues = Vector.fill(3)(random.nextDouble(-1.0, 1.0)),
        upValues = Vector.fill(3)(random.nextDouble(-1.0, 1.0)),
        label = "target"
      )
      val student = LoraLinear.wrap(base, rank = 1, alpha = 1.0, new SplittableRandom(13L), "lora")
      val optimizer = new TensorSgd(learningRate = 0.2)
      val input = randomInput(rows = 8, columns = 3, seed = 14L)
      val target = targetLayer(input)

      def loss(): Tensor = (student(input) - target).pow(2.0).mean

      val initial = loss().valueAtFlat(0)
      (0 until 400).foreach { _ =>
        student.trainableParameters.foreach(_.clearGradients())
        loss().backward()
        val _ = optimizer.step(student.trainableParameters)
      }
      student.trainableParameters.foreach(_.clearGradients())
      val finalLoss = loss().valueAtFlat(0)
      Assert.isTrue(
        finalLoss < initial / 20.0,
        s"expected at least a 20x loss reduction, got $initial -> $finalLoss"
      )
    },
    test("trainable accounting follows r times in plus out") {
      val base = Linear.random(16, 8, new SplittableRandom(15L), "base")
      val adapted = LoraLinear.wrap(base, rank = 2, alpha = 4.0, new SplittableRandom(16L), "lora")
      Assert.equal(adapted.trainableParameterCount, 2 * (16 + 8))
      val fullFineTuning = base.parameters.map(_.size).sum
      Assert.isTrue(
        adapted.trainableParameterCount < fullFineTuning / 2,
        "the adapter should train far fewer elements than the full layer"
      )
      Assert.close(adapted.scaling, 2.0, tolerance = 1e-15)
    },
    test("invalid ranks scales and shapes are rejected") {
      val base = Linear.random(4, 3, new SplittableRandom(17L), "base")
      val zeroRank = Assert.throws[IllegalArgumentException] {
        LoraLinear.wrap(base, rank = 0, alpha = 1.0, new SplittableRandom(1L), "lora")
      }
      Assert.isTrue(zeroRank.getMessage.contains("positive"))
      val fullRank = Assert.throws[IllegalArgumentException] {
        LoraLinear.wrap(base, rank = 4, alpha = 1.0, new SplittableRandom(1L), "lora")
      }
      Assert.isTrue(fullRank.getMessage.contains("exceeds"))
      val badAlpha = Assert.throws[IllegalArgumentException] {
        LoraLinear.wrap(base, rank = 2, alpha = 0.0, new SplittableRandom(1L), "lora")
      }
      Assert.isTrue(badAlpha.getMessage.contains("positive"))
      val badShape = Assert.throws[IllegalArgumentException] {
        LoraLinear.fromValues(
          base,
          rank = 2,
          alpha = 1.0,
          downValues = Vector.fill(4 * 2)(0.1),
          upValues = Vector.fill(3 * 3)(0.1),
          label = "lora"
        )
      }
      Assert.isTrue(badShape.getMessage.contains("does not match"))
      val adapted = LoraLinear.wrap(base, rank = 2, alpha = 1.0, new SplittableRandom(1L), "lora")
      val badInput = Assert.throws[IllegalArgumentException] {
        adapted(Tensor.constant(Shape(2, 3), Vector.fill(6)(1.0)))
      }
      Assert.isTrue(badInput.getMessage.contains("expected"))
    }
  )
