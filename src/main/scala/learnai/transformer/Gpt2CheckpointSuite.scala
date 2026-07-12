package learnai.transformer

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Gpt2CheckpointSuite extends TestSuite:
  override val name: String = "Gpt2Checkpoint"

  override val tests: Vector[TestCase] = specify(
    test("public checkpoint names are complete and stable") {
      val names = Gpt2Checkpoint.blockTensorNames(3)
      Assert.equal(names.size, 12)
      Assert.equal(names.distinct.size, names.size)
      Assert.isTrue(names.forall(_.startsWith("transformer.h.3.")))
    },
    test("combined c_attn columns split into query key and value in order") {
      val channels = 2
      val checkpoint = fixture(channels)
      val qkvWeight = checkpoint("transformer.h.0.attn.c_attn.weight").values
      val qkvBias = checkpoint("transformer.h.0.attn.c_attn.bias").values
      val block = Gpt2Checkpoint.loadBlock(0, channels, 1, 1e-5, checkpoint)
      Vector(
        block.attention.queryProjection,
        block.attention.keyProjection,
        block.attention.valueProjection
      ).zipWithIndex.foreach { case (projection, part) =>
        val expectedWeight = Vector.tabulate(channels * channels)(offset =>
          val row = offset / channels
          val column = offset % channels
          qkvWeight(row * 3 * channels + part * channels + column)
        )
        Assert.equal(projection.weight.values, expectedWeight)
        Assert.equal(projection.bias.values, qkvBias.slice(part * channels, (part + 1) * channels))
      }
    },
    test("loaded checkpoint values execute a finite shape-preserving forward pass") {
      val block = Gpt2Checkpoint.loadBlock(0, 2, 1, 1e-5, fixture(2))
      val output = block(Tensor.constant(Shape(2, 2), Vector(0.2, -0.1, 0.7, 0.3)))
      Assert.equal(output.shape, Shape(2, 2))
      Assert.isTrue(output.values.forall(_.isFinite))
    },
    test("missing extra and wrong-shaped tensors fail before a block exists") {
      val valid = fixture(2)
      Assert.throws[IllegalArgumentException] {
        Gpt2Checkpoint.loadBlock(0, 2, 1, 1e-5, valid - "transformer.h.0.ln_1.bias")
      }
      Assert.throws[IllegalArgumentException] {
        Gpt2Checkpoint.loadBlock(
          0,
          2,
          1,
          1e-5,
          valid + ("surprise" -> Gpt2CheckpointTensor(Shape(1), Vector(0.0)))
        )
      }
      Assert.throws[IllegalArgumentException] {
        Gpt2Checkpoint.loadBlock(
          0,
          2,
          1,
          1e-5,
          valid.updated(
            "transformer.h.0.mlp.c_proj.weight",
            Gpt2CheckpointTensor(Shape(4, 2), Vector.fill(8)(0.0))
          )
        )
      }
      ()
    }
  )

  private def fixture(channels: Int): Map[String, Gpt2CheckpointTensor] =
    val prefix = "transformer.h.0"
    def values(size: Int, start: Int): Vector[Double] =
      Vector.tabulate(size)(index => (start + index).toDouble / 100.0)
    Map(
      s"$prefix.ln_1.weight" -> Gpt2CheckpointTensor(Shape(channels), Vector.fill(channels)(1.0)),
      s"$prefix.ln_1.bias" -> Gpt2CheckpointTensor(Shape(channels), Vector.fill(channels)(0.0)),
      s"$prefix.attn.c_attn.weight" -> Gpt2CheckpointTensor(
        Shape(channels, 3 * channels),
        values(3 * channels * channels, 1)
      ),
      s"$prefix.attn.c_attn.bias" -> Gpt2CheckpointTensor(
        Shape(3 * channels),
        values(3 * channels, 30)
      ),
      s"$prefix.attn.c_proj.weight" -> Gpt2CheckpointTensor(
        Shape(channels, channels),
        values(channels * channels, 40)
      ),
      s"$prefix.attn.c_proj.bias" -> Gpt2CheckpointTensor(Shape(channels), values(channels, 50)),
      s"$prefix.ln_2.weight" -> Gpt2CheckpointTensor(Shape(channels), Vector.fill(channels)(1.0)),
      s"$prefix.ln_2.bias" -> Gpt2CheckpointTensor(Shape(channels), Vector.fill(channels)(0.0)),
      s"$prefix.mlp.c_fc.weight" -> Gpt2CheckpointTensor(
        Shape(channels, 4 * channels),
        values(4 * channels * channels, 60)
      ),
      s"$prefix.mlp.c_fc.bias" -> Gpt2CheckpointTensor(Shape(4 * channels), values(4 * channels, 90)),
      s"$prefix.mlp.c_proj.weight" -> Gpt2CheckpointTensor(
        Shape(4 * channels, channels),
        values(4 * channels * channels, 110)
      ),
      s"$prefix.mlp.c_proj.bias" -> Gpt2CheckpointTensor(Shape(channels), values(channels, 150))
    )
