package learnai.transformer

import learnai.tensor.Shape

/** One named row-major tensor read from a GPT-2 checkpoint container. */
final case class Gpt2CheckpointTensor(shape: Shape, values: Vector[Double]):
  require(values.size == shape.size, s"${values.size} values do not fill $shape")
  require(values.forall(_.isFinite), "checkpoint tensor values must be finite")

/**
 * Loads the public GPT-2/Hugging Face tensor naming and Conv1D layout into one executable block.
 */
object Gpt2Checkpoint:
  /**
   * Exact tensor names owned by block `index`; extra or missing names are rejected by `loadBlock`.
   */
  def blockTensorNames(index: Int): Vector[String] =
    require(index >= 0, s"block index must be non-negative: $index")
    val prefix = s"transformer.h.$index"
    Vector(
      s"$prefix.ln_1.weight",
      s"$prefix.ln_1.bias",
      s"$prefix.attn.c_attn.weight",
      s"$prefix.attn.c_attn.bias",
      s"$prefix.attn.c_proj.weight",
      s"$prefix.attn.c_proj.bias",
      s"$prefix.ln_2.weight",
      s"$prefix.ln_2.bias",
      s"$prefix.mlp.c_fc.weight",
      s"$prefix.mlp.c_fc.bias",
      s"$prefix.mlp.c_proj.weight",
      s"$prefix.mlp.c_proj.bias"
    )

  /**
   * Loads one block atomically by constructing fresh parameters only after every name and shape is
   * checked. GPT-2 stores Q/K/V together as `[channels, 3 * channels]`; this runtime executes three
   * separate Linear layers, so columns and bias ranges are split in Q, K, V order.
   */
  def loadBlock(
      index: Int,
      channels: Int,
      headCount: Int,
      epsilon: Double,
      tensors: Map[String, Gpt2CheckpointTensor]
  ): Gpt2Block =
    val names                                                 = blockTensorNames(index)
    require(tensors.keySet == names.toSet, missingAndExtra(names.toSet, tensors.keySet))
    val prefix                                                = s"transformer.h.$index"
    def read(suffix: String, expected: Shape): Vector[Double] =
      val name   = s"$prefix.$suffix"
      val tensor = tensors(name)
      require(tensor.shape == expected, s"$name has ${tensor.shape}, expected $expected")
      tensor.values

    val vector                                      = Shape(channels)
    val square                                      = Shape(channels, channels)
    val qkvWeight                                   = read("attn.c_attn.weight", Shape(channels, 3 * channels))
    val qkvBias                                     = read("attn.c_attn.bias", Shape(3 * channels))
    def weightPart(part: Int): Vector[Double]       = Vector.tabulate(channels * channels)(offset =>
      val row    = offset / channels
      val column = offset % channels
      qkvWeight(row * 3 * channels + part * channels + column)
    )
    def biasPart(part: Int): Vector[Double]         = qkvBias.slice(part * channels, (part + 1) * channels)
    def projection(part: Int, name: String): Linear = Linear
      .fromValues(channels, channels, weightPart(part), biasPart(part), s"$prefix.attn.$name")

    Gpt2Block.fromComponents(
      channels,
      LayerNorm.fromValues(
        channels,
        epsilon,
        read("ln_1.weight", vector),
        read("ln_1.bias", vector),
        s"$prefix.ln_1"
      ),
      CausalSelfAttention.fromProjections(
        channels,
        headCount,
        projection(0, "query"),
        projection(1, "key"),
        projection(2, "value"),
        Linear.fromValues(
          channels,
          channels,
          read("attn.c_proj.weight", square),
          read("attn.c_proj.bias", vector),
          s"$prefix.attn.c_proj"
        )
      ),
      LayerNorm.fromValues(
        channels,
        epsilon,
        read("ln_2.weight", vector),
        read("ln_2.bias", vector),
        s"$prefix.ln_2"
      ),
      Gpt2FeedForward.fromProjections(
        channels,
        Linear.fromValues(
          channels,
          4 * channels,
          read("mlp.c_fc.weight", Shape(channels, 4 * channels)),
          read("mlp.c_fc.bias", Shape(4 * channels)),
          s"$prefix.mlp.c_fc"
        ),
        Linear.fromValues(
          4 * channels,
          channels,
          read("mlp.c_proj.weight", Shape(4 * channels, channels)),
          read("mlp.c_proj.bias", vector),
          s"$prefix.mlp.c_proj"
        )
      )
    )

  private def missingAndExtra(expected: Set[String], actual: Set[String]): String =
    val missing = (expected -- actual).toVector.sorted.mkString(", ")
    val extra   = (actual -- expected).toVector.sorted.mkString(", ")
    s"checkpoint tensor names differ; missing=[$missing], extra=[$extra]"
