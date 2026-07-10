package learnai.finetune

import java.util.SplittableRandom

import learnai.optim.Initialization
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.transformer.Linear

/** Low-rank adaptation (LoRA) of one frozen dense layer.
  *
  * Full fine-tuning updates every element of a weight matrix. LoRA instead
  * freezes the base layer and learns an additive update constrained to rank
  * `r`:
  *
  * `h = base(x) + (alpha / r) * (x A) B`
  *
  * with `A: [in, r]` and `B: [r, out]`. The trainable element count drops
  * from `in * out` to `r * (in + out)`, and because the update is a plain
  * matrix product, it can be *merged* into the base weight after training,
  * leaving inference cost identical to the original layer.
  *
  * Two initialization facts carry most of the method's safety story:
  *
  *   - `B` starts at zero, so a freshly wrapped layer computes exactly the
  *     base function — adaptation begins from "no change", not from noise;
  *   - `A` starts random, so gradients through `B` are non-degenerate from
  *     the first step.
  *
  * Freezing is an *optimizer-side* contract in this codebase: the base
  * parameters still receive gradients during backward (they are on the
  * computation path), but [[trainableParameters]] exposes only the adapter
  * matrices, and only what the optimizer receives changes. The test suite
  * pins the base bitwise across training steps.
  */
final class LoraLinear private (
    val base: Linear,
    val rank: Int,
    val alpha: Double,
    val adapterDown: Tensor,
    val adapterUp: Tensor
):
  require(rank > 0, s"LoRA rank must be positive: $rank")
  require(
    rank <= math.min(base.inputChannels, base.outputChannels),
    s"rank $rank exceeds min(${base.inputChannels}, ${base.outputChannels}); " +
      "a full-rank adapter defeats the decomposition"
  )
  require(alpha > 0.0 && alpha.isFinite, s"alpha must be finite and positive: $alpha")
  require(
    adapterDown.shape == Shape(base.inputChannels, rank),
    s"adapter down projection must be [${base.inputChannels},$rank], got ${adapterDown.shape}"
  )
  require(
    adapterUp.shape == Shape(rank, base.outputChannels),
    s"adapter up projection must be [$rank,${base.outputChannels}], got ${adapterUp.shape}"
  )

  /** The effective scale applied to the low-rank update. */
  val scaling: Double = alpha / rank.toDouble

  /** Applies the frozen base plus the scaled low-rank update. */
  def apply(input: Tensor): Tensor =
    require(
      input.rank == 2 && input.shape(1) == base.inputChannels,
      s"LoRA expected [rows,${base.inputChannels}], got ${input.shape}"
    )
    base(input) + input.matmul(adapterDown).matmul(adapterUp).scale(scaling)

  /** Only the adapter matrices; hand exactly this to the optimizer. */
  def trainableParameters: Vector[Tensor] = Vector(adapterDown, adapterUp)

  /** Every parameter on the forward path, frozen or not. */
  def allParameters: Vector[Tensor] = base.parameters ++ trainableParameters

  /** Trainable elements: `r * (in + out)` versus full fine-tuning's count. */
  def trainableParameterCount: Int = adapterDown.size + adapterUp.size

  /** Folds the adapter into a standalone layer: `W' = W + s * A B`.
    *
    * The merged layer owns fresh parameter tensors and computes the same
    * function as [[apply]] up to floating-point association, with the
    * adapter's runtime cost gone. The bias is copied unchanged — LoRA never
    * touches it.
    */
  def merged(label: String): Linear =
    val update = adapterDown.matmul(adapterUp).scale(scaling)
    val mergedWeights = base.weight.values.zip(update.values).map(_ + _)
    Linear.fromValues(
      base.inputChannels,
      base.outputChannels,
      mergedWeights,
      base.bias.values,
      label
    )

object LoraLinear:
  /** Wraps a base layer with a rank-`r` adapter in the published
    * initialization: random down projection, zero up projection.
    */
  def wrap(
      base: Linear,
      rank: Int,
      alpha: Double,
      random: SplittableRandom,
      label: String
  ): LoraLinear =
    require(rank > 0, s"LoRA rank must be positive: $rank")
    val down = Initialization.xavierUniform(
      Shape(base.inputChannels, rank),
      fanIn = base.inputChannels,
      fanOut = rank,
      random,
      s"$label.down"
    )
    val up = Initialization.zeros(Shape(rank, base.outputChannels), s"$label.up")
    new LoraLinear(base, rank, alpha, down, up)

  /** Builds an adapter from explicit values for deterministic tests. */
  def fromValues(
      base: Linear,
      rank: Int,
      alpha: Double,
      downValues: Vector[Double],
      upValues: Vector[Double],
      label: String
  ): LoraLinear =
    new LoraLinear(
      base,
      rank,
      alpha,
      Tensor.parameter(Shape(base.inputChannels, rank), downValues, s"$label.down"),
      Tensor.parameter(Shape(rank, base.outputChannels), upValues, s"$label.up")
    )
