package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Tensor
import learnai.text.TokenId

/** Executable dropout-off GPT-2 architecture with learned positions and a tied logit head. */
final class Gpt2Model private (
    val config: Gpt2Config,
    val embeddings: TokenPositionEmbedding,
    val blocks: Vector[Gpt2Block],
    val finalNorm: LayerNorm
):
  require(blocks.size == config.layerCount, s"${blocks.size} blocks != ${config.layerCount}")

  /** Computes `[time, vocabulary]` logits for one non-empty sequence. */
  def logits(tokenIds: Vector[TokenId]): Tensor =
    require(tokenIds.nonEmpty, "GPT-2 requires at least one input token")
    val embedded   = embeddings(tokenIds)
    val hidden     = blocks.foldLeft(embedded)((current, block) => block(current))
    val normalized = finalNorm(hidden)
    normalized.matmul(embeddings.tokens.weight.transpose2D)

  /** Every trainable tensor exactly once; the output classifier reuses token embedding weight. */
  def parameters: Vector[Tensor] =
    embeddings.parameters ++ blocks.flatMap(_.parameters) ++ finalNorm.parameters

  def parameterCount: Long = parameters.map(_.size.toLong).sum

object Gpt2Model:
  /** Creates a deterministic educational model with GPT-2 shapes and forward operations. */
  def random(config: Gpt2Config, seed: Long, epsilon: Double = 1e-5): Gpt2Model =
    val random = new SplittableRandom(seed)
    val embeddings = new TokenPositionEmbedding(
      Embedding.random(config.vocabularySize, config.channels, random, "transformer.wte"),
      Embedding.random(config.contextLength, config.channels, random, "transformer.wpe")
    )
    val blocks = Vector.tabulate(config.layerCount)(index =>
      Gpt2Block.random(
        config.channels,
        config.headCount,
        epsilon,
        random,
        s"transformer.h.$index"
      )
    )
    fromComponents(
      config,
      embeddings,
      blocks,
      LayerNorm.create(config.channels, epsilon, "transformer.ln_f")
    )

  /** Assembles an imported model while checking architecture ownership boundaries. */
  def fromComponents(
      config: Gpt2Config,
      embeddings: TokenPositionEmbedding,
      blocks: Vector[Gpt2Block],
      finalNorm: LayerNorm
  ): Gpt2Model =
    require(embeddings.tokens.entries == config.vocabularySize, "token vocabulary mismatch")
    require(embeddings.positions.entries == config.contextLength, "position context mismatch")
    require(embeddings.channels == config.channels, "embedding channel mismatch")
    require(blocks.forall(_.channels == config.channels), "block channel mismatch")
    require(finalNorm.channels == config.channels, "final LayerNorm channel mismatch")
    new Gpt2Model(config, embeddings, blocks, finalNorm)
