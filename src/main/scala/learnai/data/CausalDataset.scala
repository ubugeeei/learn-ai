package learnai.data

import java.util.random.RandomGenerator

import learnai.text.TokenId

/** One fixed-length causal language-model training example.
  *
  * `targets(t)` is the token immediately following `inputs(t)` in the source
  * sequence. Inputs and targets always have the same non-zero length.
  */
final case class CausalExample(
    inputs: Vector[TokenId],
    targets: Vector[TokenId]
):
  require(inputs.nonEmpty, "a causal example cannot be empty")
  require(
    inputs.size == targets.size,
    s"input and target lengths differ: ${inputs.size} != ${targets.size}"
  )

/** A collection of equally shaped causal examples.
  *
  * The dataset is immutable. Batches may sample example indices with
  * replacement using a caller-owned random generator.
  */
final class CausalDataset private (
    val contextLength: Int,
    val examples: Vector[CausalExample]
):
  def size: Int = examples.size

  def isEmpty: Boolean = examples.isEmpty

  /** Samples a fixed-size batch with replacement.
    *
    * @return
    *   `Left` for an empty dataset, otherwise exactly `batchSize` examples.
    */
  def sampleBatch(
      batchSize: Int,
      random: RandomGenerator
  ): Either[String, CausalBatch] =
    require(batchSize > 0, s"batch size must be positive: $batchSize")
    if examples.isEmpty then Left("cannot sample from an empty dataset")
    else
      Right(
        CausalBatch(
          Vector.fill(batchSize)(examples(random.nextInt(examples.size)))
        )
      )

  /** Splits the stored examples into deterministic consecutive batches.
    *
    * @param dropLast
    *   when true, omit a final batch smaller than `batchSize`.
    */
  def sequentialBatches(batchSize: Int, dropLast: Boolean): Vector[CausalBatch] =
    require(batchSize > 0, s"batch size must be positive: $batchSize")
    examples
      .grouped(batchSize)
      .filter(group => !dropLast || group.size == batchSize)
      .map(group => CausalBatch(group.toVector))
      .toVector

/** A batch whose examples all share one context length. */
final case class CausalBatch(examples: Vector[CausalExample]):
  require(examples.nonEmpty, "a causal batch cannot be empty")
  private val lengths = examples.map(_.inputs.size).distinct
  require(lengths.size == 1, s"all batch examples must have one context length: $lengths")

  val batchSize: Int = examples.size
  val contextLength: Int = lengths.head
  val inputs: Vector[Vector[TokenId]] = examples.map(_.inputs)
  val targets: Vector[Vector[TokenId]] = examples.map(_.targets)

/** Train/validation datasets cut from disjoint contiguous token regions. */
final case class CausalSplit(
    training: CausalDataset,
    validation: CausalDataset,
    boundaryTokenIndex: Int
)

object CausalDataset:
  /** Creates every full sliding context from one token sequence.
    *
    * A sequence of length `N` produces `max(0, N - contextLength)` examples.
    */
  def fromTokens(tokens: Vector[TokenId], contextLength: Int): CausalDataset =
    require(contextLength > 0, s"context length must be positive: $contextLength")
    val examples = tokens
      .sliding(contextLength + 1)
      .filter(_.size == contextLength + 1)
      .map { window =>
        CausalExample(window.take(contextLength), window.drop(1))
      }
      .toVector
    new CausalDataset(contextLength, examples)

  /** Splits raw tokens first, then creates windows independently.
    *
    * This ordering prevents any example from crossing the train/validation
    * boundary. The exact number of examples on each side depends on context
    * length.
    */
  def contiguousSplit(
      tokens: Vector[TokenId],
      trainingFraction: Double,
      contextLength: Int
  ): CausalSplit =
    require(
      trainingFraction > 0.0 && trainingFraction < 1.0,
      s"training fraction must be strictly between 0 and 1: $trainingFraction"
    )
    val boundary = math.floor(tokens.size.toDouble * trainingFraction).toInt
    val trainingTokens = tokens.take(boundary)
    val validationTokens = tokens.drop(boundary)
    CausalSplit(
      fromTokens(trainingTokens, contextLength),
      fromTokens(validationTokens, contextLength),
      boundary
    )
