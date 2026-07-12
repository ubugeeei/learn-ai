package learnai.math

import java.util.random.RandomGenerator

/** A validated categorical probability distribution. */
final class Categorical private (val probabilities: VectorD):
  val size: Int = probabilities.size

  def probability(index: Int): Double = probabilities(index)

  def entropy: Double =
    var result = 0.0
    var index  = 0
    while index < size do
      val probability = probabilities(index)
      if probability > 0.0 then result -= probability * math.log(probability)
      index += 1
    Numerics.requireFinite(result, "entropy")

  def sample(random: RandomGenerator): Int =
    val threshold  = random.nextDouble()
    var cumulative = 0.0
    var index      = 0
    while index < size - 1 do
      cumulative += probabilities(index)
      if threshold < cumulative then return index
      index += 1
    size - 1

  override def equals(other: Any): Boolean = other match
    case that: Categorical => probabilities == that.probabilities
    case _                 => false

  override def hashCode(): Int = probabilities.hashCode()

  override def toString: String = s"Categorical($probabilities)"

object Categorical:
  def from(probabilities: VectorD): Either[String, Categorical] =
    if probabilities.size == 0 then Left("a categorical distribution cannot be empty")
    else
      val negativeIndex = probabilities.toVector.indexWhere(_ < 0.0)
      if negativeIndex >= 0 then
        Left(
          s"probability at index $negativeIndex must be non-negative, " +
            s"got ${probabilities(negativeIndex)}"
        )
      else
        val total = probabilities.sum
        if !Numerics
            .approximatelyEqual(total, 1.0, absoluteTolerance = 1e-9, relativeTolerance = 1e-9)
        then Left(s"probabilities must sum to 1.0, got $total")
        else Right(new Categorical(probabilities))

object Probability:
  /** Stable log(sum(exp(logits))). */
  def logSumExp(logits: VectorD): Either[String, Double] = logits.max.map { maximum =>
    val shiftedExponentials = logits.map(logit => math.exp(logit - maximum))
    Numerics.requireFinite(maximum + math.log(shiftedExponentials.sum), "log-sum-exp")
  }

  /** Stable softmax. Subtracting the maximum does not change the result. */
  def softmax(logits: VectorD): Either[String, Categorical] = logits.max.flatMap { maximum =>
    val exponentials = logits.map(logit => math.exp(logit - maximum))
    val denominator  = exponentials.sum
    Categorical.from(exponentials.scale(1.0 / denominator))
  }

  /** Negative log-likelihood for a one-hot target, computed from logits. */
  def crossEntropy(logits: VectorD, targetIndex: Int): Either[String, Double] =
    if targetIndex < 0 || targetIndex >= logits.size then
      Left(s"target index $targetIndex outside [0, ${logits.size})")
    else logSumExp(logits).map(normalizer => normalizer - logits(targetIndex))
