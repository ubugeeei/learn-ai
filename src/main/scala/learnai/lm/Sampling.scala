package learnai.lm

import java.util.random.RandomGenerator

import learnai.math.Categorical
import learnai.math.Probability
import learnai.math.VectorD
import learnai.text.TokenId

/** Immutable generation-time probability filtering policy.
  *
  * Filtering order is temperature, top-k, then nucleus top-p. At least one
  * token is always retained, and ties are resolved by smaller token ID for
  * deterministic behavior.
  */
final case class SamplingConfig(
    temperature: Double = 1.0,
    topK: Option[Int] = None,
    topP: Option[Double] = None
):
  require(
    temperature > 0.0 && temperature.isFinite,
    s"temperature must be finite and positive: $temperature"
  )
  topK.foreach(value => require(value > 0, s"top-k must be positive: $value"))
  topP.foreach { value =>
    require(value > 0.0 && value <= 1.0 && value.isFinite, s"top-p must be in (0,1]: $value")
  }

object Sampling:
  /** Converts logits to a filtered, renormalized categorical distribution. */
  def distribution(
      logits: VectorD,
      config: SamplingConfig
  ): Either[String, Categorical] =
    Probability.softmax(logits.scale(1.0 / config.temperature)).flatMap { initial =>
      val afterTopK = config.topK match
        case None => initial.probabilities
        case Some(limit) => retainTopK(initial.probabilities, math.min(limit, logits.size))

      val normalizedTopK = normalize(afterTopK)
      val afterTopP = config.topP match
        case None            => normalizedTopK
        case Some(threshold) => retainTopP(normalizedTopK, threshold)
      Categorical.from(normalize(afterTopP))
    }

  /** Samples one token ID using a caller-owned random source. */
  def sample(
      logits: VectorD,
      config: SamplingConfig,
      random: RandomGenerator
  ): Either[String, TokenId] =
    distribution(logits, config).map(distribution => TokenId(distribution.sample(random)))

  private def retainTopK(probabilities: VectorD, limit: Int): VectorD =
    val retained = probabilities.toVector.indices
      .sortBy(index => (-probabilities(index), index))
      .take(limit)
      .toSet
    VectorD.tabulate(probabilities.size) { index =>
      if retained.contains(index) then probabilities(index) else 0.0
    }

  private def retainTopP(probabilities: VectorD, threshold: Double): VectorD =
    val ranked = probabilities.toVector.indices.sortBy(index => (-probabilities(index), index))
    val retained = scala.collection.mutable.Set.empty[Int]
    var cumulative = 0.0
    var rank = 0
    while rank < ranked.size && cumulative < threshold do
      val index = ranked(rank)
      retained += index
      cumulative += probabilities(index)
      rank += 1
    VectorD.tabulate(probabilities.size) { index =>
      if retained.contains(index) then probabilities(index) else 0.0
    }

  private def normalize(probabilities: VectorD): VectorD =
    val total = probabilities.sum
    require(total > 0.0, "sampling filter removed every token")
    probabilities.scale(1.0 / total)
