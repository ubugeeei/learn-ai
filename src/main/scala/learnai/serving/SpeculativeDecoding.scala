package learnai.serving

import java.util.random.RandomGenerator

import learnai.math.Probability
import learnai.math.VectorD
import learnai.text.TokenId
import learnai.transformer.MiniGpt

/** Result of verifying one drafted token against the target distribution. */
final case class DraftVerification(accepted: Boolean, token: Int)

/**
 * The accept/reject core of speculative decoding.
 *
 * A cheap draft model proposes token `x ~ q`; the target model accepts it with probability
 * `min(1, p(x) / q(x))` and otherwise resamples from the *residual* distribution `max(p - q, 0)`
 * renormalized. The scheme's defining theorem is that the emitted token is distributed exactly as
 * `p` — not approximately, and regardless of how bad the draft is. Draft quality affects only the
 * acceptance *rate*, never the output distribution.
 *
 * [[outputDistribution]] computes the emitted distribution analytically from the implementation's
 * own accept and residual rules; the test suite asserts it equals the target elementwise, which
 * turns the paper's Theorem 1 into an executable oracle for this exact code.
 */
object SpeculativeSampling:
  private val SumTolerance = 1e-9

  /** Validates a categorical distribution: non-negative, finite, sums to 1. */
  def validateDistribution(distribution: Vector[Double], name: String): Unit =
    require(distribution.nonEmpty, s"$name distribution cannot be empty")
    distribution.zipWithIndex.foreach { case (probability, index) =>
      require(
        probability >= 0.0 && probability.isFinite,
        s"$name probability at $index must be finite and non-negative: $probability"
      )
    }
    val total = distribution.sum
    require(math.abs(total - 1.0) <= SumTolerance, s"$name distribution sums to $total, expected 1")

  /** Probability that one drafted token is accepted: `sum(min(p, q))`. */
  def acceptanceProbability(target: Vector[Double], draft: Vector[Double]): Double =
    validatePair(target, draft)
    target.zip(draft).map((p, q) => math.min(p, q)).sum

  /**
   * The renormalized excess of the target over the draft.
   *
   * Undefined (Left) when the draft already matches the target, because rejection then has zero
   * probability and no resampling rule is needed.
   */
  def residualDistribution(
      target: Vector[Double],
      draft: Vector[Double]
  ): Either[String, Vector[Double]] =
    validatePair(target, draft)
    val excess = target.zip(draft).map((p, q) => math.max(p - q, 0.0))
    val mass   = excess.sum
    if mass <= 0.0 then Left("draft equals target; the residual distribution is undefined")
    else Right(excess.map(_ / mass))

  /**
   * Analytic distribution of the emitted token under accept/reject.
   *
   * `emitted(x) = min(p(x), q(x)) + (1 - alpha) * residual(x)` with `alpha = sum(min(p, q))`. The
   * scheme is correct exactly when this equals `p`, which the suite asserts for adversarial pairs.
   */
  def outputDistribution(target: Vector[Double], draft: Vector[Double]): Vector[Double] =
    val acceptedMass      = target.zip(draft).map((p, q) => math.min(p, q))
    val rejectProbability = 1.0 - acceptedMass.sum
    residualDistribution(target, draft) match
      case Left(_)         => acceptedMass
      case Right(residual) => acceptedMass.zip(residual)
          .map((kept, extra) => kept + rejectProbability * extra)

  /**
   * Verifies one drafted token, resampling from the residual on rejection.
   *
   * The accept test uses `u * q(x) < p(x)` so no division can overflow, and rejection walks the
   * unnormalized residual with a second uniform draw scaled by the residual mass, avoiding a
   * renormalization pass. Both draws come from the caller-owned generator, and that consumption is
   * part of the deterministic stream contract (Chapter 22c).
   */
  def verifyDraftToken(
      target: Vector[Double],
      draft: Vector[Double],
      draftToken: Int,
      random: RandomGenerator
  ): DraftVerification =
    validatePair(target, draft)
    require(
      draftToken >= 0 && draftToken < draft.size,
      s"draft token $draftToken outside [0, ${draft.size})"
    )
    require(
      draft(draftToken) > 0.0,
      s"draft token $draftToken has zero draft probability and cannot have been sampled"
    )
    val uniform = random.nextDouble()
    if uniform * draft(draftToken) < target(draftToken) then
      DraftVerification(accepted = true, token = draftToken)
    else
      val excess = target.zip(draft).map((p, q) => math.max(p - q, 0.0))
      val mass   = excess.sum
      if mass <= 0.0 then
        // Numerically indistinguishable distributions: rejection probability
        // is zero up to rounding, so keep the drafted token.
        DraftVerification(accepted = true, token = draftToken)
      else
        val threshold  = random.nextDouble() * mass
        var cumulative = 0.0
        var index      = 0
        var chosen     = -1
        while index < excess.size && chosen < 0 do
          cumulative += excess(index)
          if cumulative > threshold then chosen = index
          index += 1
        DraftVerification(accepted = false, token = if chosen >= 0 then chosen else excess.size - 1)

  private def validatePair(target: Vector[Double], draft: Vector[Double]): Unit =
    validateDistribution(target, "target")
    validateDistribution(draft, "draft")
    require(target.size == draft.size, s"target size ${target.size} != draft size ${draft.size}")

/**
 * Work accounting for one speculative generation request.
 *
 * The speedup story is `targetVerificationPasses` versus emitted tokens: plain decoding runs the
 * target once per token, while speculative decoding runs it once per *round* and verifies a whole
 * draft block in that single pass. Draft work is counted separately because the draft model is
 * priced differently.
 */
final case class SpeculativeStatistics(
    rounds: Int,
    draftedTokens: Int,
    acceptedDraftTokens: Int,
    rejectedDrafts: Int,
    bonusTokens: Int,
    targetVerificationPasses: Long
):
  require(rounds >= 0, "rounds cannot be negative")
  require(draftedTokens >= 0, "drafted tokens cannot be negative")
  require(acceptedDraftTokens >= 0, "accepted tokens cannot be negative")
  require(acceptedDraftTokens <= draftedTokens, "cannot accept more than drafted")
  require(rejectedDrafts >= 0 && rejectedDrafts <= rounds, "invalid rejection count")
  require(bonusTokens >= 0, "bonus tokens cannot be negative")

  def emittedTokens: Int = acceptedDraftTokens + rejectedDrafts + bonusTokens

  def acceptanceRate: Double =
    if draftedTokens == 0 then 1.0 else acceptedDraftTokens.toDouble / draftedTokens.toDouble

final case class SpeculativeGenerationResult(
    tokens: Vector[TokenId],
    statistics: SpeculativeStatistics
)

/**
 * Draft-and-verify generation over two MiniGPT models.
 *
 * Per round the draft proposes up to `lookahead` tokens sequentially; the target then scores the
 * whole proposal in one `logits` call — the batched verification that makes the scheme profitable —
 * and each proposal is accepted or rejected in order. The first rejection emits the residual sample
 * and ends the round; a fully accepted round emits one bonus token from the target's own next
 * distribution, so even a rejected-everything pathology still emits one token per target pass.
 */
object SpeculativeDecoding:
  def generate(
      target: MiniGpt,
      draft: MiniGpt,
      prompt: Vector[TokenId],
      newTokenCount: Int,
      lookahead: Int,
      temperature: Double,
      random: RandomGenerator
  ): Either[String, SpeculativeGenerationResult] =
    require(prompt.nonEmpty, "speculative generation requires a non-empty prompt")
    require(newTokenCount >= 0, s"new token count must be non-negative: $newTokenCount")
    require(lookahead > 0, s"lookahead must be positive: $lookahead")
    require(
      lookahead < target.config.maximumContextLength,
      s"lookahead $lookahead must leave context room in ${target.config.maximumContextLength}"
    )
    require(
      temperature > 0.0 && temperature.isFinite,
      s"temperature must be finite and positive: $temperature"
    )
    require(
      target.config.vocabularySize == draft.config.vocabularySize,
      s"vocabulary mismatch: target ${target.config.vocabularySize}, " +
        s"draft ${draft.config.vocabularySize}"
    )

    var generated             = prompt
    var emitted               = 0
    var rounds                = 0
    var drafted               = 0
    var accepted              = 0
    var rejections            = 0
    var bonus                 = 0
    var error: Option[String] = None

    while emitted < newTokenCount && error.isEmpty do
      val blockSize = math.min(lookahead, newTokenCount - emitted)
      // Both phases condition on one shared window with room for the whole
      // proposal block. Under learned absolute positions, letting the draft
      // see a longer window than the verifier would shift every position
      // id and make even an identical draft diverge from the target.
      val retained  = generated.takeRight(target.config.maximumContextLength - blockSize)

      // Draft phase: sequential proposals from the draft's distributions.
      var proposals          = Vector.empty[Int]
      var draftDistributions = Vector.empty[Vector[Double]]
      while proposals.size < blockSize && error.isEmpty do
        val context = retained ++ proposals.map(TokenId(_))
        draft.nextDistribution(context, temperature) match
          case Left(problem)       => error = Some(problem)
          case Right(distribution) =>
            val probabilities = distribution.probabilities.toVector
            proposals :+= distribution.sample(random)
            draftDistributions :+= probabilities

      if error.isEmpty then
        rounds += 1
        drafted += proposals.size
        // Verification phase: one target forward scores every proposal
        // position plus the bonus position.
        val logits = target.logits(retained ++ proposals.map(TokenId(_)))

        def targetDistribution(offset: Int): Either[String, Vector[Double]] = Probability.softmax(
          VectorD.from(logits.rowValues(retained.size - 1 + offset)).scale(1.0 / temperature)
        ).map(_.probabilities.toVector)

        var index     = 0
        var roundOpen = true
        while index < proposals.size && roundOpen && error.isEmpty do
          targetDistribution(index) match
            case Left(problem)              => error = Some(problem)
            case Right(targetProbabilities) =>
              val verification = SpeculativeSampling.verifyDraftToken(
                targetProbabilities,
                draftDistributions(index),
                proposals(index),
                random
              )
              generated :+= TokenId(verification.token)
              emitted += 1
              if verification.accepted then
                accepted += 1
                index += 1
              else
                rejections += 1
                roundOpen = false

        if roundOpen && error.isEmpty && emitted < newTokenCount then
          targetDistribution(proposals.size) match
            case Left(problem)             => error = Some(problem)
            case Right(bonusProbabilities) =>
              val cumulativeTarget = random.nextDouble()
              var cumulative       = 0.0
              var token            = -1
              var candidate        = 0
              while candidate < bonusProbabilities.size && token < 0 do
                cumulative += bonusProbabilities(candidate)
                if cumulative > cumulativeTarget then token = candidate
                candidate += 1
              val chosen           = if token >= 0 then token else bonusProbabilities.size - 1
              generated :+= TokenId(chosen)
              emitted += 1
              bonus += 1

    error match
      case Some(problem) => Left(problem)
      case None          => Right(SpeculativeGenerationResult(
          generated,
          SpeculativeStatistics(rounds, drafted, accepted, rejections, bonus, rounds.toLong)
        ))
