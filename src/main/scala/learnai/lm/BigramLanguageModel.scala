package learnai.lm

import java.util.SplittableRandom
import java.util.random.RandomGenerator

import learnai.math.Categorical
import learnai.math.Probability
import learnai.math.VectorD
import learnai.optim.AdamW
import learnai.optim.Initialization
import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.text.BpeTrainer
import learnai.text.TokenId
import learnai.text.TokenId.*

/**
 * A neural bigram language model.
 *
 * Row `i` of `transitionLogits` contains unnormalized scores for the token following token `i`. The
 * model has no hidden state and a context length of exactly one token.
 */
final class BigramLanguageModel private (val vocabularySize: Int, val transitionLogits: Tensor):
  require(vocabularySize > 0, s"vocabulary size must be positive: $vocabularySize")
  require(
    transitionLogits.shape == Shape(vocabularySize, vocabularySize),
    s"transition table must have shape [$vocabularySize,$vocabularySize], " +
      s"got ${transitionLogits.shape}"
  )
  require(transitionLogits.isTrainable, "transition logits must be trainable")

  /** Returns `[N, vocabularySize]` logits for `N` current tokens. */
  def apply(inputTokenIds: Vector[TokenId]): Tensor =
    val rows = validateTokenIds(inputTokenIds, "input")
    transitionLogits.gatherRows(rows)

  /** Computes mean next-token cross entropy for aligned token pairs. */
  def loss(inputTokenIds: Vector[TokenId], targetTokenIds: Vector[TokenId]): Tensor =
    require(inputTokenIds.nonEmpty, "bigram loss requires at least one token pair")
    require(
      inputTokenIds.size == targetTokenIds.size,
      s"input and target sizes differ: ${inputTokenIds.size} != ${targetTokenIds.size}"
    )
    val targets = validateTokenIds(targetTokenIds, "target")
    apply(inputTokenIds).crossEntropy(targets)

  /** Converts one transition row into a temperature-scaled distribution. */
  def nextDistribution(
      currentTokenId: TokenId,
      temperature: Double = 1.0
  ): Either[String, Categorical] =
    require(temperature > 0.0 && temperature.isFinite, s"temperature must be finite and positive")
    val row          = validateTokenIds(Vector(currentTokenId), "current").head
    val scaledLogits = VectorD.from(transitionLogits.rowValues(row)).scale(1.0 / temperature)
    Probability.softmax(scaledLogits)

  /**
   * Samples `newTokenCount` transitions and includes the start token in the result.
   *
   * The caller owns the random generator, so equal model state, seed, start, count, and temperature
   * produce the same token sequence.
   */
  def generate(
      start: TokenId,
      newTokenCount: Int,
      temperature: Double,
      random: RandomGenerator
  ): Either[String, Vector[TokenId]] =
    require(newTokenCount >= 0, s"new token count must be non-negative: $newTokenCount")
    validateTokenIds(Vector(start), "start")
    val generated             = Vector.newBuilder[TokenId]
    generated += start
    var current               = start
    var remaining             = newTokenCount
    var error: Option[String] = None
    while remaining > 0 && error.isEmpty do
      nextDistribution(current, temperature) match
        case Left(message)       => error = Some(message)
        case Right(distribution) =>
          current = TokenId(distribution.sample(random))
          generated += current
          remaining -= 1
    error match
      case Some(message) => Left(message)
      case None          => Right(generated.result())

  def parameters: Vector[Tensor] = Vector(transitionLogits)

  private def validateTokenIds(tokenIds: Vector[TokenId], role: String): Vector[Int] = tokenIds
    .zipWithIndex.map { case (tokenId, index) =>
      val value = tokenId.value
      require(
        value >= 0 && value < vocabularySize,
        s"$role token ID $value at index $index outside [0, $vocabularySize)"
      )
      value
    }

object BigramLanguageModel:
  /** Creates a reproducibly initialized transition table. */
  def random(vocabularySize: Int, seed: Long): BigramLanguageModel =
    require(vocabularySize > 0, s"vocabulary size must be positive: $vocabularySize")
    val logits = Initialization.xavierUniform(
      Shape(vocabularySize, vocabularySize),
      fanIn = vocabularySize,
      fanOut = vocabularySize,
      new SplittableRandom(seed),
      label = "bigram.transitionLogits"
    )
    new BigramLanguageModel(vocabularySize, logits)

  /** Constructs a model from explicit row-major logits, mainly for tests and loading. */
  def fromLogits(vocabularySize: Int, values: Vector[Double]): BigramLanguageModel =
    val logits = Tensor
      .parameter(Shape(vocabularySize, vocabularySize), values, "bigram.transitionLogits")
    new BigramLanguageModel(vocabularySize, logits)

object BigramTrainer:
  /**
   * Full-batch AdamW training over aligned adjacent token pairs.
   *
   * @return
   *   loss before each parameter update, with exactly `steps` entries.
   */
  def train(
      model: BigramLanguageModel,
      inputs: Vector[TokenId],
      targets: Vector[TokenId],
      steps: Int,
      learningRate: Double
  ): Vector[Double] =
    require(steps >= 0, s"training steps must be non-negative: $steps")
    val optimizer =
      new AdamW(learningRate = learningRate, weightDecay = 0.0, maximumGradientNorm = Some(1.0))
    val history   = Vector.newBuilder[Double]
    var step      = 0
    while step < steps do
      val loss = model.loss(inputs, targets)
      history += loss.valueAtFlat(0)
      loss.backward()
      val _    = optimizer.step(model.parameters)
      step += 1
    history.result()

def trainBigram(): Unit =
  val corpus    = Vector.fill(20)("scala learns ai. ai learns scala. ").mkString
  val tokenizer = BpeTrainer.train(Vector(corpus), targetVocabularySize = 272)
  val tokens    = tokenizer.encode(corpus)
  val inputs    = tokens.dropRight(1)
  val targets   = tokens.drop(1)
  val model     = BigramLanguageModel.random(tokenizer.vocabularySize, seed = 42L)
  val history   = BigramTrainer.train(model, inputs, targets, steps = 250, learningRate = 0.05)
  println(f"initial loss: ${history.head}%.6f")
  println(f"final loss:   ${history.last}%.6f")

  val generated = model.generate(
    start = tokens.head,
    newTokenCount = 80,
    temperature = 0.7,
    random = new SplittableRandom(7L)
  )
  generated.flatMap(tokenizer.decode) match
    case Right(text)   => println(s"generated: $text")
    case Left(problem) => println(s"generated bytes were not valid UTF-8: $problem")
