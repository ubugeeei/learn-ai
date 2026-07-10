package learnai.chat

import learnai.finetune.ChatMessage
import learnai.finetune.ChatRole
import learnai.finetune.ChatSpecialTokens
import learnai.finetune.ChatTemplate
import learnai.finetune.Conversation
import learnai.finetune.SftEvaluation
import learnai.finetune.SftExample
import learnai.optim.AdamW
import learnai.text.ByteTokenizer
import learnai.text.TokenId
import learnai.text.TokenId.*
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

/** One text turn retained by the local interactive session. */
final case class LocalChatTurn(role: ChatRole, text: String):
  require(role != ChatRole.System, "interactive turns must be user or assistant messages")
  require(text.nonEmpty, "an interactive chat turn cannot be empty")

/** One deliberately small supervised conversation used by the local lab. */
final case class LocalChatPair(user: String, assistant: String):
  require(user.nonEmpty, "a local chat training prompt cannot be empty")
  require(assistant.nonEmpty, "a local chat training answer cannot be empty")

/** Fixed byte-level wire format shared by local SFT and local inference.
  *
  * UTF-8 bytes occupy IDs 0 through 255. Four chat-control tokens occupy IDs
  * 256 through 259. Keeping this mapping fixed means the checkpoint needs no
  * hidden tokenizer artifact: its vocabulary contract is visible in code.
  */
object LocalChatCodec:
  val specialTokens: ChatSpecialTokens = ChatSpecialTokens(
    system = TokenId(256),
    user = TokenId(257),
    assistant = TokenId(258),
    endOfTurn = TokenId(259)
  )
  val vocabularySize: Int = 260

  /** Encodes ordinary message text without inserting control tokens. */
  def encodeText(text: String): Vector[TokenId] = ByteTokenizer.encode(text)

  /** Decodes assistant content and refuses leaked chat-control tokens. */
  def decodeAssistant(tokens: Vector[TokenId]): Either[String, String] =
    tokens.find(_.value >= ByteTokenizer.ByteVocabularySize) match
      case Some(token) => Left(s"assistant content contains reserved token ${token.value}")
      case None        => ByteTokenizer.decode(tokens, skipSpecialTokens = false)

  /** Converts one text pair into the exact assistant-masked SFT example. */
  def trainingExample(pair: LocalChatPair): SftExample =
    val conversation = Conversation(
      Vector(
        ChatMessage(ChatRole.User, encodeText(pair.user)),
        ChatMessage(ChatRole.Assistant, encodeText(pair.assistant))
      )
    )
    ChatTemplate.trainingExample(ChatTemplate.render(conversation, specialTokens))

  /** Renders prior turns and appends the marker that asks the model to answer.
    *
    * The returned prompt ends in `<|assistant|>` but contains no generated
    * assistant content. This must match the role-marker convention used by
    * [[trainingExample]] or the model is conditioned on an unseen format.
    */
  def inferencePrompt(turns: Vector[LocalChatTurn]): Vector[TokenId] =
    require(turns.nonEmpty, "local chat inference requires at least one turn")
    require(turns.last.role == ChatRole.User, "local chat inference must end with a user turn")
    val conversation = Conversation(
      turns.map(turn => ChatMessage(turn.role, encodeText(turn.text)))
    )
    ChatTemplate.render(conversation, specialTokens).tokens :+ specialTokens.assistant

/** Why local autoregressive decoding stopped. */
enum LocalChatStopReason:
  case EndOfTurn
  case TokenLimit
  case UnexpectedControlToken

/** Observable result of one local generation request. */
final case class LocalChatReply(
    text: String,
    inputTokens: Int,
    outputTokens: Int,
    stopReason: LocalChatStopReason
):
  require(inputTokens > 0, "a reply must have at least one input token")
  require(outputTokens >= 0, "output token count cannot be negative")

/** Small boundary used to test the terminal loop without training a model. */
trait LocalChatResponder:
  def reply(history: Vector[LocalChatTurn]): Either[String, LocalChatReply]

/** Greedy decoder backed by the repository's own [[MiniGpt]].
  *
  * Greedy decoding is intentional for this first interactive lab: fixed
  * weights and input produce the same answer, making regressions observable.
  * Sampling remains available in Chapter 23 and can replace this policy.
  */
final class MiniGptChatResponder(
    model: MiniGpt,
    maximumNewTokens: Int = 32
) extends LocalChatResponder:
  require(maximumNewTokens > 0, "maximum new tokens must be positive")

  override def reply(history: Vector[LocalChatTurn]): Either[String, LocalChatReply] =
    require(history.nonEmpty, "local chat response requires at least one turn")
    val currentUserTurn = history.last
    require(currentUserTurn.role == ChatRole.User, "local chat history must end with a user turn")
    // The bundled SFT corpus contains independent one-turn conversations.
    // Supplying older turns would be an inference/training template mismatch,
    // so this first lab deliberately conditions on the current user turn only.
    val prompt = LocalChatCodec.inferencePrompt(Vector(currentUserTurn))
    var context = prompt
    var generated = Vector.empty[TokenId]
    var stopReason = LocalChatStopReason.TokenLimit
    var stopped = false
    var error: Option[String] = None

    while generated.size < maximumNewTokens && !stopped && error.isEmpty do
      model.nextDistribution(context) match
        case Left(problem) => error = Some(problem)
        case Right(distribution) =>
          val nextId = distribution.probabilities.toVector.indices.minBy { index =>
            (-distribution.probability(index), index)
          }
          val next = TokenId(nextId)
          if next == LocalChatCodec.specialTokens.endOfTurn then
            stopReason = LocalChatStopReason.EndOfTurn
            stopped = true
          else if next.value >= ByteTokenizer.ByteVocabularySize then
            stopReason = LocalChatStopReason.UnexpectedControlToken
            stopped = true
          else
            generated :+= next
            context :+= next

    error match
      case Some(problem) => Left(problem)
      case None =>
        LocalChatCodec.decodeAssistant(generated).map { text =>
          LocalChatReply(
            text = text,
            inputTokens = prompt.size,
            outputTokens = generated.size + (if stopped then 1 else 0),
            stopReason = stopReason
          )
        }

/** Hyperparameters required to reproduce the local conversational model. */
final case class LocalChatTrainingConfig(
    model: MiniGptConfig,
    epochs: Int,
    learningRate: Double,
    seed: Long,
    progressEveryEpochs: Int
):
  require(model.vocabularySize == LocalChatCodec.vocabularySize, "model vocabulary must match codec")
  require(epochs > 0, "local chat training epochs must be positive")
  require(
    learningRate > 0.0 && learningRate.isFinite,
    "local chat learning rate must be finite and positive"
  )
  require(progressEveryEpochs > 0, "progress interval must be positive")

/** One stable progress observation emitted after a complete training epoch. */
final case class LocalChatTrainingProgress(epoch: Int, meanLoss: Double)

/** The trained model and enough evidence to inspect the optimization run. */
final case class LocalChatTrainingResult(
    model: MiniGpt,
    initialMeanLoss: Double,
    finalMeanLoss: Double,
    optimizerUpdates: Int
)

/** Reproducible, intentionally tiny SFT corpus shipped with the lab. */
object LocalChatCorpus:
  val examples: Vector[LocalChatPair] = Vector(
    LocalChatPair("hello", "hello from scala."),
    LocalChatPair("who are you?", "i am a tiny local model."),
    LocalChatPair("token?", "a token is a text id."),
    LocalChatPair("attention?", "attention mixes context."),
    LocalChatPair("training?", "training adjusts weights."),
    LocalChatPair("scala?", "scala runs this model."),
    LocalChatPair("bye", "goodbye.")
  )

/** Full-parameter SFT loop for the local chat lab.
  *
  * One optimizer update is performed per conversation per epoch. The ordering
  * is fixed rather than shuffled so the checkpoint is reproducible from the
  * source, seed, and hyperparameters alone. This is a memorization lab, not a
  * statistically meaningful training recipe.
  */
object LocalChatTrainer:
  val defaultConfig: LocalChatTrainingConfig = LocalChatTrainingConfig(
    model = MiniGptConfig(
      vocabularySize = LocalChatCodec.vocabularySize,
      maximumContextLength = 48,
      channels = 12,
      headCount = 3,
      hiddenChannels = 24,
      layerCount = 1
    ),
    epochs = 80,
    learningRate = 0.02,
    seed = 2_026_071_000L,
    progressEveryEpochs = 20
  )

  /** Trains from a deterministic initialization and reports epoch boundaries. */
  def train(
      config: LocalChatTrainingConfig = defaultConfig,
      examples: Vector[LocalChatPair] = LocalChatCorpus.examples,
      onProgress: LocalChatTrainingProgress => Unit = _ => ()
  ): LocalChatTrainingResult =
    require(examples.nonEmpty, "local chat training requires at least one example")
    val trainingExamples = examples.map(LocalChatCodec.trainingExample)
    trainingExamples.zipWithIndex.foreach { case (example, index) =>
      require(
        example.inputs.size <= config.model.maximumContextLength,
        s"training example $index has ${example.inputs.size} tokens; " +
          s"model context is ${config.model.maximumContextLength}"
      )
    }

    val model = MiniGpt.random(config.model, config.seed)
    val optimizer = new AdamW(
      learningRate = config.learningRate,
      weightDecay = 0.0,
      maximumGradientNorm = Some(1.0)
    )
    val initialLoss = meanLoss(model, trainingExamples)

    var epoch = 1
    while epoch <= config.epochs do
      trainingExamples.foreach { example =>
        model.parameters.foreach(_.clearGradients())
        val loss = SftEvaluation.exampleLoss(model, example)
        loss.backward()
        val _ = optimizer.step(model.parameters)
      }
      model.parameters.foreach(_.clearGradients())
      if epoch % config.progressEveryEpochs == 0 || epoch == config.epochs then
        onProgress(LocalChatTrainingProgress(epoch, meanLoss(model, trainingExamples)))
      epoch += 1

    LocalChatTrainingResult(
      model = model,
      initialMeanLoss = initialLoss,
      finalMeanLoss = meanLoss(model, trainingExamples),
      optimizerUpdates = config.epochs * examples.size
    )

  private def meanLoss(model: MiniGpt, examples: Vector[SftExample]): Double =
    examples.map(example => SftEvaluation.exampleLoss(model, example).valueAtFlat(0)).sum /
      examples.size.toDouble
