package learnai.chat

import java.nio.file.Files
import java.nio.file.Path

import scala.io.StdIn

import learnai.finetune.ChatRole
import learnai.io.MiniGptCheckpoint
import learnai.text.Utf8

/** Minimal terminal boundary that keeps the interactive loop unit-testable. */
trait ChatTerminal:
  def readLine(prompt: String): Option[String]
  def printLine(line: String): Unit

/** Terminal implementation used by the real CLI entrypoint. */
object StandardChatTerminal extends ChatTerminal:
  override def readLine(prompt: String): Option[String] =
    print(prompt)
    Console.flush()
    Option(StdIn.readLine())

  override def printLine(line: String): Unit = println(line)

/** Stateful terminal loop; model inference remains behind [[LocalChatResponder]]. */
final class InteractiveLocalChat(
    responder: LocalChatResponder,
    terminal: ChatTerminal,
    examples: Vector[LocalChatPair] = LocalChatCorpus.examples,
    maximumInputBytes: Int = 120
):
  require(maximumInputBytes > 0, "maximum input bytes must be positive")
  private var history = Vector.empty[LocalChatTurn]

  /** Reads until `/quit` or end-of-input and never throws on a model error. */
  def run(): Unit =
    terminal.printLine("Local MiniGPT chat (Scala 3, CPU, no network)")
    terminal.printLine("Type /help for commands or /examples for prompts the tiny model learned.")
    terminal.printLine("This model is a learning-scale memorizer, not a general assistant.")
    terminal.printLine("Inference context: latest user turn only; /history is display-only.")

    var running = true
    while running do
      terminal.readLine("you> ") match
        case None =>
          terminal.printLine("bye")
          running = false
        case Some(raw) =>
          val input = raw.trim
          input match
            case ""         => ()
            case "/quit"    =>
              terminal.printLine("bye")
              running = false
            case "/help"    => printHelp()
            case "/history" => printHistory()
            case "/examples" => printExamples()
            case "/reset" =>
              history = Vector.empty
              terminal.printLine("history cleared")
            case command if command.startsWith("/") =>
              terminal.printLine(s"unknown command: $command (try /help)")
            case message => answer(message)

  private def answer(message: String): Unit =
    val byteCount = Utf8.encodeBytes(message).size
    if byteCount > maximumInputBytes then
      terminal.printLine(s"input has $byteCount UTF-8 bytes; limit is $maximumInputBytes")
    else
      val pendingHistory = history :+ LocalChatTurn(ChatRole.User, message)
      responder.reply(pendingHistory) match
        case Left(problem) =>
          terminal.printLine(s"model error: $problem")
        case Right(reply) =>
          history = pendingHistory :+ LocalChatTurn(ChatRole.Assistant, reply.text)
          terminal.printLine(s"assistant> ${reply.text}")
          terminal.printLine(
            s"[tokens: input=${reply.inputTokens}, output=${reply.outputTokens}; " +
              s"stop=${renderStop(reply.stopReason)}]"
          )

  private def printHelp(): Unit =
    terminal.printLine("/examples  show the seven prompts present in the training corpus")
    terminal.printLine("/history   show successful turns retained by this process")
    terminal.printLine("/reset     clear conversation history")
    terminal.printLine("/quit      leave the chat")

  private def printExamples(): Unit =
    terminal.printLine(examples.map(_.user).mkString("learned prompts: ", " | ", ""))

  private def printHistory(): Unit =
    if history.isEmpty then terminal.printLine("history is empty")
    else
      history.zipWithIndex.foreach { case (turn, index) =>
        terminal.printLine(s"${index + 1}. ${turn.role.toString.toLowerCase}: ${turn.text}")
      }

  private def renderStop(reason: LocalChatStopReason): String = reason match
    case LocalChatStopReason.EndOfTurn             => "end_of_turn"
    case LocalChatStopReason.TokenLimit            => "token_limit"
    case LocalChatStopReason.UnexpectedControlToken => "unexpected_control_token"

/** Loads the cached local checkpoint or performs the small SFT run once. */
object LocalChatApplication:
  val checkpointPath: Path = Path.of("target", "local-chat", "mini-gpt-chat-v4.laigpt")

  /** Builds the local model, printing every state transition relevant to a learner. */
  def loadOrTrain(terminal: ChatTerminal): MiniGptChatResponder =
    val expectedConfig = LocalChatTrainer.defaultConfig.model
    val loaded =
      if Files.isRegularFile(checkpointPath) then
        MiniGptCheckpoint.load(checkpointPath) match
          case Right((model, metadata)) if model.config == expectedConfig =>
            terminal.printLine(
              s"loaded checkpoint: $checkpointPath " +
                s"(${metadata.scalarParameters} parameters, sha256=${metadata.sha256.take(12)}...)"
            )
            Some(model)
          case Right((_, _)) =>
            terminal.printLine("cached checkpoint uses a different architecture; retraining")
            None
          case Left(problem) =>
            terminal.printLine(s"cached checkpoint was rejected: $problem")
            terminal.printLine("retraining from the fixed seed")
            None
      else None

    val model = loaded.getOrElse {
      val config = LocalChatTrainer.defaultConfig
      terminal.printLine(
        s"no usable checkpoint; training ${LocalChatCorpus.examples.size} conversations " +
          s"for ${config.epochs} epochs on the CPU"
      )
      val trained = LocalChatTrainer.train(
        config = config,
        onProgress = progress =>
          terminal.printLine(f"training epoch=${progress.epoch}%3d mean_loss=${progress.meanLoss}%.6f")
      )
      terminal.printLine(
        f"training complete: loss ${trained.initialMeanLoss}%.6f -> ${trained.finalMeanLoss}%.6f, " +
          s"updates=${trained.optimizerUpdates}, parameters=${trained.model.parameterCount}"
      )
      MiniGptCheckpoint.save(trained.model, checkpointPath) match
        case Right(metadata) =>
          terminal.printLine(
            s"saved checkpoint: $checkpointPath (sha256=${metadata.sha256.take(12)}...)"
          )
        case Left(problem) => terminal.printLine(s"checkpoint save warning: $problem")
      trained.model
    }

    new MiniGptChatResponder(model)

@main def runLocalChat(): Unit =
  val responder = LocalChatApplication.loadOrTrain(StandardChatTerminal)
  new InteractiveLocalChat(responder, StandardChatTerminal).run()
