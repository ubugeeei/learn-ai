package learnai.chat

import scala.collection.mutable.ArrayBuffer

import learnai.finetune.ChatRole
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId.*
import learnai.transformer.MiniGptConfig

object LocalChatSuite extends TestSuite:
  override val name: String = "LocalChat"

  private final class ScriptedTerminal(lines: Vector[String]) extends ChatTerminal:
    private var index = 0
    val prompts: ArrayBuffer[String] = ArrayBuffer.empty
    val output: ArrayBuffer[String] = ArrayBuffer.empty

    override def readLine(prompt: String): Option[String] =
      prompts += prompt
      if index >= lines.size then None
      else
        val line = lines(index)
        index += 1
        Some(line)

    override def printLine(line: String): Unit = output += line

  private final class RecordingResponder(
      decisions: Vector[Either[String, LocalChatReply]]
  ) extends LocalChatResponder:
    private var index = 0
    val histories: ArrayBuffer[Vector[LocalChatTurn]] = ArrayBuffer.empty

    override def reply(history: Vector[LocalChatTurn]): Either[String, LocalChatReply] =
      histories += history
      require(index < decisions.size, "scripted responder ran out of decisions")
      val decision = decisions(index)
      index += 1
      decision

  private val ok = Right(
    LocalChatReply("answer", 8, 3, LocalChatStopReason.EndOfTurn)
  )

  override val tests: Vector[TestCase] = Vector(
    test("the local codec uses four reserved IDs above every byte") {
      Assert.equal(LocalChatCodec.vocabularySize, 260)
      Assert.equal(
        LocalChatCodec.specialTokens.all.map(_.value),
        Vector(256, 257, 258, 259)
      )
      val unicode = "Scala, café, and 🚀"
      Assert.equal(
        LocalChatCodec.decodeAssistant(LocalChatCodec.encodeText(unicode)),
        Right(unicode)
      )
    },
    test("training and inference share the exact assistant role marker") {
      val pair = LocalChatPair("hi", "ok")
      val example = LocalChatCodec.trainingExample(pair)
      val prompt = LocalChatCodec.inferencePrompt(
        Vector(LocalChatTurn(ChatRole.User, pair.user))
      )
      Assert.equal(prompt.last, LocalChatCodec.specialTokens.assistant)
      Assert.equal(example.inputs.take(prompt.size), prompt)
      Assert.equal(example.trainableTargetCount, pair.assistant.length + 1)
    },
    test("assistant decoding refuses leaked role-control tokens") {
      val result = LocalChatCodec.decodeAssistant(
        LocalChatCodec.encodeText("ok") :+ LocalChatCodec.specialTokens.user
      )
      Assert.isTrue(Assert.left(result).contains("reserved token"))
    },
    test("a small real SFT run reduces mean assistant loss") {
      val config = LocalChatTrainingConfig(
        model = MiniGptConfig(
          vocabularySize = LocalChatCodec.vocabularySize,
          maximumContextLength = 16,
          channels = 4,
          headCount = 1,
          hiddenChannels = 8,
          layerCount = 1
        ),
        epochs = 30,
        learningRate = 0.04,
        seed = 19L,
        progressEveryEpochs = 10
      )
      val progress = ArrayBuffer.empty[LocalChatTrainingProgress]
      val result = LocalChatTrainer.train(
        config,
        Vector(LocalChatPair("a", "b")),
        update => progress += update
      )
      Assert.isTrue(
        result.finalMeanLoss < result.initialMeanLoss / 2.0,
        s"expected loss reduction, got ${result.initialMeanLoss} -> ${result.finalMeanLoss}"
      )
      Assert.equal(result.optimizerUpdates, 30)
      Assert.equal(progress.map(_.epoch).toVector, Vector(10, 20, 30))
    },
    test("successful turns are passed back as typed history") {
      val responder = new RecordingResponder(Vector(ok, ok))
      val terminal = new ScriptedTerminal(Vector("hello", "token?", "/quit"))
      new InteractiveLocalChat(responder, terminal).run()
      Assert.equal(responder.histories.size, 2)
      Assert.equal(
        responder.histories(1).map(_.role),
        Vector(ChatRole.User, ChatRole.Assistant, ChatRole.User)
      )
      Assert.equal(responder.histories(1).map(_.text), Vector("hello", "answer", "token?"))
      Assert.isTrue(terminal.output.exists(_.startsWith("assistant> answer")))
      Assert.isTrue(terminal.output.exists(_.contains("stop=end_of_turn")))
    },
    test("a model error is displayed and is not committed to history") {
      val responder = new RecordingResponder(Vector(Left("numerical failure"), ok))
      val terminal = new ScriptedTerminal(Vector("first", "second", "/quit"))
      new InteractiveLocalChat(responder, terminal).run()
      Assert.isTrue(terminal.output.contains("model error: numerical failure"))
      Assert.equal(responder.histories(1), Vector(LocalChatTurn(ChatRole.User, "second")))
    },
    test("reset clears history before the next request") {
      val responder = new RecordingResponder(Vector(ok, ok))
      val terminal = new ScriptedTerminal(Vector("one", "/reset", "two", "/quit"))
      new InteractiveLocalChat(responder, terminal).run()
      Assert.equal(responder.histories(1), Vector(LocalChatTurn(ChatRole.User, "two")))
      Assert.isTrue(terminal.output.contains("history cleared"))
    },
    test("help examples history limits and unknown commands are observable") {
      val responder = new RecordingResponder(Vector(ok))
      val terminal = new ScriptedTerminal(
        Vector("/help", "/examples", "/history", "/what", "abcd", "/history", "/quit")
      )
      new InteractiveLocalChat(
        responder,
        terminal,
        examples = Vector(LocalChatPair("try", "this")),
        maximumInputBytes = 4
      ).run()
      Assert.isTrue(terminal.output.exists(_.startsWith("/examples")))
      Assert.isTrue(terminal.output.contains("learned prompts: try"))
      Assert.isTrue(terminal.output.contains("history is empty"))
      Assert.isTrue(terminal.output.exists(_.startsWith("unknown command")))
      Assert.isTrue(terminal.output.contains("1. user: abcd"))
      Assert.isTrue(terminal.output.contains("2. assistant: answer"))
    },
    test("oversized UTF-8 input is refused before model invocation") {
      val responder = new RecordingResponder(Vector.empty)
      val terminal = new ScriptedTerminal(Vector("ééé", "/quit"))
      new InteractiveLocalChat(responder, terminal, maximumInputBytes = 4).run()
      Assert.equal(responder.histories.size, 0)
      Assert.isTrue(terminal.output.exists(_.contains("UTF-8 bytes")))
    },
    test("end of input exits cleanly") {
      val responder = new RecordingResponder(Vector.empty)
      val terminal = new ScriptedTerminal(Vector.empty)
      new InteractiveLocalChat(responder, terminal).run()
      Assert.equal(terminal.output.last, "bye")
      Assert.equal(terminal.prompts.toVector, Vector("you> "))
    }
  )
