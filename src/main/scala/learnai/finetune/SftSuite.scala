package learnai.finetune

import learnai.optim.AdamW
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object SftSuite extends TestSuite:
  override val name: String = "ChatTemplateSft"

  private val specials = ChatSpecialTokens(
    system = TokenId(10),
    user = TokenId(11),
    assistant = TokenId(12),
    endOfTurn = TokenId(13)
  )

  private def tokens(values: Int*): Vector[TokenId] = values.toVector.map(TokenId(_))

  private def conversation(turns: (ChatRole, Vector[TokenId])*): Conversation =
    Conversation(turns.toVector.map((role, content) => ChatMessage(role, content)))

  override val tests: Vector[TestCase] = specify(
    test("a three-turn conversation renders to the exact template layout") {
      val rendered = ChatTemplate.render(
        conversation(
          ChatRole.System    -> tokens(0),
          ChatRole.User      -> tokens(1, 2),
          ChatRole.Assistant -> tokens(3, 4)
        ),
        specials
      )
      // system: 10 0 13 | user: 11 1 2 13 | assistant: 12 3 4 13
      Assert.equal(rendered.tokens, tokens(10, 0, 13, 11, 1, 2, 13, 12, 3, 4, 13))
      Assert.equal(
        rendered.assistantSpan,
        Vector(false, false, false, false, false, false, false, false, true, true, true)
      )
    },
    test("the training example trains exactly the assistant content plus its ending") {
      val rendered = ChatTemplate.render(
        conversation(ChatRole.User -> tokens(1, 2), ChatRole.Assistant -> tokens(3, 4)),
        specials
      )
      val example  = ChatTemplate.trainingExample(rendered)
      Assert.equal(example.inputs, rendered.tokens.init)
      Assert.equal(example.targets, rendered.tokens.tail)
      // Trainable targets: 3, 4, and the assistant end-of-turn.
      Assert.equal(example.trainableTargetCount, 3)
      example.targets.zip(example.lossMask).foreach { case (target, trained) =>
        if trained then
          Assert.isTrue(
            target == TokenId(3) || target == TokenId(4) || target == specials.endOfTurn,
            s"unexpected trainable target $target"
          )
      }
      // The assistant role marker itself is conditioning, never a target.
      example.targets.zip(example.lossMask).foreach { case (target, trained) =>
        if target == specials.assistant then Assert.isTrue(!trained, "role marker was trained")
      }
    },
    test("every assistant message contributes content plus one trainable targets") {
      val multiTurn = conversation(
        ChatRole.System    -> tokens(0),
        ChatRole.User      -> tokens(1),
        ChatRole.Assistant -> tokens(2, 3, 4),
        ChatRole.User      -> tokens(5, 6),
        ChatRole.Assistant -> tokens(7)
      )
      val example   = ChatTemplate.trainingExample(ChatTemplate.render(multiTurn, specials))
      val expected  = multiTurn.assistantMessages.map(_.content.size + 1).sum
      Assert.equal(example.trainableTargetCount, expected)
      Assert.equal(expected, 6)
    },
    test("malformed conversations are rejected before any rendering") {
      val assistantFirst = Assert
        .throws[IllegalArgumentException](conversation(ChatRole.Assistant -> tokens(1)))
      Assert.isTrue(assistantFirst.getMessage.contains("cannot open"))
      val systemInMiddle = Assert.throws[IllegalArgumentException] {
        conversation(ChatRole.User -> tokens(1), ChatRole.System -> tokens(0))
      }
      Assert.isTrue(systemInMiddle.getMessage.contains("only allowed first"))
      val doubleTurn     = Assert.throws[IllegalArgumentException] {
        conversation(ChatRole.User -> tokens(1), ChatRole.User -> tokens(2))
      }
      Assert.isTrue(doubleTurn.getMessage.contains("share the role"))
      val emptyContent   = Assert
        .throws[IllegalArgumentException](ChatMessage(ChatRole.User, Vector.empty))
      Assert.isTrue(emptyContent.getMessage.contains("at least one content token"))
      val noAssistant    = Assert.throws[IllegalArgumentException] {
        ChatTemplate
          .trainingExample(ChatTemplate.render(conversation(ChatRole.User -> tokens(1)), specials))
      }
      Assert.isTrue(noAssistant.getMessage.contains("trainable target"))
    },
    test("reserved tokens are refused inside message content") {
      val embeddedEnd       = Assert.throws[IllegalArgumentException] {
        ChatTemplate.render(conversation(ChatRole.User -> tokens(1, 13)), specials)
      }
      Assert.isTrue(embeddedEnd.getMessage.contains("reserved special token"))
      val duplicateSpecials = Assert.throws[IllegalArgumentException] {
        ChatSpecialTokens(TokenId(10), TokenId(10), TokenId(12), TokenId(13))
      }
      Assert.isTrue(duplicateSpecials.getMessage.contains("distinct"))
    },
    test("supervised fine-tuning reduces the assistant-span loss of a small model") {
      val config    = MiniGptConfig(
        vocabularySize = 14,
        maximumContextLength = 12,
        channels = 8,
        headCount = 2,
        hiddenChannels = 16,
        layerCount = 1
      )
      val model     = MiniGpt.random(config, seed = 11L)
      val training  = conversation(ChatRole.User -> tokens(1, 2), ChatRole.Assistant -> tokens(3, 4))
      val example   = ChatTemplate.trainingExample(ChatTemplate.render(training, specials))
      val optimizer = new AdamW(learningRate = 0.05, weightDecay = 0.0)

      val initial   = SftEvaluation.exampleLoss(model, example).valueAtFlat(0)
      (0 until 100).foreach { _ =>
        model.parameters.foreach(_.clearGradients())
        SftEvaluation.exampleLoss(model, example).backward()
        val _ = optimizer.step(model.parameters)
      }
      model.parameters.foreach(_.clearGradients())
      val finalLoss = SftEvaluation.exampleLoss(model, example).valueAtFlat(0)
      Assert.isTrue(
        finalLoss < initial / 5.0,
        s"expected at least a 5x loss reduction, got $initial -> $finalLoss"
      )
    },
    test("held-out evaluation is a deterministic per-token weighted mean") {
      val config   = MiniGptConfig(
        vocabularySize = 14,
        maximumContextLength = 12,
        channels = 8,
        headCount = 2,
        hiddenChannels = 16,
        layerCount = 1
      )
      val model    = MiniGpt.random(config, seed = 13L)
      val heldOut  = Vector(
        conversation(ChatRole.User -> tokens(1), ChatRole.Assistant    -> tokens(2, 3)),
        conversation(ChatRole.User -> tokens(4, 5), ChatRole.Assistant -> tokens(6))
      )
      val examples = heldOut
        .map(held => ChatTemplate.trainingExample(ChatTemplate.render(held, specials)))
      val expected = examples.map { example =>
        SftEvaluation.exampleLoss(model, example).valueAtFlat(0) *
          example.trainableTargetCount.toDouble
      }.sum / examples.map(_.trainableTargetCount).sum.toDouble

      val first  = SftEvaluation.heldOutLoss(model, heldOut, specials)
      val second = SftEvaluation.heldOutLoss(model, heldOut, specials)
      Assert.equal(first, second)
      Assert.close(first, expected, tolerance = 1e-15)

      val tooLong  = conversation(
        ChatRole.User      -> tokens(1, 2, 3, 4, 5, 6),
        ChatRole.Assistant -> tokens(7, 0, 1, 2)
      )
      val overflow = Assert.throws[IllegalArgumentException] {
        SftEvaluation.heldOutLoss(model, Vector(tooLong), specials)
      }
      Assert.isTrue(overflow.getMessage.contains("exceeds model context"))
    }
  )
