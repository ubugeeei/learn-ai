package learnai.finetune

import learnai.tensor.Tensor
import learnai.text.TokenId
import learnai.transformer.MiniGpt

/** Who produced one message of a conversation. */
enum ChatRole:
  case System, User, Assistant

/** One conversation turn, already tokenized.
  *
  * Contents are token IDs rather than strings on purpose: tokenization is
  * Chapter 15's contract, and keeping the template token-level makes every
  * mask position exactly checkable in tests.
  */
final case class ChatMessage(role: ChatRole, content: Vector[TokenId]):
  require(content.nonEmpty, "a chat message requires at least one content token")

/** A structurally valid conversation.
  *
  * The shape rules are the ones most chat stacks enforce before training:
  * an optional system message first, no assistant opening, and no two
  * consecutive messages from the same role. Rejecting malformed data here —
  * before rendering — is the "data validation" half of SFT; silently
  * training on a double-assistant turn corrupts the loss mask without any
  * runtime error later.
  */
final case class Conversation(messages: Vector[ChatMessage]):
  require(messages.nonEmpty, "a conversation requires at least one message")
  require(
    messages.head.role != ChatRole.Assistant,
    "a conversation cannot open with an assistant message"
  )
  messages.zipWithIndex.foreach { case (message, index) =>
    require(
      message.role != ChatRole.System || index == 0,
      s"system message at position $index; system is only allowed first"
    )
  }
  messages.iterator.zip(messages.iterator.drop(1)).zipWithIndex.foreach {
    case ((previous, current), index) =>
      require(
        previous.role != current.role,
        s"messages ${index} and ${index + 1} share the role ${current.role}"
      )
  }

  def assistantMessages: Vector[ChatMessage] =
    messages.filter(_.role == ChatRole.Assistant)

/** The four reserved tokens a chat template needs. */
final case class ChatSpecialTokens(
    system: TokenId,
    user: TokenId,
    assistant: TokenId,
    endOfTurn: TokenId
):
  val all: Vector[TokenId] = Vector(system, user, assistant, endOfTurn)
  require(all.distinct.size == all.size, "chat special tokens must be distinct")

  def roleMarker(role: ChatRole): TokenId = role match
    case ChatRole.System    => system
    case ChatRole.User      => user
    case ChatRole.Assistant => assistant

/** A rendered conversation with its assistant-span annotation.
  *
  * `assistantSpan(i)` is true when token `i` is assistant *content* or the
  * end-of-turn of an assistant message. Role markers are never in the span:
  * the model is conditioned on "the assistant speaks next", not trained to
  * decide it.
  */
final case class RenderedConversation(
    tokens: Vector[TokenId],
    assistantSpan: Vector[Boolean]
):
  require(tokens.size == assistantSpan.size, "annotation must cover every token")
  require(tokens.size >= 2, "a rendered conversation needs at least two tokens")

/** One supervised fine-tuning example aligned for causal training. */
final case class SftExample(
    inputs: Vector[TokenId],
    targets: Vector[TokenId],
    lossMask: Vector[Boolean]
):
  require(
    inputs.size == targets.size && inputs.size == lossMask.size,
    s"input/target/mask lengths differ: ${inputs.size}/${targets.size}/${lossMask.size}"
  )
  require(lossMask.exists(identity), "an SFT example requires at least one trainable target")

  def trainableTargetCount: Int = lossMask.count(identity)

object ChatTemplate:
  /** Renders `roleMarker content... endOfTurn` per message.
    *
    * Contents must not contain any special token; a document that embeds a
    * fake `endOfTurn` would silently terminate the mask early, which is a
    * prompt-injection shaped data bug this validation refuses.
    */
  def render(
      conversation: Conversation,
      specials: ChatSpecialTokens
  ): RenderedConversation =
    conversation.messages.zipWithIndex.foreach { case (message, index) =>
      require(
        !message.content.exists(specials.all.contains),
        s"message $index contains a reserved special token"
      )
    }
    val parts = conversation.messages.map { message =>
      val isAssistant = message.role == ChatRole.Assistant
      val tokens = specials.roleMarker(message.role) +: message.content :+ specials.endOfTurn
      val span = false +: Vector.fill(message.content.size)(isAssistant) :+ isAssistant
      (tokens, span)
    }
    RenderedConversation(parts.flatMap(_._1), parts.flatMap(_._2))

  /** Aligns a rendered conversation into a causal example that trains only
    * on assistant spans.
    *
    * Position `t` predicts token `t + 1`, so the loss mask at `t` is the
    * assistant-span flag of the *target*. Every assistant message therefore
    * contributes `content.size + 1` trainable predictions (its content plus
    * its end-of-turn), which the test suite asserts as a counting
    * invariant. Conversations without an assistant message cannot form a
    * training example.
    */
  def trainingExample(rendered: RenderedConversation): SftExample =
    SftExample(
      inputs = rendered.tokens.init,
      targets = rendered.tokens.tail,
      lossMask = rendered.assistantSpan.tail
    )

object SftEvaluation:
  /** Token-weighted masked loss over a held-out conversation set.
    *
    * Each conversation contributes its masked mean loss weighted by its
    * trainable-target count, so the result is the mean loss per assistant
    * token — comparable across sets whose conversations differ in length.
    * Evaluation is deterministic: no sampling is involved anywhere.
    */
  def heldOutLoss(
      model: MiniGpt,
      conversations: Vector[Conversation],
      specials: ChatSpecialTokens
  ): Double =
    require(conversations.nonEmpty, "held-out evaluation requires at least one conversation")
    var weightedLoss = 0.0
    var trainableTargets = 0L
    conversations.foreach { conversation =>
      val example = ChatTemplate.trainingExample(ChatTemplate.render(conversation, specials))
      require(
        example.inputs.size <= model.config.maximumContextLength,
        s"conversation of ${example.inputs.size} positions exceeds model context " +
          s"${model.config.maximumContextLength}"
      )
      val loss = exampleLoss(model, example).valueAtFlat(0)
      weightedLoss += loss * example.trainableTargetCount.toDouble
      trainableTargets += example.trainableTargetCount.toLong
    }
    weightedLoss / trainableTargets.toDouble

  /** Builds one masked loss graph for training or evaluation. */
  def exampleLoss(model: MiniGpt, example: SftExample): Tensor =
    model.lossMasked(example.inputs, example.targets, example.lossMask)
