package learnai.agent

import learnai.json.JsonArray
import learnai.json.JsonBoolean
import learnai.json.JsonNull
import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.json.JsonValue

enum MessageRole:
  case System
  case User
  case Assistant

/** One typed tool invocation proposed by a language model. */
final case class ToolCall(id: String, name: String, arguments: JsonObject):
  require(id.nonEmpty, "tool call ID cannot be empty")
  require(name.nonEmpty, "tool name cannot be empty")

sealed trait ConversationItem
final case class TextMessage(role: MessageRole, content: String) extends ConversationItem
final case class AssistantToolCalls(calls: Vector[ToolCall]) extends ConversationItem:
  require(calls.nonEmpty, "an assistant tool-call item cannot be empty")

/** A tool failure safe to expose to the model as an observation. */
final case class ToolError(code: String, message: String, retryable: Boolean):
  require(code.nonEmpty, "tool error code cannot be empty")

sealed trait ToolOutcome
final case class ToolSucceeded(value: JsonValue) extends ToolOutcome
final case class ToolFailed(error: ToolError) extends ToolOutcome

/** The result paired to a prior call ID and tool name. */
final case class ToolObservation(
    callId: String,
    toolName: String,
    outcome: ToolOutcome
) extends ConversationItem

enum JsonFieldType:
  case StringValue
  case NumberValue
  case IntegerValue
  case BooleanValue
  case ObjectValue
  case ArrayValue
  case NullValue

/** One top-level tool argument contract. */
final case class ToolField(
    name: String,
    fieldType: JsonFieldType,
    description: String,
    required: Boolean
):
  require(name.nonEmpty, "tool field name cannot be empty")

/** A deliberately small JSON object schema used for tool input validation. */
final case class ToolSchema(
    fields: Vector[ToolField],
    allowAdditionalFields: Boolean = false
):
  require(fields.map(_.name).distinct.size == fields.size, "tool schema field names must be unique")

  /** Returns every validation problem; an empty vector means valid. */
  def validate(arguments: JsonObject): Vector[String] =
    val problems = Vector.newBuilder[String]
    val defined = fields.map(field => field.name -> field).toMap
    fields.filter(_.required).foreach { field =>
      if arguments.get(field.name).isEmpty then problems += s"missing required field '${field.name}'"
    }
    arguments.fields.foreach { case (name, value) =>
      defined.get(name) match
        case None if !allowAdditionalFields => problems += s"unknown field '$name'"
        case None => ()
        case Some(field) if !matches(value, field.fieldType) =>
          problems += s"field '$name' must be ${field.fieldType}, got ${value.getClass.getSimpleName}"
        case Some(_) => ()
    }
    problems.result()

  private def matches(value: JsonValue, expected: JsonFieldType): Boolean =
    (value, expected) match
      case (_: JsonString, JsonFieldType.StringValue)   => true
      case (_: JsonNumber, JsonFieldType.NumberValue)   => true
      case (JsonNumber(number), JsonFieldType.IntegerValue) => number.isWhole
      case (_: JsonBoolean, JsonFieldType.BooleanValue) => true
      case (_: JsonObject, JsonFieldType.ObjectValue)   => true
      case (_: JsonArray, JsonFieldType.ArrayValue)     => true
      case (JsonNull, JsonFieldType.NullValue)          => true
      case _                                            => false

/** Operational effect used to decide approval and retry behavior.
  *
  * Read-only and idempotent operations may be retried after an explicitly
  * retryable failure. Non-idempotent writes are never retried automatically
  * because the first attempt may have committed before its response failed.
  */
enum ToolEffect(val requiresApproval: Boolean, val safeToRetry: Boolean):
  case ReadOnly extends ToolEffect(requiresApproval = false, safeToRetry = true)
  case IdempotentWrite extends ToolEffect(requiresApproval = true, safeToRetry = true)
  case NonIdempotentWrite extends ToolEffect(requiresApproval = true, safeToRetry = false)

/** Model-visible metadata plus host-only operational policy for one capability. */
final case class ToolDefinition(
    name: String,
    description: String,
    schema: ToolSchema,
    effect: ToolEffect = ToolEffect.ReadOnly
):
  require(name.nonEmpty, "tool definition name cannot be empty")

sealed trait ApprovalDecision:
  def reason: String

final case class ApprovalGranted(reason: String) extends ApprovalDecision
final case class ApprovalDenied(reason: String) extends ApprovalDecision

/** Host-controlled authorization boundary for effectful tool calls. */
trait ToolApprover:
  def decide(call: ToolCall, definition: ToolDefinition, context: ToolContext): ApprovalDecision

object ToolApprover:
  /** Secure default: effectful tools require an approver supplied by the host. */
  val denyEffectful: ToolApprover = new ToolApprover:
    override def decide(
        call: ToolCall,
        definition: ToolDefinition,
        context: ToolContext
    ): ApprovalDecision =
      ApprovalDenied(s"${definition.effect} tool '${definition.name}' has no host approval")

  /** Explicit opt-in useful for controlled tests and trusted local workflows. */
  val allowAll: ToolApprover = new ToolApprover:
    override def decide(
        call: ToolCall,
        definition: ToolDefinition,
        context: ToolContext
    ): ApprovalDecision =
      ApprovalGranted(s"host approved ${definition.effect} tool '${definition.name}'")

/** Runtime context passed to a tool but not controlled by model arguments. */
final case class ToolContext(callId: String, agentStep: Int, attempt: Int = 1):
  require(attempt > 0, s"tool attempt must be positive: $attempt")

/** A capability that the host explicitly grants to an agent run. */
trait Tool:
  def definition: ToolDefinition
  def execute(arguments: JsonObject, context: ToolContext): Either[ToolError, JsonValue]

final case class ModelUsage(inputTokens: Long, outputTokens: Long):
  require(inputTokens >= 0L && outputTokens >= 0L, "token usage cannot be negative")
  def +(other: ModelUsage): ModelUsage =
    ModelUsage(inputTokens + other.inputTokens, outputTokens + other.outputTokens)
  def totalTokens: Long = inputTokens + outputTokens

object ModelUsage:
  val zero: ModelUsage = ModelUsage(0L, 0L)

final case class ModelRequest(
    history: Vector[ConversationItem],
    tools: Vector[ToolDefinition],
    step: Int
)

final case class ModelError(code: String, message: String, retryable: Boolean)

sealed trait ModelDecision:
  def usage: ModelUsage

final case class FinalAnswer(text: String, usage: ModelUsage) extends ModelDecision
final case class RequestTools(calls: Vector[ToolCall], usage: ModelUsage) extends ModelDecision:
  require(calls.nonEmpty, "a tool request must contain at least one call")

/** Provider-neutral boundary; adapters translate this protocol to remote APIs. */
trait LanguageModel:
  def complete(request: ModelRequest): Either[ModelError, ModelDecision]
