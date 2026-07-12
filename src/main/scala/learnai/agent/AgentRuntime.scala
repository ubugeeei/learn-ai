package learnai.agent

import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/** Hard limits that guarantee an agent run reaches a terminal state. */
final case class AgentConfig(
    maximumModelSteps: Int = 16,
    maximumToolCalls: Int = 32,
    toolTimeoutMillis: Long = 10_000L,
    maximumToolAttempts: Int = 1
):
  require(maximumModelSteps > 0, "maximum model steps must be positive")
  require(maximumToolCalls >= 0, "maximum tool calls must be non-negative")
  require(toolTimeoutMillis > 0L, "tool timeout must be positive")
  require(maximumToolAttempts > 0, "maximum tool attempts must be positive")

enum AgentStatus:
  case Completed
  case ModelFailed
  case ModelStepLimitExceeded
  case ToolCallLimitExceeded

sealed trait AgentEvent:
  def step: Int

final case class ModelInvoked(step: Int, historyItems: Int)                    extends AgentEvent
final case class ModelReturned(step: Int, decision: String, usage: ModelUsage) extends AgentEvent
final case class ToolStarted(step: Int, callId: String, toolName: String)      extends AgentEvent
final case class ToolFinished(step: Int, observation: ToolObservation)         extends AgentEvent
final case class ToolResultReused(step: Int, observation: ToolObservation)     extends AgentEvent
final case class ToolAuthorizationChecked(
    step: Int,
    callId: String,
    toolName: String,
    effect: ToolEffect,
    approved: Boolean,
    reason: String
) extends AgentEvent
final case class ToolAttemptStarted(step: Int, callId: String, toolName: String, attempt: Int)
    extends AgentEvent
final case class ToolAttemptFinished(
    step: Int,
    callId: String,
    toolName: String,
    attempt: Int,
    outcome: ToolOutcome
) extends AgentEvent
final case class ToolRetryScheduled(step: Int, callId: String, toolName: String, nextAttempt: Int)
    extends AgentEvent
final case class ToolRetrySuppressed(step: Int, callId: String, toolName: String, reason: String)
    extends AgentEvent
final case class AgentStopped(step: Int, status: AgentStatus, reason: String)  extends AgentEvent

/** Complete immutable trace returned for success and every failure mode. */
final case class AgentRun(
    status: AgentStatus,
    finalAnswer: Option[String],
    history: Vector[ConversationItem],
    events: Vector[AgentEvent],
    usage: ModelUsage,
    executedToolCalls: Int
)

/** Executes a bounded model/tool loop over explicitly granted capabilities. */
final class AgentRuntime(
    tools: Vector[Tool],
    config: AgentConfig,
    approver: ToolApprover = ToolApprover.denyEffectful
):
  private val toolNames   = tools.map(_.definition.name)
  require(toolNames.forall(_.nonEmpty), "tool names cannot be empty")
  require(toolNames.distinct.size == toolNames.size, s"tool names must be unique: $toolNames")
  private val toolsByName = tools.map(tool => tool.definition.name -> tool).toMap

  /** Runs until final answer, model failure, or a configured hard limit. */
  def run(model: LanguageModel, initialHistory: Vector[ConversationItem]): AgentRun =
    var history           = initialHistory
    var events            = Vector.empty[AgentEvent]
    var usage             = ModelUsage.zero
    var executedToolCalls = 0
    var step              = 0
    var cached            = Map.empty[String, (ToolCall, ToolObservation)]

    def stop(status: AgentStatus, reason: String, answer: Option[String] = None): AgentRun =
      events :+= AgentStopped(step, status, reason)
      AgentRun(status, answer, history, events, usage, executedToolCalls)

    while step < config.maximumModelSteps do
      events :+= ModelInvoked(step, history.size)
      val request = ModelRequest(history, tools.map(_.definition), step)
      model.complete(request) match
        case Left(error)     =>
          return stop(AgentStatus.ModelFailed, s"model error ${error.code}: ${error.message}")
        case Right(decision) =>
          usage = usage + decision.usage
          decision match
            case FinalAnswer(text, decisionUsage) =>
              events :+= ModelReturned(step, "final_answer", decisionUsage)
              history :+= TextMessage(MessageRole.Assistant, text)
              return stop(AgentStatus.Completed, "model returned a final answer", Some(text))

            case RequestTools(calls, decisionUsage) =>
              events :+= ModelReturned(step, "tool_calls", decisionUsage)
              history :+= AssistantToolCalls(calls)
              var callIndex = 0
              while callIndex < calls.size do
                val call = calls(callIndex)
                cached.get(call.id) match
                  case Some((original, observation)) if original == call =>
                    history :+= observation
                    events :+= ToolResultReused(step, observation)
                  case Some(_)                                           =>
                    val observation = ToolObservation(
                      call.id,
                      call.name,
                      ToolFailed(ToolError(
                        "conflicting_call_id",
                        s"call ID '${call.id}' was reused with different contents",
                        retryable = false
                      ))
                    )
                    history :+= observation
                    events :+= ToolFinished(step, observation)
                  case None                                              =>
                    if executedToolCalls >= config.maximumToolCalls then
                      return stop(
                        AgentStatus.ToolCallLimitExceeded,
                        s"tool call limit ${config.maximumToolCalls} reached"
                      )
                    executedToolCalls += 1
                    events :+= ToolStarted(step, call.id, call.name)
                    val execution   = executeCall(call, step)
                    events ++= execution.events
                    val observation = execution.observation
                    events :+= ToolFinished(step, observation)
                    history :+= observation
                    cached += call.id -> (call -> observation)
                callIndex += 1
      step += 1

    stop(
      AgentStatus.ModelStepLimitExceeded,
      s"model step limit ${config.maximumModelSteps} reached"
    )

  private def executeCall(call: ToolCall, step: Int): CallExecution = toolsByName
    .get(call.name) match
    case None       => CallExecution(
        ToolObservation(
          call.id,
          call.name,
          ToolFailed(ToolError("unknown_tool", s"tool '${call.name}' is not granted", false))
        ),
        Vector.empty
      )
    case Some(tool) =>
      val validationProblems = tool.definition.schema.validate(call.arguments)
      if validationProblems.nonEmpty then
        CallExecution(
          ToolObservation(
            call.id,
            call.name,
            ToolFailed(ToolError("invalid_arguments", validationProblems.mkString("; "), false))
          ),
          Vector.empty
        )
      else
        val authorization      = authorize(tool, call, step)
        val authorizationEvent = ToolAuthorizationChecked(
          step,
          call.id,
          call.name,
          tool.definition.effect,
          authorization.isInstanceOf[ApprovalGranted],
          authorization.reason
        )
        authorization match
          case ApprovalDenied(reason) => CallExecution(
              ToolObservation(
                call.id,
                call.name,
                ToolFailed(ToolError("approval_denied", reason, retryable = false))
              ),
              Vector(authorizationEvent)
            )
          case _: ApprovalGranted     =>
            val (outcome, attemptEvents) = invokeWithRetries(tool, call, step)
            CallExecution(
              ToolObservation(call.id, call.name, outcome),
              authorizationEvent +: attemptEvents
            )

  private def authorize(tool: Tool, call: ToolCall, step: Int): ApprovalDecision =
    if !tool.definition.effect.requiresApproval then
      ApprovalGranted(s"${tool.definition.effect} tools are allowed without interactive approval")
    else
      try approver.decide(call, tool.definition, ToolContext(call.id, step))
      catch
        case error: Throwable => ApprovalDenied(s"approver failed with ${error.getClass
              .getSimpleName}: ${Option(error.getMessage).getOrElse("")}".trim)

  private def invokeWithRetries(
      tool: Tool,
      call: ToolCall,
      step: Int
  ): (ToolOutcome, Vector[AgentEvent]) =
    val attemptEvents        = Vector.newBuilder[AgentEvent]
    var attempt              = 1
    var outcome: ToolOutcome =
      ToolFailed(ToolError("not_attempted", "tool was not attempted", false))
    var finished             = false
    while !finished do
      attemptEvents += ToolAttemptStarted(step, call.id, call.name, attempt)
      outcome = invokeWithTimeout(tool, call, step, attempt)
      attemptEvents += ToolAttemptFinished(step, call.id, call.name, attempt, outcome)
      outcome match
        case ToolFailed(error)
            if error.retryable && attempt < config.maximumToolAttempts &&
              tool.definition.effect.safeToRetry =>
          attempt += 1
          attemptEvents += ToolRetryScheduled(step, call.id, call.name, attempt)
        case ToolFailed(error)
            if error.retryable && attempt < config.maximumToolAttempts &&
              !tool.definition.effect.safeToRetry =>
          attemptEvents += ToolRetrySuppressed(
            step,
            call.id,
            call.name,
            s"${tool.definition.effect} is not safe to retry automatically"
          )
          finished = true
        case _ => finished = true
    outcome -> attemptEvents.result()

  private def invokeWithTimeout(tool: Tool, call: ToolCall, step: Int, attempt: Int): ToolOutcome =
    val executor      = Executors.newVirtualThreadPerTaskExecutor()
    val requestedCall = call
    try
      val future = executor.submit(
        new Callable[Either[ToolError, learnai.json.JsonValue]]:
          override def call(): Either[ToolError, learnai.json.JsonValue] = tool
            .execute(requestedCall.arguments, ToolContext(requestedCall.id, step, attempt))
      )
      try
        future.get(config.toolTimeoutMillis, TimeUnit.MILLISECONDS) match
          case Right(value) => ToolSucceeded(value)
          case Left(error)  => ToolFailed(error)
      catch
        case _: TimeoutException       =>
          future.cancel(true)
          ToolFailed(ToolError(
            "tool_timeout",
            s"tool '${call.name}' exceeded ${config.toolTimeoutMillis} ms",
            retryable = true
          ))
        case error: ExecutionException =>
          val cause = Option(error.getCause).getOrElse(error)
          ToolFailed(ToolError(
            "tool_exception",
            s"${cause.getClass.getSimpleName}: ${cause.getMessage}",
            false
          ))
    finally
      val _ = executor.shutdownNow()

final private case class CallExecution(observation: ToolObservation, events: Vector[AgentEvent])
