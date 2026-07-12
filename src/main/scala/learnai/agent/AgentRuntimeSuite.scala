package learnai.agent

import scala.collection.mutable.ArrayBuffer

import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object AgentRuntimeSuite extends TestSuite:
  override val name: String = "AgentRuntime"

  private final class ScriptedModel(decisions: Vector[Either[ModelError, ModelDecision]])
      extends LanguageModel:
    private var index = 0
    val requests: ArrayBuffer[ModelRequest] = ArrayBuffer.empty

    override def complete(request: ModelRequest): Either[ModelError, ModelDecision] =
      requests += request
      require(index < decisions.size, "scripted model ran out of decisions")
      val decision = decisions(index)
      index += 1
      decision

  private final class EchoTool(delayMillis: Long = 0L) extends Tool:
    var invocations = 0
    override val definition: ToolDefinition = ToolDefinition(
      "echo",
      "Returns the provided message.",
      ToolSchema(Vector(ToolField("message", JsonFieldType.StringValue, "Text to echo", required = true)))
    )

    override def execute(
        arguments: JsonObject,
        context: ToolContext
    ): Either[ToolError, learnai.json.JsonValue] =
      invocations += 1
      if delayMillis > 0L then Thread.sleep(delayMillis)
      Right(arguments.get("message").get)

  private final class ControlledTool(
      effect: ToolEffect,
      outcomes: Vector[Either[ToolError, learnai.json.JsonValue]]
  ) extends Tool:
    require(outcomes.nonEmpty, "controlled tool requires at least one outcome")
    var invocations = 0
    val contexts: ArrayBuffer[ToolContext] = ArrayBuffer.empty

    override val definition: ToolDefinition = ToolDefinition(
      "controlled",
      "A deterministic tool used to verify runtime policy.",
      ToolSchema(Vector.empty),
      effect
    )

    override def execute(
        arguments: JsonObject,
        context: ToolContext
    ): Either[ToolError, learnai.json.JsonValue] =
      contexts += context
      val outcome = outcomes(math.min(invocations, outcomes.size - 1))
      invocations += 1
      outcome

  private val userHistory = Vector(TextMessage(MessageRole.User, "hello"))
  private val smallUsage = ModelUsage(10, 2)

  override val tests: Vector[TestCase] = specify(
    test("a final model answer completes without executing tools") {
      val model = new ScriptedModel(Vector(Right(FinalAnswer("done", smallUsage))))
      val run = new AgentRuntime(Vector.empty, AgentConfig()).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.Completed)
      Assert.equal(run.finalAnswer, Some("done"))
      Assert.equal(run.executedToolCalls, 0)
      Assert.equal(run.usage, smallUsage)
      Assert.equal(model.requests.head.tools, Vector.empty)
    },
    test("a valid tool call becomes an observation for the next model step") {
      val tool = new EchoTool()
      val call = ToolCall("call-1", "echo", JsonObject("message" -> JsonString("hi")))
      val model = new ScriptedModel(
        Vector(
          Right(RequestTools(Vector(call), smallUsage)),
          Right(FinalAnswer("observed", ModelUsage(12, 1)))
        )
      )
      val run = new AgentRuntime(Vector(tool), AgentConfig()).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.Completed)
      Assert.equal(tool.invocations, 1)
      Assert.equal(run.executedToolCalls, 1)
      Assert.equal(run.usage, ModelUsage(22, 3))
      val observation = model.requests(1).history.collectFirst { case value: ToolObservation => value }
      Assert.equal(observation, Some(ToolObservation("call-1", "echo", ToolSucceeded(JsonString("hi")))))
    },
    test("schema errors are returned to the model without invoking the tool") {
      val tool = new EchoTool()
      val invalid = ToolCall("bad", "echo", JsonObject("message" -> JsonNumber(3)))
      val model = new ScriptedModel(
        Vector(
          Right(RequestTools(Vector(invalid), smallUsage)),
          Right(FinalAnswer("recovered", smallUsage))
        )
      )
      val run = new AgentRuntime(Vector(tool), AgentConfig()).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.Completed)
      Assert.equal(tool.invocations, 0)
      val failure = run.history.collectFirst {
        case ToolObservation(_, _, ToolFailed(error)) => error
      }
      Assert.isTrue(failure.exists(_.code == "invalid_arguments"))
    },
    test("unknown tools are capability failures visible to the model") {
      val call = ToolCall("unknown", "shell", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("ok", smallUsage)))
      )
      val run = new AgentRuntime(Vector(new EchoTool()), AgentConfig()).run(model, userHistory)
      val error = run.history.collectFirst {
        case ToolObservation(_, _, ToolFailed(problem)) => problem
      }
      Assert.isTrue(error.exists(_.code == "unknown_tool"))
      Assert.isTrue(error.exists(_.message.contains("not granted")))
    },
    test("identical call IDs reuse cached outcomes instead of repeating side effects") {
      val tool = new EchoTool()
      val call = ToolCall("stable-id", "echo", JsonObject("message" -> JsonString("once")))
      val model = new ScriptedModel(
        Vector(
          Right(RequestTools(Vector(call), smallUsage)),
          Right(RequestTools(Vector(call), smallUsage)),
          Right(FinalAnswer("done", smallUsage))
        )
      )
      val run = new AgentRuntime(Vector(tool), AgentConfig()).run(model, userHistory)
      Assert.equal(tool.invocations, 1)
      Assert.equal(run.executedToolCalls, 1)
      Assert.equal(run.events.count(_.isInstanceOf[ToolResultReused]), 1)
      Assert.equal(run.history.count(_.isInstanceOf[ToolObservation]), 2)
    },
    test("reusing a call ID with different arguments is rejected") {
      val tool = new EchoTool()
      val first = ToolCall("same", "echo", JsonObject("message" -> JsonString("one")))
      val second = ToolCall("same", "echo", JsonObject("message" -> JsonString("two")))
      val model = new ScriptedModel(
        Vector(
          Right(RequestTools(Vector(first), smallUsage)),
          Right(RequestTools(Vector(second), smallUsage)),
          Right(FinalAnswer("done", smallUsage))
        )
      )
      val run = new AgentRuntime(Vector(tool), AgentConfig()).run(model, userHistory)
      Assert.equal(tool.invocations, 1)
      Assert.isTrue(run.history.exists {
        case ToolObservation(_, _, ToolFailed(error)) => error.code == "conflicting_call_id"
        case _                                        => false
      })
    },
    test("tool timeout is a retryable observation and interrupts the task") {
      val tool = new EchoTool(delayMillis = 200L)
      val call = ToolCall("slow", "echo", JsonObject("message" -> JsonString("late")))
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("fallback", smallUsage)))
      )
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(toolTimeoutMillis = 10L)
      ).run(model, userHistory)
      val timeout = run.history.collectFirst {
        case ToolObservation(_, _, ToolFailed(error)) if error.code == "tool_timeout" => error
      }
      Assert.isTrue(timeout.exists(_.retryable))
      Assert.equal(run.status, AgentStatus.Completed)
    },
    test("effectful tools are denied unless the host grants approval") {
      val tool = new ControlledTool(
        ToolEffect.IdempotentWrite,
        Vector(Right(JsonString("written")))
      )
      val call = ToolCall("write-1", "controlled", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("not written", smallUsage)))
      )
      val run = new AgentRuntime(Vector(tool), AgentConfig()).run(model, userHistory)
      val error = run.history.collectFirst {
        case ToolObservation(_, _, ToolFailed(problem)) => problem
      }
      Assert.equal(tool.invocations, 0)
      Assert.isTrue(error.exists(_.code == "approval_denied"))
      val authorization = run.events.collectFirst { case event: ToolAuthorizationChecked => event }
      Assert.isTrue(authorization.exists(event => !event.approved))
    },
    test("an approved idempotent write executes with an auditable decision") {
      val tool = new ControlledTool(
        ToolEffect.IdempotentWrite,
        Vector(Right(JsonString("written")))
      )
      val call = ToolCall("write-2", "controlled", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("done", smallUsage)))
      )
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(),
        ToolApprover.allowAll
      ).run(model, userHistory)
      Assert.equal(tool.invocations, 1)
      Assert.isTrue(run.events.exists {
        case event: ToolAuthorizationChecked => event.approved
        case _                               => false
      })
    },
    test("retryable read-only failures use distinct bounded attempts") {
      val tool = new ControlledTool(
        ToolEffect.ReadOnly,
        Vector(
          Left(ToolError("temporary", "try again", retryable = true)),
          Right(JsonString("recovered"))
        )
      )
      val call = ToolCall("retry-1", "controlled", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("done", smallUsage)))
      )
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(maximumToolAttempts = 2)
      ).run(model, userHistory)
      Assert.equal(tool.invocations, 2)
      Assert.equal(tool.contexts.map(_.attempt).toVector, Vector(1, 2))
      Assert.equal(run.executedToolCalls, 1)
      Assert.equal(run.events.count(_.isInstanceOf[ToolRetryScheduled]), 1)
      Assert.isTrue(run.history.exists {
        case ToolObservation(_, _, ToolSucceeded(JsonString("recovered"))) => true
        case _                                                             => false
      })
    },
    test("non-idempotent writes are never retried automatically") {
      val tool = new ControlledTool(
        ToolEffect.NonIdempotentWrite,
        Vector(
          Left(ToolError("uncertain_commit", "response lost", retryable = true)),
          Right(JsonString("would duplicate"))
        )
      )
      val call = ToolCall("charge-1", "controlled", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("stopped", smallUsage)))
      )
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(maximumToolAttempts = 3),
        ToolApprover.allowAll
      ).run(model, userHistory)
      Assert.equal(tool.invocations, 1)
      Assert.equal(run.events.count(_.isInstanceOf[ToolRetrySuppressed]), 1)
    },
    test("an approver exception fails closed without invoking the tool") {
      val tool = new ControlledTool(
        ToolEffect.IdempotentWrite,
        Vector(Right(JsonString("written")))
      )
      val failingApprover = new ToolApprover:
        override def decide(
            call: ToolCall,
            definition: ToolDefinition,
            context: ToolContext
        ): ApprovalDecision = throw new IllegalStateException("approval service unavailable")
      val call = ToolCall("write-3", "controlled", JsonObject.empty)
      val model = new ScriptedModel(
        Vector(Right(RequestTools(Vector(call), smallUsage)), Right(FinalAnswer("safe", smallUsage)))
      )
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(),
        failingApprover
      ).run(model, userHistory)
      Assert.equal(tool.invocations, 0)
      Assert.isTrue(run.history.exists {
        case ToolObservation(_, _, ToolFailed(error)) =>
          error.code == "approval_denied" && error.message.contains("approver failed")
        case _ => false
      })
    },
    test("tool-call budget stops before an unapproved extra execution") {
      val tool = new EchoTool()
      val calls = Vector(
        ToolCall("one", "echo", JsonObject("message" -> JsonString("1"))),
        ToolCall("two", "echo", JsonObject("message" -> JsonString("2")))
      )
      val model = new ScriptedModel(Vector(Right(RequestTools(calls, smallUsage))))
      val run = new AgentRuntime(
        Vector(tool),
        AgentConfig(maximumToolCalls = 1)
      ).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.ToolCallLimitExceeded)
      Assert.equal(tool.invocations, 1)
      Assert.equal(run.executedToolCalls, 1)
    },
    test("model-step limit terminates a loop with a complete trace") {
      val call = ToolCall("one", "missing", JsonObject.empty)
      val model = new ScriptedModel(Vector(Right(RequestTools(Vector(call), smallUsage))))
      val run = new AgentRuntime(
        Vector.empty,
        AgentConfig(maximumModelSteps = 1)
      ).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.ModelStepLimitExceeded)
      Assert.equal(run.events.last.asInstanceOf[AgentStopped].status, AgentStatus.ModelStepLimitExceeded)
    },
    test("model errors terminate without being converted into tool observations") {
      val modelError = ModelError("provider_unavailable", "try later", retryable = true)
      val model = new ScriptedModel(Vector(Left(modelError)))
      val run = new AgentRuntime(Vector.empty, AgentConfig()).run(model, userHistory)
      Assert.equal(run.status, AgentStatus.ModelFailed)
      Assert.isTrue(run.events.last.asInstanceOf[AgentStopped].reason.contains("provider_unavailable"))
      Assert.isTrue(!run.history.exists(_.isInstanceOf[ToolObservation]))
    },
    test("schema validation reports required unknown and integer type failures together") {
      val schema = ToolSchema(
        Vector(
          ToolField("name", JsonFieldType.StringValue, "name", required = true),
          ToolField("count", JsonFieldType.IntegerValue, "count", required = true)
        )
      )
      val problems = schema.validate(
        JsonObject(
          "count" -> JsonNumber(BigDecimal("1.5")),
          "extra" -> JsonString("no")
        )
      )
      Assert.equal(problems.size, 3)
      Assert.isTrue(problems.exists(_.contains("missing required field 'name'")))
      Assert.isTrue(problems.exists(_.contains("field 'count'")))
      Assert.isTrue(problems.exists(_.contains("unknown field 'extra'")))
    }
  )
