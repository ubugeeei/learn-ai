package learnai.agent

import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object EvaluationSuite extends TestSuite:
  override val name: String = "AgentEvaluation"

  private final class ScriptedModel(decisions: Vector[Either[ModelError, ModelDecision]])
      extends LanguageModel:
    private var index = 0
    override def complete(request: ModelRequest): Either[ModelError, ModelDecision] =
      require(index < decisions.size, "scripted evaluation model ran out of decisions")
      val result = decisions(index)
      index += 1
      result

  private final class RecordingTool(effect: ToolEffect = ToolEffect.ReadOnly) extends Tool:
    var invocations = 0
    override val definition: ToolDefinition = ToolDefinition(
      "record",
      "Records one invocation.",
      ToolSchema(Vector.empty),
      effect
    )
    override def execute(
        arguments: JsonObject,
        context: ToolContext
    ): Either[ToolError, learnai.json.JsonValue] =
      invocations += 1
      Right(JsonString("recorded"))

  private val initialHistory = Vector(TextMessage(MessageRole.User, "evaluate"))

  override val tests: Vector[TestCase] = specify(
    test("a report aggregates deterministic outcome trajectory and cost checks") {
      val tool = new RecordingTool()
      val usage = ModelUsage(10, 2)
      val call = ToolCall("record-1", "record", JsonObject.empty)
      val evalCase = AgentEvalCase(
        "grounded answer",
        () => new ScriptedModel(
          Vector(
            Right(RequestTools(Vector(call), usage)),
            Right(FinalAnswer("done", usage))
          )
        ),
        initialHistory,
        Vector(
          ExpectedStatusCheck(AgentStatus.Completed),
          ExactAnswerCheck("done"),
          RequiredToolAttemptsCheck(Set("record")),
          MaximumTokensCheck(24),
          MaximumToolCallsCheck(1),
          MaximumModelStepsCheck(2)
        )
      )
      val pricing = TokenPricing(BigDecimal(2), BigDecimal(4))
      val report = new AgentEvaluator(
        new AgentRuntime(Vector(tool), AgentConfig()),
        pricing
      ).evaluate(Vector(evalCase))
      Assert.equal(report.passedCases, 1)
      Assert.close(report.passRate, 1.0)
      Assert.equal(report.totalUsage, ModelUsage(20, 4))
      Assert.equal(report.totalLogicalToolCalls, 1)
      Assert.equal(report.totalModelSteps, 2)
      Assert.equal(report.totalEstimatedCostUsd, BigDecimal("0.000056"))
      Assert.equal(tool.invocations, 1)
      Assert.isTrue(report.failedChecksByName.isEmpty)
    },
    test("failed checks remain visible by case and category") {
      val evalCase = AgentEvalCase(
        "intentional failure",
        () => new ScriptedModel(Vector(Right(FinalAnswer("actual", ModelUsage(2, 1))))),
        initialHistory,
        Vector(
          ExactAnswerCheck("expected"),
          MaximumTokensCheck(1)
        )
      )
      val report = new AgentEvaluator(
        new AgentRuntime(Vector.empty, AgentConfig())
      ).evaluate(Vector(evalCase))
      Assert.equal(report.passedCases, 0)
      Assert.close(report.passRate, 0.0)
      Assert.equal(report.failedChecksByName, Map("exact_answer" -> 1, "maximum_tokens" -> 1))
      Assert.isTrue(report.results.head.checks.forall(check => !check.passed))
    },
    test("requested and physically attempted forbidden tools are distinct checks") {
      val tool = new RecordingTool(ToolEffect.NonIdempotentWrite)
      val call = ToolCall("write-1", "record", JsonObject.empty)
      val evalCase = AgentEvalCase(
        "denied write",
        () => new ScriptedModel(
          Vector(
            Right(RequestTools(Vector(call), ModelUsage.zero)),
            Right(FinalAnswer("denied", ModelUsage.zero))
          )
        ),
        initialHistory,
        Vector(
          ForbiddenToolAttemptsCheck(Set("record")),
          ForbiddenToolRequestsCheck(Set("record")),
          RequiredToolErrorCheck("approval_denied")
        )
      )
      val result = new AgentEvaluator(
        new AgentRuntime(Vector(tool), AgentConfig())
      ).evaluate(Vector(evalCase)).results.head
      val checks = result.checks.map(check => check.name -> check.passed).toMap
      Assert.equal(tool.invocations, 0)
      Assert.equal(checks("forbidden_tool_attempts"), true)
      Assert.equal(checks("forbidden_tool_requests"), false)
      Assert.equal(checks("required_tool_error"), true)
    },
    test("multiple cases retain order and create independent model instances") {
      var modelsCreated = 0
      def evalCase(name: String): AgentEvalCase = AgentEvalCase(
        name,
        () =>
          modelsCreated += 1
          new ScriptedModel(Vector(Right(FinalAnswer(name, ModelUsage.zero)))),
        Vector.empty,
        Vector(ExactAnswerCheck(name))
      )
      val report = new AgentEvaluator(
        new AgentRuntime(Vector.empty, AgentConfig())
      ).evaluate(Vector(evalCase("first"), evalCase("second")))
      Assert.equal(report.results.map(_.name), Vector("first", "second"))
      Assert.equal(modelsCreated, 2)
      Assert.equal(report.passedCases, 2)
    },
    test("duplicate case and check names fail before model execution") {
      val factory = () => new ScriptedModel(Vector(Right(FinalAnswer("ok", ModelUsage.zero))))
      val duplicateChecks = Assert.throws[IllegalArgumentException] {
        AgentEvalCase(
          "bad checks",
          factory,
          Vector.empty,
          Vector(MaximumTokensCheck(1), MaximumTokensCheck(2))
        )
      }
      val valid = AgentEvalCase(
        "same",
        factory,
        Vector.empty,
        Vector(ExactAnswerCheck("ok"))
      )
      val duplicateCases = Assert.throws[IllegalArgumentException] {
        new AgentEvaluator(new AgentRuntime(Vector.empty, AgentConfig())).evaluate(Vector(valid, valid))
      }
      Assert.isTrue(duplicateChecks.getMessage.contains("unique"))
      Assert.isTrue(duplicateCases.getMessage.contains("unique"))
    }
  )
