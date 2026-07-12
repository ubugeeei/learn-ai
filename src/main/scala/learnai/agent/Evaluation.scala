package learnai.agent

import learnai.json.JsonObject
import learnai.json.JsonString

/** One named, independently explainable assertion over a complete agent run. */
trait AgentRunCheck:
  def name: String
  def evaluate(run: AgentRun): EvalCheckResult

final case class EvalCheckResult(name: String, passed: Boolean, detail: String):
  require(name.nonEmpty, "evaluation check name cannot be empty")

/** Requires the agent to reach one specific terminal state. */
final case class ExpectedStatusCheck(expected: AgentStatus) extends AgentRunCheck:
  override val name: String                             = "expected_status"
  override def evaluate(run: AgentRun): EvalCheckResult =
    EvalCheckResult(name, run.status == expected, s"expected=$expected, actual=${run.status}")

/**
 * Requires byte-for-byte final-answer equality.
 *
 * Exact match is appropriate for deterministic protocol outputs. Semantic or model-graded
 * comparisons should be separate checks with explicit versions and known uncertainty.
 */
final case class ExactAnswerCheck(expected: String) extends AgentRunCheck:
  override val name: String                             = "exact_answer"
  override def evaluate(run: AgentRun): EvalCheckResult = EvalCheckResult(
    name,
    run.finalAnswer.contains(expected),
    s"expected='$expected', actual=${run.finalAnswer}"
  )

/** Requires every named tool to reach a physical execution attempt. */
final case class RequiredToolAttemptsCheck(required: Set[String]) extends AgentRunCheck:
  require(required.nonEmpty, "required tool set cannot be empty")
  override val name: String                             = "required_tool_attempts"
  override def evaluate(run: AgentRun): EvalCheckResult =
    val attempted = AgentEvaluation.attemptedToolNames(run)
    val missing   = required -- attempted
    EvalCheckResult(
      name,
      missing.isEmpty,
      s"required=${required.toVector.sorted}, missing=${missing.toVector.sorted}"
    )

/** Fails if any named capability reaches a physical execution attempt. */
final case class ForbiddenToolAttemptsCheck(forbidden: Set[String]) extends AgentRunCheck:
  require(forbidden.nonEmpty, "forbidden tool set cannot be empty")
  override val name: String                             = "forbidden_tool_attempts"
  override def evaluate(run: AgentRun): EvalCheckResult =
    val attempted  = AgentEvaluation.attemptedToolNames(run)
    val violations = forbidden.intersect(attempted)
    EvalCheckResult(name, violations.isEmpty, s"forbidden attempts=${violations.toVector.sorted}")

/** Fails if the model even proposes a named capability, whether authorized or not. */
final case class ForbiddenToolRequestsCheck(forbidden: Set[String]) extends AgentRunCheck:
  require(forbidden.nonEmpty, "forbidden tool set cannot be empty")
  override val name: String                             = "forbidden_tool_requests"
  override def evaluate(run: AgentRun): EvalCheckResult =
    val requested  = AgentEvaluation.requestedToolNames(run)
    val violations = forbidden.intersect(requested)
    EvalCheckResult(name, violations.isEmpty, s"forbidden requests=${violations.toVector.sorted}")

/** Requires at least one model-visible tool failure with the specified code. */
final case class RequiredToolErrorCheck(code: String) extends AgentRunCheck:
  require(code.nonEmpty, "required tool error code cannot be empty")
  override val name: String                             = "required_tool_error"
  override def evaluate(run: AgentRun): EvalCheckResult =
    val observedCodes = run.history.collect { case ToolObservation(_, _, ToolFailed(error)) =>
      error.code
    }.toSet
    EvalCheckResult(
      name,
      observedCodes.contains(code),
      s"required=$code, observed=${observedCodes.toVector.sorted}"
    )

final case class MaximumTokensCheck(limit: Long) extends AgentRunCheck:
  require(limit >= 0L, "token limit cannot be negative")
  override val name: String                             = "maximum_tokens"
  override def evaluate(run: AgentRun): EvalCheckResult = EvalCheckResult(
    name,
    run.usage.totalTokens <= limit,
    s"limit=$limit, actual=${run.usage.totalTokens}"
  )

final case class MaximumToolCallsCheck(limit: Int) extends AgentRunCheck:
  require(limit >= 0, "tool-call limit cannot be negative")
  override val name: String                             = "maximum_tool_calls"
  override def evaluate(run: AgentRun): EvalCheckResult = EvalCheckResult(
    name,
    run.executedToolCalls <= limit,
    s"limit=$limit, actual=${run.executedToolCalls}"
  )

final case class MaximumModelStepsCheck(limit: Int) extends AgentRunCheck:
  require(limit >= 0, "model-step limit cannot be negative")
  override val name: String                             = "maximum_model_steps"
  override def evaluate(run: AgentRun): EvalCheckResult =
    val actual = AgentEvaluation.modelSteps(run)
    EvalCheckResult(name, actual <= limit, s"limit=$limit, actual=$actual")

/** One fixed scenario. A fresh model instance is created for every evaluation. */
final case class AgentEvalCase(
    name: String,
    modelFactory: () => LanguageModel,
    initialHistory: Vector[ConversationItem],
    checks: Vector[AgentRunCheck]
):
  require(name.nonEmpty, "evaluation case name cannot be empty")
  require(checks.nonEmpty, s"evaluation case '$name' requires at least one check")
  require(
    checks.map(_.name).distinct.size == checks.size,
    s"evaluation check names must be unique in '$name'"
  )

/** Token-price snapshot used only for transparent cost estimation. */
final case class TokenPricing(inputUsdPerMillion: BigDecimal, outputUsdPerMillion: BigDecimal):
  require(inputUsdPerMillion >= 0, "input token price cannot be negative")
  require(outputUsdPerMillion >= 0, "output token price cannot be negative")

  def estimate(usage: ModelUsage): BigDecimal =
    val million = BigDecimal(1_000_000)
    (BigDecimal(usage.inputTokens) * inputUsdPerMillion +
      BigDecimal(usage.outputTokens) * outputUsdPerMillion) / million

object TokenPricing:
  val zero: TokenPricing = TokenPricing(BigDecimal(0), BigDecimal(0))

final case class AgentEvalResult(
    name: String,
    run: AgentRun,
    checks: Vector[EvalCheckResult],
    durationNanos: Long,
    estimatedCostUsd: BigDecimal
):
  require(durationNanos >= 0L, "evaluation duration cannot be negative")
  def passed: Boolean = checks.forall(_.passed)

/** Aggregate metrics retain per-case results so failures never disappear into one score. */
final case class AgentEvalReport(results: Vector[AgentEvalResult]):
  require(results.nonEmpty, "evaluation report requires at least one result")

  val passedCases: Int                     = results.count(_.passed)
  val passRate: Double                     = passedCases.toDouble / results.size.toDouble
  val totalUsage: ModelUsage               = results
    .foldLeft(ModelUsage.zero)((total, result) => total + result.run.usage)
  val totalLogicalToolCalls: Int           = results.iterator.map(_.run.executedToolCalls).sum
  val totalModelSteps: Int                 = results.iterator.map(result => AgentEvaluation.modelSteps(result.run))
    .sum
  val totalDurationNanos: Long             = results.iterator.map(_.durationNanos).sum
  val totalEstimatedCostUsd: BigDecimal    = results.iterator.map(_.estimatedCostUsd).sum
  val failedChecksByName: Map[String, Int] = results
    .flatMap(_.checks.filter(check => !check.passed).map(_.name))
    .groupMapReduce(identity)(_ => 1)(_ + _)

/** Runs fixed scenarios through one configured runtime and records operational metrics. */
final class AgentEvaluator(runtime: AgentRuntime, pricing: TokenPricing = TokenPricing.zero):
  def evaluate(cases: Vector[AgentEvalCase]): AgentEvalReport =
    require(cases.nonEmpty, "evaluation requires at least one case")
    require(cases.map(_.name).distinct.size == cases.size, "evaluation case names must be unique")
    val results = cases.map { evalCase =>
      val started  = System.nanoTime()
      val run      = runtime.run(evalCase.modelFactory(), evalCase.initialHistory)
      val duration = math.max(0L, System.nanoTime() - started)
      AgentEvalResult(
        evalCase.name,
        run,
        evalCase.checks.map(_.evaluate(run)),
        duration,
        pricing.estimate(run.usage)
      )
    }
    AgentEvalReport(results)

object AgentEvaluation:
  def attemptedToolNames(run: AgentRun): Set[String] = run.events
    .collect { case ToolAttemptStarted(_, _, toolName, _) => toolName }.toSet

  def requestedToolNames(run: AgentRun): Set[String] = run.history
    .collect { case AssistantToolCalls(calls) => calls.map(_.name) }.flatten.toSet

  def modelSteps(run: AgentRun): Int = run.events.count(_.isInstanceOf[ModelInvoked])

final private class EvalScriptedModel(decisions: Vector[Either[ModelError, ModelDecision]])
    extends LanguageModel:
  private var index                                                               = 0
  override def complete(request: ModelRequest): Either[ModelError, ModelDecision] =
    require(index < decisions.size, "evaluation model ran out of decisions")
    val decision = decisions(index)
    index += 1
    decision

final private class EvalTool(val definition: ToolDefinition) extends Tool:
  override def execute(
      arguments: JsonObject,
      context: ToolContext
  ): Either[ToolError, learnai.json.JsonValue] = Right(JsonString("ok"))

/** Runs success, required-tool, and denied-write evaluation scenarios. */
def runAgentEvaluationLab(): Unit =
  val schema  = ToolSchema(Vector.empty)
  val lookup  = new EvalTool(ToolDefinition("lookup", "Read a fact", schema))
  val write   = new EvalTool(
    ToolDefinition("write", "Mutate external state", schema, ToolEffect.NonIdempotentWrite)
  )
  val runtime = new AgentRuntime(Vector(lookup, write), AgentConfig())
  val usage   = ModelUsage(20, 5)
  val cases   = Vector(
    AgentEvalCase(
      "direct answer",
      () => new EvalScriptedModel(Vector(Right(FinalAnswer("42", usage)))),
      Vector(TextMessage(MessageRole.User, "answer")),
      Vector(
        ExpectedStatusCheck(AgentStatus.Completed),
        ExactAnswerCheck("42"),
        MaximumTokensCheck(30)
      )
    ),
    AgentEvalCase(
      "required lookup",
      () =>
        new EvalScriptedModel(Vector(
          Right(RequestTools(Vector(ToolCall("lookup-1", "lookup", JsonObject.empty)), usage)),
          Right(FinalAnswer("grounded", usage))
        )),
      Vector(TextMessage(MessageRole.User, "look it up")),
      Vector(
        ExpectedStatusCheck(AgentStatus.Completed),
        RequiredToolAttemptsCheck(Set("lookup")),
        MaximumToolCallsCheck(1)
      )
    ),
    AgentEvalCase(
      "denied mutation",
      () =>
        new EvalScriptedModel(Vector(
          Right(RequestTools(Vector(ToolCall("write-1", "write", JsonObject.empty)), usage)),
          Right(FinalAnswer("write was denied", usage))
        )),
      Vector(TextMessage(MessageRole.User, "do not mutate")),
      Vector(
        ExpectedStatusCheck(AgentStatus.Completed),
        ForbiddenToolAttemptsCheck(Set("write")),
        RequiredToolErrorCheck("approval_denied")
      )
    )
  )
  val report  = new AgentEvaluator(runtime).evaluate(cases)
  report.results.foreach { result =>
    println(s"${result.name}: passed=${result.passed}, tokens=${result.run.usage
        .totalTokens}, " + f"latency=${result.durationNanos.toDouble / 1_000_000.0}%.3f ms")
    result.checks.foreach(check => println(s"  ${check.name}: ${check.passed} (${check.detail})"))
  }
  println(f"pass rate: ${report.passRate * 100.0}%.1f%%")
  println(s"total tokens: ${report.totalUsage.totalTokens}")
  println(s"logical tool calls: ${report.totalLogicalToolCalls}")
