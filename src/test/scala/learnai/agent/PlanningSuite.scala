package learnai.agent

import scala.collection.mutable.ArrayBuffer

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object PlanningSuite extends TestSuite:
  override val name: String = "PlanningAgent"

  private final class OneDecisionModel(decision: Either[ModelError, ModelDecision])
      extends LanguageModel:
    private var invoked = false

    override def complete(request: ModelRequest): Either[ModelError, ModelDecision] =
      require(!invoked, "one-decision model was invoked more than once")
      invoked = true
      decision

  private final class RecordingFactory(
      decide: TaskExecutionContext => Either[ModelError, ModelDecision]
  ) extends TaskModelFactory:
    val contexts: ArrayBuffer[TaskExecutionContext] = ArrayBuffer.empty

    override def create(context: TaskExecutionContext): LanguageModel =
      contexts += context
      new OneDecisionModel(decide(context))

  private def plan(goal: String, tasks: Vector[PlanTask]): TaskPlan =
    Assert.right(TaskPlan.create(goal, tasks))

  private def runtime: AgentRuntime = new AgentRuntime(Vector.empty, AgentConfig())

  override val tests: Vector[TestCase] = Vector(
    test("plan validation rejects duplicate missing and cyclic dependencies") {
      val duplicate = TaskPlan.create(
        "goal",
        Vector(PlanTask("same", "first"), PlanTask("same", "second"))
      )
      val missing = TaskPlan.create(
        "goal",
        Vector(PlanTask("task", "work", Vector("unknown")))
      )
      val cyclic = TaskPlan.create(
        "goal",
        Vector(
          PlanTask("a", "first", Vector("b")),
          PlanTask("b", "second", Vector("a"))
        )
      )
      Assert.isTrue(Assert.left(duplicate).exists(_.contains("duplicate")))
      Assert.isTrue(Assert.left(missing).exists(_.contains("unknown")))
      Assert.isTrue(Assert.left(cyclic).exists(_.contains("cycle")))
    },
    test("ready tasks execute in declared dependency order") {
      val graph = plan(
        "publish a brief",
        Vector(
          PlanTask("collect", "collect facts"),
          PlanTask("write", "write brief", Vector("collect"))
        )
      )
      val factory = new RecordingFactory(context =>
        Right(FinalAnswer(s"answer-${context.task.id}", ModelUsage(3, 2)))
      )
      val result = new PlanningAgent(runtime, PlanningConfig()).run(
        graph,
        Vector(TextMessage(MessageRole.User, "start")),
        factory
      )
      Assert.equal(result.status, PlanningStatus.Completed)
      Assert.equal(factory.contexts.map(_.task.id).toVector, Vector("collect", "write"))
      Assert.equal(
        factory.contexts.last.dependencyAnswers,
        Map("collect" -> "answer-collect")
      )
      Assert.equal(result.usage, ModelUsage(6, 4))
      Assert.equal(result.checkpoint.completedAnswers.keySet, Set("collect", "write"))
    },
    test("a failed worker is recreated within the bounded task attempt budget") {
      val graph = plan("recover", Vector(PlanTask("unstable", "finish despite one failure")))
      val factory = new RecordingFactory(context =>
        if context.attempt == 1 then
          Left(ModelError("temporary", "retry", retryable = true))
        else Right(FinalAnswer("recovered", ModelUsage(2, 1)))
      )
      val result = new PlanningAgent(runtime, PlanningConfig(maximumTaskAttempts = 2)).run(
        graph,
        Vector.empty,
        factory
      )
      Assert.equal(result.status, PlanningStatus.Completed)
      Assert.equal(result.attempts.map(_.attempt), Vector(1, 2))
      Assert.equal(result.taskResults.head.answer, Some("recovered"))
      Assert.equal(result.taskResults.head.attempts, 2)
    },
    test("failed branches block descendants but not independent work") {
      val graph = plan(
        "branching work",
        Vector(
          PlanTask("fail", "cannot complete"),
          PlanTask("blocked", "depends on failure", Vector("fail")),
          PlanTask("independent", "can still complete")
        )
      )
      val factory = new RecordingFactory(context =>
        if context.task.id == "fail" then
          Left(ModelError("permanent", "stop", retryable = false))
        else Right(FinalAnswer("independent answer", ModelUsage(1, 1)))
      )
      val result = new PlanningAgent(runtime, PlanningConfig(maximumTaskAttempts = 1)).run(
        graph,
        Vector.empty,
        factory
      )
      val statuses = result.taskResults.map(result => result.task.id -> result.status).toMap
      Assert.equal(result.status, PlanningStatus.Failed)
      Assert.equal(statuses("fail"), PlannedTaskStatus.Failed)
      Assert.equal(statuses("blocked"), PlannedTaskStatus.Blocked)
      Assert.equal(statuses("independent"), PlannedTaskStatus.Succeeded)
      Assert.equal(factory.contexts.map(_.task.id).toVector, Vector("fail", "independent"))
    },
    test("a dependency-closed checkpoint skips completed task effects") {
      val graph = plan(
        "resume",
        Vector(
          PlanTask("first", "already complete"),
          PlanTask("second", "continue here", Vector("first"))
        )
      )
      val checkpoint = Assert.right(
        PlanCheckpoint.create(graph, Map("first" -> "durable answer"))
      )
      val factory = new RecordingFactory(context =>
        Right(FinalAnswer("new answer", ModelUsage(1, 1)))
      )
      val result = new PlanningAgent(runtime, PlanningConfig()).run(
        graph,
        Vector.empty,
        factory,
        checkpoint
      )
      Assert.equal(factory.contexts.map(_.task.id).toVector, Vector("second"))
      Assert.equal(factory.contexts.head.dependencyAnswers("first"), "durable answer")
      Assert.equal(result.taskResults.head.attempts, 0)
      Assert.equal(result.events.count(_.isInstanceOf[TaskRecovered]), 1)
      Assert.equal(result.checkpoint.completedAnswers.size, 2)
    },
    test("checkpoint validation rejects unknown and dependency-incomplete state") {
      val graph = plan(
        "validate resume",
        Vector(
          PlanTask("first", "first"),
          PlanTask("second", "second", Vector("first"))
        )
      )
      val unknown = Assert.left(PlanCheckpoint.create(graph, Map("missing" -> "value")))
      val incomplete = Assert.left(PlanCheckpoint.create(graph, Map("second" -> "value")))
      Assert.isTrue(unknown.exists(_.contains("unknown")))
      Assert.isTrue(incomplete.exists(_.contains("missing completed dependency")))
    },
    test("dependency outputs are labeled as untrusted conversation data") {
      val graph = plan(
        "safe context",
        Vector(
          PlanTask("source", "produce data"),
          PlanTask("consumer", "consume data", Vector("source"))
        )
      )
      val histories = ArrayBuffer.empty[Vector[ConversationItem]]
      val factory = new TaskModelFactory:
        override def create(context: TaskExecutionContext): LanguageModel = new LanguageModel:
          override def complete(request: ModelRequest): Either[ModelError, ModelDecision] =
            histories += request.history
            Right(FinalAnswer(s"answer-${context.task.id}", ModelUsage.zero))
      val _ = new PlanningAgent(runtime, PlanningConfig()).run(graph, Vector.empty, factory)
      val downstream = histories.last.collect { case TextMessage(_, content) => content }.mkString("\n")
      Assert.isTrue(downstream.contains("untrusted data"))
      Assert.isTrue(downstream.contains("answer-source"))
    }
  )
