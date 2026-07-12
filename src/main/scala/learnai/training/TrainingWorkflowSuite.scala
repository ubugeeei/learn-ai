package learnai.training

import java.nio.charset.StandardCharsets
import java.nio.file.Files

import learnai.io.MiniGptCheckpoint
import learnai.io.TrainingBundle
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object TrainingWorkflowSuite extends TestSuite:
  override val name: String = "TrainingWorkflow"

  override val tests: Vector[TestCase] = specify(
    test("command parser requires explicit input and output paths") {
      Assert.isTrue(TrainingCommand.parse(Nil).isLeft)
      Assert.isTrue(TrainingCommand.parse(List("--input", "corpus.txt")).isLeft)
    },
    test("command parser rejects unknown duplicate and malformed options") {
      Assert.isTrue(TrainingCommand.parse(List("--wat", "1")).isLeft)
      Assert.isTrue(TrainingCommand.parse(List("--input", "a", "--input", "b")).isLeft)
      Assert.isTrue(TrainingCommand.parse(List("--updates")).isLeft)
    },
    test("a real corpus produces verified inference and resume artifacts") {
      val directory = Files.createTempDirectory("learnai-workflow-")
      val input = directory.resolve("corpus.txt")
      val output = directory.resolve("run")
      Files.writeString(input, Vector.fill(20)("small models can still use real files. ").mkString, StandardCharsets.UTF_8)
      val result = Assert.right(
        TrainingWorkflow.run(
          TrainingWorkflowConfig(
            input = input,
            output = output,
            contextLength = 8,
            channels = 4,
            heads = 1,
            hiddenChannels = 8,
            layers = 1,
            updates = 2,
            batchSize = 2,
            microBatchSize = 1
          )
        )
      )
      Assert.equal(result.completedUpdates, 2)
      Assert.isTrue(Files.size(output.resolve("manifest.json")) > 0L)
      Assert.equal(Files.readAllLines(output.resolve("metrics.jsonl")).size(), 2)
      Assert.isTrue(MiniGptCheckpoint.load(output.resolve("model.laigpt")).isRight)
      val bundle = Assert.right(TrainingBundle.load(output.resolve("training.laibnd")))
      Assert.equal(bundle.experimentId, result.experimentId)
      Assert.equal(bundle.state.completedUpdates, 2)
    }
  )
