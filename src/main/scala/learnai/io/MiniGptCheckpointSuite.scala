package learnai.io

import java.nio.file.Files

import learnai.text.TokenId
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig
import learnai.transformer.MiniGptTrainer
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object MiniGptCheckpointSuite extends TestSuite:
  override val name: String = "MiniGptCheckpoint"

  private val config = MiniGptConfig(
    vocabularySize = 5,
    maximumContextLength = 4,
    channels = 4,
    headCount = 2,
    hiddenChannels = 8,
    layerCount = 1
  )

  override val tests: Vector[TestCase] = specify(
    test("save and load preserve config labels parameters and logits exactly") {
      val directory = Files.createTempDirectory("learn-ai-checkpoint-roundtrip")
      val path = directory.resolve("model.lai")
      try
        val model = MiniGpt.random(config, seed = 42L)
        val inputs = Vector(0, 1, 0, 2).map(TokenId(_))
        val targets = Vector(1, 0, 2, 0).map(TokenId(_))
        val _ = MiniGptTrainer.trainSequence(model, inputs, targets, steps = 3, learningRate = 0.01)
        val logitsBefore = model.logits(inputs).values

        val saved = Assert.right(MiniGptCheckpoint.save(model, path))
        val (loaded, loadedMetadata) = Assert.right(MiniGptCheckpoint.load(path))

        Assert.equal(loaded.config, model.config)
        Assert.equal(loaded.parameters.map(_.label), model.parameters.map(_.label))
        Assert.equal(loaded.parameters.flatMap(_.values), model.parameters.flatMap(_.values))
        Assert.equal(loaded.logits(inputs).values, logitsBefore)
        Assert.equal(loadedMetadata, saved)
        Assert.equal(saved.sha256.length, 64)
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("one corrupted byte is rejected by the checksum before parsing") {
      val directory = Files.createTempDirectory("learn-ai-checkpoint-corrupt")
      val path = directory.resolve("model.lai")
      try
        val model = MiniGpt.random(config, seed = 1L)
        Assert.isTrue(MiniGptCheckpoint.save(model, path).isRight)
        val bytes = Files.readAllBytes(path)
        bytes(12) = (bytes(12) ^ 0x01).toByte
        Files.write(path, bytes)
        val error = Assert.left(MiniGptCheckpoint.load(path))
        Assert.isTrue(error.contains("SHA-256 mismatch"))
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("truncated files return an error instead of a partially loaded model") {
      val directory = Files.createTempDirectory("learn-ai-checkpoint-truncated")
      val path = directory.resolve("model.lai")
      try
        Files.write(path, Array[Byte](1, 2, 3, 4))
        val error = Assert.left(MiniGptCheckpoint.load(path))
        Assert.isTrue(error.contains("too short"))
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("checkpoint save replaces an existing file atomically") {
      val directory = Files.createTempDirectory("learn-ai-checkpoint-replace")
      val path = directory.resolve("model.lai")
      try
        Files.writeString(path, "old incomplete contents")
        val model = MiniGpt.random(config, seed = 9L)
        Assert.isTrue(MiniGptCheckpoint.save(model, path).isRight)
        Assert.isTrue(MiniGptCheckpoint.load(path).isRight)
        Assert.isTrue(Files.size(path) > "old incomplete contents".length)
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("Tensor assignment validates all values before mutation") {
      val model = MiniGpt.random(config, seed = 3L)
      val parameter = model.parameters.head
      val original = parameter.values
      val invalid = original.updated(0, Double.NaN)
      val error = Assert.throws[IllegalArgumentException] {
        parameter.assignParameterValues(invalid)
      }
      Assert.isTrue(error.getMessage.contains("must be finite"))
      Assert.equal(parameter.values, original)
    }
  )
