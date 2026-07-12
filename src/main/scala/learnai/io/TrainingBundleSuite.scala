package learnai.io

import java.nio.file.Files

import learnai.data.CausalDataset
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.training.AdamWTrainingConfig
import learnai.training.MiniGptTrainingConfig
import learnai.training.ResumableMiniGptTraining
import learnai.training.WarmupCosineLearningRate
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object TrainingBundleSuite extends TestSuite:
  override val name: String = "TrainingBundle"

  private val modelConfig = MiniGptConfig(
    vocabularySize = 4,
    maximumContextLength = 3,
    channels = 4,
    headCount = 2,
    hiddenChannels = 8,
    layerCount = 1
  )

  private val tokens = Vector.fill(20)(Vector(0, 1, 2, 3)).flatten.map(TokenId(_))
  private val split = CausalDataset.contiguousSplit(tokens, trainingFraction = 0.75, contextLength = 3)

  private val trainingConfig = MiniGptTrainingConfig(
    totalUpdates = 8,
    batchSize = 4,
    microBatchSize = 2,
    validationEveryUpdates = 3,
    maximumValidationBatches = 3,
    batchSeed = 23L,
    learningRateSchedule = WarmupCosineLearningRate(peak = 0.03, minimum = 0.003, warmupUpdates = 3),
    optimizer = AdamWTrainingConfig(weightDecay = 0.01, maximumGradientNorm = Some(1.0))
  )

  private val experimentId = "a" * 64
  private val otherExperimentId = "b" * 64

  override val tests: Vector[TestCase] = specify(
    test("a mid-run state survives the disk round trip exactly") {
      val directory = Files.createTempDirectory("learn-ai-bundle-roundtrip")
      val path = directory.resolve("run.laibundle")
      try
        val model = MiniGpt.random(modelConfig, seed = 3L)
        val initial = ResumableMiniGptTraining.freshState(model, split, trainingConfig)
        val (_, state) =
          ResumableMiniGptTraining.train(model, split, trainingConfig, initial, updates = 3)

        val saved = Assert.right(TrainingBundle.save(state, experimentId, path))
        val loaded = Assert.right(TrainingBundle.load(path))

        Assert.equal(loaded.state, state)
        Assert.equal(loaded.experimentId, experimentId)
        Assert.equal(loaded.metadata, saved)
        Assert.equal(saved.completedUpdates, 3)
        Assert.equal(saved.sha256.length, 64)
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("resuming from disk matches the straight run bitwise") {
      val directory = Files.createTempDirectory("learn-ai-bundle-resume")
      val path = directory.resolve("run.laibundle")
      try
        val straightModel = MiniGpt.random(modelConfig, seed = 3L)
        val (straightRun, _) =
          ResumableMiniGptTraining.trainFromStart(straightModel, split, trainingConfig)

        val firstModel = MiniGpt.random(modelConfig, seed = 3L)
        val initial = ResumableMiniGptTraining.freshState(firstModel, split, trainingConfig)
        val (firstMetrics, middle) =
          ResumableMiniGptTraining.train(firstModel, split, trainingConfig, initial, updates = 4)
        Assert.isTrue(TrainingBundle.save(middle, experimentId, path).isRight)

        // Simulate a new process: nothing survives except the file and the
        // caller's own configuration-derived experiment identity.
        val restored = Assert.right(TrainingBundle.loadForResume(path, experimentId))
        val resumedModel = MiniGpt.random(modelConfig, seed = 999L)
        val (secondMetrics, _) = ResumableMiniGptTraining.train(
          resumedModel,
          split,
          trainingConfig,
          restored,
          updates = 4
        )

        Assert.equal(firstMetrics ++ secondMetrics, straightRun.steps)
        Assert.equal(
          resumedModel.parameters.map(_.values),
          straightModel.parameters.map(_.values)
        )
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("a bundle from a different experiment is refused on resume") {
      val directory = Files.createTempDirectory("learn-ai-bundle-identity")
      val path = directory.resolve("run.laibundle")
      try
        val model = MiniGpt.random(modelConfig, seed = 5L)
        val state = ResumableMiniGptTraining.freshState(model, split, trainingConfig)
        Assert.isTrue(TrainingBundle.save(state, experimentId, path).isRight)
        val refusal = Assert.left(TrainingBundle.loadForResume(path, otherExperimentId))
        Assert.isTrue(refusal.contains("refusing to continue"))
        Assert.isTrue(refusal.contains(experimentId))
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("corrupted truncated and mislabeled files fail verification") {
      val directory = Files.createTempDirectory("learn-ai-bundle-corrupt")
      val path = directory.resolve("run.laibundle")
      try
        val model = MiniGpt.random(modelConfig, seed = 7L)
        val state = ResumableMiniGptTraining.freshState(model, split, trainingConfig)
        Assert.isTrue(TrainingBundle.save(state, experimentId, path).isRight)

        val original = Files.readAllBytes(path)
        val corrupted = original.clone()
        corrupted(20) = (corrupted(20) ^ 0x01).toByte
        Files.write(path, corrupted)
        Assert.isTrue(Assert.left(TrainingBundle.load(path)).contains("SHA-256 mismatch"))

        Files.write(path, original.take(original.length / 2))
        Assert.isTrue(TrainingBundle.load(path).isLeft)

        val wrongMagic = original.clone()
        wrongMagic(0) = 'X'.toByte
        Files.write(path, wrongMagic)
        // Any header edit also breaks the checksum; the point is rejection.
        Assert.isTrue(TrainingBundle.load(path).isLeft)
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    },
    test("save validates the experiment identity format") {
      val directory = Files.createTempDirectory("learn-ai-bundle-id-format")
      val path = directory.resolve("run.laibundle")
      try
        val model = MiniGpt.random(modelConfig, seed = 9L)
        val state = ResumableMiniGptTraining.freshState(model, split, trainingConfig)
        val error = Assert.left(TrainingBundle.save(state, "not-a-hash", path))
        Assert.isTrue(error.contains("hexadecimal"))
        Assert.isTrue(!Files.exists(path), "a rejected save must not create the file")
      finally
        val _ = Files.deleteIfExists(path)
        val _ = Files.deleteIfExists(directory)
    }
  )
