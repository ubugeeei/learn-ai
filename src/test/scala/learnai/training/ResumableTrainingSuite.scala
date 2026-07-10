package learnai.training

import learnai.data.CausalDataset
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object ResumableTrainingSuite extends TestSuite:
  override val name: String = "ResumableMiniGptTraining"

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

  private def trainingConfig(totalUpdates: Int): MiniGptTrainingConfig =
    MiniGptTrainingConfig(
      totalUpdates = totalUpdates,
      batchSize = 4,
      microBatchSize = 2,
      validationEveryUpdates = 3,
      maximumValidationBatches = 3,
      batchSeed = 17L,
      learningRateSchedule = WarmupCosineLearningRate(
        peak = 0.03,
        minimum = 0.003,
        warmupUpdates = math.min(3, totalUpdates)
      ),
      optimizer = AdamWTrainingConfig(weightDecay = 0.01, maximumGradientNorm = Some(1.0))
    )

  override val tests: Vector[TestCase] = Vector(
    test("every split point resumes bitwise-identically to the straight run") {
      val config = trainingConfig(totalUpdates = 8)
      val straightModel = MiniGpt.random(modelConfig, seed = 3L)
      val (straightRun, straightState) =
        ResumableMiniGptTraining.trainFromStart(straightModel, split, config)
      val straightWeights = straightModel.parameters.map(_.values)

      (1 until config.totalUpdates).foreach { splitPoint =>
        val firstModel = MiniGpt.random(modelConfig, seed = 3L)
        val initial = ResumableMiniGptTraining.freshState(firstModel, split, config)
        val (firstMetrics, middleState) =
          ResumableMiniGptTraining.train(firstModel, split, config, initial, splitPoint)

        // Resume into a model with *different* random weights: the restore
        // must overwrite them completely, or the state is not the model.
        val resumedModel = MiniGpt.random(modelConfig, seed = 777L + splitPoint)
        val (secondMetrics, finalState) = ResumableMiniGptTraining.train(
          resumedModel,
          split,
          config,
          middleState,
          config.totalUpdates - splitPoint
        )

        Assert.equal(firstMetrics ++ secondMetrics, straightRun.steps)
        Assert.equal(resumedModel.parameters.map(_.values), straightWeights)
        Assert.equal(finalState, straightState)
      }
    },
    test("the fresh state captures the untouched model and stream origin") {
      val config = trainingConfig(totalUpdates = 8)
      val model = MiniGpt.random(modelConfig, seed = 5L)
      val state = ResumableMiniGptTraining.freshState(model, split, config)
      Assert.equal(state.completedUpdates, 0)
      Assert.equal(state.tokensSeen, 0L)
      Assert.equal(state.randomState, config.batchSeed)
      Assert.equal(state.parameterValues, model.parameters.map(_.values))
      Assert.equal(state.bestValidationLoss, state.initialValidationLoss)
      Assert.equal(state.optimizer.step, 0L)
      val expectedValidation = MiniGptTraining.validationLoss(
        model,
        split.validation,
        config.batchSize,
        config.maximumValidationBatches
      )
      Assert.close(state.initialValidationLoss, expectedValidation, tolerance = 0.0)
    },
    test("a longer resumable run learns and validates on the configured cadence") {
      val config = trainingConfig(totalUpdates = 30).copy(validationEveryUpdates = 5)
      val model = MiniGpt.random(modelConfig, seed = 6L)
      val (run, finalState) = ResumableMiniGptTraining.trainFromStart(model, split, config)
      Assert.equal(run.steps.map(_.update), Vector.range(1, 31))
      Assert.equal(
        run.steps.flatMap(step => step.validationLoss.map(_ => step.update)),
        Vector.range(5, 31, 5)
      )
      Assert.isTrue(
        run.finalValidationLoss < run.initialValidationLoss,
        s"validation ${run.finalValidationLoss} did not improve on ${run.initialValidationLoss}"
      )
      Assert.equal(finalState.completedUpdates, 30)
      Assert.equal(finalState.tokensSeen, run.totalTokensSeen)
      Assert.isTrue(
        finalState.randomState != config.batchSeed,
        "the random stream must have advanced past its seed"
      )
    },
    test("chunk boundaries do not change validation bookkeeping") {
      // The interval is 3 and the split is 4, so the resumed chunk owns the
      // validations at updates 6 and 8 while the first chunk owns update 3.
      val config = trainingConfig(totalUpdates = 8)
      val model = MiniGpt.random(modelConfig, seed = 7L)
      val initial = ResumableMiniGptTraining.freshState(model, split, config)
      val (firstMetrics, middle) =
        ResumableMiniGptTraining.train(model, split, config, initial, 4)
      val (secondMetrics, _) =
        ResumableMiniGptTraining.train(model, split, config, middle, 4)
      Assert.equal(
        firstMetrics.flatMap(step => step.validationLoss.map(_ => step.update)),
        Vector(3)
      )
      Assert.equal(
        secondMetrics.flatMap(step => step.validationLoss.map(_ => step.update)),
        Vector(6, 8)
      )
    },
    test("invalid chunk sizes states and architectures are rejected") {
      val config = trainingConfig(totalUpdates = 8)
      val model = MiniGpt.random(modelConfig, seed = 8L)
      val state = ResumableMiniGptTraining.freshState(model, split, config)

      val zeroUpdates = Assert.throws[IllegalArgumentException] {
        ResumableMiniGptTraining.train(model, split, config, state, updates = 0)
      }
      Assert.isTrue(zeroUpdates.getMessage.contains("positive"))

      val overrun = Assert.throws[IllegalArgumentException] {
        ResumableMiniGptTraining.train(model, split, config, state, updates = 9)
      }
      Assert.isTrue(overrun.getMessage.contains("exceeds"))

      val widerModel = MiniGpt.random(modelConfig.copy(channels = 8, hiddenChannels = 16), seed = 8L)
      val wrongArchitecture = Assert.throws[IllegalArgumentException] {
        ResumableMiniGptTraining.train(widerModel, split, config, state, updates = 1)
      }
      Assert.isTrue(wrongArchitecture.getMessage.contains("does not match"))

      val corruptBookkeeping = Assert.throws[IllegalArgumentException] {
        state.copy(bestValidationUpdate = 1)
      }
      Assert.isTrue(corruptBookkeeping.getMessage.contains("outside"))
    }
  )
