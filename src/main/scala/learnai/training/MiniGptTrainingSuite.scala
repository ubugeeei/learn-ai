package learnai.training

import learnai.data.CausalBatch
import learnai.data.CausalDataset
import learnai.data.CausalSplit
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object MiniGptTrainingSuite extends TestSuite:
  override val name: String = "MiniGptTraining"

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

  override val tests: Vector[TestCase] = specify(
    test("constant and warmup-cosine schedules have explicit endpoint semantics") {
      val constant = ConstantLearningRate(0.02)
      Assert.equal(Vector.tabulate(4)(constant.learningRate(_, 4)), Vector.fill(4)(0.02))

      val schedule = WarmupCosineLearningRate(peak = 1.0, minimum = 0.0, warmupUpdates = 2)
      val values = Vector.tabulate(6)(schedule.learningRate(_, 6))
      Assert.equal(values.take(3), Vector(0.5, 1.0, 1.0))
      Assert.close(values(3), 0.75)
      Assert.close(values(4), 0.25)
      Assert.close(values(5), 0.0)
    },
    test("batch mean loss is the arithmetic mean of independently evaluated examples") {
      val model = MiniGpt.random(modelConfig, seed = 1L)
      val examples = split.validation.examples.take(3)
      val batch = CausalBatch(examples)
      val expected = examples.map(example => model.loss(example.inputs, example.targets).valueAtFlat(0)).sum /
        examples.size.toDouble
      Assert.close(MiniGptTraining.meanLoss(model, batch).valueAtFlat(0), expected, tolerance = 1e-12)
    },
    test("validation loss weights a short final batch by its example count") {
      val model = MiniGpt.random(modelConfig, seed = 2L)
      val dataset = CausalDataset.fromTokens(Vector.range(0, 8).map(index => TokenId(index % 4)), 3)
      Assert.equal(dataset.size, 5)
      val expected = dataset.examples
        .map(example => model.loss(example.inputs, example.targets).valueAtFlat(0))
        .sum / dataset.size.toDouble
      val actual = MiniGptTraining.validationLoss(model, dataset, batchSize = 4, maximumBatches = 2)
      Assert.close(actual, expected, tolerance = 1e-12)
    },
    test("seeded training is exactly reproducible across metrics and parameters") {
      val config = trainingConfig(totalUpdates = 8)
      val firstModel = MiniGpt.random(modelConfig, seed = 3L)
      val secondModel = MiniGpt.random(modelConfig, seed = 3L)
      val first = MiniGptTraining.train(firstModel, split, config)
      val second = MiniGptTraining.train(secondModel, split, config)
      Assert.equal(first, second)
      Assert.equal(firstModel.parameters.map(_.values), secondModel.parameters.map(_.values))
      Assert.isTrue(firstModel.parameters.forall(_.gradients.forall(_ == 0.0)))
    },
    test("training records scheduled validation tokens gradients and loss reduction") {
      val model = MiniGpt.random(modelConfig, seed = 4L)
      val config = trainingConfig(totalUpdates = 60)
      val run = MiniGptTraining.train(model, split, config)
      Assert.equal(run.steps.map(_.update), Vector.range(1, 61))
      Assert.equal(
        run.steps.flatMap(step => step.validationLoss.map(_ => step.update)),
        Vector.range(5, 61, 5)
      )
      Assert.equal(run.steps.map(_.tokensSeen), Vector.tabulate(60)(index => (index + 1L) * 12L))
      Assert.equal(run.totalTokensSeen, 720L)
      Assert.isTrue(
        run.bestValidationLoss < run.initialValidationLoss,
        s"best validation ${run.bestValidationLoss} did not improve initial ${run.initialValidationLoss}"
      )
      Assert.isTrue(
        run.finalValidationLoss < run.initialValidationLoss * 0.5,
        s"final validation ${run.finalValidationLoss} was not below half of initial ${run.initialValidationLoss}"
      )
      Assert.isTrue(run.steps.forall(step => step.gradientNorm.isFinite && step.gradientNorm >= 0.0))
      Assert.isTrue(run.steps.forall(step => step.gradientScale > 0.0 && step.gradientScale <= 1.0))
    },
    test("invalid accumulation and empty dataset configurations fail before updates") {
      val configError = Assert.throws[IllegalArgumentException] {
        trainingConfig(totalUpdates = 2).copy(batchSize = 3, microBatchSize = 2)
      }
      val model = MiniGpt.random(modelConfig, seed = 5L)
      val empty = CausalDataset.fromTokens(Vector(TokenId(0), TokenId(1)), contextLength = 3)
      val datasetError = Assert.throws[IllegalArgumentException] {
        MiniGptTraining.train(model, CausalSplit(empty, split.validation, 0), trainingConfig(2))
      }
      val scheduleError = Assert.throws[IllegalArgumentException] {
        WarmupCosineLearningRate(0.1, 0.0, warmupUpdates = 3).learningRate(0, totalUpdates = 2)
      }
      Assert.isTrue(configError.getMessage.contains("divisible"))
      Assert.isTrue(datasetError.getMessage.contains("training requires"))
      Assert.isTrue(scheduleError.getMessage.contains("exceed"))
    }
  )

  private def trainingConfig(totalUpdates: Int): MiniGptTrainingConfig =
    MiniGptTrainingConfig(
      totalUpdates = totalUpdates,
      batchSize = 4,
      microBatchSize = 2,
      validationEveryUpdates = 5,
      maximumValidationBatches = 3,
      batchSeed = 99L,
      learningRateSchedule = WarmupCosineLearningRate(
        peak = 0.03,
        minimum = 0.003,
        warmupUpdates = math.min(3, totalUpdates)
      ),
      optimizer = AdamWTrainingConfig(weightDecay = 0.0, maximumGradientNorm = Some(1.0))
    )
