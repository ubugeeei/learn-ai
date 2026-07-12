package learnai.distributed

import learnai.data.CausalDataset
import learnai.tensor.Shape
import learnai.tensor.Tensor
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

object DataParallelSuite extends TestSuite:
  override val name: String = "DataParallel"

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

  private def trainingConfig(totalUpdates: Int, batchSize: Int, microBatchSize: Int): MiniGptTrainingConfig =
    MiniGptTrainingConfig(
      totalUpdates = totalUpdates,
      batchSize = batchSize,
      microBatchSize = microBatchSize,
      validationEveryUpdates = 3,
      maximumValidationBatches = 3,
      batchSeed = 29L,
      learningRateSchedule = WarmupCosineLearningRate(peak = 0.03, minimum = 0.003, warmupUpdates = 2),
      optimizer = AdamWTrainingConfig(weightDecay = 0.01, maximumGradientNorm = Some(1.0))
    )

  override val tests: Vector[TestCase] = specify(
    test("data-parallel training matches the single-process trainer to rounding") {
      // Equality is deliberately NOT asserted bitwise: sequential
      // accumulation interleaves element additions inside each backward,
      // while ranks finish partial gradients before the reduction adds
      // them. Same terms, different association, last-ulp differences.
      val config = trainingConfig(totalUpdates = 6, batchSize = 4, microBatchSize = 2)
      val singleModel = MiniGpt.random(modelConfig, seed = 3L)
      val (singleRun, _) = ResumableMiniGptTraining.trainFromStart(singleModel, split, config)

      val parallel = DataParallelMiniGpt.train(modelConfig, 3L, split, config, worldSize = 2)

      parallel.metrics.map(_.trainingLoss).zip(singleRun.steps.map(_.trainingLoss)).foreach {
        case (parallelLoss, singleLoss) =>
          Assert.close(parallelLoss, singleLoss, tolerance = 1e-12)
      }
      parallel.metrics.map(_.gradientNorm).zip(singleRun.steps.map(_.gradientNorm)).foreach {
        case (parallelNorm, singleNorm) =>
          Assert.close(parallelNorm, singleNorm, tolerance = 1e-12)
      }
      Assert.equal(
        parallel.metrics.map(_.learningRate),
        singleRun.steps.map(_.learningRate)
      )
      singleModel.parameters.map(_.values).flatten
        .zip(parallel.replicas.head.parameters.map(_.values).flatten)
        .foreach { case (single, replicated) =>
          Assert.close(replicated, single, tolerance = 1e-10)
        }
    },
    test("replicas stay bitwise identical even with several microbatches per rank") {
      val config = trainingConfig(totalUpdates = 4, batchSize = 8, microBatchSize = 2)
      val parallel = DataParallelMiniGpt.train(modelConfig, 5L, split, config, worldSize = 2)
      val reference = parallel.replicas.head.parameters.map(_.values)
      parallel.replicas.tail.foreach { replica =>
        Assert.equal(replica.parameters.map(_.values), reference)
      }

      // Against the single process, association differs, so equality is
      // only up to rounding; the gap must be tiny, not zero.
      val singleModel = MiniGpt.random(modelConfig, seed = 5L)
      val (singleRun, _) = ResumableMiniGptTraining.trainFromStart(singleModel, split, config)
      parallel.metrics.map(_.trainingLoss).zip(singleRun.steps.map(_.trainingLoss)).foreach {
        case (parallelLoss, singleLoss) =>
          Assert.close(parallelLoss, singleLoss, tolerance = 1e-12)
      }
      singleModel.parameters.map(_.values).flatten
        .zip(parallel.replicas.head.parameters.map(_.values).flatten)
        .foreach { case (single, replicated) =>
          Assert.close(replicated, single, tolerance = 1e-9)
        }
    },
    test("collectives combine contributions in rank order with recorded traces") {
      val collectives = new Collectives(worldSize = 3)
      val sum = collectives.allReduceSum(
        Vector(Vector(1.0, 2.0), Vector(10.0, 20.0), Vector(100.0, 200.0))
      )
      Assert.equal(sum, Vector(111.0, 222.0))
      val mean = collectives.allReduceMean(
        Vector(Vector(3.0), Vector(6.0), Vector(9.0))
      )
      Assert.close(mean.head, 6.0, tolerance = 1e-15)
      val copies = collectives.broadcast(Vector(7.0, 8.0))
      Assert.equal(copies, Vector.fill(3)(Vector(7.0, 8.0)))

      Assert.equal(collectives.traces.size, 3)
      Assert.equal(collectives.traces.map(_.operation),
        Vector("all-reduce-sum", "all-reduce-sum", "broadcast"))
      Assert.equal(collectives.traces.head.logicalPayloadBytes, 16L)
    },
    test("ring all-reduce bytes follow the chunked two-phase formula") {
      // N = 4, 10 elements: chunks of ceil(10/4) = 3 elements; each rank
      // sends (N-1) chunks twice: 2 * 3 * 3 * 8 = 144 bytes.
      Assert.equal(new Collectives(4).ringAllReduceBytesPerRank(10), 144L)
      // A single rank moves nothing.
      Assert.equal(new Collectives(1).ringAllReduceBytesPerRank(1000), 0L)
      // Large world sizes approach 2x the buffer per rank.
      val bytes = new Collectives(64).ringAllReduceBytesPerRank(64_000)
      Assert.isTrue(bytes < 2L * 64_000L * 8L + 64L * 8L, s"ring bytes too large: $bytes")
    },
    test("training records one gradient bucket and one loss reduction per update") {
      val config = trainingConfig(totalUpdates = 3, batchSize = 4, microBatchSize = 2)
      val parallel = DataParallelMiniGpt.train(modelConfig, 7L, split, config, worldSize = 2)
      Assert.equal(parallel.collectiveTraces.size, 6)
      val gradientBuckets = parallel.collectiveTraces.filter(_.elementCount > 1)
      Assert.equal(gradientBuckets.size, 3)
      gradientBuckets.foreach { trace =>
        Assert.equal(trace.elementCount, parallel.replicas.head.parameterCount)
      }
      Assert.equal(parallel.collectiveTraces.count(_.elementCount == 1), 3)
    },
    test("invalid world sizes contributions and gradient assignments are rejected") {
      val badWorld = Assert.throws[IllegalArgumentException](new Collectives(0))
      Assert.isTrue(badWorld.getMessage.contains("positive"))

      val collectives = new Collectives(2)
      val wrongCount = Assert.throws[IllegalArgumentException] {
        collectives.allReduceSum(Vector(Vector(1.0)))
      }
      Assert.isTrue(wrongCount.getMessage.contains("expected 2"))
      val ragged = Assert.throws[IllegalArgumentException] {
        collectives.allReduceSum(Vector(Vector(1.0, 2.0), Vector(1.0)))
      }
      Assert.isTrue(ragged.getMessage.contains("elements"))

      val config = trainingConfig(totalUpdates = 2, batchSize = 4, microBatchSize = 2)
      val indivisible = Assert.throws[IllegalArgumentException] {
        DataParallelMiniGpt.train(modelConfig, 1L, split, config, worldSize = 4)
      }
      Assert.isTrue(indivisible.getMessage.contains("divisible"))

      val constant = Tensor.constant(Shape(2), Vector(1.0, 2.0))
      val notTrainable = Assert.throws[IllegalArgumentException] {
        constant.assignGradients(Vector(0.1, 0.2))
      }
      Assert.isTrue(notTrainable.getMessage.contains("trainable"))
      val parameter = Tensor.parameter(Shape(2), Vector(1.0, 2.0), "x")
      val wrongSize = Assert.throws[IllegalArgumentException] {
        parameter.assignGradients(Vector(0.1))
      }
      Assert.isTrue(wrongSize.getMessage.contains("does not match"))
    }
  )
