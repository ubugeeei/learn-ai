package learnai.distributed

import learnai.data.CausalBatch
import learnai.data.CausalSplit
import learnai.optim.AdamW
import learnai.random.SplitMix64
import learnai.training.MiniGptTraining
import learnai.training.MiniGptTrainingConfig
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

/** Metrics for one synchronous data-parallel update. */
final case class DataParallelUpdateMetrics(
    update: Int,
    learningRate: Double,
    trainingLoss: Double,
    gradientNorm: Double
)

/** Result of a simulated synchronous data-parallel run. */
final case class DataParallelRun(
    replicas: Vector[MiniGpt],
    metrics: Vector[DataParallelUpdateMetrics],
    collectiveTraces: Vector[CollectiveTrace]
)

/** Synchronous data-parallel MiniGPT training over simulated ranks.
  *
  * Each rank owns a full model replica and a shard of every global batch;
  * one gradient all-reduce per update keeps the replicas identical. The
  * simulation preserves the two properties that make real data parallelism
  * debuggable:
  *
  *   - *replica identity*: replicas start from the same seed, receive the
  *     same reduced gradients in the same order, and run identical
  *     optimizer steps, so their weights must stay bit-for-bit equal —
  *     any divergence is a bug, never noise;
  *   - *single-process equivalence*: microbatch loss is scaled by
  *     `microBatchSize / batchSize` exactly as in Chapter 22b, so the
  *     all-reduce **sum** of rank gradients is the mean-over-batch
  *     gradient, and results match the Chapter 22c trainer to rounding.
  *     The match is deliberately *not* bitwise: sequential accumulation
  *     interleaves element additions inside each backward pass, while
  *     ranks complete their partial gradients before the reduction adds
  *     them — a different association of the same terms. Chasing bitwise
  *     equality across parallelism strategies is a losing game; bitwise
  *     *replica identity within one strategy* is the invariant worth
  *     enforcing, and this simulation does.
  *
  * Gradients are flattened into one bucket per update (in parameter
  * order), all-reduced once, and written back with `assignGradients` —
  * the same fuse-then-reduce structure real frameworks use to amortize
  * per-collective latency.
  */
object DataParallelMiniGpt:
  def train(
      modelConfig: MiniGptConfig,
      modelSeed: Long,
      split: CausalSplit,
      config: MiniGptTrainingConfig,
      worldSize: Int
  ): DataParallelRun =
    require(worldSize > 0, s"world size must be positive: $worldSize")
    require(
      config.microBatchesPerUpdate % worldSize == 0,
      s"${config.microBatchesPerUpdate} microbatches per update must be divisible " +
        s"by world size $worldSize"
    )
    val replicas = Vector.fill(worldSize)(MiniGpt.random(modelConfig, modelSeed))
    MiniGptTraining.validateDatasets(replicas.head, split.training, split.validation)
    val optimizers = Vector.fill(worldSize) {
      new AdamW(
        learningRate = config.learningRateSchedule.learningRate(0, config.totalUpdates),
        beta1 = config.optimizer.beta1,
        beta2 = config.optimizer.beta2,
        epsilon = config.optimizer.epsilon,
        weightDecay = config.optimizer.weightDecay,
        maximumGradientNorm = config.optimizer.maximumGradientNorm
      )
    }
    val collectives = new Collectives(worldSize)
    // One shared stream stands in for identically seeded per-rank loaders.
    val random = SplitMix64.seeded(config.batchSeed)
    val microBatchesPerRank = config.microBatchesPerUpdate / worldSize
    val metrics = Vector.newBuilder[DataParallelUpdateMetrics]

    var updateIndex = 0
    while updateIndex < config.totalUpdates do
      val batch = split.training.sampleBatch(config.batchSize, random).fold(
        problem => throw new IllegalStateException(problem),
        identity
      )
      val microBatches = batch.examples.grouped(config.microBatchSize).toVector

      val rankResults = Vector.tabulate(worldSize) { rank =>
        val replica = replicas(rank)
        replica.parameters.foreach(_.clearGradients())
        val owned =
          microBatches.slice(rank * microBatchesPerRank, (rank + 1) * microBatchesPerRank)
        var weightedLoss = 0.0
        owned.foreach { examples =>
          val microBatch = CausalBatch(examples.toVector)
          val loss = MiniGptTraining.meanLoss(replica, microBatch)
          weightedLoss += loss.valueAtFlat(0) * microBatch.batchSize.toDouble
          loss
            .scale(microBatch.batchSize.toDouble / config.batchSize.toDouble)
            .backwardAccumulating()
        }
        val flatGradients = replica.parameters.flatMap(_.gradients)
        (weightedLoss, flatGradients)
      }

      // One fused gradient bucket and one scalar loss reduction per update.
      val reducedGradients = collectives.allReduceSum(rankResults.map(_._2))
      val reducedLoss = collectives.allReduceSum(rankResults.map(result => Vector(result._1)))

      val learningRate =
        config.learningRateSchedule.learningRate(updateIndex, config.totalUpdates)
      var gradientNorm = 0.0
      replicas.zip(optimizers).foreach { case (replica, optimizer) =>
        var offset = 0
        replica.parameters.foreach { parameter =>
          parameter.assignGradients(reducedGradients.slice(offset, offset + parameter.size))
          offset += parameter.size
        }
        val stats = optimizer.stepAtLearningRate(replica.parameters, learningRate)
        gradientNorm = stats.gradientNorm
        replica.parameters.foreach(_.clearGradients())
      }

      metrics += DataParallelUpdateMetrics(
        update = updateIndex + 1,
        learningRate = learningRate,
        trainingLoss = reducedLoss.head / config.batchSize.toDouble,
        gradientNorm = gradientNorm
      )
      updateIndex += 1

    DataParallelRun(replicas, metrics.result(), collectives.traces)
