package learnai.training

import java.util.SplittableRandom

import learnai.data.CausalBatch
import learnai.data.CausalDataset
import learnai.data.CausalSplit
import learnai.optim.AdamW
import learnai.optim.OptimizerStats
import learnai.tensor.Tensor
import learnai.transformer.MiniGpt

/** Learning-rate policy evaluated at a zero-based optimizer update index. */
sealed trait LearningRateSchedule:
  def learningRate(updateIndex: Int, totalUpdates: Int): Double

/** Keeps one learning rate for every optimizer update. */
final case class ConstantLearningRate(value: Double) extends LearningRateSchedule:
  require(value >= 0.0 && value.isFinite, s"learning rate must be finite and non-negative: $value")

  override def learningRate(updateIndex: Int, totalUpdates: Int): Double =
    LearningRateSchedule.validateIndex(updateIndex, totalUpdates)
    value

/** Linear warmup followed by cosine decay to a minimum learning rate.
  *
  * Warmup update `i` uses `peak * (i + 1) / warmupUpdates`. The decay region
  * includes the peak at progress zero and reaches `minimum` on its final
  * update. When warmup consumes every update, the schedule ends at `peak`.
  */
final case class WarmupCosineLearningRate(
    peak: Double,
    minimum: Double,
    warmupUpdates: Int
) extends LearningRateSchedule:
  require(peak > 0.0 && peak.isFinite, s"peak learning rate must be finite and positive: $peak")
  require(
    minimum >= 0.0 && minimum.isFinite,
    s"minimum learning rate must be finite and non-negative: $minimum"
  )
  require(minimum <= peak, s"minimum learning rate $minimum cannot exceed peak $peak")
  require(warmupUpdates >= 0, s"warmup updates must be non-negative: $warmupUpdates")

  override def learningRate(updateIndex: Int, totalUpdates: Int): Double =
    LearningRateSchedule.validateIndex(updateIndex, totalUpdates)
    require(
      warmupUpdates <= totalUpdates,
      s"warmup updates $warmupUpdates exceed total updates $totalUpdates"
    )
    if warmupUpdates > 0 && updateIndex < warmupUpdates then
      peak * (updateIndex + 1).toDouble / warmupUpdates.toDouble
    else
      val decayUpdates = totalUpdates - warmupUpdates
      if decayUpdates == 0 then peak
      else
        val decayIndex = updateIndex - warmupUpdates
        val progress =
          if decayUpdates == 1 then 1.0
          else decayIndex.toDouble / (decayUpdates - 1).toDouble
        minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.Pi * progress))

object LearningRateSchedule:
  private[training] def validateIndex(updateIndex: Int, totalUpdates: Int): Unit =
    require(totalUpdates > 0, s"total updates must be positive: $totalUpdates")
    require(
      updateIndex >= 0 && updateIndex < totalUpdates,
      s"update index $updateIndex outside [0, $totalUpdates)"
    )

/** AdamW hyperparameters whose state is owned by one training run. */
final case class AdamWTrainingConfig(
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    epsilon: Double = 1e-8,
    weightDecay: Double = 0.01,
    maximumGradientNorm: Option[Double] = Some(1.0)
):
  require(beta1 >= 0.0 && beta1 < 1.0 && beta1.isFinite, s"beta1 must be in [0,1): $beta1")
  require(beta2 >= 0.0 && beta2 < 1.0 && beta2.isFinite, s"beta2 must be in [0,1): $beta2")
  require(epsilon > 0.0 && epsilon.isFinite, s"epsilon must be finite and positive: $epsilon")
  require(
    weightDecay >= 0.0 && weightDecay.isFinite,
    s"weight decay must be finite and non-negative: $weightDecay"
  )
  maximumGradientNorm.foreach { maximum =>
    require(
      maximum > 0.0 && maximum.isFinite,
      s"maximum gradient norm must be finite and positive: $maximum"
    )
  }

/** Complete deterministic control configuration for the local training loop. */
final case class MiniGptTrainingConfig(
    totalUpdates: Int,
    batchSize: Int,
    microBatchSize: Int,
    validationEveryUpdates: Int,
    maximumValidationBatches: Int,
    batchSeed: Long,
    learningRateSchedule: LearningRateSchedule,
    optimizer: AdamWTrainingConfig = AdamWTrainingConfig()
):
  require(totalUpdates > 0, s"total updates must be positive: $totalUpdates")
  require(batchSize > 0, s"batch size must be positive: $batchSize")
  require(microBatchSize > 0, s"microbatch size must be positive: $microBatchSize")
  require(
    batchSize % microBatchSize == 0,
    s"batch size $batchSize must be divisible by microbatch size $microBatchSize"
  )
  require(
    validationEveryUpdates > 0,
    s"validation interval must be positive: $validationEveryUpdates"
  )
  require(
    maximumValidationBatches > 0,
    s"maximum validation batches must be positive: $maximumValidationBatches"
  )
  val microBatchesPerUpdate: Int = batchSize / microBatchSize

/** Metrics recorded after one optimizer update.
  *
  * Training loss is measured on the sampled batch before the update.
  * Validation loss, when present, is measured deterministically after the
  * update. `tokensSeen` counts target tokens used for parameter updates and
  * excludes validation work.
  */
final case class TrainingStepMetrics(
    update: Int,
    learningRate: Double,
    trainingLoss: Double,
    validationLoss: Option[Double],
    gradientNorm: Double,
    gradientScale: Double,
    tokensSeen: Long
)

/** Complete local training result retaining every step metric and validation decision. */
final case class MiniGptTrainingRun(
    config: MiniGptTrainingConfig,
    initialValidationLoss: Double,
    steps: Vector[TrainingStepMetrics],
    bestValidationUpdate: Int,
    bestValidationLoss: Double
):
  require(steps.size == config.totalUpdates, "training run must retain one metric per update")
  val finalTrainingLoss: Double = steps.last.trainingLoss
  val finalValidationLoss: Double = steps.reverseIterator
    .flatMap(_.validationLoss)
    .nextOption()
    .getOrElse(initialValidationLoss)
  val totalTokensSeen: Long = steps.last.tokensSeen

object MiniGptTraining:
  /** Trains with replacement-sampled batches and exact weighted microbatch accumulation.
    *
    * Validation uses consecutive batches and never samples from the training
    * random stream. The initial validation loss is recorded before update one;
    * subsequent validation runs occur at the configured interval and always on
    * the final update. The returned model is left with cleared gradients.
    *
    * This chapter does not yet implement resumable random/optimizer state.
    * Re-running from update zero with the same model initialization, data,
    * configuration, and seed is deterministic; resuming mid-run is Chapter
    * 22c's separate contract.
    */
  def train(
      model: MiniGpt,
      split: CausalSplit,
      config: MiniGptTrainingConfig
  ): MiniGptTrainingRun =
    validateDatasets(model, split.training, split.validation)
    val parameters = model.parameters
    val random = new SplittableRandom(config.batchSeed)
    val initialLearningRate = config.learningRateSchedule.learningRate(0, config.totalUpdates)
    val optimizer = new AdamW(
      learningRate = initialLearningRate,
      beta1 = config.optimizer.beta1,
      beta2 = config.optimizer.beta2,
      epsilon = config.optimizer.epsilon,
      weightDecay = config.optimizer.weightDecay,
      maximumGradientNorm = config.optimizer.maximumGradientNorm
    )
    val initialValidation = validationLoss(
      model,
      split.validation,
      config.batchSize,
      config.maximumValidationBatches
    )
    var bestValidationLoss = initialValidation
    var bestValidationUpdate = 0
    var tokensSeen = 0L
    val metrics = Vector.newBuilder[TrainingStepMetrics]

    var updateIndex = 0
    while updateIndex < config.totalUpdates do
      parameters.foreach(_.clearGradients())
      val batch = split.training.sampleBatch(config.batchSize, random).fold(
        problem => throw new IllegalStateException(problem),
        identity
      )
      val microBatches = batch.examples.grouped(config.microBatchSize).toVector
      var weightedTrainingLoss = 0.0
      microBatches.foreach { examples =>
        val microBatch = CausalBatch(examples.toVector)
        val loss = meanLoss(model, microBatch)
        weightedTrainingLoss += loss.valueAtFlat(0) * microBatch.batchSize.toDouble
        loss
          .scale(microBatch.batchSize.toDouble / config.batchSize.toDouble)
          .backwardAccumulating()
      }
      val learningRate = config.learningRateSchedule.learningRate(updateIndex, config.totalUpdates)
      val optimizerStats = optimizer.stepAtLearningRate(parameters, learningRate)
      parameters.foreach(_.clearGradients())

      val completedUpdate = updateIndex + 1
      tokensSeen = Math.addExact(
        tokensSeen,
        Math.multiplyExact(config.batchSize.toLong, batch.contextLength.toLong)
      )
      val shouldValidate =
        completedUpdate % config.validationEveryUpdates == 0 ||
          completedUpdate == config.totalUpdates
      val currentValidation =
        if shouldValidate then
          Some(
            validationLoss(
              model,
              split.validation,
              config.batchSize,
              config.maximumValidationBatches
            )
          )
        else None
      currentValidation.foreach { value =>
        if value < bestValidationLoss then
          bestValidationLoss = value
          bestValidationUpdate = completedUpdate
      }
      metrics += metric(
        completedUpdate,
        learningRate,
        weightedTrainingLoss / config.batchSize.toDouble,
        currentValidation,
        optimizerStats,
        tokensSeen
      )
      updateIndex += 1

    MiniGptTrainingRun(
      config,
      initialValidation,
      metrics.result(),
      bestValidationUpdate,
      bestValidationLoss
    )

  /** Builds one scalar mean loss over all examples in a batch. */
  def meanLoss(model: MiniGpt, batch: CausalBatch): Tensor =
    val losses = batch.examples.map(example => model.loss(example.inputs, example.targets))
    losses.reduce(_ + _).scale(1.0 / batch.batchSize.toDouble)

  /** Computes deterministic example-weighted loss over consecutive validation batches. */
  def validationLoss(
      model: MiniGpt,
      dataset: CausalDataset,
      batchSize: Int,
      maximumBatches: Int
  ): Double =
    require(batchSize > 0, s"validation batch size must be positive: $batchSize")
    require(maximumBatches > 0, s"maximum validation batches must be positive: $maximumBatches")
    val batches = dataset.sequentialBatches(batchSize, dropLast = false).take(maximumBatches)
    require(batches.nonEmpty, "validation requires at least one example")
    var weightedLoss = 0.0
    var examples = 0
    batches.foreach { batch =>
      weightedLoss += meanLoss(model, batch).valueAtFlat(0) * batch.batchSize.toDouble
      examples += batch.batchSize
    }
    weightedLoss / examples.toDouble

  private def validateDatasets(
      model: MiniGpt,
      training: CausalDataset,
      validation: CausalDataset
  ): Unit =
    require(!training.isEmpty, "training requires at least one example")
    require(!validation.isEmpty, "validation requires at least one example")
    require(
      training.contextLength == validation.contextLength,
      s"training context ${training.contextLength} differs from validation ${validation.contextLength}"
    )
    require(
      training.contextLength <= model.config.maximumContextLength,
      s"dataset context ${training.contextLength} exceeds model maximum ${model.config.maximumContextLength}"
    )

  private def metric(
      completedUpdate: Int,
      learningRate: Double,
      trainingLoss: Double,
      validationLoss: Option[Double],
      optimizerStats: OptimizerStats,
      tokensSeen: Long
  ): TrainingStepMetrics =
    TrainingStepMetrics(
      completedUpdate,
      learningRate,
      trainingLoss,
      validationLoss,
      optimizerStats.gradientNorm,
      optimizerStats.gradientScale,
      tokensSeen
    )
