package learnai.training

import learnai.data.CausalBatch
import learnai.data.CausalSplit
import learnai.optim.AdamW
import learnai.optim.AdamWSnapshot
import learnai.random.SplitMix64
import learnai.tensor.Tensor
import learnai.transformer.MiniGpt

/** Everything required to continue a training run exactly where it stopped.
  *
  * The five resume ingredients from the chapter contract map onto fields:
  *
  *   - *model*: `parameterValues`, in `model.parameters` order;
  *   - *optimizer*: `optimizer`, the complete AdamW moment/step snapshot;
  *   - *scheduler*: `completedUpdates` — every schedule in this course is a
  *     pure function of the global update index, so the index is the state;
  *   - *data cursor*: `randomState` — batches are sampled with replacement
  *     from the RNG stream, so the stream position is the cursor;
  *   - *random state*: `randomState`, the full SplitMix64 counter.
  *
  * `initialValidationLoss` and the best-validation bookkeeping are carried so
  * a resumed run reports the same summary a straight run would. The state is
  * an in-memory value; Chapter 25's versioned checksummed format shows how
  * to persist it, and doing so is this chapter's first exercise.
  */
final case class MiniGptTrainingState(
    completedUpdates: Int,
    tokensSeen: Long,
    bestValidationUpdate: Int,
    bestValidationLoss: Double,
    initialValidationLoss: Double,
    randomState: Long,
    optimizer: AdamWSnapshot,
    parameterValues: Vector[Vector[Double]]
):
  require(completedUpdates >= 0, s"completed updates must be non-negative: $completedUpdates")
  require(tokensSeen >= 0L, s"tokens seen must be non-negative: $tokensSeen")
  require(
    bestValidationUpdate >= 0 && bestValidationUpdate <= completedUpdates,
    s"best validation update $bestValidationUpdate outside [0, $completedUpdates]"
  )
  require(
    bestValidationLoss.isFinite && initialValidationLoss.isFinite,
    "validation losses in a training state must be finite"
  )
  require(parameterValues.nonEmpty, "a training state must carry at least one parameter tensor")

object ResumableMiniGptTraining:
  /** Captures the complete state of a run that has not performed any update.
    *
    * The initial validation loss is computed here, exactly as the Chapter
    * 22b trainer records it before its first update.
    */
  def freshState(
      model: MiniGpt,
      split: CausalSplit,
      config: MiniGptTrainingConfig
  ): MiniGptTrainingState =
    MiniGptTraining.validateDatasets(model, split.training, split.validation)
    val initialValidation = MiniGptTraining.validationLoss(
      model,
      split.validation,
      config.batchSize,
      config.maximumValidationBatches
    )
    MiniGptTrainingState(
      completedUpdates = 0,
      tokensSeen = 0L,
      bestValidationUpdate = 0,
      bestValidationLoss = initialValidation,
      initialValidationLoss = initialValidation,
      randomState = config.batchSeed,
      optimizer = AdamWSnapshot.zero(model.parameters),
      parameterValues = model.parameters.map(_.values)
    )

  /** Runs `updates` optimizer updates starting from a captured state.
    *
    * The state is restored into the supplied model before the first update,
    * so resuming into a freshly constructed (even differently initialized)
    * model of the same architecture is exact. The loop body is the Chapter
    * 22b algorithm — replacement-sampled batches, weighted microbatch
    * accumulation, scheduled learning rates, interval validation — driven by
    * a SplitMix64 stream whose position travels inside the state.
    *
    * Exactness contract, verified bitwise by the test suite: for any split
    * point `k`, `train(_, s0, n)` produces the same metrics and final
    * parameters as `train(_, s0, k)` followed by `train(_, sk, n - k)`.
    *
    * @return
    *   metrics for the executed chunk and the state after its last update.
    */
  def train(
      model: MiniGpt,
      split: CausalSplit,
      config: MiniGptTrainingConfig,
      state: MiniGptTrainingState,
      updates: Int
  ): (Vector[TrainingStepMetrics], MiniGptTrainingState) =
    require(updates > 0, s"update count must be positive: $updates")
    require(
      state.completedUpdates + updates <= config.totalUpdates,
      s"resuming ${state.completedUpdates} + $updates updates exceeds total ${config.totalUpdates}"
    )
    MiniGptTraining.validateDatasets(model, split.training, split.validation)
    val parameters = model.parameters
    restoreParameters(parameters, state)

    val optimizer = new AdamW(
      learningRate = config.learningRateSchedule.learningRate(0, config.totalUpdates),
      beta1 = config.optimizer.beta1,
      beta2 = config.optimizer.beta2,
      epsilon = config.optimizer.epsilon,
      weightDecay = config.optimizer.weightDecay,
      maximumGradientNorm = config.optimizer.maximumGradientNorm
    )
    optimizer.restore(parameters, state.optimizer)
    val random = SplitMix64.fromState(state.randomState)

    var bestValidationLoss = state.bestValidationLoss
    var bestValidationUpdate = state.bestValidationUpdate
    var tokensSeen = state.tokensSeen
    val metrics = Vector.newBuilder[TrainingStepMetrics]

    var updateIndex = state.completedUpdates
    val endIndex = state.completedUpdates + updates
    while updateIndex < endIndex do
      parameters.foreach(_.clearGradients())
      val batch = split.training.sampleBatch(config.batchSize, random).fold(
        problem => throw new IllegalStateException(problem),
        identity
      )
      val microBatches = batch.examples.grouped(config.microBatchSize).toVector
      var weightedTrainingLoss = 0.0
      microBatches.foreach { examples =>
        val microBatch = CausalBatch(examples.toVector)
        val loss = MiniGptTraining.meanLoss(model, microBatch)
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
            MiniGptTraining.validationLoss(
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
      metrics += TrainingStepMetrics(
        completedUpdate,
        learningRate,
        weightedTrainingLoss / config.batchSize.toDouble,
        currentValidation,
        optimizerStats.gradientNorm,
        optimizerStats.gradientScale,
        tokensSeen
      )
      updateIndex += 1

    val nextState = MiniGptTrainingState(
      completedUpdates = endIndex,
      tokensSeen = tokensSeen,
      bestValidationUpdate = bestValidationUpdate,
      bestValidationLoss = bestValidationLoss,
      initialValidationLoss = state.initialValidationLoss,
      randomState = random.state,
      optimizer = optimizer.snapshot(parameters),
      parameterValues = parameters.map(_.values)
    )
    (metrics.result(), nextState)

  /** Runs a complete training run in one chunk and assembles the summary.
    *
    * This is the resumable counterpart of `MiniGptTraining.train`. The two
    * trainers intentionally use different random generators, so their batch
    * sequences — and therefore their metrics — are not comparable run to
    * run; each is deterministic under its own contract.
    */
  def trainFromStart(
      model: MiniGpt,
      split: CausalSplit,
      config: MiniGptTrainingConfig
  ): (MiniGptTrainingRun, MiniGptTrainingState) =
    val initial = freshState(model, split, config)
    val (metrics, finalState) = train(model, split, config, initial, config.totalUpdates)
    val run = MiniGptTrainingRun(
      config,
      initial.initialValidationLoss,
      metrics,
      finalState.bestValidationUpdate,
      finalState.bestValidationLoss
    )
    (run, finalState)

  /** Writes checkpointed values back into the model's parameter tensors. */
  private def restoreParameters(
      parameters: Vector[Tensor],
      state: MiniGptTrainingState
  ): Unit =
    require(
      state.parameterValues.size == parameters.size,
      s"state carries ${state.parameterValues.size} parameter tensors, " +
        s"model owns ${parameters.size}"
    )
    parameters.zip(state.parameterValues).foreach { case (parameter, values) =>
      parameter.assignParameterValues(values)
    }
