package learnai.training

import learnai.data.CausalDataset
import learnai.diagnostics.RuntimeFingerprint
import learnai.experiment.CorpusFingerprint
import learnai.experiment.ExperimentManifest
import learnai.experiment.ExperimentSpecification
import learnai.text.ByteTokenizer
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

/** Runs deterministic batch training with validation, accumulation, scheduling, and metrics. */
@main def runMiniGptTrainingLab(): Unit =
  val corpus = Vector.fill(40)("to be or not to be. ").mkString
  val tokens = ByteTokenizer.encode(corpus, addBeginOfText = true, addEndOfText = true)
  val contextLength = 8
  val split = CausalDataset.contiguousSplit(tokens, trainingFraction = 0.8, contextLength)
  val model = MiniGpt.random(
    MiniGptConfig(
      vocabularySize = ByteTokenizer.VocabularySize,
      maximumContextLength = contextLength,
      channels = 8,
      headCount = 2,
      hiddenChannels = 16,
      layerCount = 1
    ),
    seed = 123L
  )
  val config = MiniGptTrainingConfig(
    totalUpdates = 40,
    batchSize = 4,
    microBatchSize = 2,
    validationEveryUpdates = 5,
    maximumValidationBatches = 4,
    batchSeed = 456L,
    learningRateSchedule = WarmupCosineLearningRate(
      peak = 0.02,
      minimum = 0.002,
      warmupUpdates = 5
    ),
    optimizer = AdamWTrainingConfig(weightDecay = 0.0, maximumGradientNorm = Some(1.0))
  )
  val manifest = ExperimentManifest(
    ExperimentSpecification(
      name = "byte-level-mini-gpt-training-lab",
      modelSeed = 123L,
      model = model.config,
      training = config,
      corpus = CorpusFingerprint.fromText(
        "repeated-shakespeare-fragment",
        corpus,
        tokenCount = tokens.size.toLong,
        trainingExamples = split.training.size.toLong,
        validationExamples = split.validation.size.toLong
      ),
      codeRevision = sys.props.getOrElse("learnai.codeRevision", "unrecorded"),
      environmentRevision = sys.props.getOrElse("learnai.environmentRevision", "unrecorded")
    ),
    RuntimeFingerprint.current
  )
  val run = MiniGptTraining.train(model, split, config)

  println(s"experiment ID:      ${manifest.experimentId}")
  println(s"training examples:   ${split.training.size}")
  println(s"validation examples: ${split.validation.size}")
  println(f"initial validation:  ${run.initialValidationLoss}%.6f")
  run.steps.foreach { step =>
    step.validationLoss.foreach { validation =>
      println(
        f"update=${step.update}%3d tokens=${step.tokensSeen}%5d " +
          f"lr=${step.learningRate}%.6f train=${step.trainingLoss}%.6f " +
          f"validation=$validation%.6f gradient=${step.gradientNorm}%.4f " +
          f"clip=${step.gradientScale}%.4f"
      )
    }
  }
  println(f"best validation:     ${run.bestValidationLoss}%.6f at update ${run.bestValidationUpdate}%d")
  println(f"final validation:    ${run.finalValidationLoss}%.6f")
  println(s"total tokens seen:   ${run.totalTokensSeen}")
