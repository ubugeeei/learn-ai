package learnai.training

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

import scala.util.control.NonFatal

import learnai.data.CausalDataset
import learnai.diagnostics.RuntimeFingerprint
import learnai.experiment.CorpusFingerprint
import learnai.experiment.ExperimentManifest
import learnai.experiment.ExperimentSpecification
import learnai.io.MiniGptCheckpoint
import learnai.io.TrainingBundle
import learnai.json.JsonNull
import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.text.ByteTokenizer
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

/** User-facing configuration for a complete local training workflow. */
final case class TrainingWorkflowConfig(
    input: Path,
    output: Path,
    contextLength: Int = 32,
    channels: Int = 32,
    heads: Int = 4,
    hiddenChannels: Int = 64,
    layers: Int = 2,
    updates: Int = 100,
    batchSize: Int = 8,
    microBatchSize: Int = 2,
    learningRate: Double = 0.003,
    modelSeed: Long = 1L,
    batchSeed: Long = 2L,
    trainingFraction: Double = 0.9,
    codeRevision: String = "working-tree",
    environmentRevision: String = "local"
)

/** Durable run identity and summary returned after every artifact verifies. */
final case class TrainingWorkflowResult(
    experimentId: String,
    output: Path,
    completedUpdates: Int,
    tokensSeen: Long,
    bestValidationLoss: Double
)

/**
 * Connects the educational components into a file-to-artifacts workflow.
 *
 * The command consumes a real UTF-8 corpus and writes four independently inspectable artifacts: an
 * experiment manifest, per-update JSONL metrics, an inference checkpoint, and a complete
 * exact-resume training bundle.
 */
object TrainingWorkflow:
  /**
   * Trains one model from a UTF-8 file and persists a complete run directory.
   *
   * @return
   *   a verified run summary, or a contextual error without a partial success
   */
  def run(config: TrainingWorkflowConfig): Either[String, TrainingWorkflowResult] =
    try
      validate(config)
      val corpus = Files.readString(config.input, StandardCharsets.UTF_8)
      require(corpus.nonEmpty, s"input corpus is empty: ${config.input}")
      val tokens = ByteTokenizer.encode(corpus, addBeginOfText = true, addEndOfText = true)
      val split  = CausalDataset
        .contiguousSplit(tokens, config.trainingFraction, config.contextLength)
      require(!split.training.isEmpty, "training split contains no full context windows")
      require(!split.validation.isEmpty, "validation split contains no full context windows")

      val modelConfig    = MiniGptConfig(
        ByteTokenizer.VocabularySize,
        config.contextLength,
        config.channels,
        config.heads,
        config.hiddenChannels,
        config.layers
      )
      val trainingConfig = MiniGptTrainingConfig(
        totalUpdates = config.updates,
        batchSize = config.batchSize,
        microBatchSize = config.microBatchSize,
        validationEveryUpdates = math.max(1, config.updates / 10),
        maximumValidationBatches = 32,
        batchSeed = config.batchSeed,
        learningRateSchedule = WarmupCosineLearningRate(
          peak = config.learningRate,
          minimum = config.learningRate * 0.1,
          warmupUpdates = math.min(math.max(1, config.updates / 20), config.updates)
        )
      )
      val manifest       = ExperimentManifest(
        ExperimentSpecification(
          name = config.input.getFileName.toString,
          modelSeed = config.modelSeed,
          model = modelConfig,
          training = trainingConfig,
          corpus = CorpusFingerprint.fromText(
            config.input.getFileName.toString,
            corpus,
            tokens.size.toLong,
            split.training.size.toLong,
            split.validation.size.toLong
          ),
          codeRevision = config.codeRevision,
          environmentRevision = config.environmentRevision
        ),
        RuntimeFingerprint.current
      )

      Files.createDirectories(config.output)
      val model             = MiniGpt.random(modelConfig, config.modelSeed)
      val (run, finalState) = ResumableMiniGptTraining.trainFromStart(model, split, trainingConfig)
      writeText(config.output.resolve("manifest.json"), manifest.render + "\n")
      writeText(
        config.output.resolve("metrics.jsonl"),
        run.steps.iterator.map(metricJson).mkString("", "\n", "\n")
      )
      MiniGptCheckpoint.save(model, config.output.resolve("model.laigpt"))
        .fold(problem => throw new IllegalStateException(problem), identity)
      TrainingBundle
        .save(finalState, manifest.experimentId, config.output.resolve("training.laibnd"))
        .fold(problem => throw new IllegalStateException(problem), identity)
      Right(TrainingWorkflowResult(
        manifest.experimentId,
        config.output.toAbsolutePath,
        finalState.completedUpdates,
        finalState.tokensSeen,
        finalState.bestValidationLoss
      ))
    catch case NonFatal(error) => Left(s"training workflow failed: ${error.getMessage}")

  private def validate(config: TrainingWorkflowConfig): Unit =
    require(Files.isRegularFile(config.input), s"input is not a regular file: ${config.input}")
    require(config.contextLength > 0, "context length must be positive")
    require(config.channels > 0, "channels must be positive")
    require(config.heads > 0 && config.channels     % config.heads == 0, "heads must divide channels")
    require(config.hiddenChannels > 0, "hidden channels must be positive")
    require(config.layers > 0, "layers must be positive")
    require(config.updates > 0, "updates must be positive")
    require(config.batchSize > 0, "batch size must be positive")
    require(
      config.microBatchSize > 0 && config.batchSize % config.microBatchSize == 0,
      "microbatch size must divide batch size"
    )
    require(
      config.learningRate > 0.0 && config.learningRate.isFinite,
      "learning rate must be positive"
    )
    require(
      config.trainingFraction > 0.0 && config.trainingFraction < 1.0,
      "training fraction must be between zero and one"
    )

  private def metricJson(metric: TrainingStepMetrics): String = JsonObject(
    "update"          -> JsonNumber(metric.update),
    "learning_rate"   -> JsonNumber(BigDecimal.decimal(metric.learningRate)),
    "training_loss"   -> JsonNumber(BigDecimal.decimal(metric.trainingLoss)),
    "validation_loss" -> metric.validationLoss.map(value => JsonNumber(BigDecimal.decimal(value)))
      .getOrElse(JsonNull),
    "gradient_norm"   -> JsonNumber(BigDecimal.decimal(metric.gradientNorm)),
    "gradient_scale"  -> JsonNumber(BigDecimal.decimal(metric.gradientScale)),
    "tokens_seen"     -> JsonNumber(metric.tokensSeen)
  ).render

  private def writeText(path: Path, content: String): Unit =
    val _ = Files.writeString(path, content, StandardCharsets.UTF_8)

/** Strict parser for the `train` command. Unknown and duplicate options fail. */
object TrainingCommand:
  /** Human-readable contract for every supported training option. */
  val help: String =
    """Usage: learnai.Main train --input <corpus.txt> --output <directory> [options]
      |
      |Options:
      |  --context <n>       context length (default: 32)
      |  --channels <n>      embedding width (default: 32)
      |  --heads <n>         attention heads (default: 4)
      |  --hidden <n>        feed-forward width (default: 64)
      |  --layers <n>        Transformer blocks (default: 2)
      |  --updates <n>       optimizer updates (default: 100)
      |  --batch-size <n>    examples per update (default: 8)
      |  --microbatch <n>    examples per backward pass (default: 2)
      |  --learning-rate <x> peak learning rate (default: 0.003)
      |  --model-seed <n>    parameter seed (default: 1)
      |  --batch-seed <n>    data-order seed (default: 2)
      |""".stripMargin

  /** Parses strict name/value pairs into a validated workflow configuration. */
  def parse(arguments: List[String]): Either[String, TrainingWorkflowConfig] = collect(arguments)
    .flatMap { options =>
      for
        input        <- required(options, "--input").map(Path.of(_))
        output       <- required(options, "--output").map(Path.of(_))
        context      <- integer(options, "--context", 32)
        channels     <- integer(options, "--channels", 32)
        heads        <- integer(options, "--heads", 4)
        hidden       <- integer(options, "--hidden", 64)
        layers       <- integer(options, "--layers", 2)
        updates      <- integer(options, "--updates", 100)
        batchSize    <- integer(options, "--batch-size", 8)
        microbatch   <- integer(options, "--microbatch", 2)
        learningRate <- decimal(options, "--learning-rate", 0.003)
        modelSeed    <- long(options, "--model-seed", 1L)
        batchSeed    <- long(options, "--batch-seed", 2L)
      yield TrainingWorkflowConfig(
        input,
        output,
        context,
        channels,
        heads,
        hidden,
        layers,
        updates,
        batchSize,
        microbatch,
        learningRate,
        modelSeed,
        batchSeed
      )
    }

  private def collect(arguments: List[String]): Either[String, Map[String, String]] =
    val known = Set(
      "--input",
      "--output",
      "--context",
      "--channels",
      "--heads",
      "--hidden",
      "--layers",
      "--updates",
      "--batch-size",
      "--microbatch",
      "--learning-rate",
      "--model-seed",
      "--batch-seed"
    )
    arguments.grouped(2).foldLeft[Either[String, Map[String, String]]](Right(Map.empty)) {
      case (result, List(name, value)) => result.flatMap { values =>
          if !known.contains(name) then Left(s"unknown option: $name")
          else if values.contains(name) then Left(s"duplicate option: $name")
          else Right(values.updated(name, value))
        }
      case (_, remainder)              => Left(s"option requires a value: ${remainder.mkString(" ")}")
    }

  private def required(values: Map[String, String], name: String): Either[String, String] = values
    .get(name).toRight(s"missing required option: $name")

  private def integer(
      values: Map[String, String],
      name: String,
      default: Int
  ): Either[String, Int] = parseNumber(values.getOrElse(name, default.toString), name)(_.toInt)

  private def long(values: Map[String, String], name: String, default: Long): Either[String, Long] =
    parseNumber(values.getOrElse(name, default.toString), name)(_.toLong)

  private def decimal(
      values: Map[String, String],
      name: String,
      default: Double
  ): Either[String, Double] =
    parseNumber(values.getOrElse(name, default.toString), name)(_.toDouble)

  private def parseNumber[A](raw: String, name: String)(parse: String => A): Either[String, A] =
    try Right(parse(raw))
    catch case _: NumberFormatException => Left(s"$name expects a number, got '$raw'")
