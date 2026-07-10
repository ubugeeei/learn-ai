package learnai.experiment

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat

import learnai.diagnostics.RuntimeFingerprint
import learnai.json.JsonNull
import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.training.AdamWTrainingConfig
import learnai.training.ConstantLearningRate
import learnai.training.LearningRateSchedule
import learnai.training.MiniGptTrainingConfig
import learnai.training.WarmupCosineLearningRate
import learnai.transformer.MiniGptConfig

/** Stable identity and content summary for the token corpus used by an experiment. */
final case class CorpusFingerprint(
    name: String,
    sha256: String,
    tokenCount: Long,
    trainingExamples: Long,
    validationExamples: Long
):
  require(name.nonEmpty, "corpus name cannot be empty")
  require(
    sha256.matches("[0-9a-f]{64}"),
    s"corpus SHA-256 must contain 64 lowercase hexadecimal characters: $sha256"
  )
  require(tokenCount >= 0L, s"token count must be non-negative: $tokenCount")
  require(trainingExamples >= 0L, s"training example count must be non-negative: $trainingExamples")
  require(
    validationExamples >= 0L,
    s"validation example count must be non-negative: $validationExamples"
  )

object CorpusFingerprint:
  /** Hashes exact UTF-8 source text; production corpus builders hash immutable shard bytes. */
  def fromText(
      name: String,
      text: String,
      tokenCount: Long,
      trainingExamples: Long,
      validationExamples: Long
  ): CorpusFingerprint =
    CorpusFingerprint(
      name,
      Sha256.hex(text.getBytes(StandardCharsets.UTF_8)),
      tokenCount,
      trainingExamples,
      validationExamples
    )

/** Logical experiment inputs that must remain stable across repeated executions.
  *
  * Runtime measurements are deliberately excluded. `experimentId` identifies
  * the intended model/data/training/code/environment configuration, while an
  * `ExperimentManifest` records where one execution happened.
  */
final case class ExperimentSpecification(
    name: String,
    modelSeed: Long,
    model: MiniGptConfig,
    training: MiniGptTrainingConfig,
    corpus: CorpusFingerprint,
    codeRevision: String,
    environmentRevision: String
):
  require(name.nonEmpty, "experiment name cannot be empty")
  require(codeRevision.nonEmpty, "code revision cannot be empty")
  require(environmentRevision.nonEmpty, "environment revision cannot be empty")

  /** Canonical insertion-ordered JSON used as the experiment identity payload. */
  val canonicalJson: JsonObject = ExperimentJson.specification(this)
  val experimentId: String = Sha256.hex(canonicalJson.render.getBytes(StandardCharsets.UTF_8))

/** One execution manifest joining logical identity with an observed runtime. */
final case class ExperimentManifest(
    specification: ExperimentSpecification,
    runtime: RuntimeFingerprint
):
  val experimentId: String = specification.experimentId

  /** Deterministic JSON suitable for logging or a future atomic artifact writer. */
  val json: JsonObject = JsonObject(
    "schema_version" -> JsonNumber(1),
    "experiment_id" -> JsonString(experimentId),
    "specification" -> specification.canonicalJson,
    "runtime" -> ExperimentJson.runtime(runtime)
  )

  def render: String = json.render

private object ExperimentJson:
  def specification(value: ExperimentSpecification): JsonObject = JsonObject(
    "name" -> JsonString(value.name),
    "model_seed" -> JsonNumber(value.modelSeed),
    "model" -> model(value.model),
    "training" -> training(value.training),
    "corpus" -> corpus(value.corpus),
    "code_revision" -> JsonString(value.codeRevision),
    "environment_revision" -> JsonString(value.environmentRevision)
  )

  def runtime(value: RuntimeFingerprint): JsonObject = JsonObject(
    "java_runtime_version" -> JsonString(value.javaRuntimeVersion),
    "java_vm_name" -> JsonString(value.javaVmName),
    "java_vm_version" -> JsonString(value.javaVmVersion),
    "operating_system" -> JsonString(value.operatingSystem),
    "architecture" -> JsonString(value.architecture),
    "available_processors" -> JsonNumber(value.availableProcessors)
  )

  private def model(value: MiniGptConfig): JsonObject = JsonObject(
    "vocabulary_size" -> JsonNumber(value.vocabularySize),
    "maximum_context_length" -> JsonNumber(value.maximumContextLength),
    "channels" -> JsonNumber(value.channels),
    "head_count" -> JsonNumber(value.headCount),
    "hidden_channels" -> JsonNumber(value.hiddenChannels),
    "layer_count" -> JsonNumber(value.layerCount),
    "normalization_epsilon" -> decimal(value.normalizationEpsilon)
  )

  private def training(value: MiniGptTrainingConfig): JsonObject = JsonObject(
    "total_updates" -> JsonNumber(value.totalUpdates),
    "batch_size" -> JsonNumber(value.batchSize),
    "microbatch_size" -> JsonNumber(value.microBatchSize),
    "validation_every_updates" -> JsonNumber(value.validationEveryUpdates),
    "maximum_validation_batches" -> JsonNumber(value.maximumValidationBatches),
    "batch_seed" -> JsonNumber(value.batchSeed),
    "learning_rate_schedule" -> learningRate(value.learningRateSchedule),
    "optimizer" -> optimizer(value.optimizer)
  )

  private def learningRate(value: LearningRateSchedule): JsonObject =
    value match
      case ConstantLearningRate(rate) =>
        JsonObject(
          "kind" -> JsonString("constant"),
          "value" -> decimal(rate)
        )
      case WarmupCosineLearningRate(peak, minimum, warmupUpdates) =>
        JsonObject(
          "kind" -> JsonString("warmup_cosine"),
          "peak" -> decimal(peak),
          "minimum" -> decimal(minimum),
          "warmup_updates" -> JsonNumber(warmupUpdates)
        )

  private def optimizer(value: AdamWTrainingConfig): JsonObject = JsonObject(
    "kind" -> JsonString("adamw"),
    "beta1" -> decimal(value.beta1),
    "beta2" -> decimal(value.beta2),
    "epsilon" -> decimal(value.epsilon),
    "weight_decay" -> decimal(value.weightDecay),
    "maximum_gradient_norm" -> value.maximumGradientNorm
      .map(maximum => decimal(maximum))
      .getOrElse(JsonNull)
  )

  private def corpus(value: CorpusFingerprint): JsonObject = JsonObject(
    "name" -> JsonString(value.name),
    "sha256" -> JsonString(value.sha256),
    "token_count" -> JsonNumber(value.tokenCount),
    "training_examples" -> JsonNumber(value.trainingExamples),
    "validation_examples" -> JsonNumber(value.validationExamples)
  )

  private def decimal(value: Double): JsonNumber =
    JsonNumber(BigDecimal.decimal(value))

private object Sha256:
  def hex(bytes: Array[Byte]): String =
    HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
