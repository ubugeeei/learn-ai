package learnai.experiment

import learnai.diagnostics.RuntimeFingerprint
import learnai.json.JsonParser
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.training.AdamWTrainingConfig
import learnai.training.MiniGptTrainingConfig
import learnai.training.WarmupCosineLearningRate
import learnai.transformer.MiniGptConfig

object ExperimentManifestSuite extends TestSuite:
  override val name: String = "ExperimentManifest"

  override val tests: Vector[TestCase] = specify(
    test("text corpus fingerprint uses exact UTF-8 SHA-256 bytes") {
      val fingerprint = CorpusFingerprint.fromText("fixture", "abc", 3L, 2L, 1L)
      Assert.equal(
        fingerprint.sha256,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
      )
      Assert.equal(fingerprint.tokenCount, 3L)
    },
    test("equal logical specifications have equal canonical JSON and identity") {
      val first = specification()
      val second = specification()
      Assert.equal(first.canonicalJson, second.canonicalJson)
      Assert.equal(first.experimentId, second.experimentId)
      Assert.equal(JsonParser.parse(first.canonicalJson.render), Right(first.canonicalJson))
    },
    test("changing seed data code environment or training configuration changes identity") {
      val original = specification()
      val variants = Vector(
        original.copy(modelSeed = 8L),
        original.copy(corpus = original.corpus.copy(tokenCount = 101L)),
        original.copy(codeRevision = "def456"),
        original.copy(environmentRevision = "flake-2"),
        original.copy(training = original.training.copy(batchSize = 8, microBatchSize = 2))
      )
      Assert.equal(variants.map(_.experimentId).distinct.size, variants.size)
      Assert.isTrue(variants.forall(_.experimentId != original.experimentId))
    },
    test("runtime fingerprint changes execution manifest but not logical experiment identity") {
      val spec = specification()
      val first = ExperimentManifest(spec, runtime("vm-a"))
      val second = ExperimentManifest(spec, runtime("vm-b"))
      Assert.equal(first.experimentId, second.experimentId)
      Assert.isTrue(first.render != second.render)
      Assert.equal(JsonParser.parse(first.render), Right(first.json))
    },
    test("manifest rendering is stable and names every reproducibility boundary") {
      val manifest = ExperimentManifest(specification(), runtime("vm"))
      val renderedAgain = ExperimentManifest(specification(), runtime("vm")).render
      Assert.equal(manifest.render, renderedAgain)
      Vector(
        "schema_version",
        "experiment_id",
        "model_seed",
        "learning_rate_schedule",
        "batch_seed",
        "sha256",
        "code_revision",
        "environment_revision",
        "java_runtime_version"
      ).foreach(field => Assert.isTrue(manifest.render.contains(s"\"$field\""), s"missing $field"))
    },
    test("invalid corpus hashes and missing revisions fail before a manifest exists") {
      val hashError = Assert.throws[IllegalArgumentException] {
        CorpusFingerprint("bad", "ABC", 0L, 0L, 0L)
      }
      val revisionError = Assert.throws[IllegalArgumentException] {
        specification().copy(codeRevision = "")
      }
      Assert.isTrue(hashError.getMessage.contains("64 lowercase"))
      Assert.isTrue(revisionError.getMessage.contains("code revision"))
    }
  )

  private def specification(): ExperimentSpecification =
    ExperimentSpecification(
      name = "fixture-run",
      modelSeed = 7L,
      model = MiniGptConfig(4, 3, 4, 2, 8, 1),
      training = MiniGptTrainingConfig(
        totalUpdates = 10,
        batchSize = 4,
        microBatchSize = 2,
        validationEveryUpdates = 2,
        maximumValidationBatches = 3,
        batchSeed = 11L,
        learningRateSchedule = WarmupCosineLearningRate(0.02, 0.001, 2),
        optimizer = AdamWTrainingConfig(weightDecay = 0.0)
      ),
      corpus = CorpusFingerprint.fromText("fixture", "0 1 2 3", 100L, 70L, 20L),
      codeRevision = "abc123",
      environmentRevision = "flake-1"
    )

  private def runtime(vm: String): RuntimeFingerprint =
    RuntimeFingerprint("java-21", vm, "1", "test-os", "test-arch", 2)
