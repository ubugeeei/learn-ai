package learnai.io

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

import scala.util.control.NonFatal

import learnai.optim.AdamWSnapshot
import learnai.training.MiniGptTrainingState

/** Metadata returned after a training bundle is written or verified. */
final case class TrainingBundleMetadata(
    formatVersion: Int,
    experimentId: String,
    completedUpdates: Int,
    sha256: String
)

/** A verified bundle: the experiment it belongs to plus the captured state. */
final case class LoadedTrainingBundle(
    experimentId: String,
    state: MiniGptTrainingState,
    metadata: TrainingBundleMetadata
)

/** Versioned, checksummed persistence for Chapter 22c training state.
  *
  * The bundle stores the *complete* `MiniGptTrainingState` — parameter
  * values, AdamW moments and step, schedule position, and the SplitMix64
  * counter — plus the Chapter 22a `experimentId` that identifies the
  * model/data/training configuration the state belongs to. Resuming under a
  * different configuration is refused by identity comparison rather than
  * hoped away: the caller states which experiment it is about to continue,
  * and a mismatched bundle is an error, not a warning.
  *
  * All floating-point values travel through `DataOutputStream.writeDouble`,
  * which writes exact IEEE 754 bits. Decimal text formats are not an
  * acceptable substitute here: a value that rounds through text breaks the
  * bitwise resume contract this format exists to preserve.
  *
  * The layout follows Chapter 25's checkpoint discipline: magic, version,
  * validated payload, and a trailing SHA-256 over everything before it,
  * written atomically via a temporary file.
  */
object TrainingBundle:
  private val Magic = "LAIBND01".getBytes(StandardCharsets.US_ASCII)
  private val FormatVersion = 1
  private val ChecksumBytes = 32
  private val MaximumFileBytes = 256L * 1024L * 1024L
  private val MaximumTensorCount = 100_000
  private val MaximumScalarsPerTensor = 10_000_000
  private val ExperimentIdPattern = "[0-9a-f]{64}"

  /** Serializes the state and atomically replaces `path`. */
  def save(
      state: MiniGptTrainingState,
      experimentId: String,
      path: Path
  ): Either[String, TrainingBundleMetadata] =
    try
      require(
        experimentId.matches(ExperimentIdPattern),
        s"experiment id must be 64 lowercase hexadecimal characters: $experimentId"
      )
      val payloadBuffer = new ByteArrayOutputStream()
      val output = new DataOutputStream(payloadBuffer)
      output.write(Magic)
      output.writeInt(FormatVersion)
      writeString(output, experimentId)
      output.writeInt(state.completedUpdates)
      output.writeLong(state.tokensSeen)
      output.writeInt(state.bestValidationUpdate)
      output.writeDouble(state.bestValidationLoss)
      output.writeDouble(state.initialValidationLoss)
      output.writeLong(state.randomState)
      writeValueGroups(output, state.parameterValues)
      output.writeLong(state.optimizer.step)
      writeValueGroups(output, state.optimizer.firstMoments)
      writeValueGroups(output, state.optimizer.secondMoments)
      output.flush()

      val payload = payloadBuffer.toByteArray
      val checksum = sha256(payload)
      val complete = payload ++ checksum
      require(complete.length <= MaximumFileBytes, s"bundle exceeds $MaximumFileBytes bytes")
      writeAtomically(path, complete)

      Right(
        TrainingBundleMetadata(
          FormatVersion,
          experimentId,
          state.completedUpdates,
          hex(checksum)
        )
      )
    catch
      case NonFatal(error) => Left(s"could not save training bundle: ${error.getMessage}")

  /** Loads and fully verifies a bundle without any expectation about identity. */
  def load(path: Path): Either[String, LoadedTrainingBundle] =
    try
      val fileSize = Files.size(path)
      require(fileSize > ChecksumBytes, "training bundle is too short")
      require(fileSize <= MaximumFileBytes, s"bundle exceeds $MaximumFileBytes bytes")
      val complete = Files.readAllBytes(path)
      val payload = complete.take(complete.length - ChecksumBytes)
      val storedChecksum = complete.takeRight(ChecksumBytes)
      val computedChecksum = sha256(payload)
      require(
        MessageDigest.isEqual(storedChecksum, computedChecksum),
        "training bundle SHA-256 mismatch"
      )

      val input = new DataInputStream(new ByteArrayInputStream(payload))
      val magic = input.readNBytes(Magic.length)
      require(java.util.Arrays.equals(magic, Magic), "unknown training bundle magic")
      val version = input.readInt()
      require(version == FormatVersion, s"unsupported training bundle version $version")
      val experimentId = readString(input)
      require(
        experimentId.matches(ExperimentIdPattern),
        s"stored experiment id is not a SHA-256 hash: $experimentId"
      )
      val completedUpdates = input.readInt()
      val tokensSeen = input.readLong()
      val bestValidationUpdate = input.readInt()
      val bestValidationLoss = input.readDouble()
      val initialValidationLoss = input.readDouble()
      val randomState = input.readLong()
      val parameterValues = readValueGroups(input)
      val optimizerStep = input.readLong()
      val firstMoments = readValueGroups(input)
      val secondMoments = readValueGroups(input)
      require(input.available() == 0, s"bundle payload has ${input.available()} trailing bytes")

      val state = MiniGptTrainingState(
        completedUpdates = completedUpdates,
        tokensSeen = tokensSeen,
        bestValidationUpdate = bestValidationUpdate,
        bestValidationLoss = bestValidationLoss,
        initialValidationLoss = initialValidationLoss,
        randomState = randomState,
        optimizer = AdamWSnapshot(optimizerStep, firstMoments, secondMoments),
        parameterValues = parameterValues
      )
      Right(
        LoadedTrainingBundle(
          experimentId,
          state,
          TrainingBundleMetadata(version, experimentId, completedUpdates, hex(computedChecksum))
        )
      )
    catch
      case NonFatal(error) => Left(s"could not load training bundle: ${error.getMessage}")

  /** Loads a bundle and refuses it unless it belongs to the expected experiment.
    *
    * This is the resume entry point: the caller derives
    * `expectedExperimentId` from its *current* configuration (Chapter 22a),
    * so any drift in model, data, training, code, or environment identity
    * produces an explicit refusal instead of a silently different run.
    */
  def loadForResume(
      path: Path,
      expectedExperimentId: String
  ): Either[String, MiniGptTrainingState] =
    load(path).flatMap { bundle =>
      if bundle.experimentId == expectedExperimentId then Right(bundle.state)
      else
        Left(
          s"training bundle belongs to experiment ${bundle.experimentId}, " +
            s"but resume expected $expectedExperimentId; refusing to continue"
        )
    }

  private def writeValueGroups(output: DataOutputStream, groups: Vector[Vector[Double]]): Unit =
    require(groups.size <= MaximumTensorCount, s"bundle stores too many tensors: ${groups.size}")
    output.writeInt(groups.size)
    groups.foreach { values =>
      require(
        values.size <= MaximumScalarsPerTensor,
        s"tensor stores too many scalars: ${values.size}"
      )
      output.writeInt(values.size)
      values.foreach(output.writeDouble)
    }

  private def readValueGroups(input: DataInputStream): Vector[Vector[Double]] =
    val groupCount = input.readInt()
    require(
      groupCount >= 0 && groupCount <= MaximumTensorCount,
      s"invalid tensor count $groupCount"
    )
    Vector.fill(groupCount) {
      val scalarCount = input.readInt()
      require(
        scalarCount >= 0 && scalarCount <= MaximumScalarsPerTensor,
        s"invalid scalar count $scalarCount"
      )
      Vector.fill(scalarCount)(input.readDouble())
    }

  private def writeAtomically(path: Path, bytes: Array[Byte]): Unit =
    val absolute = path.toAbsolutePath
    val parent = absolute.getParent
    Files.createDirectories(parent)
    val temporary = Files.createTempFile(parent, s".${absolute.getFileName}.", ".tmp")
    try
      Files.write(temporary, bytes)
      try
        Files.move(
          temporary,
          absolute,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING
        )
      catch
        case _: AtomicMoveNotSupportedException =>
          Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING)
    finally
      val _ = Files.deleteIfExists(temporary)

  private def writeString(output: DataOutputStream, value: String): Unit =
    val bytes = value.getBytes(StandardCharsets.UTF_8)
    output.writeInt(bytes.length)
    output.write(bytes)

  private def readString(input: DataInputStream): String =
    val length = input.readInt()
    require(length >= 0 && length <= 1024, s"invalid string byte length $length")
    val bytes = input.readNBytes(length)
    require(bytes.length == length, s"truncated string: expected $length bytes, got ${bytes.length}")
    new String(bytes, StandardCharsets.UTF_8)

  private def sha256(bytes: Array[Byte]): Array[Byte] =
    MessageDigest.getInstance("SHA-256").digest(bytes)

  private def hex(bytes: Array[Byte]): String =
    bytes.iterator.map(byte => f"${byte & 0xff}%02x").mkString
