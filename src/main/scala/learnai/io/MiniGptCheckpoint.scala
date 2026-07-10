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

import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

/** Metadata returned after a checkpoint is verified or written. */
final case class CheckpointMetadata(
    formatVersion: Int,
    parameterTensors: Int,
    scalarParameters: Int,
    sha256: String
)

/** Versioned, checksummed MiniGPT inference checkpoint I/O.
  *
  * The format stores architecture config, stable parameter labels, shapes, and
  * row-major `Double` values. It intentionally excludes tokenizer artifacts,
  * optimizer state, RNG state, and data-loader state; those are required for a
  * resumable training bundle but not for this inference checkpoint.
  */
object MiniGptCheckpoint:
  private val Magic = "LAIGPT01".getBytes(StandardCharsets.US_ASCII)
  private val FormatVersion = 1
  private val ChecksumBytes = 32
  private val MaximumFileBytes = 128L * 1024L * 1024L
  private val MaximumScalarParameters = 10_000_000L
  private val MaximumLabelBytes = 16 * 1024

  /** Serializes and atomically replaces `path` after building a full checksum. */
  def save(model: MiniGpt, path: Path): Either[String, CheckpointMetadata] =
    try
      val parameters = model.parameters
      require(
        parameters.map(_.label).distinct.size == parameters.size,
        "checkpoint parameter labels must be unique"
      )
      val payloadBuffer = new ByteArrayOutputStream()
      val output = new DataOutputStream(payloadBuffer)
      output.write(Magic)
      output.writeInt(FormatVersion)
      writeConfig(output, model.config)
      output.writeInt(parameters.size)
      parameters.foreach { parameter =>
        writeString(output, parameter.label)
        output.writeInt(parameter.shape.rank)
        parameter.shape.dimensions.foreach(output.writeInt)
        output.writeInt(parameter.size)
        parameter.values.foreach(output.writeDouble)
      }
      output.flush()
      val payload = payloadBuffer.toByteArray
      val checksum = sha256(payload)
      val complete = payload ++ checksum
      require(complete.length <= MaximumFileBytes, s"checkpoint exceeds $MaximumFileBytes bytes")

      val absolute = path.toAbsolutePath
      val parent = absolute.getParent
      Files.createDirectories(parent)
      val temporary = Files.createTempFile(parent, s".${absolute.getFileName}.", ".tmp")
      try
        Files.write(temporary, complete)
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

      Right(
        CheckpointMetadata(
          FormatVersion,
          parameters.size,
          model.parameterCount,
          hex(checksum)
        )
      )
    catch
      case NonFatal(error) => Left(s"could not save checkpoint: ${error.getMessage}")

  /** Loads a verified checkpoint into a newly constructed MiniGPT model. */
  def load(path: Path): Either[String, (MiniGpt, CheckpointMetadata)] =
    try
      val fileSize = Files.size(path)
      require(fileSize > ChecksumBytes, "checkpoint is too short")
      require(fileSize <= MaximumFileBytes, s"checkpoint exceeds $MaximumFileBytes bytes")
      val complete = Files.readAllBytes(path)
      val payload = complete.take(complete.length - ChecksumBytes)
      val storedChecksum = complete.takeRight(ChecksumBytes)
      val computedChecksum = sha256(payload)
      require(
        MessageDigest.isEqual(storedChecksum, computedChecksum),
        "checkpoint SHA-256 mismatch"
      )

      val input = new DataInputStream(new ByteArrayInputStream(payload))
      val magic = input.readNBytes(Magic.length)
      require(java.util.Arrays.equals(magic, Magic), "unknown checkpoint magic")
      val version = input.readInt()
      require(version == FormatVersion, s"unsupported checkpoint version $version")
      val config = readConfig(input)
      validateConfigSize(config)
      val model = MiniGpt.random(config, seed = 0L)
      val expected = model.parameters
      val tensorCount = input.readInt()
      require(tensorCount == expected.size, s"parameter tensor count $tensorCount != ${expected.size}")

      expected.foreach { parameter =>
        val label = readString(input)
        require(label == parameter.label, s"parameter label '$label' != '${parameter.label}'")
        val rank = input.readInt()
        require(rank >= 0 && rank <= 8, s"invalid parameter rank $rank for '$label'")
        val dimensions = Vector.fill(rank)(input.readInt())
        require(
          dimensions == parameter.shape.dimensions,
          s"parameter '$label' shape $dimensions != ${parameter.shape.dimensions}"
        )
        val scalarCount = input.readInt()
        require(scalarCount == parameter.size, s"parameter '$label' size $scalarCount != ${parameter.size}")
        val values = Vector.fill(scalarCount)(input.readDouble())
        parameter.assignParameterValues(values)
      }
      require(input.available() == 0, s"checkpoint payload has ${input.available()} trailing bytes")

      Right(
        model -> CheckpointMetadata(
          version,
          expected.size,
          model.parameterCount,
          hex(computedChecksum)
        )
      )
    catch
      case NonFatal(error) => Left(s"could not load checkpoint: ${error.getMessage}")

  private def writeConfig(output: DataOutputStream, config: MiniGptConfig): Unit =
    output.writeInt(config.vocabularySize)
    output.writeInt(config.maximumContextLength)
    output.writeInt(config.channels)
    output.writeInt(config.headCount)
    output.writeInt(config.hiddenChannels)
    output.writeInt(config.layerCount)
    output.writeDouble(config.normalizationEpsilon)

  private def readConfig(input: DataInputStream): MiniGptConfig =
    MiniGptConfig(
      vocabularySize = input.readInt(),
      maximumContextLength = input.readInt(),
      channels = input.readInt(),
      headCount = input.readInt(),
      hiddenChannels = input.readInt(),
      layerCount = input.readInt(),
      normalizationEpsilon = input.readDouble()
    )

  private def validateConfigSize(config: MiniGptConfig): Unit =
    val vocabularyEmbedding = config.vocabularySize.toLong * config.channels
    val positionEmbedding = config.maximumContextLength.toLong * config.channels
    val attention = 4L * (config.channels.toLong * config.channels + config.channels)
    val norms = 2L * config.channels
    val feedForward =
      config.channels.toLong * config.hiddenChannels + config.hiddenChannels +
        config.hiddenChannels.toLong * config.channels + config.channels
    val blocks = config.layerCount.toLong * (attention + norms + feedForward)
    val total = vocabularyEmbedding + positionEmbedding + blocks + config.channels
    require(
      total > 0L && total <= MaximumScalarParameters,
      s"checkpoint config requests $total scalars; limit is $MaximumScalarParameters"
    )

  private def writeString(output: DataOutputStream, value: String): Unit =
    val bytes = value.getBytes(StandardCharsets.UTF_8)
    require(bytes.length <= MaximumLabelBytes, s"label exceeds $MaximumLabelBytes bytes")
    output.writeInt(bytes.length)
    output.write(bytes)

  private def readString(input: DataInputStream): String =
    val length = input.readInt()
    require(length >= 0 && length <= MaximumLabelBytes, s"invalid label byte length $length")
    val bytes = input.readNBytes(length)
    require(bytes.length == length, s"truncated label: expected $length bytes, got ${bytes.length}")
    new String(bytes, StandardCharsets.UTF_8)

  private def sha256(bytes: Array[Byte]): Array[Byte] =
    MessageDigest.getInstance("SHA-256").digest(bytes)

  private def hex(bytes: Array[Byte]): String =
    bytes.iterator.map(byte => f"${byte & 0xff}%02x").mkString
