package learnai.data

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.time.Instant
import java.util.HexFormat

import learnai.json.JsonArray
import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonString

/** Provenance supplied by the corpus owner rather than guessed from file contents. */
final case class CorpusProvenance(source: String, license: String, collectedAt: Instant):
  require(source.nonEmpty, "corpus source cannot be empty")
  require(license.nonEmpty, "corpus license cannot be empty")

/** One immutable shard's relative name, global byte interval, length, and digest. */
final case class CorpusShard(name: String, startByte: Long, byteLength: Long, sha256: String):
  require(name.matches("shard-[0-9]{5}\\.bin"), s"invalid shard name: $name")
  require(startByte >= 0, s"negative shard start: $startByte")
  require(byteLength > 0, s"non-positive shard length: $byteLength")
  require(sha256.matches("[0-9a-f]{64}"), s"invalid shard SHA-256: $sha256")
  val endByteExclusive: Long = Math.addExact(startByte, byteLength)

/** Stable corpus identity plus the exact ordered shard layout needed for replay. */
final case class CorpusShardManifest(
    schemaVersion: Int,
    name: String,
    provenance: CorpusProvenance,
    totalBytes: Long,
    sha256: String,
    shards: Vector[CorpusShard]
):
  require(schemaVersion == 1, s"unsupported corpus manifest schema: $schemaVersion")
  require(name.nonEmpty, "corpus name cannot be empty")
  require(totalBytes > 0, s"corpus must contain bytes: $totalBytes")
  require(sha256.matches("[0-9a-f]{64}"), s"invalid corpus SHA-256: $sha256")
  require(shards.nonEmpty, "corpus manifest requires at least one shard")
  shards.zipWithIndex.foreach { case (shard, index) =>
    val expectedStart = if index == 0 then 0L else shards(index - 1).endByteExclusive
    require(shard.startByte == expectedStart, s"shard $index starts at ${shard.startByte}, expected $expectedStart")
  }
  require(shards.last.endByteExclusive == totalBytes, "shards do not cover total corpus bytes")

  /** Deterministic JSON artifact; field order is part of the teaching format. */
  def json: JsonObject = JsonObject(
    "schema_version" -> JsonNumber(schemaVersion),
    "name" -> JsonString(name),
    "source" -> JsonString(provenance.source),
    "license" -> JsonString(provenance.license),
    "collected_at" -> JsonString(provenance.collectedAt.toString),
    "total_bytes" -> JsonNumber(totalBytes),
    "sha256" -> JsonString(sha256),
    "shards" -> JsonArray(shards.map(shard => JsonObject(
      "name" -> JsonString(shard.name),
      "start_byte" -> JsonNumber(shard.startByte),
      "byte_length" -> JsonNumber(shard.byteLength),
      "sha256" -> JsonString(shard.sha256)
    )))
  )

/** Exact next-byte location for bounded reads and checkpointed resume. */
final case class CorpusCursor(shardIndex: Int, offsetInShard: Long):
  require(shardIndex >= 0, s"negative shard index: $shardIndex")
  require(offsetInShard >= 0, s"negative shard offset: $offsetInShard")

final case class CorpusRead(bytes: Vector[Byte], next: CorpusCursor, endOfCorpus: Boolean)

/** Builds immutable UTF-8-safe shard files and publishes a deterministic manifest atomically. */
object CorpusShardBuilder:
  def build(
      name: String,
      text: String,
      provenance: CorpusProvenance,
      output: Path,
      maximumShardBytes: Int
  ): CorpusShardManifest =
    require(maximumShardBytes > 0, s"maximum shard bytes must be positive: $maximumShardBytes")
    val codePoints = text.codePoints().toArray.toVector.map(codePoint =>
      new String(Character.toChars(codePoint)).getBytes(java.nio.charset.StandardCharsets.UTF_8).toVector
    )
    require(codePoints.nonEmpty, "corpus text cannot be empty")
    val tooLarge = codePoints.indexWhere(_.size > maximumShardBytes)
    require(tooLarge < 0, s"code point $tooLarge exceeds shard limit $maximumShardBytes")
    Files.createDirectories(output)

    val groups = Vector.newBuilder[Vector[Byte]]
    var current = Vector.empty[Byte]
    codePoints.foreach { encoded =>
      if current.nonEmpty && current.size + encoded.size > maximumShardBytes then
        groups += current
        current = Vector.empty
      current ++= encoded
    }
    if current.nonEmpty then groups += current
    val payloads = groups.result()
    var globalOffset = 0L
    val shards = payloads.zipWithIndex.map { case (payload, index) =>
      val fileName = f"shard-$index%05d.bin"
      atomicWrite(output.resolve(fileName), payload.toArray)
      val shard = CorpusShard(fileName, globalOffset, payload.size.toLong, digest(payload.toArray))
      globalOffset = shard.endByteExclusive
      shard
    }
    val allBytes = payloads.flatten.toArray
    val manifest = CorpusShardManifest(1, name, provenance, allBytes.length.toLong, digest(allBytes), shards)
    atomicWrite(output.resolve("corpus-manifest.json"), manifest.json.render.getBytes(java.nio.charset.StandardCharsets.UTF_8))
    manifest

  private def atomicWrite(path: Path, bytes: Array[Byte]): Unit =
    val temporary = Files.createTempFile(path.getParent, s".${path.getFileName}.", ".tmp")
    try
      Files.write(temporary, bytes)
      try
        val _ = Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      catch case _: java.nio.file.AtomicMoveNotSupportedException =>
        val _ = Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING)
    finally
      val _ = Files.deleteIfExists(temporary)

  private[data] def digest(bytes: Array[Byte]): String = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))

/** Verifies immutable shard identity and reads at most a caller-provided number of bytes. */
final class CorpusShardReader(manifest: CorpusShardManifest, directory: Path):
  /** Hashes every shard and the concatenated corpus, rejecting truncation or corruption. */
  def verify(): Either[String, Unit] =
    val digest = MessageDigest.getInstance("SHA-256")
    manifest.shards.zipWithIndex.foldLeft[Either[String, Unit]](Right(())) {
      case (failure @ Left(_), _) => failure
      case (Right(_), (shard, index)) =>
        val path = directory.resolve(shard.name)
        if !Files.isRegularFile(path) then Left(s"missing shard $index: ${shard.name}")
        else
          val length = Files.size(path)
          if length != shard.byteLength then Left(s"shard $index length $length != ${shard.byteLength}")
          else
            val shardDigest = MessageDigest.getInstance("SHA-256")
            val input = Files.newInputStream(path)
            try
              val buffer = new Array[Byte](8192)
              var count = input.read(buffer)
              while count >= 0 do
                if count > 0 then
                  shardDigest.update(buffer, 0, count)
                  digest.update(buffer, 0, count)
                count = input.read(buffer)
              val actual = HexFormat.of().formatHex(shardDigest.digest())
              Either.cond(actual == shard.sha256, (), s"shard $index SHA-256 mismatch")
            finally input.close()
    }.flatMap(_ => Either.cond(HexFormat.of().formatHex(digest.digest()) == manifest.sha256, (), "corpus SHA-256 mismatch"))

  /** Reads across shard boundaries without returning more than `maximumBytes`. */
  def read(cursor: CorpusCursor, maximumBytes: Int): Either[String, CorpusRead] =
    require(maximumBytes > 0, s"maximum read bytes must be positive: $maximumBytes")
    if cursor.shardIndex > manifest.shards.size then Left(s"shard index ${cursor.shardIndex} exceeds ${manifest.shards.size}")
    else if cursor.shardIndex == manifest.shards.size then
      if cursor.offsetInShard == 0 then Right(CorpusRead(Vector.empty, cursor, endOfCorpus = true))
      else Left("end cursor offset must be zero")
    else
      val initialShard = manifest.shards(cursor.shardIndex)
      if cursor.offsetInShard > initialShard.byteLength then Left(s"offset ${cursor.offsetInShard} exceeds shard length ${initialShard.byteLength}")
      else
        var shardIndex = cursor.shardIndex
        var offset = cursor.offsetInShard.toInt
        val output = Vector.newBuilder[Byte]
        var remaining = maximumBytes
        while shardIndex < manifest.shards.size && remaining > 0 do
          val shard = manifest.shards(shardIndex)
          val count = math.min(remaining, shard.byteLength.toInt - offset)
          val buffer = ByteBuffer.allocate(count)
          val channel = FileChannel.open(directory.resolve(shard.name))
          try
            channel.position(offset.toLong)
            while buffer.hasRemaining && channel.read(buffer) >= 0 do ()
          finally channel.close()
          output ++= buffer.array().take(buffer.position())
          remaining -= buffer.position()
          offset += buffer.position()
          if offset == shard.byteLength.toInt then
            shardIndex += 1
            offset = 0
        val next = CorpusCursor(shardIndex, offset.toLong)
        Right(CorpusRead(output.result(), next, shardIndex == manifest.shards.size))

/** Creates and reads a temporary manifest/shard set so the complete boundary is observable. */
def runCorpusShardLab(): Unit =
  val directory = Files.createTempDirectory("learn-ai-corpus-")
  val manifest = CorpusShardBuilder.build(
    "demo",
    "ScalaとAIを学ぶ",
    CorpusProvenance("course-demo", "CC0-1.0", Instant.parse("2026-01-01T00:00:00Z")),
    directory,
    maximumShardBytes = 8
  )
  val reader = new CorpusShardReader(manifest, directory)
  println(s"corpus SHA-256: ${manifest.sha256}")
  println(s"shards: ${manifest.shards.size}, bytes: ${manifest.totalBytes}")
  println(s"verification: ${reader.verify()}")
  println(s"first 5 bytes: ${reader.read(CorpusCursor(0, 0), 5).toOption.get.bytes.map(_ & 0xff).mkString(",")}")
