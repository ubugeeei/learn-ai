package learnai.data

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.time.Instant

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object CorpusShardsSuite extends TestSuite:
  override val name: String = "CorpusShards"

  private val provenance = CorpusProvenance(
    "https://example.test/corpus",
    "CC-BY-4.0",
    Instant.parse("2026-01-02T03:04:05Z")
  )

  override val tests: Vector[TestCase] = specify(
    test("builder publishes contiguous UTF-8-safe shards and stable provenance") {
      val directory = Files.createTempDirectory("corpus-shards-")
      val manifest = CorpusShardBuilder.build("mixed", "a日🚀b", provenance, directory, 4)
      Assert.equal(manifest.totalBytes, "a日🚀b".getBytes(StandardCharsets.UTF_8).length.toLong)
      Assert.equal(manifest.provenance, provenance)
      Assert.equal(manifest.shards.map(_.startByte), Vector(0L, 4L, 8L))
      manifest.shards.foreach(shard =>
        Assert.right(learnai.text.Utf8.decodeBytes(Files.readAllBytes(directory.resolve(shard.name)).map(_ & 0xff)))
      )
      Assert.isTrue(Files.isRegularFile(directory.resolve("corpus-manifest.json")))
    },
    test("equal bytes and provenance create equal deterministic manifests") {
      val first = CorpusShardBuilder.build("same", "abcdef", provenance, Files.createTempDirectory("corpus-a-"), 2)
      val second = CorpusShardBuilder.build("same", "abcdef", provenance, Files.createTempDirectory("corpus-b-"), 2)
      Assert.equal(first, second)
      Assert.equal(first.json.render, second.json.render)
    },
    test("bounded reads cross shards and resume without gaps or duplication") {
      val directory = Files.createTempDirectory("corpus-resume-")
      val text = "abcdefghij"
      val manifest = CorpusShardBuilder.build("resume", text, provenance, directory, 3)
      val reader = new CorpusShardReader(manifest, directory)
      var cursor = CorpusCursor(0, 0)
      val bytes = Vector.newBuilder[Byte]
      var ended = false
      while !ended do
        val read = Assert.right(reader.read(cursor, 2))
        Assert.isTrue(read.bytes.size <= 2)
        bytes ++= read.bytes
        cursor = read.next
        ended = read.endOfCorpus
      Assert.equal(new String(bytes.result().toArray, StandardCharsets.UTF_8), text)
      Assert.equal(cursor, CorpusCursor(manifest.shards.size, 0))
    },
    test("verification detects one corrupted byte") {
      val directory = Files.createTempDirectory("corpus-corrupt-")
      val manifest = CorpusShardBuilder.build("corrupt", "abcdefgh", provenance, directory, 4)
      val path = directory.resolve(manifest.shards.head.name)
      val bytes = Files.readAllBytes(path)
      bytes(0) = (bytes(0) ^ 1).toByte
      Files.write(path, bytes)
      val error = Assert.left(new CorpusShardReader(manifest, directory).verify())
      Assert.isTrue(error.contains("SHA-256 mismatch"))
    },
    test("verification detects truncation and missing shards") {
      val truncatedDirectory = Files.createTempDirectory("corpus-truncated-")
      val truncated = CorpusShardBuilder.build("truncated", "abcdefgh", provenance, truncatedDirectory, 4)
      Files.write(truncatedDirectory.resolve(truncated.shards.head.name), Array[Byte](1))
      Assert.isTrue(Assert.left(new CorpusShardReader(truncated, truncatedDirectory).verify()).contains("length"))

      val missingDirectory = Files.createTempDirectory("corpus-missing-")
      val missing = CorpusShardBuilder.build("missing", "abcdefgh", provenance, missingDirectory, 4)
      Files.delete(missingDirectory.resolve(missing.shards.last.name))
      Assert.isTrue(Assert.left(new CorpusShardReader(missing, missingDirectory).verify()).contains("missing"))
    },
    test("invalid cursors and read budgets fail at the boundary") {
      val directory = Files.createTempDirectory("corpus-cursor-")
      val manifest = CorpusShardBuilder.build("cursor", "abcd", provenance, directory, 2)
      val reader = new CorpusShardReader(manifest, directory)
      Assert.throws[IllegalArgumentException](reader.read(CorpusCursor(0, 0), 0))
      Assert.isTrue(reader.read(CorpusCursor(0, 3), 1).isLeft)
      Assert.isTrue(reader.read(CorpusCursor(manifest.shards.size + 1, 0), 1).isLeft)
    },
    test("a code point larger than the shard budget is rejected") {
      Assert.throws[IllegalArgumentException](
        CorpusShardBuilder.build("tiny", "🚀", provenance, Files.createTempDirectory("corpus-tiny-"), 3)
      )
      ()
    }
  )
