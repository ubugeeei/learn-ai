package learnai.retrieval

import learnai.agent.ToolContext
import learnai.json.JsonArray
import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object RetrievalSuite extends TestSuite:
  override val name: String = "Retrieval"

  private val documents = Vector(
    SourceDocument("scala", "Scala", "Scala has immutable values and expressive static types."),
    SourceDocument("garden", "Garden", "Tomatoes need sunlight and regular watering in the garden."),
    SourceDocument("agent", "Agents", "An agent invokes validated tools and records observations.")
  )

  override val tests: Vector[TestCase] = Vector(
    test("chunker covers the source with the configured overlap") {
      val document = SourceDocument("doc", "Document", "abcdefghij")
      val chunks = TextChunker.chunk(document, maximumCharacters = 4, overlapCharacters = 1)
      Assert.equal(chunks.map(_.text), Vector("abcd", "defg", "ghij", "j"))
      Assert.equal(chunks.map(chunk => chunk.startOffset -> chunk.endOffset), Vector(0 -> 4, 3 -> 7, 6 -> 10, 9 -> 10))
      chunks.foreach { chunk =>
        Assert.equal(chunk.text, document.text.substring(chunk.startOffset, chunk.endOffset))
      }
    },
    test("chunker never combines text from different documents") {
      val index = VectorIndex.build(documents, new HashingEmbedder(128), 20, 5)
      Assert.isTrue(index.size >= documents.size)
      val results = Assert.right(index.search("Scala", 20))
      results.foreach(result => Assert.isTrue(documents.exists(_.id == result.chunk.documentId)))
    },
    test("hashing embeddings are deterministic normalized and case-insensitive") {
      val embedder = new HashingEmbedder(256)
      val first = embedder.embed("Scala TYPES scala")
      val second = embedder.embed("scala types SCALA")
      Assert.equal(first, second)
      Assert.close(first.norm, 1.0)
      Assert.close(embedder.embed("---").norm, 0.0)
    },
    test("cosine search ranks token-overlapping content first") {
      val index = VectorIndex.build(documents, new HashingEmbedder(512), 200, 0)
      val result = Assert.right(index.search("immutable Scala types", 2))
      Assert.equal(result.head.chunk.documentId, "scala")
      Assert.isTrue(result.head.score > result.last.score)
    },
    test("search order is deterministic when scores tie") {
      val tied = Vector(
        SourceDocument("b", "B", "same words"),
        SourceDocument("a", "A", "same words")
      )
      val index = VectorIndex.build(tied, new HashingEmbedder(64), 100, 0)
      val results = Assert.right(index.search("same words", 2))
      Assert.equal(results.map(_.chunk.id), Vector("a#chunk-0", "b#chunk-0"))
    },
    test("empty indexes and tokenless queries return explicit errors") {
      val empty = VectorIndex.build(Vector.empty, new HashingEmbedder(32), 10, 0)
      val index = VectorIndex.build(documents, new HashingEmbedder(32), 100, 0)
      Assert.isTrue(empty.search("query", 1).left.exists(_.contains("empty")))
      Assert.isTrue(index.search("---", 1).left.exists(_.contains("no indexable")))
    },
    test("search tool exposes citation metadata in structured JSON") {
      val index = VectorIndex.build(documents, new HashingEmbedder(512), 200, 0)
      val tool = new SearchTool(index, resultCount = 1)
      val output = Assert.right(
        tool.execute(
          JsonObject("query" -> JsonString("validated tools observations")),
          ToolContext("call-1", agentStep = 0)
        )
      )
      val result = output.asInstanceOf[JsonArray].values.head.asInstanceOf[JsonObject]
      Assert.equal(result.get("document_id"), Some(JsonString("agent")))
      Assert.isTrue(result.get("chunk_id").nonEmpty)
      Assert.isTrue(result.get("start_offset").nonEmpty)
      Assert.isTrue(result.get("text").nonEmpty)
    }
  )
