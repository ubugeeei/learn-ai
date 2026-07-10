package learnai.retrieval

import java.util.Locale

import learnai.agent.JsonFieldType
import learnai.agent.Tool
import learnai.agent.ToolContext
import learnai.agent.ToolDefinition
import learnai.agent.ToolError
import learnai.agent.ToolField
import learnai.agent.ToolSchema
import learnai.json.JsonArray
import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonString
import learnai.json.JsonValue
import learnai.math.VectorD

/** One source artifact whose stable ID is suitable for citations. */
final case class SourceDocument(id: String, title: String, text: String):
  require(id.nonEmpty, "document ID cannot be empty")
  require(title.nonEmpty, "document title cannot be empty")

/** A contiguous source span; offsets use UTF-16 indices from the source String. */
final case class TextChunk(
    id: String,
    documentId: String,
    documentTitle: String,
    startOffset: Int,
    endOffset: Int,
    text: String
):
  require(startOffset >= 0 && endOffset >= startOffset, "invalid chunk offsets")

object TextChunker:
  /** Splits a document into fixed-size overlapping spans without crossing documents. */
  def chunk(
      document: SourceDocument,
      maximumCharacters: Int,
      overlapCharacters: Int
  ): Vector[TextChunk] =
    require(maximumCharacters > 0, "maximum chunk characters must be positive")
    require(overlapCharacters >= 0, "chunk overlap cannot be negative")
    require(overlapCharacters < maximumCharacters, "chunk overlap must be smaller than chunk size")
    if document.text.isEmpty then Vector.empty
    else
      val step = maximumCharacters - overlapCharacters
      val chunks = Vector.newBuilder[TextChunk]
      var start = 0
      var index = 0
      while start < document.text.length do
        val end = math.min(start + maximumCharacters, document.text.length)
        chunks += TextChunk(
          s"${document.id}#chunk-$index",
          document.id,
          document.title,
          start,
          end,
          document.text.substring(start, end)
        )
        start += step
        index += 1
      chunks.result()

/** A deterministic bag-of-words hashing embedder used to teach retrieval mechanics.
  *
  * It is not a semantic neural embedding model. Tokens that hash to the same
  * bucket collide; signed hashing reduces systematic positive collision bias.
  */
final class HashingEmbedder(val dimensions: Int):
  require(dimensions > 0, s"embedding dimensions must be positive: $dimensions")

  /** Produces an L2-normalized vector, or the zero vector for text without tokens. */
  def embed(text: String): VectorD =
    val values = Array.fill(dimensions)(0.0)
    tokenize(text).foreach { token =>
      val hash = token.hashCode
      val bucket = (hash & 0x7fffffff) % dimensions
      val sign = if (hash & 0x80000000) == 0 then 1.0 else -1.0
      values(bucket) += sign
    }
    val raw = VectorD.from(values)
    if raw.norm == 0.0 then raw else raw.scale(1.0 / raw.norm)

  private def tokenize(text: String): Vector[String] =
    text
      .toLowerCase(Locale.ROOT)
      .split("[^\\p{L}\\p{N}]+")
      .iterator
      .filter(_.nonEmpty)
      .toVector

final case class SearchResult(chunk: TextChunk, score: Double)

/** Immutable in-memory cosine index over precomputed normalized chunk vectors. */
final class VectorIndex private (
    val embedder: HashingEmbedder,
    private val entries: Vector[(TextChunk, VectorD)]
):
  val size: Int = entries.size

  /** Returns score-descending results with chunk ID as deterministic tie-breaker. */
  def search(query: String, resultCount: Int): Either[String, Vector[SearchResult]] =
    require(resultCount > 0, s"result count must be positive: $resultCount")
    if entries.isEmpty then Left("cannot search an empty index")
    else
      val queryVector = embedder.embed(query)
      if queryVector.norm == 0.0 then Left("query contains no indexable tokens")
      else
        Right(
          entries
            .map { case (chunk, vector) => SearchResult(chunk, queryVector.dot(vector)) }
            .sortBy(result => (-result.score, result.chunk.id))
            .take(resultCount)
        )

object VectorIndex:
  /** Chunks and indexes each document independently. */
  def build(
      documents: Vector[SourceDocument],
      embedder: HashingEmbedder,
      maximumChunkCharacters: Int,
      overlapCharacters: Int
  ): VectorIndex =
    require(documents.map(_.id).distinct.size == documents.size, "document IDs must be unique")
    val chunks = documents.flatMap { document =>
      TextChunker.chunk(document, maximumChunkCharacters, overlapCharacters)
    }
    new VectorIndex(embedder, chunks.map(chunk => chunk -> embedder.embed(chunk.text)))

/** Read-only retrieval capability that returns source offsets with every result. */
final class SearchTool(index: VectorIndex, resultCount: Int) extends Tool:
  require(resultCount > 0, "search tool result count must be positive")

  override val definition: ToolDefinition = ToolDefinition(
    "search_documents",
    "Search indexed documents and return source chunks with citation metadata.",
    ToolSchema(
      Vector(ToolField("query", JsonFieldType.StringValue, "Search query", required = true))
    )
  )

  override def execute(
      arguments: JsonObject,
      context: ToolContext
  ): Either[ToolError, JsonValue] =
    val query = arguments.get("query").collect { case JsonString(value) => value }.get
    index.search(query, resultCount) match
      case Left(message) => Left(ToolError("search_failed", message, retryable = false))
      case Right(results) =>
        Right(
          JsonArray(
            results.map { result =>
              JsonObject(
                "chunk_id" -> JsonString(result.chunk.id),
                "document_id" -> JsonString(result.chunk.documentId),
                "title" -> JsonString(result.chunk.documentTitle),
                "start_offset" -> JsonNumber(result.chunk.startOffset),
                "end_offset" -> JsonNumber(result.chunk.endOffset),
                "score" -> JsonNumber(BigDecimal(result.score)),
                "text" -> JsonString(result.chunk.text)
              )
            }
          )
        )
