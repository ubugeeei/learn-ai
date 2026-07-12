package learnai.text

import java.util.regex.Pattern

import scala.collection.mutable

import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.json.JsonParser

/** Exact GPT-2 byte-to-Unicode table, vocabulary IDs, and ranked BPE merge rules. */
final class Gpt2Tokenizer private (
    val encoder: Map[String, TokenId],
    val merges: Map[(String, String), Int]
):
  private val decoder = encoder.iterator.map { case (piece, id) => id.value -> piece }.toMap

  /** Encodes text with GPT-2 pre-tokenization followed by lowest-rank adjacent-pair merging. */
  def encode(text: String): Either[String, Vector[TokenId]] =
    val encodedBytes = Utf8.encodeBytes(text).map(Gpt2Tokenizer.byteEncoder)
    val transformed  = new String(encodedBytes.toArray)
    val matcher      = Gpt2Tokenizer.TokenPattern.matcher(transformed)
    val output       = Vector.newBuilder[TokenId]
    var error        = Option.empty[String]
    while matcher.find() && error.isEmpty do
      val pieces = mergePiece(matcher.group())
      pieces.foreach(piece =>
        if error.isEmpty then
          encoder.get(piece) match
            case Some(token) => output += token
            case None        => error = Some(s"GPT-2 vocabulary has no token for piece '$piece'")
      )
    error.toLeft(output.result())

  /** Reverses vocabulary pieces and the byte-to-Unicode table, then decodes strict UTF-8. */
  def decode(tokens: IterableOnce[TokenId]): Either[String, String] =
    val bytes    = Vector.newBuilder[Int]
    val iterator = tokens.iterator
    var error    = Option.empty[String]
    while iterator.hasNext && error.isEmpty do
      val id = iterator.next().value
      decoder.get(id) match
        case None        => error = Some(s"GPT-2 token ID $id is absent from the decoder")
        case Some(piece) => piece.foreach(character =>
            Gpt2Tokenizer.byteDecoder.get(character) match
              case Some(byte) => bytes += byte
              case None       =>
                error = Some(f"character U+${character.toInt}%04X is not a GPT-2 byte symbol")
          )
    error match
      case Some(message) => Left(message)
      case None          => Utf8.decodeBytes(bytes.result())

  private def mergePiece(piece: String): Vector[String] =
    var symbols  = piece.iterator.map(_.toString).toVector
    var continue = true
    while symbols.size > 1 && continue do
      val ranked = symbols.sliding(2).zipWithIndex.flatMap { case (pair, index) =>
        merges.get(pair(0) -> pair(1)).map(rank => (rank, index, pair(0), pair(1)))
      }.toVector
      if ranked.isEmpty then continue = false
      else
        val (_, _, left, right) = ranked.minBy { case (rank, index, _, _) => (rank, index) }
        val replaced            = Vector.newBuilder[String]
        var index               = 0
        while index < symbols.size do
          if index + 1 < symbols.size && symbols(index) == left && symbols(index + 1) == right then
            replaced += left + right
            index += 2
          else
            replaced += symbols(index)
            index += 1
        symbols = replaced.result()
    symbols

object Gpt2Tokenizer:
  // Java supports the Unicode categories used by GPT-2; the alternatives preserve GPT-2 ordering.
  private val TokenPattern = Pattern.compile(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE
  )

  private val visibleBytes                = (33 to 126).toVector ++ (161 to 172).toVector ++ (174 to 255).toVector
  private val invisibleBytes              = (0 to 255).filterNot(visibleBytes.toSet).toVector
  private val byteEncoder: Map[Int, Char] =
    (visibleBytes.map(byte => byte -> byte.toChar) ++ invisibleBytes.zipWithIndex.map {
      case (byte, index) => byte -> (256 + index).toChar
    }).toMap
  private val byteDecoder: Map[Char, Int] = byteEncoder.iterator.map(_.swap).toMap

  /** Returns the public GPT-2 Unicode stand-in for one unsigned byte. */
  def byteSymbol(byte: Int): Char =
    require(byte >= 0 && byte <= 255, s"byte outside [0, 256): $byte")
    byteEncoder(byte)

  /** Parses official `encoder.json` and `vocab.bpe` contents without external JSON dependencies. */
  def fromArtifacts(encoderJson: String, vocabularyBpe: String): Either[String, Gpt2Tokenizer] =
    for
      parsed     <- JsonParser.parse(encoderJson, maximumInputCharacters = 20_000_000)
      fields     <- parsed match
                      case JsonObject(values) => Right(values)
                      case _                  => Left("GPT-2 encoder.json root must be an object")
      entries    <- fields.foldLeft[Either[String, Vector[(String, TokenId)]]](Right(Vector.empty)) {
                      case (result, (piece, JsonNumber(number)))
                          if number.isValidInt && number.isWhole =>
                        result.map(_ :+ (piece -> TokenId(number.toInt)))
                      case (_, (piece, _)) =>
                        Left(s"GPT-2 vocabulary ID for '$piece' must be an integer")
                    }
      _          <- Either.cond(
                      entries.map(_._2.value).distinct.size == entries.size,
                      (),
                      "GPT-2 encoder.json contains duplicate token IDs"
                    )
      mergePairs <- parseMerges(vocabularyBpe)
    yield new Gpt2Tokenizer(entries.toMap, mergePairs.zipWithIndex.toMap)

  private def parseMerges(content: String): Either[String, Vector[(String, String)]] =
    val lines = content.linesIterator.toVector
    if lines.isEmpty || !lines.head.startsWith("#version:") then
      Left("GPT-2 vocab.bpe must begin with a #version header")
    else
      lines.tail.filter(_.nonEmpty).zipWithIndex
        .foldLeft[Either[String, Vector[(String, String)]]](Right(Vector.empty)) {
          case (result, (line, index)) => line.split(" ", -1).toVector match
              case Vector(left, right) if left.nonEmpty && right.nonEmpty =>
                result.flatMap(current =>
                  Either.cond(
                    !current.contains(left -> right),
                    current :+ (left -> right),
                    s"duplicate GPT-2 merge at line ${index + 2}: $line"
                  )
                )
              case _                                                      => Left(s"invalid GPT-2 merge at line ${index + 2}: '$line'")
        }
