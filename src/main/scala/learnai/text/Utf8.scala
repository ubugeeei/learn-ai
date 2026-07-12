package learnai.text

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets

/** Non-negative integer identifier whose representation stays allocation-free. */
opaque type TokenId = Int

object TokenId:
  def apply(value: Int): TokenId =
    require(value >= 0, s"token ID must be non-negative: $value")
    value

  extension (tokenId: TokenId) def value: Int = tokenId

/** Strict UTF-8 conversion that reports malformed input instead of replacement. */
object Utf8:
  def encodeBytes(text: String): Vector[Int] = text.getBytes(StandardCharsets.UTF_8).iterator
    .map(byte => byte & 0xff).toVector

  def decodeBytes(unsignedBytes: IterableOnce[Int]): Either[String, String] =
    val values       = unsignedBytes.iterator.toArray
    val invalidIndex = values.indexWhere(value => value < 0 || value > 255)
    if invalidIndex >= 0 then
      Left(s"byte at index $invalidIndex outside [0, 256): ${values(invalidIndex)}")
    else
      val bytes   = values.map(_.toByte)
      val decoder = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
      try Right(decoder.decode(ByteBuffer.wrap(bytes)).toString)
      catch
        case error: java.nio.charset.CharacterCodingException =>
          Left(s"invalid UTF-8 byte sequence: ${error.getMessage}")

/** Reversible byte tokenizer with explicit begin/end special tokens. */
object ByteTokenizer:
  val ByteVocabularySize: Int = 256
  val BeginOfText: TokenId    = TokenId(256)
  val EndOfText: TokenId      = TokenId(257)
  val VocabularySize: Int     = 258

  def encode(
      text: String,
      addBeginOfText: Boolean = false,
      addEndOfText: Boolean = false
  ): Vector[TokenId] =
    val bytes         = Utf8.encodeBytes(text).map(TokenId(_))
    val withBeginning = if addBeginOfText then BeginOfText +: bytes else bytes
    if addEndOfText then withBeginning :+ EndOfText else withBeginning

  def decode(
      tokenIds: IterableOnce[TokenId],
      skipSpecialTokens: Boolean = true
  ): Either[String, String] =
    val values        = tokenIds.iterator.map(_.value).toVector
    val ordinaryBytes = if skipSpecialTokens then values.filter(_ < ByteVocabularySize) else values
    Utf8.decodeBytes(ordinaryBytes)
