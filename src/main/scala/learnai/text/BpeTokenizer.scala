package learnai.text

import scala.collection.mutable

final case class BpeMerge(left: TokenId, right: TokenId, result: TokenId)

final class BpeTokenizer private (
    val merges: Vector[BpeMerge],
    private val tokenBytes: Vector[Vector[Int]]
):
  val vocabularySize: Int = tokenBytes.size

  def encode(text: String): Vector[TokenId] =
    var symbols = Utf8.encodeBytes(text)
    merges.foreach { merge =>
      symbols = BpeTokenizer
        .replacePair(symbols, merge.left.value, merge.right.value, merge.result.value)
    }
    symbols.map(TokenId(_))

  def decode(tokenIds: IterableOnce[TokenId]): Either[String, String] =
    val bytes                 = Vector.newBuilder[Int]
    val iterator              = tokenIds.iterator
    var error: Option[String] = None
    while iterator.hasNext && error.isEmpty do
      val tokenId = iterator.next().value
      if tokenId < 0 || tokenId >= vocabularySize then
        error = Some(s"token ID $tokenId outside vocabulary [0, $vocabularySize)")
      else bytes ++= tokenBytes(tokenId)
    error match
      case Some(message) => Left(message)
      case None          => Utf8.decodeBytes(bytes.result())

object BpeTokenizer:
  val BaseVocabularySize: Int = 256

  def fromMerges(merges: Vector[BpeMerge]): BpeTokenizer =
    val expansions = mutable.ArrayBuffer
      .from(Vector.tabulate(BaseVocabularySize)(byte => Vector(byte)))
    merges.zipWithIndex.foreach { case (merge, index) =>
      val expectedResult = BaseVocabularySize + index
      require(
        merge.result.value == expectedResult,
        s"merge $index result must be token $expectedResult, got ${merge.result.value}"
      )
      require(
        merge.left.value < expectedResult && merge.right.value < expectedResult,
        s"merge $index may reference only earlier tokens: $merge"
      )
      expansions += expansions(merge.left.value) ++ expansions(merge.right.value)
    }
    new BpeTokenizer(merges, expansions.toVector)

  private[text] def replacePair(
      symbols: Vector[Int],
      left: Int,
      right: Int,
      result: Int
  ): Vector[Int] =
    val replaced = Vector.newBuilder[Int]
    var index    = 0
    while index < symbols.size do
      if index + 1 < symbols.size && symbols(index) == left && symbols(index + 1) == right then
        replaced += result
        index += 2
      else
        replaced += symbols(index)
        index += 1
    replaced.result()

object BpeTrainer:
  def train(corpus: Vector[String], targetVocabularySize: Int): BpeTokenizer =
    require(
      targetVocabularySize >= BpeTokenizer.BaseVocabularySize,
      s"target vocabulary must be at least ${BpeTokenizer.BaseVocabularySize}: " +
        s"$targetVocabularySize"
    )

    var sequences   = corpus.map(Utf8.encodeBytes)
    val merges      = Vector.newBuilder[BpeMerge]
    var nextToken   = BpeTokenizer.BaseVocabularySize
    var canContinue = true

    while nextToken < targetVocabularySize && canContinue do
      val counts = mutable.Map.empty[(Int, Int), Int].withDefaultValue(0)
      sequences.foreach { sequence =>
        var index = 0
        while index + 1 < sequence.size do
          val pair = sequence(index) -> sequence(index + 1)
          counts.update(pair, counts(pair) + 1)
          index += 1
      }

      if counts.isEmpty then canContinue = false
      else
        val ((left, right), _) = counts.minBy { case ((left, right), count) =>
          (-count, left, right)
        }
        val merge              = BpeMerge(TokenId(left), TokenId(right), TokenId(nextToken))
        merges += merge
        sequences = sequences
          .map(sequence => BpeTokenizer.replacePair(sequence, left, right, nextToken))
        nextToken += 1

    BpeTokenizer.fromMerges(merges.result())
