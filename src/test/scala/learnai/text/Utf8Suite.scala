package learnai.text

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Utf8Suite extends TestSuite:
  override val name: String = "Utf8Tokenizer"

  override val tests: Vector[TestCase] = Vector(
    test("UTF-8 tokenizer round-trips ASCII Japanese and emoji") {
      val examples = Vector("hello", "大規模言語モデル", "Scala 3 🚀")
      examples.foreach { text =>
        Assert.equal(ByteTokenizer.decode(ByteTokenizer.encode(text)), Right(text))
      }
    },
    test("token count follows encoded bytes rather than visible characters") {
      Assert.equal(ByteTokenizer.encode("A").size, 1)
      Assert.equal(ByteTokenizer.encode("あ").size, 3)
      Assert.equal(ByteTokenizer.encode("🚀").size, 4)
    },
    test("special tokens can be added and skipped during decoding") {
      val tokens = ByteTokenizer.encode("ai", addBeginOfText = true, addEndOfText = true)
      Assert.equal(tokens.head, ByteTokenizer.BeginOfText)
      Assert.equal(tokens.last, ByteTokenizer.EndOfText)
      Assert.equal(ByteTokenizer.decode(tokens), Right("ai"))
      Assert.isTrue(ByteTokenizer.decode(tokens, skipSpecialTokens = false).isLeft)
    },
    test("decoder rejects malformed UTF-8 instead of replacing it silently") {
      val continuationWithoutLeader = Vector(TokenId(0x80))
      Assert.isTrue(ByteTokenizer.decode(continuationWithoutLeader).isLeft)
    },
    test("token IDs cannot be negative") {
      val error = Assert.throws[IllegalArgumentException](TokenId(-1))
      Assert.isTrue(error.getMessage.contains("non-negative"))
    }
  )
