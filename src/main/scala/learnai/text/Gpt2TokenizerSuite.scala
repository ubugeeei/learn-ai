package learnai.text

import learnai.json.JsonNumber
import learnai.json.JsonObject
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Gpt2TokenizerSuite extends TestSuite:
  override val name: String = "Gpt2Tokenizer"

  override val tests: Vector[TestCase] = specify(
    test("bytes-to-Unicode is a bijection over all 256 byte values") {
      val symbols = Vector.tabulate(256)(Gpt2Tokenizer.byteSymbol)
      Assert.equal(symbols.distinct.size, 256)
      Assert.equal(Gpt2Tokenizer.byteSymbol(32), 'Ġ')
      Assert.equal(Gpt2Tokenizer.byteSymbol(65), 'A')
    },
    test("ranked merges produce a known hello token and preserve leading space") {
      val base      = baseArtifacts()
      val extra     = Vector("he", "ll", "hell", "hello")
      val encoder   = encoderJson(base ++ extra)
      val merges    = "#version: 0.2\nh e\nl l\nhe ll\nhell o\n"
      val tokenizer = Assert.right(Gpt2Tokenizer.fromArtifacts(encoder, merges))
      val helloId   = TokenId(base.size + 3)
      Assert.equal(Assert.right(tokenizer.encode("hello")), Vector(helloId))
      val spaced    = Assert.right(tokenizer.encode(" hello"))
      Assert.equal(spaced.head, TokenId(32))
      Assert.equal(Assert.right(tokenizer.decode(spaced)), " hello")
    },
    test("base byte artifacts round-trip multilingual strict UTF-8") {
      val tokenizer = Assert
        .right(Gpt2Tokenizer.fromArtifacts(encoderJson(baseArtifacts()), "#version: 0.2\n"))
      val text      = "ScalaとAI 🚀 café"
      val tokens    = Assert.right(tokenizer.encode(text))
      Assert.equal(tokens.size, Utf8.encodeBytes(text).size)
      Assert.equal(Assert.right(tokenizer.decode(tokens)), text)
    },
    test("artifact parser rejects malformed IDs headers and duplicate merges") {
      Assert.isTrue(Gpt2Tokenizer.fromArtifacts("[]", "#version: 0.2\n").isLeft)
      Assert.isTrue(Gpt2Tokenizer.fromArtifacts("{\"a\":1.5}", "#version: 0.2\n").isLeft)
      Assert.isTrue(Gpt2Tokenizer.fromArtifacts("{\"a\":0}", "a b\n").isLeft)
      Assert.isTrue(Gpt2Tokenizer.fromArtifacts("{\"a\":0}", "#version: 0.2\na b\na b\n").isLeft)
    },
    test("decode rejects IDs absent from the public vocabulary") {
      val tokenizer = Assert
        .right(Gpt2Tokenizer.fromArtifacts(encoderJson(baseArtifacts()), "#version: 0.2\n"))
      Assert.isTrue(tokenizer.decode(Vector(TokenId(999))).isLeft)
    }
  )

  private def baseArtifacts(): Vector[String] = Vector
    .tabulate(256)(byte => Gpt2Tokenizer.byteSymbol(byte).toString)

  private def encoderJson(pieces: Vector[String]): String = JsonObject(pieces.zipWithIndex.map {
    case (piece, id) => piece -> JsonNumber(BigDecimal(id))
  }).render
