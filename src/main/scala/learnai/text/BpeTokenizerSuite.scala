package learnai.text

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object BpeTokenizerSuite extends TestSuite:
  override val name: String = "BpeTokenizer"

  override val tests: Vector[TestCase] = specify(
    test("trained BPE round-trips Unicode text") {
      val corpus    = Vector("banana bandana", "café résumé", "Scala 🚀 Scala 🚀")
      val tokenizer = BpeTrainer.train(corpus, targetVocabularySize = 280)
      corpus.foreach(text => Assert.equal(tokenizer.decode(tokenizer.encode(text)), Right(text)))
    },
    test("frequent byte pairs reduce token count") {
      val text      = "banana banana banana"
      val tokenizer = BpeTrainer.train(Vector(text), targetVocabularySize = 264)
      val byteCount = ByteTokenizer.encode(text).size
      Assert.isTrue(tokenizer.encode(text).size < byteCount)
      Assert.isTrue(tokenizer.merges.nonEmpty)
    },
    test("training is deterministic when pair frequencies tie") {
      val corpus = Vector("abab cdcd")
      val first  = BpeTrainer.train(corpus, 260)
      val second = BpeTrainer.train(corpus, 260)
      Assert.equal(first.merges, second.merges)
    },
    test("a corpus without pairs leaves the base vocabulary unchanged") {
      val tokenizer = BpeTrainer.train(Vector("", "a", ""), 300)
      Assert.equal(tokenizer.vocabularySize, 256)
    },
    test("decoder rejects token IDs outside its learned vocabulary") {
      val tokenizer = BpeTrainer.train(Vector("abcabc"), 258)
      val error     = Assert.left(tokenizer.decode(Vector(TokenId(999))))
      Assert.isTrue(error.contains("outside vocabulary"))
    },
    test("merge tables must be topologically ordered") {
      val invalid = Vector(BpeMerge(TokenId(256), TokenId(97), TokenId(256)))
      val error   = Assert.throws[IllegalArgumentException](BpeTokenizer.fromMerges(invalid))
      Assert.isTrue(error.getMessage.contains("earlier tokens"))
    }
  )
