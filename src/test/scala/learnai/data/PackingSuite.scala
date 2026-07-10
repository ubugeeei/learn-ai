package learnai.data

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite
import learnai.text.TokenId
import learnai.text.TokenId.*
import learnai.transformer.MiniGpt
import learnai.transformer.MiniGptConfig

object PackingSuite extends TestSuite:
  override val name: String = "SequencePacking"

  private val separator = TokenId(9)
  private val padding = TokenId(8)

  private def tokens(values: Int*): Vector[TokenId] = values.toVector.map(TokenId(_))

  override val tests: Vector[TestCase] = Vector(
    test("a hand-packed two-document example matches window by window") {
      val result = SequencePacking.pack(
        Vector(tokens(0, 1, 2), tokens(3, 4)),
        contextLength = 4,
        separator,
        padding
      )
      // stream: 0 1 2 S 3 4 S  (7 tokens, 6 predictions, 2 windows, 2 pads)
      Assert.equal(result.packedStreamLength, 7)
      Assert.equal(result.paddingTokenCount, 2)
      Assert.equal(result.examples.size, 2)

      val first = result.examples(0)
      Assert.equal(first.inputs, tokens(0, 1, 2, 9))
      Assert.equal(first.targets, tokens(1, 2, 9, 3))
      // Predicting the separator is real signal; predicting the next
      // document's opening from the separator is not.
      Assert.equal(first.lossMask, Vector(true, true, true, false))

      val second = result.examples(1)
      Assert.equal(second.inputs, tokens(3, 4, 9, 8))
      Assert.equal(second.targets, tokens(4, 9, 8, 8))
      Assert.equal(second.lossMask, Vector(true, true, false, false))

      // One trainable prediction per real document token.
      Assert.equal(result.unmaskedTargetCount, 5)
    },
    test("unmasked pairs are exactly the within-document transitions plus endings") {
      val documents = Vector(
        tokens(0, 1, 2, 3, 4),
        tokens(5),
        tokens(6, 7, 0, 1),
        tokens(2, 3, 4, 5, 6, 7)
      )
      val result = SequencePacking.pack(documents, contextLength = 3, separator, padding)

      val observedPairs = result.examples.flatMap { example =>
        example.inputs.lazyZip(example.targets).lazyZip(example.lossMask).collect {
          case (input, target, true) => (input, target)
        }
      }
      val expectedPairs = documents.flatMap { document =>
        val terminated = document :+ separator
        terminated.init.zip(terminated.tail)
      }
      Assert.equal(observedPairs.sorted(using pairOrdering), expectedPairs.sorted(using pairOrdering))
      Assert.equal(result.unmaskedTargetCount, documents.map(_.size).sum)
    },
    test("every masked position is a padding target or a separator input") {
      val documents = Vector(tokens(0, 1), tokens(2, 3, 4, 5, 6), tokens(7))
      val result = SequencePacking.pack(documents, contextLength = 4, separator, padding)
      result.examples.foreach { example =>
        example.inputs.lazyZip(example.targets).lazyZip(example.lossMask).foreach {
          (input, target, keep) =>
            if !keep then
              Assert.isTrue(
                target == padding || input == separator,
                s"masked position with input $input and target $target has no reason"
              )
            else
              Assert.isTrue(
                target != padding && input != separator,
                s"trainable position with input $input and target $target should be masked"
              )
        }
      }
    },
    test("degenerate windows are exposed and filterable rather than hidden") {
      // Context length one with single-token documents produces windows
      // whose only input is a separator; they carry zero trainable targets.
      val result = SequencePacking.pack(
        Vector(tokens(0), tokens(1)),
        contextLength = 1,
        separator,
        padding
      )
      Assert.isTrue(
        result.examples.exists(_.unmaskedTargetCount == 0),
        "this construction should produce an all-masked window"
      )
      Assert.isTrue(result.trainableExamples.forall(_.unmaskedTargetCount > 0))
      Assert.equal(
        result.trainableExamples.map(_.unmaskedTargetCount).sum,
        result.unmaskedTargetCount
      )
    },
    test("a packed window trains MiniGPT through the masked loss") {
      val config = MiniGptConfig(
        vocabularySize = 10,
        maximumContextLength = 4,
        channels = 4,
        headCount = 2,
        hiddenChannels = 8,
        layerCount = 1
      )
      val model = MiniGpt.random(config, seed = 5L)
      val result = SequencePacking.pack(
        Vector(tokens(0, 1, 2), tokens(3, 4)),
        contextLength = 4,
        separator,
        padding
      )
      val window = result.examples(1)
      val loss = model.lossMasked(window.inputs, window.targets, window.lossMask)
      Assert.isTrue(loss.valueAtFlat(0).isFinite, "masked loss must be finite")
      loss.backward()
      Assert.isTrue(
        model.parameters.exists(_.gradients.exists(_ != 0.0)),
        "unmasked positions must produce gradients"
      )
    },
    test("invalid documents and configurations are rejected") {
      val noDocuments = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector.empty, contextLength = 4, separator, padding)
      }
      Assert.isTrue(noDocuments.getMessage.contains("at least one document"))
      val emptyDocument = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector(tokens(1), Vector.empty), 4, separator, padding)
      }
      Assert.isTrue(emptyDocument.getMessage.contains("empty"))
      val sameSpecials = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector(tokens(1)), 4, separator, separator)
      }
      Assert.isTrue(sameSpecials.getMessage.contains("differ"))
      val separatorInside = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector(tokens(1, 9)), 4, separator, padding)
      }
      Assert.isTrue(separatorInside.getMessage.contains("separator"))
      val paddingInside = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector(tokens(1, 8)), 4, separator, padding)
      }
      Assert.isTrue(paddingInside.getMessage.contains("padding"))
      val badContext = Assert.throws[IllegalArgumentException] {
        SequencePacking.pack(Vector(tokens(1)), 0, separator, padding)
      }
      Assert.isTrue(badContext.getMessage.contains("positive"))
    }
  )

  private val pairOrdering: Ordering[(TokenId, TokenId)] =
    Ordering.by(pair => (pair._1.value, pair._2.value))
