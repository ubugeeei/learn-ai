package learnai.data

import java.util.SplittableRandom

import learnai.text.TokenId
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object CausalDatasetSuite extends TestSuite:
  override val name: String = "CausalDataset"

  override val tests: Vector[TestCase] = Vector(
    test("sliding windows align each target one token ahead") {
      val tokens = Vector(10, 11, 12, 13, 14).map(TokenId(_))
      val dataset = CausalDataset.fromTokens(tokens, contextLength = 3)
      Assert.equal(dataset.size, 2)
      Assert.equal(
        dataset.examples.head,
        CausalExample(
          Vector(10, 11, 12).map(TokenId(_)),
          Vector(11, 12, 13).map(TokenId(_))
        )
      )
      Assert.equal(
        dataset.examples.last,
        CausalExample(
          Vector(11, 12, 13).map(TokenId(_)),
          Vector(12, 13, 14).map(TokenId(_))
        )
      )
    },
    test("a sequence no longer than its context produces no full example") {
      val exact = CausalDataset.fromTokens(Vector(1, 2, 3).map(TokenId(_)), 3)
      val shorter = CausalDataset.fromTokens(Vector(1, 2).map(TokenId(_)), 3)
      Assert.isTrue(exact.isEmpty)
      Assert.isTrue(shorter.isEmpty)
    },
    test("contiguous split creates no window crossing its boundary") {
      import learnai.text.TokenId.*

      val tokens = Vector.range(0, 20).map(TokenId(_))
      val split = CausalDataset.contiguousSplit(tokens, trainingFraction = 0.6, contextLength = 3)
      val trainingIds = split.training.examples.flatMap(example => example.inputs ++ example.targets)
      val validationIds = split.validation.examples.flatMap(example => example.inputs ++ example.targets)
      Assert.isTrue(trainingIds.forall(_.value < split.boundaryTokenIndex))
      Assert.isTrue(validationIds.forall(_.value >= split.boundaryTokenIndex))
    },
    test("sampling is deterministic for equal random seeds") {
      val dataset = CausalDataset.fromTokens(Vector.range(0, 12).map(TokenId(_)), 2)
      val first = Assert.right(dataset.sampleBatch(8, new SplittableRandom(17L)))
      val second = Assert.right(dataset.sampleBatch(8, new SplittableRandom(17L)))
      Assert.equal(first, second)
      Assert.equal(first.batchSize, 8)
      Assert.equal(first.contextLength, 2)
    },
    test("sampling an empty dataset returns a descriptive error") {
      val empty = CausalDataset.fromTokens(Vector.empty, 4)
      Assert.equal(
        empty.sampleBatch(2, new SplittableRandom(1L)),
        Left("cannot sample from an empty dataset")
      )
    },
    test("sequential batches can keep or drop a short final batch") {
      val dataset = CausalDataset.fromTokens(Vector.range(0, 8).map(TokenId(_)), 2)
      Assert.equal(dataset.size, 6)
      Assert.equal(dataset.sequentialBatches(4, dropLast = false).map(_.batchSize), Vector(4, 2))
      Assert.equal(dataset.sequentialBatches(4, dropLast = true).map(_.batchSize), Vector(4))
    },
    test("invalid context batch and split parameters are rejected") {
      val contextError = Assert.throws[IllegalArgumentException] {
        CausalDataset.fromTokens(Vector(TokenId(1)), contextLength = 0)
      }
      val splitError = Assert.throws[IllegalArgumentException] {
        CausalDataset.contiguousSplit(Vector(TokenId(1)), 1.0, 1)
      }
      val dataset = CausalDataset.fromTokens(Vector(TokenId(1), TokenId(2)), 1)
      val batchError = Assert.throws[IllegalArgumentException] {
        dataset.sampleBatch(0, new SplittableRandom(1L))
      }
      Assert.isTrue(contextError.getMessage.contains("context length"))
      Assert.isTrue(splitError.getMessage.contains("training fraction"))
      Assert.isTrue(batchError.getMessage.contains("batch size"))
    }
  )
