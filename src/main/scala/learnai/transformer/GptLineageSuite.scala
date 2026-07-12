package learnai.transformer

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object GptLineageSuite extends TestSuite:
  override val name: String = "GptLineage"

  override val tests: Vector[TestCase] = specify(
    test("GPT-2 Small inventory matches the published 124M checkpoint exactly") {
      Assert.equal(GptLineage.parameterInventory(GptLineage.Gpt2Small).total, 124439808L)
    },
    test("all four OpenAI GPT-2 configurations match checkpoint parameter inventories") {
      val totals = Vector(
        GptLineage.Gpt2Small  -> 124439808L,
        GptLineage.Gpt2Medium -> 354823168L,
        GptLineage.Gpt2Large  -> 774030080L,
        GptLineage.Gpt2Xl     -> 1557611200L
      )
      totals.foreach { case (config, expected) =>
        Assert.equal(GptLineage.parameterInventory(config).total, expected)
      }
    },
    test("tied language-model head is not counted as a second vocabulary matrix") {
      val config    = Gpt2Config(10, 8, 4, 2, 1)
      val inventory = GptLineage.parameterInventory(config)
      Assert.equal(inventory.tokenEmbedding, 40L)
      Assert.equal(inventory.total, 324L)
    },
    test("invalid head and channel ownership is rejected at configuration boundary") {
      val _ = Assert.throws[IllegalArgumentException](Gpt2Config(100, 16, 10, 3, 2))
    },
    test("compatibility report names normalization activation tokenizer and weight loading gaps") {
      val report = GptLineage.miniGptCompatibilityDifferences.mkString(" ").toLowerCase
      Vector("layernorm", "gelu", "tokenizer", "weight").foreach(term =>
        Assert.isTrue(report.contains(term), s"compatibility report omitted $term")
      )
    }
  )
