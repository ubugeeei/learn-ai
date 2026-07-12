package learnai.transformer

import learnai.tensor.Shape
import learnai.text.TokenId
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Gpt2ModelSuite extends TestSuite:
  override val name: String = "Gpt2Model"

  override val tests: Vector[TestCase] = specify(
    test("full-model parameters exactly match the closed GPT-2 inventory") {
      val config = Gpt2Config(11, 8, 4, 2, 2)
      val model  = Gpt2Model.random(config, 7)
      Assert.equal(model.parameterCount, GptLineage.parameterInventory(config).total)
      Assert.equal(model.parameters.distinct.size, model.parameters.size)
      Assert.isTrue(!model.parameters.exists(_.label.contains("lm_head")))
    },
    test("logits expose one vocabulary row per token") {
      val model  = Gpt2Model.random(Gpt2Config(7, 6, 4, 2, 1), 9)
      val logits = model.logits(Vector(TokenId(1), TokenId(3), TokenId(2)))
      Assert.equal(logits.shape, Shape(3, 7))
      Assert.isTrue(logits.values.forall(_.isFinite))
    },
    test("changing a future token cannot alter earlier full-model logits") {
      val model      = Gpt2Model.random(Gpt2Config(9, 6, 4, 2, 2), 13)
      val first      = model.logits(Vector(TokenId(1), TokenId(2), TokenId(3))).values
      val second     = model.logits(Vector(TokenId(1), TokenId(2), TokenId(8))).values
      val vocabulary = 9
      Assert.equal(first.take(2 * vocabulary), second.take(2 * vocabulary))
    },
    test("tied logits send gradient back to the token embedding") {
      val model = Gpt2Model.random(Gpt2Config(7, 5, 4, 2, 1), 17)
      model.logits(Vector(TokenId(1), TokenId(2))).sum.backward()
      Assert.isTrue(model.embeddings.tokens.weight.gradients.exists(_ != 0.0))
    },
    test("empty and over-context sequences fail at the model boundary") {
      val model = Gpt2Model.random(Gpt2Config(7, 2, 4, 2, 1), 19)
      Assert.throws[IllegalArgumentException](model.logits(Vector.empty))
      Assert
        .throws[IllegalArgumentException](model.logits(Vector(TokenId(1), TokenId(2), TokenId(3))))
      ()
    }
  )
