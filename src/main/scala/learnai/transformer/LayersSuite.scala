package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.text.TokenId
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object LayersSuite extends TestSuite:
  override val name: String = "TransformerLayers"

  override val tests: Vector[TestCase] = specify(
    test("embedding lookup has time by channel shape") {
      val embedding = Embedding.fromValues(
        entries = 3,
        channels = 2,
        values = Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        label = "embedding"
      )
      val output    = embedding(Vector(2, 0, 2))
      Assert.equal(output.shape, Shape(3, 2))
      Assert.equal(output.values, Vector(5.0, 6.0, 1.0, 2.0, 5.0, 6.0))
    },
    test("token and position embeddings make repeated tokens position-dependent") {
      val tokenEmbedding    = Embedding.fromValues(2, 2, Vector(1.0, 2.0, 3.0, 4.0), "token")
      val positionEmbedding = Embedding
        .fromValues(3, 2, Vector(0.0, 0.0, 10.0, 20.0, 100.0, 200.0), "position")
      val combined          = new TokenPositionEmbedding(tokenEmbedding, positionEmbedding)
      val output            = combined(Vector(TokenId(1), TokenId(1), TokenId(1)))
      Assert.equal(output.values, Vector(3.0, 4.0, 13.0, 24.0, 103.0, 204.0))
    },
    test("linear applies one affine transform to every row") {
      val linear = Linear.fromValues(
        inputChannels = 2,
        outputChannels = 3,
        weights = Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        biases = Vector(0.5, 1.0, 1.5),
        label = "linear"
      )
      val input  = Tensor.constant(Shape(2, 2), Vector(1.0, 0.0, 0.0, 1.0))
      val output = linear(input)
      Assert.equal(output.shape, Shape(2, 3))
      Assert.equal(output.values, Vector(1.5, 3.0, 4.5, 4.5, 6.0, 7.5))
    },
    test("RMSNorm makes each row mean square approximately one") {
      val norm   = RmsNorm.create(channels = 2, epsilon = 1e-12, label = "norm")
      val input  = Tensor.constant(Shape(2, 2), Vector(3.0, 4.0, 5.0, 12.0))
      val output = norm(input)
      Vector(0, 1).foreach { row =>
        val meanSquare = output.rowValues(row).map(value => value * value).sum / 2.0
        Assert.close(meanSquare, 1.0, tolerance = 1e-10)
      }
    },
    test("RMSNorm input and scale gradients match finite differences") {
      val input = Tensor.parameter(Shape(1, 2), Vector(0.7, -1.2), "input")
      val norm  = RmsNorm.create(2, epsilon = 1e-5, label = "norm")
      val loss  = norm(input).pow(2.0).sum
      loss.backward()

      val step                       = 1e-5
      def raw(first: Double): Double =
        val second  = -1.2
        val inverse = 1.0 / math.sqrt((first * first + second * second) / 2.0 + 1e-5)
        math.pow(first * inverse, 2.0) + math.pow(second * inverse, 2.0)
      val numeric                    = (raw(0.7 + step) - raw(0.7 - step)) / (2.0 * step)
      Assert.close(input.gradientAt(0, 0), numeric, tolerance = 1e-8)
      Assert.isTrue(norm.scale.gradients.forall(_.isFinite))
    },
    test("embedding rejects out-of-range IDs and excessive context") {
      val random      = new SplittableRandom(1L)
      val tokens      = Embedding.random(3, 2, random, "tokens")
      val positions   = Embedding.random(2, 2, random, "positions")
      val combined    = new TokenPositionEmbedding(tokens, positions)
      val idError     = Assert.throws[IllegalArgumentException](tokens(Vector(3)))
      val lengthError = Assert
        .throws[IllegalArgumentException](combined(Vector(TokenId(0), TokenId(1), TokenId(2))))
      Assert.isTrue(idError.getMessage.contains("outside"))
      Assert.isTrue(lengthError.getMessage.contains("exceeds"))
    }
  )
