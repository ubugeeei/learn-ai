package learnai.tensor

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object TensorSuite extends TestSuite:
  override val name: String = "Tensor"

  override val tests: Vector[TestCase] = Vector(
    test("shape maps coordinates to row-major offsets") {
      val shape = Shape(2, 3, 4)
      Assert.equal(shape.size, 24)
      Assert.equal(shape.offset(Vector(1, 2, 3)), 23)
      Assert.equal(shape.coordinates(23), Vector(1, 2, 3))
      Assert.equal(Shape.scalar.size, 1)
    },
    test("element-wise backward accumulates shared tensor paths") {
      val left = Tensor.parameter(Shape(2), Vector(2.0, 3.0), "left")
      val right = Tensor.parameter(Shape(2), Vector(4.0, 5.0), "right")
      val loss = (left.hadamard(right) + left).sum
      loss.backward()
      Assert.equal(left.gradients, Vector(5.0, 6.0))
      Assert.equal(right.gradients, Vector(2.0, 3.0))
    },
    test("separate microbatch graphs accumulate only trainable leaf gradients") {
      val accumulated = Tensor.parameter(Shape(2), Vector(1.0, -2.0), "accumulated")
      accumulated.clearGradients()
      accumulated.hadamard(Tensor.constant(Shape(2), Vector(2.0, 3.0))).sum
        .scale(0.5)
        .backwardAccumulating()
      accumulated.hadamard(Tensor.constant(Shape(2), Vector(4.0, -1.0))).sum
        .scale(0.5)
        .backwardAccumulating()

      val fullBatch = Tensor.parameter(Shape(2), Vector(1.0, -2.0), "fullBatch")
      val combined = (
        fullBatch.hadamard(Tensor.constant(Shape(2), Vector(2.0, 3.0))).sum +
          fullBatch.hadamard(Tensor.constant(Shape(2), Vector(4.0, -1.0))).sum
      ).scale(0.5)
      combined.backward()

      Assert.equal(accumulated.gradients, fullBatch.gradients)
      Assert.equal(accumulated.gradients, Vector(3.0, 1.0))
      accumulated.pow(2.0).sum.backward()
      Assert.equal(accumulated.gradients, Vector(2.0, -4.0))
    },
    test("matrix multiplication backward computes both operand gradients") {
      val left = Tensor.parameter(Shape(2, 2), Vector(1.0, 2.0, 3.0, 4.0), "left")
      val right = Tensor.parameter(Shape(2, 1), Vector(5.0, 6.0), "right")
      val output = left.matmul(right)
      Assert.equal(output.shape, Shape(2, 1))
      Assert.equal(output.values, Vector(17.0, 39.0))

      output.sum.backward()
      Assert.equal(left.gradients, Vector(5.0, 6.0, 5.0, 6.0))
      Assert.equal(right.gradients, Vector(4.0, 6.0))
    },
    test("reshape and transpose preserve gradient routing") {
      val input = Tensor.parameter(Shape(2, 3), Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), "x")
      val output = input.transpose2D.reshape(Shape(6)).sum
      output.backward()
      Assert.equal(input.gradients, Vector.fill(6)(1.0))
    },
    test("autodiff matches a finite difference for a matrix element") {
      val left = Tensor.parameter(Shape(1, 2), Vector(0.7, -0.2), "left")
      val right = Tensor.constant(Shape(2, 1), Vector(1.5, -2.0))
      val loss = left.matmul(right).pow(2.0).mean
      loss.backward()

      val step = 1e-5
      def raw(first: Double): Double =
        val output = first * 1.5 + -0.2 * -2.0
        output * output
      val numeric = (raw(0.7 + step) - raw(0.7 - step)) / (2.0 * step)
      Assert.close(left.gradientAt(0, 0), numeric, tolerance = 1e-8)
    },
    test("binary operations reject implicit broadcasting") {
      val error = Assert.throws[IllegalArgumentException] {
        Tensor.fill(Shape(2, 1), 1.0) + Tensor.fill(Shape(2), 1.0)
      }
      Assert.isTrue(error.getMessage.contains("requires equal shapes"))
    },
    test("gatherRows backward accumulates repeated row selections") {
      val table = Tensor.parameter(
        Shape(3, 2),
        Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "table"
      )
      val selected = table.gatherRows(Vector(2, 0, 2))
      Assert.equal(selected.values, Vector(5.0, 6.0, 1.0, 2.0, 5.0, 6.0))
      selected.sum.backward()
      Assert.equal(table.gradients, Vector(1.0, 1.0, 0.0, 0.0, 2.0, 2.0))
    },
    test("cross entropy is stable and each row gradient sums to zero") {
      val logits = Tensor.parameter(
        Shape(2, 3),
        Vector(10000.0, 10001.0, 9999.0, -4.0, 2.0, 1.0),
        "logits"
      )
      val loss = logits.crossEntropy(Vector(1, 2))
      Assert.isTrue(loss.valueAtFlat(0).isFinite)
      loss.backward()
      Assert.close(logits.gradients.take(3).sum, 0.0)
      Assert.close(logits.gradients.drop(3).sum, 0.0)
    },
    test("cross entropy gradient agrees with a finite difference") {
      val raw = Vector(0.2, -0.3, 1.1)
      val logits = Tensor.parameter(Shape(1, 3), raw, "logits")
      val loss = logits.crossEntropy(Vector(2))
      loss.backward()

      val step = 1e-5
      def rawLoss(first: Double): Double =
        val values = Vector(first, raw(1), raw(2))
        val maximum = values.max
        maximum + math.log(values.map(value => math.exp(value - maximum)).sum) - raw(2)
      val numeric = (rawLoss(raw.head + step) - rawLoss(raw.head - step)) / (2.0 * step)
      Assert.close(logits.gradientAt(0, 0), numeric, tolerance = 1e-8)
    },
    test("addRowVector reduces row gradients back into the vector") {
      val input = Tensor.parameter(Shape(2, 3), Vector.fill(6)(0.0), "input")
      val row = Tensor.parameter(Shape(3), Vector(1.0, 2.0, 3.0), "row")
      val output = input.addRowVector(row)
      Assert.equal(output.values, Vector(1.0, 2.0, 3.0, 1.0, 2.0, 3.0))
      output.sum.backward()
      Assert.equal(input.gradients, Vector.fill(6)(1.0))
      Assert.equal(row.gradients, Vector(2.0, 2.0, 2.0))
    },
    test("softmaxRows is stable and its backward is shift-invariant") {
      val logits = Tensor.parameter(Shape(2, 3), Vector(10000.0, 10001.0, 9999.0, 1.0, 1.0, 1.0), "x")
      val probabilities = logits.softmaxRows
      Assert.close(probabilities.rowValues(0).sum, 1.0)
      Assert.close(probabilities.rowValues(1).sum, 1.0)
      probabilities.hadamard(
        Tensor.constant(Shape(2, 3), Vector(1.0, 2.0, 4.0, -1.0, 3.0, 2.0))
      ).sum.backward()
      Assert.close(logits.gradients.take(3).sum, 0.0)
      Assert.close(logits.gradients.drop(3).sum, 0.0)
    },
    test("causal mask blocks future gradients") {
      val scores = Tensor.parameter(Shape(3, 3), Vector.range(0, 9).map(_.toDouble), "scores")
      scores.causalMask().sum.backward()
      Assert.equal(
        scores.gradients,
        Vector(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
      )
    },
    test("slice and concatenate columns round-trip values and gradients") {
      val input = Tensor.parameter(Shape(2, 4), Vector.range(0, 8).map(_.toDouble), "input")
      val joined = Tensor.concatenateColumns(
        Vector(input.sliceColumns(0, 1), input.sliceColumns(1, 4))
      )
      Assert.equal(joined.values, input.values)
      joined.sum.backward()
      Assert.equal(input.gradients, Vector.fill(8)(1.0))
    },
    test("backward requires a scalar output") {
      val tensor = Tensor.fill(Shape(2), 1.0)
      val error = Assert.throws[IllegalArgumentException](tensor.backward())
      val accumulatingError = Assert.throws[IllegalArgumentException](tensor.backwardAccumulating())
      Assert.isTrue(error.getMessage.contains("scalar output"))
      Assert.isTrue(accumulatingError.getMessage.contains("scalar output"))
    }
  )
