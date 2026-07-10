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
    test("backward requires a scalar output") {
      val tensor = Tensor.fill(Shape(2), 1.0)
      val error = Assert.throws[IllegalArgumentException](tensor.backward())
      Assert.isTrue(error.getMessage.contains("scalar output"))
    }
  )
