package learnai.quantization

import learnai.math.MatrixD
import learnai.math.VectorD
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object Int8QuantizationSuite extends TestSuite:
  override val name: String = "Int8Quantization"

  override val tests: Vector[TestCase] = specify(
    test("all-zero rows round-trip exactly with a valid scale") {
      val original = MatrixD.zeros(2, 3)
      val quantized = QuantizedInt8Matrix.quantize(original)
      Assert.equal(quantized.dequantize, original)
      Assert.equal(quantized.rowScales, Vector(1.0, 1.0))
      Assert.isTrue((0 until 2).forall(row => (0 until 3).forall(column => quantized.quantized(row, column) == 0)))
    },
    test("row extrema map to the symmetric int8 range") {
      val original = MatrixD.fromRows(Vector(VectorD(-2.0, 0.0, 2.0)))
      val quantized = QuantizedInt8Matrix.quantize(original)
      Assert.equal(quantized.quantized(0, 0), -127)
      Assert.equal(quantized.quantized(0, 1), 0)
      Assert.equal(quantized.quantized(0, 2), 127)
      Assert.equal(quantized.dequantize, original)
    },
    test("reconstruction error is at most half a row scale before rounding ties") {
      val original = MatrixD.fromRows(
        Vector(
          VectorD(-1.0, -0.3, 0.2, 1.0),
          VectorD(-100.0, -1.0, 2.0, 100.0)
        )
      )
      val quantized = QuantizedInt8Matrix.quantize(original)
      val reconstructed = quantized.dequantize
      (0 until original.rows).foreach { row =>
        (0 until original.columns).foreach { column =>
          val error = math.abs(original(row, column) - reconstructed(row, column))
          Assert.isTrue(error <= quantized.rowScales(row) / 2.0 + 1e-12)
        }
      }
    },
    test("quantized matvec approximates the Double reference") {
      val original = MatrixD.fromRows(
        Vector(
          VectorD(0.1, -0.4, 0.9),
          VectorD(10.0, -20.0, 30.0)
        )
      )
      val input = VectorD(2.0, -1.0, 0.5)
      val reference = original.matvec(input)
      val approximate = QuantizedInt8Matrix.quantize(original).matvec(input)
      Assert.close(approximate(0), reference(0), tolerance = 0.02)
      Assert.close(approximate(1), reference(1), tolerance = 0.5)
    },
    test("metrics report zero for an exact reconstruction") {
      val matrix = MatrixD.fromRows(Vector(VectorD(1.0, 2.0), VectorD(3.0, 4.0)))
      Assert.equal(QuantizationMetrics.compare(matrix, matrix), QuantizationError(0.0, 0.0, 0.0))
    },
    test("int8 payload is smaller once rows are wide enough") {
      val quantized = QuantizedInt8Matrix.quantize(MatrixD.zeros(8, 64))
      Assert.equal(quantized.storageBytes, 8L * 64L + 8L * 8L)
      Assert.equal(quantized.doubleStorageBytes, 8L * 64L * 8L)
      Assert.isTrue(quantized.storageBytes < quantized.doubleStorageBytes)
    },
    test("matvec rejects a vector with the wrong channel count") {
      val matrix = QuantizedInt8Matrix.quantize(MatrixD.zeros(2, 3))
      val error = Assert.throws[IllegalArgumentException](matrix.matvec(VectorD(1.0, 2.0)))
      Assert.isTrue(error.getMessage.contains("shape mismatch"))
    }
  )
