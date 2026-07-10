package learnai.transformer

import java.util.SplittableRandom

import learnai.tensor.Shape
import learnai.tensor.Tensor
import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object TiledAttentionSuite extends TestSuite:
  override val name: String = "TiledAttention"

  /** Independent two-pass stable softmax used as the reference oracle. */
  private def referenceSoftmax(scores: Vector[Double]): Vector[Double] =
    val maximum = scores.max
    val exponentials = scores.map(score => math.exp(score - maximum))
    val total = exponentials.sum
    exponentials.map(_ / total)

  /** Independent naive attention for one row: full scores, then softmax. */
  private def referenceAttendRow(
      query: Vector[Double],
      keys: Vector[Vector[Double]],
      values: Vector[Vector[Double]],
      scale: Double
  ): Vector[Double] =
    val weights = referenceSoftmax(
      keys.map(key => key.zip(query).map(_ * _).sum * scale)
    )
    Vector.tabulate(query.size) { channel =>
      weights.zip(values).map((weight, value) => weight * value(channel)).sum
    }

  private def randomRows(random: SplittableRandom, rows: Int, channels: Int): Vector[Vector[Double]] =
    Vector.fill(rows)(Vector.fill(channels)(random.nextDouble(-1.0, 1.0)))

  override val tests: Vector[TestCase] = Vector(
    test("streamed softmax equals the two-pass reference for every tile size") {
      val random = new SplittableRandom(1L)
      val scores = Vector.fill(11)(random.nextDouble(-4.0, 4.0))
      val expected = referenceSoftmax(scores)
      Vector(1, 2, 3, 5, 11, 64).foreach { tileSize =>
        val streamed = TiledAttention.softmaxStreamed(scores, tileSize)
        streamed.zip(expected).foreach { case (actual, reference) =>
          Assert.close(actual, reference, tolerance = 1e-14)
        }
      }
    },
    test("the rescaling recurrence survives scores that would overflow naive exp") {
      // exp(10001) overflows a Double; the running-maximum rescale never
      // exponentiates a positive number, so the result stays exact.
      val scores = Vector(10000.0, 10001.0, 9999.0, -10000.0)
      val expected = referenceSoftmax(scores)
      val streamed = TiledAttention.softmaxStreamed(scores, tileSize = 2)
      streamed.zip(expected).foreach { case (actual, reference) =>
        Assert.close(actual, reference, tolerance = 1e-14)
        Assert.isTrue(actual.isFinite, "streamed softmax produced a non-finite weight")
      }
    },
    test("folding tiles in any split yields one normalizer state") {
      val scores = Vector(0.5, -1.25, 3.0, 2.0, -0.75)
      val whole = TiledAttention.foldTile(TiledAttention.OnlineSoftmaxState.empty, scores)
      val split = scores.grouped(2).foldLeft(TiledAttention.OnlineSoftmaxState.empty) {
        (state, tile) => TiledAttention.foldTile(state, tile.toVector)
      }
      val singles = scores.foldLeft(TiledAttention.OnlineSoftmaxState.empty) {
        (state, score) => TiledAttention.foldTile(state, Vector(score))
      }
      Assert.equal(whole.maximum, split.maximum)
      Assert.equal(whole.maximum, singles.maximum)
      Assert.close(split.denominator, whole.denominator, tolerance = 1e-14)
      Assert.close(singles.denominator, whole.denominator, tolerance = 1e-14)
    },
    test("one tiled query row matches the naive full-row reference") {
      val random = new SplittableRandom(2L)
      val channels = 4
      val positions = 7
      val query = Vector.fill(channels)(random.nextDouble(-1.0, 1.0))
      val keys = randomRows(random, positions, channels)
      val values = randomRows(random, positions, channels)
      val scale = 1.0 / math.sqrt(channels.toDouble)
      val expected = referenceAttendRow(query, keys, values, scale)
      (1 to positions + 2).foreach { tileSize =>
        val tiled = TiledAttention.attendRowTiled(query, keys, values, scale, tileSize)
        tiled.zip(expected).foreach { case (actual, reference) =>
          Assert.close(actual, reference, tolerance = 1e-12)
        }
      }
    },
    test("tiled causal attention matches the materializing Tensor path") {
      // Cross-implementation oracle: the Chapter 12/19 graph builds the full
      // score matrix, masks it, and row-softmaxes; the tiled path never
      // materializes it. Both must agree row for row.
      val random = new SplittableRandom(3L)
      val time = 5
      val channels = 4
      val queries = randomRows(random, time, channels)
      val keys = randomRows(random, time, channels)
      val values = randomRows(random, time, channels)
      val scale = 1.0 / math.sqrt(channels.toDouble)

      val queryTensor = Tensor.constant(Shape(time, channels), queries.flatten)
      val keyTensor = Tensor.constant(Shape(time, channels), keys.flatten)
      val valueTensor = Tensor.constant(Shape(time, channels), values.flatten)
      val reference = queryTensor
        .matmul(keyTensor.transpose2D)
        .scale(scale)
        .causalMask()
        .softmaxRows
        .matmul(valueTensor)

      val tiled = TiledAttention.causalAttentionTiled(queries, keys, values, scale, tileSize = 2)
      (0 until time).foreach { row =>
        tiled(row).zip(reference.rowValues(row)).foreach { case (actual, expected) =>
          Assert.close(actual, expected, tolerance = 1e-9)
        }
      }
    },
    test("causality holds by construction: future rows cannot leak backwards") {
      val random = new SplittableRandom(4L)
      val queries = randomRows(random, 4, 3)
      val keys = randomRows(random, 4, 3)
      val values = randomRows(random, 4, 3)
      val changedKeys = keys.updated(3, Vector(50.0, -50.0, 25.0))
      val changedValues = values.updated(3, Vector(-9.0, 9.0, -9.0))
      val scale = 1.0 / math.sqrt(3.0)
      val original = TiledAttention.causalAttentionTiled(queries, keys, values, scale, 2)
      val perturbed =
        TiledAttention.causalAttentionTiled(queries, changedKeys, changedValues, scale, 2)
      (0 until 3).foreach { row =>
        Assert.equal(perturbed(row), original(row))
      }
    },
    test("materialization accounting shows the tile-versus-context trade") {
      Assert.equal(TiledAttention.materializedFloatsPerRow(channels = 64, tileSize = 16), 82L)
      // At long contexts the untiled row holds contextLength scores; the
      // tiled row's peak is independent of context length entirely.
      val tiledPeak = TiledAttention.materializedFloatsPerRow(channels = 64, tileSize = 16)
      Assert.isTrue(tiledPeak < 4096L, "tiled peak must not scale with context length")
    },
    test("invalid tiles scores and shapes are rejected") {
      val emptyTile = Assert.throws[IllegalArgumentException] {
        TiledAttention.foldTile(TiledAttention.OnlineSoftmaxState.empty, Vector.empty)
      }
      Assert.isTrue(emptyTile.getMessage.contains("at least one score"))
      val nanScore = Assert.throws[IllegalArgumentException] {
        TiledAttention.foldTile(TiledAttention.OnlineSoftmaxState.empty, Vector(Double.NaN))
      }
      Assert.isTrue(nanScore.getMessage.contains("NaN"))
      val zeroTile = Assert.throws[IllegalArgumentException] {
        TiledAttention.softmaxStreamed(Vector(1.0), tileSize = 0)
      }
      Assert.isTrue(zeroTile.getMessage.contains("positive"))
      val mismatchedKey = Assert.throws[IllegalArgumentException] {
        TiledAttention.attendRowTiled(
          Vector(1.0, 2.0),
          Vector(Vector(1.0)),
          Vector(Vector(1.0, 2.0)),
          scale = 1.0,
          tileSize = 1
        )
      }
      Assert.isTrue(mismatchedKey.getMessage.contains("width"))
      val mismatchedRows = Assert.throws[IllegalArgumentException] {
        TiledAttention.causalAttentionTiled(
          Vector(Vector(1.0)),
          Vector(Vector(1.0), Vector(2.0)),
          Vector(Vector(1.0), Vector(2.0)),
          scale = 1.0,
          tileSize = 1
        )
      }
      Assert.isTrue(mismatchedRows.getMessage.contains("row counts differ"))
    }
  )
