package learnai.serving

import java.util.SplittableRandom

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object PagedKvCacheSuite extends TestSuite:
  override val name: String = "PagedKvCache"

  private def randomRow(random: SplittableRandom, channels: Int): Vector[Double] =
    Vector.fill(channels)(random.nextDouble(-1.0, 1.0))

  override val tests: Vector[TestCase] = Vector(
    test("interleaved sequences read back exactly what each one appended") {
      // Oracle: a trivially correct list of appended rows per sequence.
      // Interleaving forces the two page tables to alternate pool pages, so
      // any cross-sequence indexing bug corrupts one of the comparisons.
      val pool = new KvPagePool(pageCount = 8, pageSize = 2, channels = 3)
      val random = new SplittableRandom(1L)
      val first = PagedKvSequence.empty(pool)
      val second = PagedKvSequence.empty(pool)
      var firstReference = Vector.empty[(Vector[Double], Vector[Double])]
      var secondReference = Vector.empty[(Vector[Double], Vector[Double])]

      (0 until 5).foreach { step =>
        val firstRow = (randomRow(random, 3), randomRow(random, 3))
        Assert.isTrue(first.append(firstRow._1, firstRow._2).isRight)
        firstReference :+= firstRow
        if step < 3 then
          val secondRow = (randomRow(random, 3), randomRow(random, 3))
          Assert.isTrue(second.append(secondRow._1, secondRow._2).isRight)
          secondReference :+= secondRow
      }

      def verify(sequence: PagedKvSequence, reference: Vector[(Vector[Double], Vector[Double])]): Unit =
        Assert.equal(sequence.length, reference.size)
        reference.zipWithIndex.foreach { case ((key, value), position) =>
          (0 until 3).foreach { channel =>
            Assert.equal(sequence.keyAt(position, channel), key(channel))
            Assert.equal(sequence.valueAt(position, channel), value(channel))
          }
        }
      verify(first, firstReference)
      verify(second, secondReference)
    },
    test("pages are allocated exactly on boundaries and waste is bounded") {
      val pool = new KvPagePool(pageCount = 8, pageSize = 4, channels = 2)
      val sequence = PagedKvSequence.empty(pool)
      val row = Vector(1.0, 2.0)
      Assert.equal(sequence.mappedPageCount, 0)

      Assert.isTrue(sequence.append(row, row).isRight)
      Assert.equal(sequence.mappedPageCount, 1)
      Assert.equal(sequence.wastedSlots, 3)

      (0 until 3).foreach(_ => Assert.isTrue(sequence.append(row, row).isRight))
      Assert.equal(sequence.mappedPageCount, 1)
      Assert.equal(sequence.wastedSlots, 0)

      Assert.isTrue(sequence.append(row, row).isRight)
      Assert.equal(sequence.mappedPageCount, 2)
      Assert.equal(sequence.wastedSlots, 3)
      Assert.equal(pool.allocatedPageCount, 2)

      // Internal fragmentation never reaches a full page.
      (0 until 9).foreach { _ =>
        Assert.isTrue(sequence.wastedSlots < pool.pageSize)
        Assert.isTrue(sequence.append(row, row).isRight)
      }
    },
    test("pool exhaustion is an explicit error and released pages are reused") {
      val pool = new KvPagePool(pageCount = 2, pageSize = 2, channels = 1)
      val greedy = PagedKvSequence.empty(pool)
      (0 until 4).foreach(_ => Assert.isTrue(greedy.append(Vector(1.0), Vector(2.0)).isRight))
      Assert.equal(pool.freePageCount, 0)

      val starved = PagedKvSequence.empty(pool)
      val refused = Assert.left(starved.append(Vector(3.0), Vector(4.0)))
      Assert.isTrue(refused.contains("exhausted"))
      Assert.equal(starved.length, 0)

      greedy.release()
      Assert.equal(pool.freePageCount, 2)
      Assert.isTrue(starved.append(Vector(3.0), Vector(4.0)).isRight)
      Assert.equal(starved.keyAt(0, 0), 3.0)
    },
    test("forking shares full prefix pages and copies only the partial page") {
      val pool = new KvPagePool(pageCount = 8, pageSize = 2, channels = 2)
      val random = new SplittableRandom(2L)
      val parent = PagedKvSequence.empty(pool)
      val rows = Vector.fill(5)((randomRow(random, 2), randomRow(random, 2)))
      rows.foreach { (key, value) => Assert.isTrue(parent.append(key, value).isRight) }
      Assert.equal(pool.allocatedPageCount, 3)

      val child = Assert.right(parent.fork())
      // Two full pages shared, one partial page copied for the child.
      Assert.equal(pool.allocatedPageCount, 4)
      Assert.equal(child.length, 5)
      rows.zipWithIndex.foreach { case ((key, value), position) =>
        (0 until 2).foreach { channel =>
          Assert.equal(child.keyAt(position, channel), key(channel))
          Assert.equal(child.valueAt(position, channel), value(channel))
        }
      }

      // Divergence: each side appends its own row without disturbing the other.
      Assert.isTrue(parent.append(Vector(10.0, 11.0), Vector(12.0, 13.0)).isRight)
      Assert.isTrue(child.append(Vector(-10.0, -11.0), Vector(-12.0, -13.0)).isRight)
      Assert.equal(parent.keyAt(5, 0), 10.0)
      Assert.equal(child.keyAt(5, 0), -10.0)
      Assert.equal(parent.keyAt(4, 0), rows(4)._1(0))
      Assert.equal(child.keyAt(4, 0), rows(4)._1(0))

      // Releasing the parent keeps shared pages alive for the child.
      parent.release()
      Assert.isTrue(pool.allocatedPageCount > 0)
      Assert.equal(child.keyAt(0, 0), rows(0)._1(0))
      child.release()
      Assert.equal(pool.allocatedPageCount, 0)
      Assert.equal(pool.freePageCount, 8)
    },
    test("a fork that cannot copy its partial page fails atomically") {
      val pool = new KvPagePool(pageCount = 2, pageSize = 2, channels = 1)
      val parent = PagedKvSequence.empty(pool)
      (0 until 3).foreach { index =>
        Assert.isTrue(parent.append(Vector(index.toDouble), Vector(0.0)).isRight)
      }
      Assert.equal(pool.freePageCount, 0)

      val refused = Assert.left(parent.fork())
      Assert.isTrue(refused.contains("exhausted"))
      // Atomicity: no leaked reference counts, parent untouched.
      Assert.equal(pool.allocatedPageCount, 2)
      Assert.equal(parent.length, 3)
      Assert.equal(parent.keyAt(2, 0), 2.0)
      parent.release()
      Assert.equal(pool.freePageCount, 2)
    },
    test("a fork at a page boundary shares everything and allocates nothing") {
      val pool = new KvPagePool(pageCount = 2, pageSize = 2, channels = 1)
      val parent = PagedKvSequence.empty(pool)
      (0 until 2).foreach { index =>
        Assert.isTrue(parent.append(Vector(index.toDouble), Vector(0.0)).isRight)
      }
      Assert.equal(pool.allocatedPageCount, 1)
      val child = Assert.right(parent.fork())
      Assert.equal(pool.allocatedPageCount, 1)
      Assert.equal(child.keyAt(1, 0), 1.0)
      parent.release()
      child.release()
      Assert.equal(pool.freePageCount, 2)
    },
    test("byte accounting lives at the pool and follows the page formulas") {
      val pool = new KvPagePool(pageCount = 4, pageSize = 8, channels = 6)
      Assert.equal(pool.pagePayloadBytes, 2L * 8L * 6L * 8L)
      Assert.equal(pool.totalPayloadBytes, 4L * pool.pagePayloadBytes)
      Assert.equal(pool.allocatedPayloadBytes, 0L)
      val sequence = PagedKvSequence.empty(pool)
      Assert.isTrue(sequence.append(Vector.fill(6)(1.0), Vector.fill(6)(2.0)).isRight)
      Assert.equal(pool.allocatedPayloadBytes, pool.pagePayloadBytes)
    },
    test("released sequences and invalid arguments are rejected") {
      val pool = new KvPagePool(pageCount = 2, pageSize = 2, channels = 2)
      val sequence = PagedKvSequence.empty(pool)
      Assert.isTrue(sequence.append(Vector(1.0, 2.0), Vector(3.0, 4.0)).isRight)

      val wrongWidth = Assert.throws[IllegalArgumentException] {
        sequence.append(Vector(1.0), Vector(3.0, 4.0))
      }
      Assert.isTrue(wrongWidth.getMessage.contains("width"))
      val outsidePosition = Assert.throws[IllegalArgumentException](sequence.keyAt(1, 0))
      Assert.isTrue(outsidePosition.getMessage.contains("outside"))
      val outsideChannel = Assert.throws[IllegalArgumentException](sequence.valueAt(0, 2))
      Assert.isTrue(outsideChannel.getMessage.contains("outside"))

      sequence.release()
      Assert.isTrue(sequence.isReleased)
      Vector(
        Assert.throws[IllegalArgumentException](sequence.keyAt(0, 0)),
        Assert.throws[IllegalArgumentException] {
          sequence.append(Vector(1.0, 2.0), Vector(3.0, 4.0))
        },
        Assert.throws[IllegalArgumentException](sequence.fork()),
        Assert.throws[IllegalArgumentException](sequence.release())
      ).foreach { error =>
        Assert.isTrue(error.getMessage.contains("released"))
      }

      val badPool = Assert.throws[IllegalArgumentException] {
        new KvPagePool(pageCount = 0, pageSize = 2, channels = 2)
      }
      Assert.isTrue(badPool.getMessage.contains("positive"))
    }
  )
