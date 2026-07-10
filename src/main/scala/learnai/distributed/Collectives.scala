package learnai.distributed

import scala.collection.mutable.ArrayBuffer

/** Record of one collective operation, kept for byte-level accounting.
  *
  * `logicalPayloadBytes` is the size of the reduced result every rank ends
  * up holding. `ringBytesPerRank` is what a chunked ring algorithm actually
  * moves per rank to get there — the number a network budget must use,
  * because collectives cost bandwidth per *rank*, not per result.
  */
final case class CollectiveTrace(
    operation: String,
    worldSize: Int,
    elementCount: Int,
    logicalPayloadBytes: Long,
    ringBytesPerRank: Long
):
  require(worldSize > 0, s"world size must be positive: $worldSize")
  require(elementCount > 0, s"element count must be positive: $elementCount")

/** Deterministic in-process simulation of the collectives one training
  * step needs.
  *
  * Real collectives run over a network with nondeterministic arrival
  * order; frameworks then fix a reduction order so replicas stay
  * bit-identical. This simulation keeps that essential property —
  * contributions are always combined in rank order 0, 1, ..., N-1 — while
  * running in one process, so tests can demand *bitwise* replica
  * equivalence instead of tolerances. Every call appends a
  * [[CollectiveTrace]], because the byte accounting is half the chapter.
  */
final class Collectives(val worldSize: Int):
  require(worldSize > 0, s"world size must be positive: $worldSize")

  private val log = ArrayBuffer.empty[CollectiveTrace]

  /** Every collective performed so far, in execution order. */
  def traces: Vector[CollectiveTrace] = log.toVector

  /** Element-wise sum of one contribution per rank, combined in rank order.
    *
    * Fixed order is the determinism contract: floating-point addition is
    * not associative, so "sum in arrival order" would make replicas
    * diverge run to run even with identical inputs.
    */
  def allReduceSum(contributions: Vector[Vector[Double]]): Vector[Double] =
    validate(contributions)
    val elementCount = contributions.head.size
    val result = contributions.head.toArray
    var rank = 1
    while rank < worldSize do
      val contribution = contributions(rank)
      var index = 0
      while index < elementCount do
        result(index) += contribution(index)
        index += 1
      rank += 1
    record("all-reduce-sum", elementCount)
    result.toVector

  /** All-reduce sum divided by the world size on every rank. */
  def allReduceMean(contributions: Vector[Vector[Double]]): Vector[Double] =
    allReduceSum(contributions).map(_ / worldSize.toDouble)

  /** Replicates one root value to every rank. */
  def broadcast(root: Vector[Double]): Vector[Vector[Double]] =
    require(root.nonEmpty, "broadcast requires at least one element")
    record("broadcast", root.size)
    Vector.fill(worldSize)(root)

  /** Bytes one rank moves in a chunked ring all-reduce.
    *
    * The ring splits the buffer into `worldSize` chunks of
    * `ceil(elements / worldSize)` elements. Reduce-scatter sends
    * `worldSize - 1` chunks per rank and all-gather sends the same again:
    * `2 (N-1) * chunkBytes`, which approaches `2x` the buffer for large
    * `N` — the reason gradient buckets are sized in tens of megabytes
    * rather than per tensor.
    */
  def ringAllReduceBytesPerRank(elementCount: Int): Long =
    require(elementCount > 0, s"element count must be positive: $elementCount")
    val chunkElements = (elementCount + worldSize - 1) / worldSize
    2L * (worldSize - 1).toLong * chunkElements.toLong * java.lang.Double.BYTES.toLong

  private def record(operation: String, elementCount: Int): Unit =
    log += CollectiveTrace(
      operation,
      worldSize,
      elementCount,
      Math.multiplyExact(elementCount.toLong, java.lang.Double.BYTES.toLong),
      ringAllReduceBytesPerRank(elementCount)
    )

  private def validate(contributions: Vector[Vector[Double]]): Unit =
    require(
      contributions.size == worldSize,
      s"expected $worldSize contributions, got ${contributions.size}"
    )
    require(contributions.head.nonEmpty, "collective requires at least one element")
    val elementCount = contributions.head.size
    contributions.zipWithIndex.foreach { case (contribution, rank) =>
      require(
        contribution.size == elementCount,
        s"rank $rank contributed ${contribution.size} elements, expected $elementCount"
      )
    }
