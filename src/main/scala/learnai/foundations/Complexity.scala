package learnai.foundations

/**
 * Result of an algorithm together with a deterministic unit-of-work count.
 *
 * Counting comparisons avoids confusing machine noise with asymptotic growth.
 */
final case class CountedResult[+A](value: A, comparisons: Int):
  require(comparisons >= 0, s"comparison count must be non-negative: $comparisons")

/** Small reference algorithms used to connect loops, memory layout, and cost models. */
object Complexity:
  /** Scans left to right and returns the first matching index. */
  def linearSearch[A](values: IndexedSeq[A], target: A): CountedResult[Option[Int]] =
    var index = 0
    while index < values.size do
      if values(index) == target then return CountedResult(Some(index), index + 1)
      index += 1
    CountedResult(None, values.size)

  /** Searches a sorted sequence by discarding half of the remaining range per comparison. */
  def binarySearch(values: IndexedSeq[Int], target: Int): CountedResult[Option[Int]] =
    var low         = 0
    var high        = values.size - 1
    var comparisons = 0
    while low <= high do
      val middle = low + (high - low) / 2
      comparisons += 1
      val value  = values(middle)
      if value == target then return CountedResult(Some(middle), comparisons)
      else if value < target then low = middle + 1
      else high = middle - 1
    CountedResult(None, comparisons)

  /** Row-major flat offset for a logical matrix coordinate. */
  def rowMajorOffset(row: Int, column: Int, columns: Int): Int =
    require(row >= 0, s"row must be non-negative: $row")
    require(column >= 0 && column < columns, s"column $column outside [0, $columns)")
    Math.addExact(Math.multiplyExact(row, columns), column)

  /** Flat-index visitation order for row-first or column-first matrix traversal. */
  def traversalOffsets(rows: Int, columns: Int, rowFirst: Boolean): Vector[Int] =
    require(rows >= 0 && columns >= 0, s"matrix dimensions must be non-negative: $rows x $columns")
    if rowFirst then Vector.tabulate(Math.multiplyExact(rows, columns))(identity)
    else
      Vector.tabulate(columns, rows)((column, row) => rowMajorOffset(row, column, columns)).flatten

  /**
   * Total element copies made by a doubling array while appending `elements` values.
   *
   * Capacity starts at one. Every growth copies all existing elements, making individual growth
   * expensive but total append cost linear.
   */
  def doublingArrayCopies(elements: Int): Long =
    require(elements >= 0, s"element count must be non-negative: $elements")
    var capacity = 1
    var size     = 0
    var copies   = 0L
    while size < elements do
      if size == capacity then
        copies = Math.addExact(copies, size.toLong)
        capacity = Math.multiplyExact(capacity, 2)
      size += 1
    copies

  /** Payload bytes for densely stored 64-bit floating-point values. */
  def doublePayloadBytes(elements: Long): Long =
    require(elements >= 0L, s"element count must be non-negative: $elements")
    Math.multiplyExact(elements, java.lang.Double.BYTES.toLong)

/** Prints comparison growth and memory-layout evidence without relying on benchmark timing. */
def runComplexityLab(): Unit =
  println("size | linear miss | binary miss | Double payload")
  Vector(8, 64, 1024, 16384).foreach { size =>
    val values = Vector.range(0, size)
    val linear = Complexity.linearSearch(values, -1)
    val binary = Complexity.binarySearch(values, -1)
    println(f"$size%5d | ${linear.comparisons}%11d | ${binary.comparisons}%11d | " + f"${Complexity
        .doublePayloadBytes(size.toLong)}%14d B")
  }
  println(s"row-first offsets:    ${Complexity.traversalOffsets(2, 3, rowFirst = true)}")
  println(s"column-first offsets: ${Complexity.traversalOffsets(2, 3, rowFirst = false)}")
