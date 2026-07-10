package learnai.serving

import scala.collection.mutable.ArrayBuffer

/** A fixed pool of key/value pages shared by every request on one server.
  *
  * Chapter 24's cache reserves `maximumContextLength` rows per request up
  * front, so a server's request capacity is decided by the *worst-case*
  * context even when most requests are short. Paged storage (vLLM's
  * PagedAttention insight) splits KV storage into fixed-size pages and
  * hands them out on demand: a request holds pages proportional to its
  * *actual* length, and waste is bounded by one partial page per sequence.
  *
  * The pool owns all storage and a reference count per page, so full pages
  * can be shared between sequences (prefix reuse). Two invariants make
  * sharing safe without copy-on-write machinery:
  *
  *   - appends only ever write into a sequence's final, partially filled
  *     page;
  *   - [[PagedKvSequence.fork]] shares only *full* pages and copies the
  *     partial one, so any page with reference count above one is full and
  *     therefore immutable.
  *
  * Byte attribution lives here and not on sequences: with sharing, summing
  * per-sequence footprints double-counts, and the pool's allocated page
  * count is the ground truth a scheduler must budget against.
  */
final class KvPagePool(val pageCount: Int, val pageSize: Int, val channels: Int):
  require(pageCount > 0, s"page count must be positive: $pageCount")
  require(pageSize > 0, s"page size must be positive: $pageSize")
  require(channels > 0, s"page channels must be positive: $channels")

  private val slotCount = Math.multiplyExact(pageCount, pageSize)
  private val keys = new Array[Double](Math.multiplyExact(slotCount, channels))
  private val values = new Array[Double](Math.multiplyExact(slotCount, channels))
  private val referenceCounts = new Array[Int](pageCount)
  private val freePages = ArrayBuffer.range(0, pageCount)

  def freePageCount: Int = freePages.size

  def allocatedPageCount: Int = pageCount - freePages.size

  /** Payload bytes of one page: `2 * pageSize * channels * 8`. */
  def pagePayloadBytes: Long =
    2L * pageSize.toLong * channels.toLong * java.lang.Double.BYTES.toLong

  /** Ground-truth allocated payload across all sequences, sharing included once. */
  def allocatedPayloadBytes: Long =
    Math.multiplyExact(allocatedPageCount.toLong, pagePayloadBytes)

  def totalPayloadBytes: Long = Math.multiplyExact(pageCount.toLong, pagePayloadBytes)

  private[serving] def referenceCount(page: Int): Int = referenceCounts(page)

  private[serving] def allocate(): Either[String, Int] =
    if freePages.isEmpty then
      Left(s"page pool exhausted: all $pageCount pages are allocated")
    else
      val page = freePages.remove(freePages.size - 1)
      referenceCounts(page) = 1
      Right(page)

  private[serving] def retain(page: Int): Unit =
    require(referenceCounts(page) > 0, s"cannot retain unallocated page $page")
    referenceCounts(page) += 1

  private[serving] def release(page: Int): Unit =
    require(referenceCounts(page) > 0, s"cannot release unallocated page $page")
    referenceCounts(page) -= 1
    if referenceCounts(page) == 0 then freePages += page

  private[serving] def writeRow(
      page: Int,
      slot: Int,
      key: Vector[Double],
      value: Vector[Double]
  ): Unit =
    val offset = rowOffset(page, slot)
    var channel = 0
    while channel < channels do
      keys(offset + channel) = key(channel)
      values(offset + channel) = value(channel)
      channel += 1

  private[serving] def copyRow(
      fromPage: Int,
      toPage: Int,
      slot: Int
  ): Unit =
    val from = rowOffset(fromPage, slot)
    val to = rowOffset(toPage, slot)
    System.arraycopy(keys, from, keys, to, channels)
    System.arraycopy(values, from, values, to, channels)

  private[serving] def keyAt(page: Int, slot: Int, channel: Int): Double =
    keys(rowOffset(page, slot) + channel)

  private[serving] def valueAt(page: Int, slot: Int, channel: Int): Double =
    values(rowOffset(page, slot) + channel)

  private def rowOffset(page: Int, slot: Int): Int =
    require(page >= 0 && page < pageCount, s"page $page outside [0, $pageCount)")
    require(slot >= 0 && slot < pageSize, s"slot $slot outside [0, $pageSize)")
    (page * pageSize + slot) * channels

/** One request's logically contiguous KV sequence over pool pages.
  *
  * The page table maps logical position `p` to page `table(p / pageSize)`,
  * slot `p % pageSize` — the same indirection a paged attention kernel
  * performs per score. Sequences are single-owner and append-only;
  * [[release]] returns pages to the pool and permanently retires the
  * sequence, which is the *mechanism* a serving scheduler's eviction or
  * preemption policy (Chapter 27e) invokes.
  */
final class PagedKvSequence private (val pool: KvPagePool):
  private val pageTable = ArrayBuffer.empty[Int]
  private var currentLength = 0
  private var active = true

  def length: Int = currentLength

  def isReleased: Boolean = !active

  /** Pages currently mapped by this sequence, sharing counted once each. */
  def mappedPageCount: Int = pageTable.size

  /** Slots reserved for this sequence, used or not. */
  def allocatedSlots: Int = pageTable.size * pool.pageSize

  /** Internal fragmentation: reserved slots the sequence does not use.
    * Bounded by `pageSize - 1` for any sequence length — the whole point
    * of small pages versus Chapter 24's full-context reservation.
    */
  def wastedSlots: Int = allocatedSlots - currentLength

  /** Appends one key/value row, allocating a fresh page on a page boundary. */
  def append(key: Vector[Double], value: Vector[Double]): Either[String, Unit] =
    requireActive()
    require(key.size == pool.channels, s"key width ${key.size} != channels ${pool.channels}")
    require(
      value.size == pool.channels,
      s"value width ${value.size} != channels ${pool.channels}"
    )
    val slot = currentLength % pool.pageSize
    val pageReady =
      if slot == 0 then pool.allocate().map(page => pageTable += page)
      else Right(())
    pageReady.map { _ =>
      pool.writeRow(pageTable.last, slot, key, value)
      currentLength += 1
    }

  def keyAt(position: Int, channel: Int): Double =
    pool.keyAt(pageFor(position), position % pool.pageSize, requireChannel(channel))

  def valueAt(position: Int, channel: Int): Double =
    pool.valueAt(pageFor(position), position % pool.pageSize, requireChannel(channel))

  /** Creates a sequence sharing this one's full prefix pages.
    *
    * Full pages are shared by reference count; a partially filled final
    * page is copied into a freshly allocated page so both sequences can
    * keep appending independently. On pool exhaustion the fork fails
    * atomically: every retained page is released and the parent is
    * untouched.
    */
  def fork(): Either[String, PagedKvSequence] =
    requireActive()
    val child = new PagedKvSequence(pool)
    val fullPages = currentLength / pool.pageSize
    var index = 0
    while index < fullPages do
      val page = pageTable(index)
      pool.retain(page)
      child.pageTable += page
      index += 1

    val partialSlots = currentLength % pool.pageSize
    val partialCopied =
      if partialSlots == 0 then Right(())
      else
        pool.allocate() match
          case Left(problem) =>
            child.pageTable.foreach(pool.release)
            child.pageTable.clear()
            Left(problem)
          case Right(fresh) =>
            val source = pageTable(fullPages)
            var slot = 0
            while slot < partialSlots do
              pool.copyRow(source, fresh, slot)
              slot += 1
            child.pageTable += fresh
            Right(())
    partialCopied.map { _ =>
      child.currentLength = currentLength
      child
    }

  /** Returns every mapped page to the pool and retires the sequence. */
  def release(): Unit =
    requireActive()
    pageTable.foreach(pool.release)
    pageTable.clear()
    currentLength = 0
    active = false

  private def pageFor(position: Int): Int =
    requireActive()
    require(
      position >= 0 && position < currentLength,
      s"position $position outside [0, $currentLength)"
    )
    pageTable(position / pool.pageSize)

  private def requireChannel(channel: Int): Int =
    require(
      channel >= 0 && channel < pool.channels,
      s"channel $channel outside [0, ${pool.channels})"
    )
    channel

  private def requireActive(): Unit =
    require(active, "sequence has been released back to the pool")

object PagedKvSequence:
  /** Starts an empty sequence; pages are allocated lazily on first append. */
  def empty(pool: KvPagePool): PagedKvSequence = new PagedKvSequence(pool)
