package learnai.foundations

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.AtomicMoveNotSupportedException
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

import scala.concurrent.duration.FiniteDuration
import scala.concurrent.duration.DurationInt
import scala.util.control.NonFatal

/** Observable JVM process facts relevant to capacity and reproducibility reports. */
final case class JvmSnapshot(
    availableProcessors: Int,
    maximumHeapBytes: Long,
    allocatedHeapBytes: Long,
    freeAllocatedHeapBytes: Long,
    javaVersion: String,
    vmName: String
)

/** Bounded results from work executed on an owned worker thread. */
enum BoundedResult[+A]:
  case Completed(value: A)
  case TimedOut
  case Failed(message: String)

/** Explicit JVM boundaries for runtime observation, bounded concurrency, and atomic files. */
object JvmSystems:
  /** Captures process-level values without claiming they equal live object usage. */
  def snapshot(): JvmSnapshot =
    val runtime = Runtime.getRuntime
    JvmSnapshot(
      runtime.availableProcessors(),
      runtime.maxMemory(),
      runtime.totalMemory(),
      runtime.freeMemory(),
      sys.props.getOrElse("java.runtime.version", "unknown"),
      sys.props.getOrElse("java.vm.name", "unknown")
    )

  /** Executes one task with a deadline and always shuts down its executor. */
  def runBounded[A](timeout: FiniteDuration)(body: => A): BoundedResult[A] =
    require(timeout.length > 0L, s"timeout must be positive: $timeout")
    val executor = Executors.newSingleThreadExecutor()
    val future   = executor.submit(
      new Callable[A]:
        override def call(): A = body
    )
    try BoundedResult.Completed(future.get(timeout.length, timeout.unit))
    catch
      case _: TimeoutException =>
        future.cancel(true)
        BoundedResult.TimedOut
      case NonFatal(error)     => BoundedResult.Failed(Option(error.getCause).getOrElse(error).toString)
    finally
      executor.shutdownNow()
      val _ = executor.awaitTermination(1, TimeUnit.SECONDS)

  /** Writes UTF-8 through a sibling temporary file and atomically replaces the destination. */
  def writeUtf8Atomically(path: Path, content: String): Either[String, Long] =
    try
      val absolute  = path.toAbsolutePath
      val parent    = Option(absolute.getParent).getOrElse(Path.of(".").toAbsolutePath)
      Files.createDirectories(parent)
      val temporary = Files.createTempFile(parent, s".${absolute.getFileName}.", ".tmp")
      try
        val bytes = content.getBytes(StandardCharsets.UTF_8)
        val _     = Files.write(temporary, bytes)
        try
          val _ = Files.move(
            temporary,
            absolute,
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING
          )
        catch
          case _: AtomicMoveNotSupportedException =>
            val _ = Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING)
        Right(bytes.length.toLong)
      finally
        val _ = Files.deleteIfExists(temporary)
    catch case NonFatal(error) => Left(s"atomic UTF-8 write failed: ${error.getMessage}")

/** Prints JVM capacity facts and demonstrates bounded task ownership. */
def runJvmSystemsLab(): Unit =
  val current = JvmSystems.snapshot()
  println(s"Java:          ${current.javaVersion}")
  println(s"VM:            ${current.vmName}")
  println(s"processors:    ${current.availableProcessors}")
  println(s"maximum heap:  ${current.maximumHeapBytes} bytes")
  println(s"allocated heap:${current.allocatedHeapBytes} bytes")
  println(s"bounded task:  ${JvmSystems.runBounded(1.second)(21 * 2)}")
