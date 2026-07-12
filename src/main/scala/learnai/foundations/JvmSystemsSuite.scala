package learnai.foundations

import java.nio.charset.StandardCharsets
import java.nio.file.Files

import scala.concurrent.duration.DurationInt

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object JvmSystemsSuite extends TestSuite:
  override val name: String = "JvmSystems"

  override val tests: Vector[TestCase] = specify(
    test("runtime snapshot reports internally valid capacity values") {
      val snapshot = JvmSystems.snapshot()
      Assert.isTrue(snapshot.availableProcessors > 0)
      Assert.isTrue(snapshot.maximumHeapBytes > 0L)
      Assert.isTrue(snapshot.allocatedHeapBytes > 0L)
      Assert.isTrue(snapshot.freeAllocatedHeapBytes >= 0L)
      Assert.isTrue(snapshot.javaVersion.nonEmpty)
      Assert.isTrue(snapshot.vmName.nonEmpty)
    },
    test("bounded execution returns a completed value") {
      Assert.equal(JvmSystems.runBounded(1.second)(6 * 7), BoundedResult.Completed(42))
    },
    test("bounded execution reports failure without throwing through the boundary") {
      val result = JvmSystems.runBounded(1.second)(throw new IllegalStateException("broken"))
      result match
        case BoundedResult.Failed(message) => Assert.isTrue(message.contains("broken"), message)
        case other                         => throw new AssertionError(s"expected failure, got $other")
    },
    test("bounded execution interrupts work after its deadline") {
      val result = JvmSystems.runBounded(1.millis)(Thread.sleep(1000L))
      Assert.equal(result, BoundedResult.TimedOut)
    },
    test("atomic UTF-8 write replaces content and reports exact bytes") {
      val directory = Files.createTempDirectory("learnai-jvm-")
      val path      = directory.resolve("result.txt")
      Assert.right(JvmSystems.writeUtf8Atomically(path, "old"))
      val bytes     = Assert.right(JvmSystems.writeUtf8Atomically(path, "新しい"))
      Assert.equal(bytes, "新しい".getBytes(StandardCharsets.UTF_8).length.toLong)
      Assert.equal(Files.readString(path, StandardCharsets.UTF_8), "新しい")
    }
  )
