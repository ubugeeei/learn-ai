package learnai.testing

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

import scala.jdk.CollectionConverters.*

/**
 * Guards the repository-level evidence behind claims of implemented coverage.
 *
 * This suite does not claim that passing tests make a topic complete. It makes structural omissions
 * visible: orphaned suites, undocumented source files, missing translations, and a curriculum that
 * hides planned work.
 */
object RepositoryCoverageSuite extends TestSuite:
  override val name: String = "RepositoryCoverage"

  override val tests: Vector[TestCase] = specify(
    test("every colocated feature suite names an existing implementation") {
      val root      = Path.of("").toAbsolutePath.normalize()
      val scalaRoot = root.resolve("src/main/scala/learnai")
      val orphans   = scalaFiles(scalaRoot).filter(_.getFileName.toString.endsWith("Suite.scala"))
        .filter { suite =>
          val implementationName = suite.getFileName.toString.stripSuffix("Suite.scala") + ".scala"
          !Files.isRegularFile(suite.resolveSibling(implementationName))
        }.map(root.relativize)

      Assert.isTrue(orphans.isEmpty, s"orphaned colocated suites:\n${orphans.mkString("\n")}")
    },
    test("every production source file explains at least one public contract") {
      val root         = Path.of("").toAbsolutePath.normalize()
      val scalaRoot    = root.resolve("src/main/scala/learnai")
      val undocumented = scalaFiles(scalaRoot)
        .filterNot(_.getFileName.toString.endsWith("Suite.scala"))
        .filterNot(path => Files.readString(path, StandardCharsets.UTF_8).contains("/**"))
        .map(root.relativize)

      Assert.isTrue(
        undocumented.isEmpty,
        s"production files without a Scaladoc contract:\n${undocumented.mkString("\n")}"
      )
    },
    test("visual beginner material retains diagrams and a plain-language glossary") {
      val root  = Path.of("").toAbsolutePath.normalize()
      val paths = Vector(
        root.resolve("docs/00-guide/09-visual-map-and-glossary.md"),
        root.resolve("docs/ja/00-guide/09-visual-map-and-glossary.md")
      )
      paths
        .foreach(path => Assert.isTrue(Files.isRegularFile(path), s"missing visual guide: $path"))
      paths.foreach { path =>
        val content      = Files.readString(path, StandardCharsets.UTF_8)
        val diagrams     = "```mermaid".r.findAllIn(content).length
        val glossaryRows = content.linesIterator.count(_.startsWith("| ")) - 2
        Assert.isTrue(diagrams >= 4, s"$path has $diagrams diagrams; expected at least 4")
        Assert.isTrue(
          glossaryRows >= 18,
          s"$path has $glossaryRows glossary rows; expected at least 18"
        )
      }
    },
    test("the curriculum reports implemented and planned scope explicitly") {
      val root        = Path.of("").toAbsolutePath.normalize()
      val curriculum  = Files.readString(root.resolve("docs/00-guide/01-curriculum.md"))
      val implemented = curriculum.linesIterator.count(_.contains("| ✅ |"))
      val planned     = curriculum.linesIterator.count(_.contains("| ⬜ |"))
      Assert.isTrue(implemented > 0, "curriculum reports no implemented items")
      Assert.isTrue(planned > 0, "curriculum must not hide unfinished scope")
      Assert.equal(implemented, 55)
      Assert.equal(planned, 22)
    }
  )

  private def scalaFiles(root: Path): Vector[Path] =
    val files = Files.walk(root)
    try files.iterator().asScala
        .filter(path => Files.isRegularFile(path) && path.toString.endsWith(".scala")).toVector
        .sortBy(_.toString)
    finally files.close()
