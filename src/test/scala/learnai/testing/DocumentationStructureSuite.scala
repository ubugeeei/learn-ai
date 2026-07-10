package learnai.testing

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import scala.jdk.CollectionConverters.*

/** Guards implemented hands-on chapters against regressing into short conceptual summaries.
  *
  * Structure and length are only minimum signals; review must still judge the
  * correctness of examples, equations, source mapping, and test oracles.
  */
object DocumentationStructureSuite extends TestSuite:
  override val name: String = "DocumentationStructure"

  private val requiredHeadingFragments = Vector(
    "Implementation walkthrough",
    "Reading",
    "Debugging checklist"
  )
  private val minimumWordCount = 700

  override val tests: Vector[TestCase] = Vector(
    test("implemented chapters retain hands-on explanation layers") {
      val root = Path.of("").toAbsolutePath.normalize()
      val problems = implementedChapters(root).flatMap { path =>
        val content = Files.readString(path, StandardCharsets.UTF_8)
        val headings = content.linesIterator.filter(_.startsWith("## ")).toVector
        val missingHeadings = requiredHeadingFragments.filterNot { fragment =>
          headings.exists(_.contains(fragment))
        }
        val wordCount = content.split("\\s+").count(_.nonEmpty)
        val relative = root.relativize(path)
        val headingProblems = missingHeadings.map(fragment => s"$relative: missing '$fragment' section")
        val lengthProblems =
          if wordCount >= minimumWordCount then Vector.empty
          else Vector(s"$relative: $wordCount words; minimum is $minimumWordCount")
        headingProblems ++ lengthProblems
      }

      Assert.isTrue(
        problems.isEmpty,
        s"implemented documentation lost required depth:\n${problems.mkString("\n")}"
      )
    },
    test("Markdown math uses GitHub-compatible delimiters") {
      val root = Path.of("").toAbsolutePath.normalize()
      val legacyDelimiters = Vector("\\(", "\\)", "\\[", "\\]")
      val problems = markdownFiles(root).flatMap { path =>
        Files
          .readAllLines(path, StandardCharsets.UTF_8)
          .asScala
          .zipWithIndex
          .flatMap { case (line, index) =>
            legacyDelimiters
              .filter(line.contains)
              .map(delimiter => s"${root.relativize(path)}:${index + 1}: legacy '$delimiter' delimiter")
          }
      }

      Assert.isTrue(
        problems.isEmpty,
        s"Markdown math must use dollar delimiters for GitHub rendering:\n${problems.mkString("\n")}"
      )
    }
  )

  private def markdownFiles(root: Path): Vector[Path] =
    val docsRoot = root.resolve("docs")
    Assert.isTrue(Files.isDirectory(docsRoot), s"documentation directory not found: $docsRoot")
    val files = Files.walk(docsRoot)
    try
      files.iterator().asScala
        .filter(file => Files.isRegularFile(file) && file.toString.endsWith(".md"))
        .toVector
        .sortBy(_.toString)
    finally files.close()

  private def implementedChapters(root: Path): Vector[Path] =
    val chapterDirectories = Vector(
      "01-foundations",
      "02-math",
      "03-learning",
      "04-text",
      "05-transformer",
      "06-inference",
      "07-frontier",
      "08-agents"
    )
    chapterDirectories.flatMap { directory =>
      val path = root.resolve("docs").resolve(directory)
      Assert.isTrue(Files.isDirectory(path), s"chapter directory not found: $path")
      val files = Files.list(path)
      try
        files.iterator().asScala
          .filter(file => Files.isRegularFile(file) && file.toString.endsWith(".md"))
          .toVector
          .sortBy(_.toString)
      finally files.close()
    }
