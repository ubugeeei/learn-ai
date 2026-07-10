package learnai.testing

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.lang.Character.UnicodeScript
import scala.jdk.CollectionConverters.*

/** Guards the English learning materials against accidental Japanese prose regressions. */
object DocumentationLanguageSuite extends TestSuite:
  override val name: String = "DocumentationLanguage"

  override val tests: Vector[TestCase] = Vector(
    test("script detection recognizes Japanese writing systems") {
      Assert.isTrue(containsJapaneseScript("\u3042"))
      Assert.isTrue(containsJapaneseScript("\u30A2"))
      Assert.isTrue(containsJapaneseScript("\u8A00"))
      Assert.isTrue(!containsJapaneseScript("English, mathematics, and Scala 3"))
      Assert.isTrue(!containsJapaneseScript("café, €10, and 🚀"))
    },
    test("all learning documentation is written in English") {
      val root = Path.of("").toAbsolutePath.normalize()
      val violations = documentationFiles(root).flatMap { path =>
        Files
          .readAllLines(path, StandardCharsets.UTF_8)
          .asScala
          .zipWithIndex
          .collect {
            case (line, index) if containsJapaneseScript(line) =>
              s"${root.relativize(path)}:${index + 1}"
          }
      }

      Assert.isTrue(
        violations.isEmpty,
        s"Japanese script found in English documentation: ${violations.mkString(", ")}"
      )
    }
  )

  private def documentationFiles(root: Path): Vector[Path] =
    val topLevel = Vector(root.resolve("README.md"), root.resolve("CONTRIBUTING.md"))
    val docsRoot = root.resolve("docs")
    Assert.isTrue(Files.isDirectory(docsRoot), s"documentation directory not found: $docsRoot")

    val paths = Files.walk(docsRoot)
    try
      topLevel ++ paths.iterator().asScala
        .filter(path => Files.isRegularFile(path) && path.toString.endsWith(".md"))
        .toVector
        .sortBy(_.toString)
    finally paths.close()

  private def containsJapaneseScript(text: String): Boolean =
    text.codePoints().anyMatch { codePoint =>
      UnicodeScript.of(codePoint) match
        case UnicodeScript.HIRAGANA | UnicodeScript.KATAKANA | UnicodeScript.HAN => true
        case _                                                                   => false
    }
