package learnai.testing

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.lang.Character.UnicodeScript
import scala.jdk.CollectionConverters.*

/**
 * Guards the language split of the learning materials.
 *
 * English sources live under `docs/` and are the canonical text; Japanese translations mirror them
 * under `docs/ja/`. Three properties are enforced: canonical documentation stays English, every
 * translation has an existing English source (no orphans that would silently drift), and every
 * translation actually contains Japanese prose (a translation file accidentally saved in English
 * would otherwise pass unnoticed).
 */
object DocumentationLanguageSuite extends TestSuite:
  override val name: String = "DocumentationLanguage"

  override val tests: Vector[TestCase] = specify(
    test("script detection recognizes Japanese writing systems") {
      Assert.isTrue(containsJapaneseScript("\u3042"))
      Assert.isTrue(containsJapaneseScript("\u30A2"))
      Assert.isTrue(containsJapaneseScript("\u8A00"))
      Assert.isTrue(!containsJapaneseScript("English, mathematics, and Scala 3"))
      Assert.isTrue(!containsJapaneseScript("café, €10, and 🚀"))
    },
    test("all canonical learning documentation is written in English") {
      val root       = Path.of("").toAbsolutePath.normalize()
      val violations = documentationFiles(root).flatMap { path =>
        Files.readAllLines(path, StandardCharsets.UTF_8).asScala.zipWithIndex.collect {
          case (line, index) if containsJapaneseScript(line) =>
            s"${root.relativize(path)}:${index + 1}"
        }
      }

      Assert.isTrue(
        violations.isEmpty,
        s"Japanese script found in English documentation: ${violations.mkString(", ")}"
      )
    },
    test("every Japanese translation mirrors an existing English source") {
      val root            = Path.of("").toAbsolutePath.normalize()
      val translationRoot = root.resolve("docs").resolve("ja")
      val orphans         = translationFiles(root).filterNot { path =>
        val relative = translationRoot.relativize(path)
        Files.isRegularFile(root.resolve("docs").resolve(relative))
      }
      Assert.isTrue(
        orphans.isEmpty,
        s"translations without an English source: ${orphans.mkString(", ")}"
      )
    },
    test("every Japanese translation actually contains Japanese prose") {
      val untranslated = translationFiles(Path.of("").toAbsolutePath.normalize())
        .filterNot(path => containsJapaneseScript(Files.readString(path, StandardCharsets.UTF_8)))
      Assert.isTrue(
        untranslated.isEmpty,
        s"translation files without Japanese text: ${untranslated.mkString(", ")}"
      )
    }
  )

  private def documentationFiles(root: Path): Vector[Path] =
    val topLevel          = Vector(root.resolve("README.md"), root.resolve("CONTRIBUTING.md"))
    val docsRoot          = root.resolve("docs")
    val translationRoot   = docsRoot.resolve("ja")
    // The canonical translation policy must quote Japanese renderings in
    // its glossary, so it is the one canonical file allowed to contain
    // Japanese script.
    val glossaryException = docsRoot.resolve("00-guide").resolve("07-translation-policy.md")
    Assert.isTrue(Files.isDirectory(docsRoot), s"documentation directory not found: $docsRoot")

    val paths = Files.walk(docsRoot)
    try topLevel ++ paths.iterator().asScala.filter { path =>
        Files.isRegularFile(path) && path.toString.endsWith(".md") &&
        !path.startsWith(translationRoot) && path != glossaryException
      }.toVector.sortBy(_.toString)
    finally paths.close()

  private def translationFiles(root: Path): Vector[Path] =
    val translationRoot = root.resolve("docs").resolve("ja")
    if !Files.isDirectory(translationRoot) then Vector.empty
    else
      val paths = Files.walk(translationRoot)
      try paths.iterator().asScala
          .filter(path => Files.isRegularFile(path) && path.toString.endsWith(".md")).toVector
          .sortBy(_.toString)
      finally paths.close()

  private def containsJapaneseScript(text: String): Boolean = text.codePoints()
    .anyMatch { codePoint =>
      UnicodeScript.of(codePoint) match
        case UnicodeScript.HIRAGANA | UnicodeScript.KATAKANA | UnicodeScript.HAN => true
        case _                                                                   => false
    }
