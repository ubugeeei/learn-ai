package learnai.json

import learnai.testing.Assert
import learnai.testing.TestCase
import learnai.testing.TestSuite

object JsonSuite extends TestSuite:
  override val name: String = "Json"

  override val tests: Vector[TestCase] = specify(
    test("parser handles every JSON value type and nested structure") {
      val input = """{"null":null,"bool":true,"number":-12.5e2,"string":"ok","array":[1,false]}"""
      val value = Assert.right(JsonParser.parse(input))
      Assert.equal(
        value,
        JsonObject(
          "null"   -> JsonNull,
          "bool"   -> JsonBoolean(true),
          "number" -> JsonNumber(BigDecimal(-1250)),
          "string" -> JsonString("ok"),
          "array"  -> JsonArray(Vector(JsonNumber(1), JsonBoolean(false)))
        )
      )
    },
    test("render and parse round-trip escaped text and emoji") {
      val original = JsonObject(
        "text"    -> JsonString("quote=\" slash=\\ newline=\n rocket=🚀"),
        "control" -> JsonString("\u0001")
      )
      Assert.equal(JsonParser.parse(original.render), Right(original))
    },
    test("Unicode surrogate escapes decode into one code point") {
      Assert.equal(JsonParser.parse("\"\\uD83D\\uDE80\""), Right(JsonString("🚀")))
    },
    test("parser rejects duplicate fields trailing commas and trailing data") {
      Assert.isTrue(JsonParser.parse("{\"x\":1,\"x\":2}").left.exists(_.contains("duplicate")))
      Assert.isTrue(JsonParser.parse("[1,]").left.exists(_.contains("trailing comma")))
      Assert.isTrue(JsonParser.parse("{} false").left.exists(_.contains("trailing")))
    },
    test("number grammar rejects leading zeros and incomplete fractions or exponents") {
      Vector("01", "1.", "1e", "-", "+1").foreach { invalid =>
        Assert.isTrue(JsonParser.parse(invalid).isLeft, s"accepted invalid number $invalid")
      }
    },
    test("string grammar rejects invalid escapes control characters and unpaired surrogates") {
      Assert.isTrue(JsonParser.parse("\"\\x\"").isLeft)
      Assert.isTrue(JsonParser.parse("\"line\nbreak\"").isLeft)
      Assert.isTrue(JsonParser.parse("\"\\uD83D\"").isLeft)
      Assert.isTrue(JsonParser.parse("\"\\uDE80\"").isLeft)
    },
    test("parser enforces configurable input and nesting limits") {
      val inputLimit = JsonParser.parse("[1,2,3]", maximumInputCharacters = 4)
      val depthLimit = JsonParser.parse("[[[0]]]", maximumDepth = 2)
      Assert.isTrue(inputLimit.left.exists(_.contains("limit")))
      Assert.isTrue(depthLimit.left.exists(_.contains("depth")))
    },
    test("renderer normalizes decimal spelling without losing value") {
      Assert.equal(JsonNumber(BigDecimal("1.2300")).render, "1.23")
      Assert.equal(JsonNumber(BigDecimal("0.000")).render, "0")
      Assert.equal(
        JsonParser.parse(JsonNumber(BigDecimal("1e100")).render),
        Right(JsonNumber(BigDecimal("1e100")))
      )
    },
    test("JsonObject preserves insertion order for deterministic rendering") {
      val value = JsonObject("z" -> JsonNumber(1), "a" -> JsonNumber(2))
      Assert.equal(value.render, "{\"z\":1,\"a\":2}")
      Assert.equal(value.fieldNames, Vector("z", "a"))
    }
  )
