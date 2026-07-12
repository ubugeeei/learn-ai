package learnai.json

import scala.collection.mutable.StringBuilder

/** A dependency-free JSON abstract syntax tree. */
sealed trait JsonValue:
  /** Renders compact RFC 8259-compatible JSON. */
  final def render: String = JsonRenderer.render(this)

case object JsonNull                                  extends JsonValue
final case class JsonBoolean(value: Boolean)          extends JsonValue
final case class JsonNumber(value: BigDecimal)        extends JsonValue
final case class JsonString(value: String)            extends JsonValue
final case class JsonArray(values: Vector[JsonValue]) extends JsonValue

/** A JSON object with stable insertion order and unique field names. */
final case class JsonObject(fields: Vector[(String, JsonValue)]) extends JsonValue:
  require(
    fields.map(_._1).distinct.size == fields.size,
    s"JSON object fields must be unique: ${fields.map(_._1)}"
  )

  def get(name: String): Option[JsonValue] = fields.find(_._1 == name).map(_._2)

  def fieldNames: Vector[String] = fields.map(_._1)

object JsonObject:
  val empty: JsonObject = JsonObject(Vector.empty)

  def apply(fields: (String, JsonValue)*): JsonObject = new JsonObject(fields.toVector)

/** Strict JSON parser with explicit input-size and nesting-depth limits. */
object JsonParser:
  val DefaultMaximumInputCharacters: Int = 1_000_000
  val DefaultMaximumDepth: Int           = 64

  /** Parses exactly one JSON value and rejects trailing non-whitespace data. */
  def parse(
      input: String,
      maximumInputCharacters: Int = DefaultMaximumInputCharacters,
      maximumDepth: Int = DefaultMaximumDepth
  ): Either[String, JsonValue] =
    require(maximumInputCharacters > 0, "maximum input characters must be positive")
    require(maximumDepth > 0, "maximum depth must be positive")
    if input.length > maximumInputCharacters then
      Left(s"JSON input has ${input.length} characters; limit is $maximumInputCharacters")
    else
      try
        val cursor = new Cursor(input, maximumDepth)
        val value  = cursor.parseDocument()
        Right(value)
      catch case failure: JsonParseFailure => Left(failure.getMessage)

  final private class Cursor(input: String, maximumDepth: Int):
    private var index = 0

    def parseDocument(): JsonValue =
      skipWhitespace()
      val value = parseValue(depth = 0)
      skipWhitespace()
      if index != input.length then fail(s"unexpected trailing character '${input(index)}'")
      value

    private def parseValue(depth: Int): JsonValue =
      if depth > maximumDepth then fail(s"JSON nesting exceeds maximum depth $maximumDepth")
      if index >= input.length then fail("expected a JSON value, reached end of input")
      input(index) match
        case 'n'                                   => parseLiteral("null", JsonNull)
        case 't'                                   => parseLiteral("true", JsonBoolean(true))
        case 'f'                                   => parseLiteral("false", JsonBoolean(false))
        case '"'                                   => JsonString(parseString())
        case '['                                   => parseArray(depth + 1)
        case '{'                                   => parseObject(depth + 1)
        case '-'                                   => parseNumber()
        case digit if digit >= '0' && digit <= '9' => parseNumber()
        case other                                 => fail(s"unexpected character '$other' while parsing a value")

    private def parseLiteral(expected: String, value: JsonValue): JsonValue =
      if !input.startsWith(expected, index) then fail(s"expected '$expected'")
      index += expected.length
      value

    private def parseArray(depth: Int): JsonArray =
      index += 1
      skipWhitespace()
      val values = Vector.newBuilder[JsonValue]
      if consumeIf(']') then JsonArray(Vector.empty)
      else
        var done = false
        while !done do
          skipWhitespace()
          values += parseValue(depth)
          skipWhitespace()
          if consumeIf(']') then done = true
          else
            expect(',')
            skipWhitespace()
            if peekContains(']') then fail("trailing comma in JSON array")
        JsonArray(values.result())

    private def parseObject(depth: Int): JsonObject =
      index += 1
      skipWhitespace()
      val fields = Vector.newBuilder[(String, JsonValue)]
      val names  = scala.collection.mutable.Set.empty[String]
      if consumeIf('}') then JsonObject.empty
      else
        var done = false
        while !done do
          skipWhitespace()
          if !peekContains('"') then fail("JSON object field name must be a string")
          val name = parseString()
          if names.contains(name) then fail(s"duplicate JSON object field '$name'")
          names += name
          skipWhitespace()
          expect(':')
          skipWhitespace()
          fields += name -> parseValue(depth)
          skipWhitespace()
          if consumeIf('}') then done = true
          else
            expect(',')
            skipWhitespace()
            if peekContains('}') then fail("trailing comma in JSON object")
        JsonObject(fields.result())

    private def parseString(): String =
      expect('"')
      val result = new StringBuilder()
      var closed = false
      while index < input.length && !closed do
        val character = input(index)
        index += 1
        character match
          case '"'                      => closed = true
          case '\\'                     => parseEscape(result)
          case control if control < ' ' => fail("unescaped control character in JSON string")
          case ordinary                 => result.append(ordinary)
      if !closed then fail("unterminated JSON string")
      result.result()

    private def parseEscape(result: StringBuilder): Unit =
      if index >= input.length then fail("unterminated JSON string escape")
      val escaped = input(index)
      index += 1
      escaped match
        case '"'   => result.append('"')
        case '\\'  => result.append('\\')
        case '/'   => result.append('/')
        case 'b'   => result.append('\b')
        case 'f'   => result.append('\f')
        case 'n'   => result.append('\n')
        case 'r'   => result.append('\r')
        case 't'   => result.append('\t')
        case 'u'   => appendUnicodeEscape(result)
        case other => fail(s"invalid JSON string escape '\\$other'")

    private def appendUnicodeEscape(result: StringBuilder): Unit =
      val first = readHexCodeUnit()
      if Character.isHighSurrogate(first.toChar) then
        if index + 2 > input.length || input(index) != '\\' || input(index + 1) != 'u' then
          fail("high surrogate must be followed by a low-surrogate Unicode escape")
        index += 2
        val second = readHexCodeUnit()
        if !Character.isLowSurrogate(second.toChar) then fail("invalid low surrogate")
        result.append(first.toChar)
        result.append(second.toChar)
      else if Character.isLowSurrogate(first.toChar) then
        fail("low surrogate cannot appear without a preceding high surrogate")
      else result.append(first.toChar)

    private def readHexCodeUnit(): Int =
      if index + 4 > input.length then fail("truncated Unicode escape")
      var value = 0
      var count = 0
      while count < 4 do
        val digit = Character.digit(input(index), 16)
        if digit < 0 then fail(s"invalid hex digit '${input(index)}' in Unicode escape")
        value = value * 16 + digit
        index += 1
        count += 1
      value

    private def parseNumber(): JsonNumber =
      val start = index
      consumeIf('-')
      if index >= input.length then fail("incomplete JSON number")
      if consumeIf('0') then
        if index < input.length && input(index).isDigit then fail("leading zero in JSON number")
      else
        requireDigitOneToNine()
        while index < input.length && input(index).isDigit do index += 1

      if consumeIf('.') then
        requireDigit()
        while index < input.length && input(index).isDigit do index += 1

      if index < input.length && (input(index) == 'e' || input(index) == 'E') then
        index += 1
        if index < input.length && (input(index) == '+' || input(index) == '-') then index += 1
        requireDigit()
        while index < input.length && input(index).isDigit do index += 1

      val raw = input.substring(start, index)
      try JsonNumber(BigDecimal(raw))
      catch case _: NumberFormatException => fail(s"invalid JSON number '$raw'")

    private def requireDigitOneToNine(): Unit =
      if index >= input.length || input(index) < '1' || input(index) > '9' then
        fail("expected digit 1-9 in JSON number")
      index += 1

    private def requireDigit(): Unit =
      if index >= input.length || !input(index).isDigit then fail("expected digit in JSON number")
      index += 1

    private def expect(expected: Char): Unit =
      if index >= input.length || input(index) != expected then fail(s"expected '$expected'")
      index += 1

    private def consumeIf(expected: Char): Boolean =
      if index < input.length && input(index) == expected then
        index += 1
        true
      else false

    private def peekContains(expected: Char): Boolean = index < input.length &&
      input(index) == expected

    private def skipWhitespace(): Unit = while index < input.length &&
      (input(index) == ' ' || input(index) == '\n' || input(index) == '\r' || input(index) == '\t')
    do index += 1

    private def fail(message: String): Nothing =
      throw JsonParseFailure(s"$message at character $index")

  final private case class JsonParseFailure(message: String) extends RuntimeException(message)

private object JsonRenderer:
  def render(value: JsonValue): String = value match
    case JsonNull           => "null"
    case JsonBoolean(value) => value.toString
    case JsonNumber(value)  => renderNumber(value)
    case JsonString(value)  => renderString(value)
    case JsonArray(values)  => values.map(render).mkString("[", ",", "]")
    case JsonObject(fields) => fields.map { case (name, fieldValue) =>
        s"${renderString(name)}:${render(fieldValue)}"
      }.mkString("{", ",", "}")

  private def renderNumber(value: BigDecimal): String =
    val stripped = value.bigDecimal.stripTrailingZeros()
    if stripped.signum() == 0 then "0" else stripped.toPlainString

  private def renderString(value: String): String =
    val output = new StringBuilder("\"")
    value.foreach {
      case '"'                      => output.append("\\\"")
      case '\\'                     => output.append("\\\\")
      case '\b'                     => output.append("\\b")
      case '\f'                     => output.append("\\f")
      case '\n'                     => output.append("\\n")
      case '\r'                     => output.append("\\r")
      case '\t'                     => output.append("\\t")
      case control if control < ' ' => output.append(f"\\u${control.toInt}%04x")
      case ordinary                 => output.append(ordinary)
    }
    output.append('"').result()
