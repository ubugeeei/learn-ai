package learnai.foundations

/** Small examples used by the Scala introduction.
  *
  * Keeping the examples as ordinary functions makes every result callable from
  * later tests. Only `runScalaTour` performs console I/O.
  */
object ScalaTour:
  final case class Observation(label: String, value: Double)

  def square(value: Double): Double = value * value

  def mean(values: Vector[Double]): Either[String, Double] =
    if values.isEmpty then Left("mean requires at least one value")
    else Right(values.sum / values.size.toDouble)

  def describeSign(value: Double): String =
    value match
      case number if number < 0.0 => "negative"
      case 0.0                    => "zero"
      case _                      => "positive"

  def normalizeLabel(raw: String): String =
    raw.trim.toLowerCase

@main def runScalaTour(): Unit =
  val observations = Vector(
    ScalaTour.Observation("first loss", 2.0),
    ScalaTour.Observation("second loss", 1.0),
    ScalaTour.Observation("third loss", 0.5)
  )
  val losses = observations.map(observation => observation.value)

  ScalaTour.mean(losses) match
    case Right(average) => println(f"mean loss: $average%.3f")
    case Left(problem)  => println(s"could not compute the mean: $problem")

  println(s"the last loss is ${ScalaTour.describeSign(losses.last)}")
