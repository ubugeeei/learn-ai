package learnai.math

object Calculus:
  val DefaultStep: Double = 1e-5

  /** Central finite-difference approximation of a scalar derivative. */
  def derivative(function: Double => Double, at: Double, step: Double = DefaultStep): Double =
    validatePointAndStep(at, step)
    val above = Numerics.requireFinite(function(at + step), "function(at + step)")
    val below = Numerics.requireFinite(function(at - step), "function(at - step)")
    Numerics.requireFinite((above - below) / (2.0 * step), "derivative")

  /** One central finite difference for every input dimension. */
  def gradient(function: VectorD => Double, at: VectorD, step: Double = DefaultStep): VectorD =
    validateStep(step)
    VectorD.tabulate(at.size) { index =>
      val above = Numerics.requireFinite(
        function(at.updated(index, at(index) + step)),
        s"function above dimension $index"
      )
      val below = Numerics.requireFinite(
        function(at.updated(index, at(index) - step)),
        s"function below dimension $index"
      )
      (above - below) / (2.0 * step)
    }

  def directionalDerivative(
      function: VectorD => Double,
      at: VectorD,
      direction: VectorD,
      step: Double = DefaultStep
  ): Either[String, Double] =
    if at.size != direction.size then
      Left(s"point and direction sizes differ: ${at.size} != ${direction.size}")
    else if direction.norm == 0.0 then Left("direction must be non-zero")
    else
      validateStep(step)
      val unitDirection = direction.scale(1.0 / direction.norm)
      val above         = at + unitDirection.scale(step)
      val below         = at - unitDirection.scale(step)
      val aboveValue    = Numerics.requireFinite(function(above), "function above direction")
      val belowValue    = Numerics.requireFinite(function(below), "function below direction")
      Right((aboveValue - belowValue) / (2.0 * step))

  private def validatePointAndStep(at: Double, step: Double): Unit =
    Numerics.requireFinite(at, "derivative point")
    validateStep(step)

  private def validateStep(step: Double): Unit =
    Numerics.requireFinite(step, "finite-difference step")
    require(step > 0.0, s"finite-difference step must be positive: $step")

def runGradientLab(): Unit =
  val function = (point: VectorD) =>
    val x = point(0)
    val y = point(1)
    x * x + 3.0 * x * y + y * y

  val point    = VectorD(2.0, -1.0)
  val gradient = Calculus.gradient(function, point)
  println(s"point:    $point")
  println(s"gradient: $gradient")
  println("expected: VectorD(1.0, 4.0)")
