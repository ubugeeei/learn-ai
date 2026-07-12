package learnai

import learnai.agent.runAgentEvaluationLab
import learnai.agent.runPlanningAgentLab
import learnai.foundations.runScalaTour
import learnai.foundations.runComplexityLab
import learnai.foundations.runJvmSystemsLab
import learnai.learning.runGradientDescentLab
import learnai.lm.trainBigram
import learnai.math.runFloatingPointLab
import learnai.math.runGradientLab
import learnai.math.runLinearAlgebraLab
import learnai.math.runStatisticsLab
import learnai.nn.trainXor
import learnai.quantization.runInt8QuantizationLab
import learnai.training.runMiniGptTrainingLab
import learnai.training.TrainingCommand
import learnai.training.TrainingWorkflow
import learnai.transformer.runKvCacheLab
import learnai.transformer.runMiniGptDiagnostics
import learnai.transformer.trainMiniGpt

/**
 * The single public entrypoint for every executable lesson.
 *
 * Lesson functions remain close to the implementation they teach, while this dispatcher makes
 * discovery possible without knowing package names.
 */
object Main:
  final private case class Lesson(name: String, description: String, run: () => Unit)

  private val lessons = Vector(
    Lesson("foundations", "Scala values, collections, and explicit errors", () => runScalaTour()),
    Lesson("complexity", "count work, bytes, locality, and array growth", () => runComplexityLab()),
    Lesson(
      "jvm",
      "inspect heap capacity, deadlines, threads, and atomic files",
      () => runJvmSystemsLab()
    ),
    Lesson(
      "floating-point",
      "rounding error and stable numerical operations",
      () => runFloatingPointLab()
    ),
    Lesson("calculus", "finite differences and analytical gradients", () => runGradientLab()),
    Lesson(
      "linear-algebra",
      "inspect rank, eigen residuals, singular values, and conditioning",
      () => runLinearAlgebraLab()
    ),
    Lesson(
      "statistics",
      "estimate uncertainty and probability calibration",
      () => runStatisticsLab()
    ),
    Lesson(
      "gradient-descent",
      "fit a scalar parameter by optimization",
      () => runGradientDescentLab()
    ),
    Lesson("xor", "train a scalar-autodiff multilayer perceptron", () => trainXor()),
    Lesson("bigram", "train and sample a bigram language model", () => trainBigram()),
    Lesson("model", "train and sample the smallest causal Transformer", () => trainMiniGpt()),
    Lesson(
      "training",
      "run batched MiniGPT training and validation",
      () => runMiniGptTrainingLab()
    ),
    Lesson(
      "diagnostics",
      "check gradients, parameter bytes, and timing",
      () => runMiniGptDiagnostics()
    ),
    Lesson("cache", "compare full-prefix and KV-cached decoding", () => runKvCacheLab()),
    Lesson("quantization", "quantize a matrix to signed int8", () => runInt8QuantizationLab()),
    Lesson("agent", "exercise tool and permission outcomes", () => runAgentEvaluationLab()),
    Lesson("planning", "run task-graph planning and recovery", () => runPlanningAgentLab())
  )

  def main(arguments: Array[String]): Unit = arguments.toList match
    case Nil | "help" :: Nil | "--help" :: Nil | "-h" :: Nil => printHelp()
    case "train" :: ("--help" | "-h") :: Nil                 => println(TrainingCommand.help)
    case "train" :: options                                  => runTrainingWorkflow(options)
    case command :: Nil                                      => lessons.find(_.name == command) match
        case Some(lesson) => lesson.run()
        case None         =>
          Console.err.println(s"Unknown lesson: $command\n")
          printHelp()
          sys.exit(2)
    case _                                                   =>
      Console.err.println("Expected exactly one lesson name.\n")
      printHelp()
      sys.exit(2)

  private def printHelp(): Unit =
    println("learn-ai — executable lessons\n")
    println("Usage: sbt 'runMain learnai.Main <lesson>'\n")
    println("Lessons:")
    lessons.foreach(lesson => println(f"  ${lesson.name}%-18s ${lesson.description}"))
    println("\nWorkflow:")
    println("  train              train on a UTF-8 file and write reproducible artifacts")
    println("\nRun 'learnai.Main train --help' for workflow options.")

  private def runTrainingWorkflow(arguments: List[String]): Unit =
    val result =
      for
        config    <- TrainingCommand.parse(arguments)
        completed <- TrainingWorkflow.run(config)
      yield completed
    result match
      case Left(problem)    =>
        Console.err.println(problem)
        Console.err.println()
        Console.err.println(TrainingCommand.help)
        sys.exit(2)
      case Right(completed) =>
        println(s"experiment:      ${completed.experimentId}")
        println(s"artifacts:       ${completed.output}")
        println(s"updates:         ${completed.completedUpdates}")
        println(s"tokens seen:     ${completed.tokensSeen}")
        println(f"best validation: ${completed.bestValidationLoss}%.6f")
