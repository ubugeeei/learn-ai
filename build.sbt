ThisBuild / organization := "dev.ubugeeei"
ThisBuild / version      := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.3.6"

lazy val root = project.in(file(".")).settings(
  name                                       := "learn-ai",
  scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Wvalue-discard"),
  Compile / run / fork                       := true,
  // A lesson and its tests live in the same package directory. Production
  // compilation ignores *Suite.scala; the Test configuration adds only those
  // colocated suites to the ordinary test sources.
  Compile / unmanagedSources / excludeFilter := HiddenFileFilter || "*Suite.scala",
  Test / unmanagedSourceDirectories += (Compile / scalaSource).value,
  Test / unmanagedSources / excludeFilter    := new FileFilter {
    override def accept(file: File): Boolean = file.getAbsolutePath.contains("/src/main/") &&
      !file.getName.endsWith("Suite.scala")
  },
  Test / parallelExecution                   := false,
  Test / test                                := (Test / runMain).toTask(" learnai.testing.AllTests").value
)

addCommandAlias("check", ";scalafmtCheckAll;scalafmtSbtCheck;clean;compile;test")
