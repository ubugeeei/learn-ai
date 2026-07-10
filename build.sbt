ThisBuild / organization := "dev.ubugeeei"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.3.6"

lazy val root = project
  .in(file("."))
  .settings(
    name := "learn-ai",
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Wvalue-discard"
    ),
    Compile / run / fork := true,
    Test / parallelExecution := false
  )

addCommandAlias("check", ";clean;compile;Test/compile")
