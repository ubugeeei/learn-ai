# 01 — Nix and the Scala 3 development environment

## What you will build

A reproducible development environment that uses the same major JDK, sbt, and
Scala compiler versions on every supported operating system.

- `flake.nix` defines the tools.
- `flake.lock` pins the exact Nix package revision.
- `project/build.properties` pins sbt.
- `build.sbt` pins Scala and compiler settings.

## Why reproducibility matters

Machine-learning results depend on more than source code. Runtime, random
state, input data, compiler behavior, and parameter values can all change an
observation. If the environment is not pinned, a later loss difference may be
caused by infrastructure rather than an experiment.

| Target | Pinned in | Purpose |
| --- | --- | --- |
| Nix packages | `flake.lock` | Exact JDK and sbt distribution |
| JDK | `flake.nix` | Compile and execute JVM programs |
| sbt | `project/build.properties` | Build-tool implementation |
| Scala | `build.sbt` | Scala 3 compiler and standard library |

The sbt package supplied by Nix is a launcher. The project-level sbt version is
selected separately, which is why both layers are pinned.

## Install Nix

Nix is the only prerequisite outside this repository. Install a distribution
with flakes enabled, then verify it:

```console
$ nix --version
nix (Nix) 2.x.x
```

The exact minor version need not match. `nix develop` must be available.

## Enter the shell

From the repository root:

```console
$ nix develop
learn-ai: Java openjdk version "21..."
learn-ai: run 'sbt check' to verify the workspace
```

The first run downloads Nix packages and Maven artifacts. Later runs reuse
local caches.

Inspect the selected tools:

```console
$ java -version
$ sbt --version
$ sbt 'show scalaVersion'
```

The baseline is JDK 21, sbt 1.12.11, and Scala 3.3.6. Nix dependencies remain
fixed until `flake.lock` is deliberately updated.

## Read the build

```scala
ThisBuild / scalaVersion := "3.3.6"
```

Read `:=` as assigning the setting on the left to the value on the right.
`scalacOptions` enables warnings for deprecated APIs, unsafe type operations,
and discarded values so mistakes appear near their source.

## Verification

```console
$ nix develop -c sbt check
```

Exit code `0` means success. Shell commands normally use non-zero exit codes
for failure.

## Common failures

### `experimental Nix feature 'flakes' is disabled`

Enable `nix-command` and `flakes` in the configuration for your Nix
distribution.

### A JDK other than 21 appears

You are probably outside the development shell. Run:

```console
$ nix develop -c java -version
```

### sbt downloads fail

The first build requires network access to Maven Central. Check proxy,
certificate, and repository access, then retry.

## Exercises

1. Compare `which java` inside and outside the Nix shell.
2. Find `locked.rev` in `flake.lock` and explain what it pins.
3. Confirm that `build.sbt` and `sbt 'show scalaVersion'` agree.

## Completion criteria

- `nix develop -c sbt check` succeeds.
- You can explain the different responsibilities of Nix and sbt.
- You can explain why version changes require experiment revalidation.
