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

## Implementation walkthrough

Open `flake.nix` and read it from the outside inward. `description` is metadata.
`inputs.nixpkgs.url` names the package collection revision family; the exact
revision is selected in `flake.lock`. `supportedSystems` is explicit so an
unsupported machine fails during evaluation rather than producing a partially
configured shell.

`forEachSystem` imports `nixpkgs` once for each supported platform. The default
development shell then selects `jdk21` and `sbt` from that same package set:

```nix
let
  jdk = pkgs.jdk21;
in pkgs.mkShell {
  packages = [ jdk pkgs.sbt ];
  JAVA_HOME = jdk.home;
}
```

The local name `jdk` is important: the package in `PATH` and `JAVA_HOME` cannot
silently refer to different JDKs. `sbt` reads `project/build.properties`, then
`build.sbt` fixes Scala `3.3.6`. There are therefore four distinct pins to
understand:

| Layer | File | Responsibility |
| --- | --- | --- |
| package universe | `flake.lock` | exact Nix input revision |
| JDK and sbt packages | `flake.nix` | native runtime/tool binaries |
| sbt launcher | `project/build.properties` | build-tool version |
| Scala compiler | `build.sbt` | source-language version |

Trace one command carefully. `nix develop -c sbt check` evaluates the flake,
builds or downloads the shell closure, sets environment variables, and then
executes `sbt check` inside that environment. The `check` alias expands to
`clean`, `compile`, and `test`; it does not mean `nix flake check`.

## Reading the verification output

Check the first reported Java version before the Scala output. If it is not the
JDK selected by the flake, the command probably ran outside `nix develop`.
During compilation, sbt reports how many main and test sources it compiled.
The final custom test-runner line reports passed, failed, and total cases. A
cached compilation is normal; `check` begins with `clean` specifically so the
full verification does not rely on stale class files.

The lockfile test is operational rather than a Scala unit test: deleting
`flake.lock` makes dependency resolution non-reproducible even if existing
compiled classes still run. Commit lock changes only when intentionally
updating the toolchain.

## Debugging checklist

1. If `nix` is missing, verify installation before touching Scala files.
2. If flake evaluation fails, run `nix develop --show-trace` and read the first
   project-owned frame.
3. If Java is wrong, compare `which java`, `java -version`, and `JAVA_HOME`
   inside the Nix shell.
4. If sbt cannot resolve Scala, inspect `flake.lock`, network access, and the
   exact launcher/compiler versions rather than upgrading randomly.
5. If only tests fail, the environment is working; move to the first failing
   test instead of rebuilding Nix.

## Exercises

1. Compare `which java` inside and outside the Nix shell.
2. Find `locked.rev` in `flake.lock` and explain what it pins.
3. Confirm that `build.sbt` and `sbt 'show scalaVersion'` agree.

## Completion criteria

- `nix develop -c sbt check` succeeds.
- You can explain the different responsibilities of Nix and sbt.
- You can explain why version changes require experiment revalidation.
