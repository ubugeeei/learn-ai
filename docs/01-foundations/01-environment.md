# 01 — Nix と Scala 3 の開発環境

## この章で作るもの

どの対応 OS でも同じ major version の JDK、sbt、Scala compiler を使う開発環境を作ります。
環境は `flake.nix`、実際に解決された Nix package は `flake.lock`、Scala compiler は
`build.sbt`、sbt 自身は `project/build.properties` で固定します。

## なぜ再現可能性が必要か

機械学習の実験では、コード以外にも実行環境、乱数、入力データ、parameter が結果へ影響します。
最初に環境を固定しないと、後で loss の差が実装変更によるものか、環境差によるものかを判定
できません。

この教材が固定する範囲は次のとおりです。

| 対象 | 固定する場所 | 役割 |
| --- | --- | --- |
| Nix packages | `flake.lock` | JDK と sbt の配布物を固定する |
| JDK | `flake.nix` の `jdk21` | Scala program を compile・実行する |
| sbt | `project/build.properties` | build tool 自身を固定する |
| Scala | `build.sbt` | Scala 3 compiler と標準 library を固定する |

依存を二段階で固定するのは、sbt が Nix package として提供される launcher と、project ごとに
選ぶ実際の sbt version を分けているためです。

## Nix を用意する

この章で唯一、リポジトリの外に必要なものが Nix です。flake が有効な Nix を installation
guide に従って導入してください。導入済みなら次で確認できます。

```console
$ nix --version
nix (Nix) 2.x.x
```

version の細部は一致しなくても構いません。`nix develop` が使えることが条件です。

## 開発 shell に入る

リポジトリ root で実行します。

```console
$ nix develop
learn-ai: Java openjdk version "21..."
learn-ai: run 'sbt check' to verify the workspace
```

初回は Nix package と、sbt が使う Scala 関連 artifact を取得するため時間がかかります。二回目
以降は cache が使われます。

shell 内で version を確認します。

```console
$ java -version
$ sbt --version
$ sbt 'show scalaVersion'
```

本教材は JDK 21、sbt 1.12.11、Scala 3.3.6 を基準にします。Nix の lock を明示的に更新しない
限り、別の日に clone しても JDK と sbt package は同じ revision から解決されます。

## build の構造を読む

`build.sbt` の設定は root project 全体へ適用されます。

```scala
ThisBuild / scalaVersion := "3.3.6"
```

`:=` の左辺が設定項目、右辺が設定値です。ここでは「この build の Scala version を文字列
`3.3.6` にする」と読めば十分です。

`scalacOptions` は compiler が潜在的な問題を警告するための設定です。教材では、廃止予定 API、
危険な型検査、意図せず値を捨てた箇所を早期に見つけます。

## 確認

```console
$ nix develop -c sbt check
```

終了 code が `0` なら完了です。終了 code は shell command の成功・失敗を表す整数で、通常
`0` が成功、それ以外が失敗です。

## よくある失敗

### `experimental Nix feature 'flakes' is disabled`

Nix の設定で `nix-command` と `flakes` を有効にします。設定方法は Nix の配布形態により異なる
ため、導入した Nix の guide を確認してください。

### JDK 21 以外が表示される

開発 shell の外で command を実行している可能性があります。`nix develop -c java -version` なら
flake 内の JDK を直接使えます。

### sbt の download が失敗する

初回 build には Maven artifact を取得する network 接続が必要です。proxy、証明書、Maven
Central への接続を確認してから再実行します。

## 演習

1. shell の内側と外側で `which java` の結果を比較してください。
2. `flake.lock` の `locked.rev` を探し、何を固定しているか説明してください。
3. `build.sbt` の `scalaVersion` と `sbt 'show scalaVersion'` が一致することを確認してください。

## 完了条件

- `nix develop -c sbt check` が成功する
- `flake.nix` と `build.sbt` の責務の違いを説明できる
- version を更新したとき、実験結果を再確認すべき理由を説明できる
