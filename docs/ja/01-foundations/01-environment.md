# 01 — Nix と Scala 3 開発環境

## この章で作るもの

サポートされているすべてのオペレーティングシステム上で、同じメジャーバージョンの
JDK・sbt・Scala コンパイラを使う、再現可能 (reproducible) な開発環境を構築します。

- `flake.nix` がツール群を定義します。
- `flake.lock` が Nix パッケージのリビジョンを正確に固定します。
- `project/build.properties` が sbt を固定します。
- `build.sbt` が Scala とコンパイラ設定を固定します。

## 再現性が重要な理由

機械学習の結果は、ソースコードだけで決まるものではありません。ランタイム、乱数の
状態、入力データ、コンパイラの挙動、パラメータの値のいずれもが観測結果を変えうる
のです。環境が固定されていなければ、後から現れた損失 (loss) の差は、実験ではなく
インフラに起因している可能性があります。

| 対象 | 固定する場所 | 目的 |
| --- | --- | --- |
| Nix パッケージ | `flake.lock` | JDK と sbt ディストリビューションの正確な指定 |
| JDK | `flake.nix` | JVM プログラムのコンパイルと実行 |
| sbt | `project/build.properties` | ビルドツールの実装 |
| Scala | `build.sbt` | Scala 3 コンパイラと標準ライブラリ |

Nix が提供する sbt パッケージはランチャーです。プロジェクトレベルの sbt バージョン
は別途選択されるため、両方の層を固定しています。

## Nix のインストール

このリポジトリの外で必要となる前提条件は Nix だけです。flakes が有効化された
ディストリビューションをインストールし、次のように確認します。

```console
$ nix --version
nix (Nix) 2.x.x
```

マイナーバージョンまで一致している必要はありません。`nix develop` が使えることが
条件です。

## シェルに入る

リポジトリのルートで次を実行します。

```console
$ nix develop
learn-ai: Java openjdk version "21..."
learn-ai: run 'sbt check' to verify the workspace
```

初回実行時には Nix パッケージと Maven アーティファクトをダウンロードします。
2 回目以降はローカルキャッシュが再利用されます。

選択されたツールを確認します。

```console
$ java -version
$ sbt --version
$ sbt 'show scalaVersion'
```

ベースラインは JDK 21、sbt 1.12.11、Scala 3.3.6 です。`flake.lock` を意図的に
更新しない限り、Nix の依存関係は固定されたままです。

## ビルド定義を読む

```scala
ThisBuild / scalaVersion := "3.3.6"
```

`:=` は、左辺の設定に右辺の値を代入すると読みます。`scalacOptions` は、非推奨
API・安全でない型操作・破棄された値に対する警告を有効にし、ミスがその発生源の
近くで顕在化するようにしています。

## 検証

```console
$ nix develop -c sbt check
```

終了コード `0` は成功を意味します。シェルコマンドは通常、失敗を非ゼロの終了
コードで表します。

## よくある失敗

### `experimental Nix feature 'flakes' is disabled`

使用している Nix ディストリビューションの設定で `nix-command` と `flakes` を
有効にしてください。

### 21 以外の JDK が表示される

おそらく開発シェルの外にいます。次を実行してください。

```console
$ nix develop -c java -version
```

### sbt のダウンロードが失敗する

初回ビルドには Maven Central へのネットワークアクセスが必要です。プロキシ、
証明書、リポジトリへのアクセスを確認してから再試行してください。

## 実装ウォークスルー

`flake.nix` を開き、外側から内側へ読み進めてください。`description` はメタデータ
です。`inputs.nixpkgs.url` はパッケージコレクションのリビジョン系列を指定し、
正確なリビジョンは `flake.lock` で選択されます。`supportedSystems` を明示して
いるのは、サポート外のマシンでは、部分的に構成されたシェルが生成されるのではなく
評価の段階で失敗するようにするためです。

`forEachSystem` は、サポートされる各プラットフォームごとに `nixpkgs` を一度
インポートします。そのうえでデフォルトの開発シェルは、同じパッケージセットから
`jdk21` と `sbt` を選択します。

```nix
let
  jdk = pkgs.jdk21;
in pkgs.mkShell {
  packages = [ jdk pkgs.sbt ];
  JAVA_HOME = jdk.home;
}
```

ローカル名 `jdk` が重要です。`PATH` 上のパッケージと `JAVA_HOME` が、気づかない
うちに異なる JDK を指すことはありえません。`sbt` は `project/build.properties`
を読み、続いて `build.sbt` が Scala `3.3.6` を固定します。したがって、理解すべき
固定 (pin) は 4 つの層に分かれています。

| 層 | ファイル | 責務 |
| --- | --- | --- |
| パッケージの全体集合 | `flake.lock` | Nix 入力の正確なリビジョン |
| JDK と sbt のパッケージ | `flake.nix` | ネイティブなランタイム/ツールのバイナリ |
| sbt ランチャー | `project/build.properties` | ビルドツールのバージョン |
| Scala コンパイラ | `build.sbt` | ソース言語のバージョン |

一つのコマンドを丁寧に追ってみましょう。`nix develop -c sbt check` は、flake を
評価し、シェルのクロージャをビルドまたはダウンロードし、環境変数を設定したうえで、
その環境の中で `sbt check` を実行します。`check` エイリアスは `clean`・`compile`・
`test` に展開されます。`nix flake check` を意味するのではありません。

## 検証出力の読み方

Scala の出力より先に、最初に報告される Java のバージョンを確認してください。
それが flake で選択した JDK でなければ、コマンドはおそらく `nix develop` の外で
実行されています。コンパイル中、sbt はメインソースとテストソースをそれぞれ何件
コンパイルしたかを報告します。最後のカスタムテストランナーの行は、成功・失敗・
合計のケース数を報告します。コンパイルがキャッシュされているのは正常です。
`check` が `clean` から始まるのは、完全な検証が古いクラスファイルに依存しない
ようにするためです。

lockfile のテストは Scala のユニットテストではなく運用上のものです。`flake.lock`
を削除すると、既存のコンパイル済みクラスが動き続けたとしても、依存関係の解決は
再現不可能になります。lock の変更をコミットするのは、意図的にツールチェーンを
更新するときだけにしてください。

## デバッグチェックリスト

1. `nix` が見つからない場合は、Scala のファイルに触れる前にインストールを確認する。
2. flake の評価が失敗する場合は、`nix develop --show-trace` を実行し、プロジェクト
   側の最初のフレームを読む。
3. Java が正しくない場合は、Nix シェル内で `which java`・`java -version`・
   `JAVA_HOME` を比較する。
4. sbt が Scala を解決できない場合は、闇雲にアップグレードするのではなく、
   `flake.lock`、ネットワークアクセス、ランチャー/コンパイラの正確なバージョンを
   調べる。
5. テストだけが失敗する場合、環境は機能しています。Nix を再構築するのではなく、
   最初に失敗したテストに取り組む。

## 演習

1. Nix シェルの内側と外側で `which java` を比較してください。
2. `flake.lock` の中の `locked.rev` を見つけ、それが何を固定しているか説明して
   ください。
3. `build.sbt` と `sbt 'show scalaVersion'` が一致することを確認してください。

## 完了条件

- `nix develop -c sbt check` が成功する。
- Nix と sbt の責務の違いを説明できる。
- バージョン変更が実験の再検証を必要とする理由を説明できる。
