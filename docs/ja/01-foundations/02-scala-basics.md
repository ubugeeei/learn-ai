# 02 — LLM 実装に必要な Scala 3

## この章で作るもの

損失 (loss) の観測値を記録し、その平均を計算する小さなコマンドラインプログラムを
作ります。このコースを通して使う、値・型・関数・コレクション・分岐・エラー表現を
ここで導入します。

ソース: `src/main/scala/learnai/foundations/ScalaTour.scala`。

## プログラムは値を変換する

プログラムを、型付きの入力から型付きの出力への変換として捉えてください。

```scala
def square(value: Double): Double = value * value
```

| 部分 | 意味 |
| --- | --- |
| `def` | 関数を定義する |
| `square` | 関数名 |
| `value` | 入力の名前 |
| 最初の `Double` | 入力の型 |
| 最後の `Double` | 出力の型 |
| `value * value` | 出力の式 |

`square(3.0)` は `9.0` を返します。同じ入力に対して同じ出力を返し、外部の状態を
変更しない関数を**純粋関数** (pure function) と呼びます。純粋関数はテストしやすい
ため、数値計算コードの中心となります。

## `val` は値に一つの安定した名前を与える

```scala
val losses = Vector(2.0, 1.0, 0.5)
```

`val` は再代入できません。コンパイラは `Vector[Double]` を推論しますが、型を明示
的に書くこともできます。

```scala
val losses: Vector[Double] = Vector(2.0, 1.0, 0.5)
```

型は、値の集合とそれらに許される操作を記述するものです。型があることで、
コンパイラは実行前に多くの意味の不一致を検出して弾くことができます。

## 関連するフィールドをまとめる

```scala
final case class Observation(label: String, value: Double)
```

`Observation` はラベルと数値の測定値をひとまとめにします。case class は、
コンストラクタ、フィールドアクセス、値としての等価性、読みやすい表示を提供
します。`final` は、この学習用の型がサブクラス化を想定していないことを表明して
います。

## コレクションを変換する

`Vector` は順序付きの不変 (immutable) コレクションです。

```scala
val losses = observations.map(observation => observation.value)
```

`map` はすべての要素に `Observation => Double` を適用し、同じ長さの新しい
`Vector[Double]` を返します。

```text
Vector[Observation] -- map(Observation => Double) --> Vector[Double]
```

このコンテナ変換の考え方は、後の章でバッチやテンソルにもそのまま適用されます。

## 未定義の平均を表現する

$n$ 個の値に対して:

$$
\operatorname{mean}(x)=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

空のコレクションは $n=0$ なので、その平均は未定義です。`Double` だけを返す設計
ではこの失敗を表現できません。`Either` を使います。

```scala
def mean(values: Vector[Double]): Either[String, Double] =
  if values.isEmpty then Left("mean requires at least one value")
  else Right(values.sum / values.size.toDouble)
```

- `Left(problem)` は失敗の理由を含みます。
- `Right(value)` は結果を含みます。

呼び出し側は両方のケースを処理します。

```scala
mean(losses) match
  case Right(average) => println(average)
  case Left(problem)  => println(problem)
```

エージェントの章でも、JSON のパース、モデル呼び出し、ツール実行に同じ考え方を
使います。

## I/O を境界に留める

`@main` 関数はプログラムのエントリポイントです。

```scala
@main def runScalaTour(): Unit =
  println("visible side effect")
```

`println` はコンソールの状態を変えるので副作用です。`Unit` は、返される
データ値ではなく副作用こそが目的であることを示します。計算は純粋に保ち、I/O は
入口と出口の境界に集中させてください。

## 実行する

```console
$ nix develop -c sbt 'runMain learnai.foundations.runScalaTour'
mean loss: 1.167
the last loss is positive
```

## 実装ウォークスルー

`ScalaTour.scala` を宣言順に読んでください。`object ScalaTour` は一つの名前空間
を作ります。そのメソッドは純粋なので、テストからも `@main` エントリポイントからも
容易に呼び出せます。

`square` は、もっとも単純な型付き関数を示します。

```scala
def square(value: Double): Double = value * value
```

パラメータと戻り値の型注釈が公開契約 (public contract) を構成します。`=` の後の
式が戻り値であり、`return` 文は必要ありません。

`mean` はエラーチャネルを導入します。

```scala
def mean(values: Vector[Double]): Either[String, Double] =
  if values.isEmpty then Left("mean requires a non-empty collection")
  else Right(values.sum / values.size.toDouble)
```

空集合の平均は数学的に未定義です。`0.0` を返すと、正しく見えるだけの数値を
でっち上げてしまいます。例外を投げると、想定内の不正な入力を合成しにくくなり
ます。`Either[String, Double]` は、呼び出し側に `Left` か `Right` の処理を強制
します。`size` が `Int` であるため、`toDouble` を明示的に書いています。

`describeSign` は 3 分岐の式です。生成される文字列はちょうど一つなので、
コンパイラは分岐の結果型を推論できます。`normalizeLabel` は不変な `String` に
対するメソッドチェーンの例です。`trim` と `toLowerCase` はそれぞれ新しい値を
返します。

最後に、`@main def runScalaTour()` は JVM のエントリポイントにコンパイルされ
ます。テストと同じ関数を呼び出しており、デモの中に二つ目の実装があるわけでは
ありません。`Either` を出力する際は、エラーチャネルが抽象的なままにならない
よう、`Right` と `Left` の両方を意図的に観察してください。

## テストの読み方

`ScalaTourSuite` は振る舞いごとにケースを分けています。square のテストは、答えが
手計算で自明な値を使います。mean のテストは通常のベクトルと空という境界の両方を
カバーします。符号のテストは、一つの正の例で条件分岐全体をカバーできると仮定する
のではなく、3 つの分岐すべてを列挙します。ラベルのテストは外側の空白と大文字
小文字の混在の両方を使うため、どちらかの操作を削除するとテストが壊れます。

関数を追加するときは、まずすべての分岐に一つずつ例を書き、不正な境界ごとに一つ
ずつ入力を書いてください。スイートの `AllTests` への登録は別の手順です。登録
されていないファイルは、緑色に見えても実行されていません。

## デバッグチェックリスト

1. 型エラーが出たら、キャストを追加する前に、要求された型と実際の型を読む。
2. 未処理の `Either` があれば、`Left` と `Right` の両方をパターンマッチする。
3. インデントのエラーが出たら、同レベルの式を揃え、外側の `object` やメソッドが
   どこで終わっているかを確認する。
4. メインクラスのエラーが出たら、sbt が表示する完全修飾名を使う。
5. 予期せず可変な結果が出たら、`var` や可変コレクションを検索する。ツアーの
   関数にはどちらも不要です。

## 演習

1. `square(-3.0)` が正になる理由を説明してください。
2. `mean(Vector.empty)` を呼び出し、その `Left` のメッセージを出力してください。
3. `Observation` にステップ番号を追加し、そのコンストラクタ呼び出しを更新して
   ください。
4. `minimum` を `Either[String, Double]` として実装してください。
5. 二つの損失を受け取る純粋な `isImproving` 関数を実装してください。

## 完了条件

- ソースの中から、値・関数・引数・戻り値・型を指し示せる。
- `Vector.map` が入力要素と出力要素をどう対応づけるか説明できる。
- 空の平均が `Either` を返す理由を説明できる。
- 純粋な計算と I/O を区別できる。
