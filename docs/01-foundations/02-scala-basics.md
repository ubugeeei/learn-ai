# 02 — LLM 実装に必要な Scala 3 入門

## この章で作るもの

loss の観測値を保持し、平均値を計算する小さな command-line program を作ります。この一例から、
以降の実装で繰り返し使う値、型、関数、collection、分岐、error の表現を学びます。

対象コードは `src/main/scala/learnai/foundations/ScalaTour.scala` です。

## program は値を変換する

program を「入力を受け取り、規則に従って出力へ変換するもの」と捉えます。関数の最小例は
`square` です。

```scala
def square(value: Double): Double = value * value
```

左から順に読みます。

| 部分 | 意味 |
| --- | --- |
| `def` | 関数を定義する |
| `square` | 関数名 |
| `value` | 入力を参照する名前 |
| `Double` | 64 bit 浮動小数点数という入力の型 |
| 最後の `Double` | 出力の型 |
| `value * value` | 出力を計算する式 |

関数は `square(3.0)` のように呼び、結果は `9.0` です。同じ入力に対して同じ出力を返し、外部を
変更しない関数を **pure function** と呼びます。pure function は小さい入力で検証しやすいため、
数値計算の中心に置きます。

## `val` は一度だけ名前を付ける

```scala
val losses = Vector(2.0, 1.0, 0.5)
```

`val` で定義した名前を別の値へ再代入することはできません。変更を減らすと、「この値を最後に
書き換えた場所」を探さずに済みます。

Scala compiler は右辺から型を推論するため、上の `losses` は `Vector[Double]` になります。
明示すると次の形です。

```scala
val losses: Vector[Double] = Vector(2.0, 1.0, 0.5)
```

型は値の集合と、許される操作を表します。`Double` には乗算があり、`String` には文字列の結合が
あります。違う意味の値を誤って混ぜると、実行前に compiler が拒否します。

## 複数の値を一つの型にまとめる

```scala
final case class Observation(label: String, value: Double)
```

`Observation` は label と数値を一組にします。

```scala
val observation = Observation("training loss", 1.25)
println(observation.label)
println(observation.value)
```

`case class` は constructor、field へのアクセス、値としての比較、読みやすい文字列表現を生成
します。`final` は、この教材では別の class に継承させないという設計意思です。

## collection を変換する

`Vector` は順序のある immutable collection です。

```scala
val losses = observations.map(observation => observation.value)
```

`map` は各 `Observation` を `Double` へ変換し、同じ要素数の `Vector[Double]` を返します。
元の `observations` は変更しません。

```text
Vector[Observation] -- map(Observation => Double) --> Vector[Double]
```

この「container の各要素に同じ関数を適用する」考えは、後で batch 内の sample や Tensor の
要素を変換するときにも現れます。

## 空の平均値という不正な入力を表す

平均は要素数で合計を割ります。

\[
\operatorname{mean}(x) = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

要素がなければ \(n=0\) となり、平均は定義できません。戻り値を常に `Double` にすると、失敗を
表現できません。そこで成功と失敗のどちらかを持つ `Either` を使います。

```scala
def mean(values: Vector[Double]): Either[String, Double] =
  if values.isEmpty then Left("mean requires at least one value")
  else Right(values.sum / values.size.toDouble)
```

- `Left(problem)` は失敗理由
- `Right(value)` は計算結果

呼び出し側は両方を処理します。

```scala
mean(losses) match
  case Right(average) => println(average)
  case Left(problem)  => println(problem)
```

この `match` は、考えられる形ごとに処理を分けます。後の agent 実装では tool の成功と失敗、
JSON parse の成功と失敗を同じ考え方で扱います。

## I/O は境界へ置く

`@main` を付けた `runScalaTour` が program の入口です。

```scala
@main def runScalaTour(): Unit =
  // ...
  println("visible side effect")
```

`println` は console という外部状態を変える **side effect** です。戻り値の `Unit` は、有用な
計算結果ではなく effect が目的であることを示します。本教材では計算を pure function にし、
I/O を入口・出口へ寄せます。

## 実行

```console
$ nix develop -c sbt 'runMain learnai.foundations.runScalaTour'
mean loss: 1.167
the last loss is positive
```

source を変更したらもう一度実行してください。sbt が変更部分を compile します。

## 演習

1. `square(-3.0)` が正になる理由を、掛け算の規則から説明してください。
2. `mean(Vector.empty)` を呼び、`Left` の message を表示してください。
3. `Observation` に step 番号を追加し、三つの constructor 呼び出しを修正してください。
4. `minimum` を `Either[String, Double]` で実装してください。
5. loss が一つ前より減ったかを表す `isImproving` を pure function として実装してください。

## 完了条件

- `val`、関数、引数、戻り値、型をコード上で指し示せる
- `Vector.map` が入力と出力の各要素をどう対応させるか説明できる
- 空 collection の平均を `Either` で表す理由を説明できる
- pure function と I/O を含む関数を区別できる
