# 03 — 外部ライブラリなしのテストとデバッグ

## この章で作るもの

関数を実行し、期待値と比較し、失敗時に build を失敗させる最小の test runner を作ります。
対象コードは `src/test/scala/learnai/testing` です。

一般的な Scala project ではテスト用 library を使いますが、この教材では最初の runner だけを
自作します。目的は library の再発明ではなく、「テストも入力、実行、比較、報告からなる普通の
program である」ことを観察するためです。

## テストの三段階

テストは Arrange、Act、Assert の三段階で読むと整理できます。

```scala
test("mean averages non-empty values") {
  // Arrange: 入力を準備する
  val values = Vector(2.0, 1.0, 0.5)

  // Act: 調べたい操作を一回行う
  val result = ScalaTour.mean(values)

  // Assert: 実際の結果を期待値と比較する
  Assert.close(Assert.right(result), 7.0 / 6.0)
}
```

各テストでは一つの振る舞いを調べます。失敗時にテスト名を読めば、壊れた性質が分かる名前を
付けます。

## test case は関数を値として保持する

```scala
final case class TestCase(name: String, run: () => Unit)
```

`run` の型 `() => Unit` は、引数を受け取らず、実行すると effect を起こし得る関数です。
`test` method の引数 `body: => Unit` は **by-name parameter** で、`TestCase` を組み立てた時点
では body を実行しません。`() => body` という関数に包み、runner が呼ぶまで遅延します。

これにより runner は次の処理を行えます。

1. test case を順番に選ぶ
2. `run()` を呼ぶ
3. 正常終了なら pass と記録する
4. exception が発生したら fail と記録する
5. 最後に集計する

## 浮動小数点数を完全一致で比べない

実数の多くは、有限の 2 進数では正確に表現できません。

```scala
0.1 + 0.2 == 0.3 // false になる処理系がある
```

そこで、差の絶対値が許容誤差以下かを調べます。

\[
\lvert \text{actual} - \text{expected} \rvert \leq \text{tolerance}
\]

```scala
Assert.close(actual, expected, tolerance = 1e-9)
```

許容誤差を無条件に大きくすると間違いも通るため、値の規模と演算回数を考えて決めます。大きさが
大きく異なる値を扱う章では、絶対誤差と相対誤差を組み合わせます。

## error path もテストする

正常な入力だけでは、関数の契約を十分に確認できません。

```scala
test("mean rejects an empty collection") {
  val result = ScalaTour.mean(Vector.empty)
  Assert.equal(Assert.left(result), "mean requires at least one value")
}
```

後の数値計算では、shape 不一致、範囲外の index、空の Tensor、非有限値を意図的に入力します。
agent では、不正な tool 引数、timeout、予算超過、壊れた model 応答を入力します。

## runner の登録を明示する

`AllTests` は suite の一覧を明示的に持ちます。

```scala
private val suites: Vector[TestSuite] = Vector(
  ScalaTourSuite
)
```

reflection による自動検索を使わないため、新しい suite を追加したらここにも追加します。登録忘れは
欠点ですが、何が実行されるかを一ファイルで確認でき、外部 dependency も不要です。

## 実行

```console
$ nix develop -c sbt test
[pass] ScalaTour / square multiplies a value by itself
...
5 passed, 0 failed, 5 total
```

compile と全テストを clean な状態から行う場合は次を使います。

```console
$ nix develop -c sbt check
```

## 意図的に失敗させて読む

`square` の期待値 `9.0` を一時的に `8.0` へ変え、`sbt test` を実行してください。次の情報を
見つけます。

- 失敗した suite と test 名
- expected と actual
- sbt の終了が failure になること

確認後は必ず `9.0` に戻します。失敗するテストを一度見ると、成功時に何を保証しているかが明確に
なります。

## 演習

1. `Assert.isTrue` を使う test を一つ追加してください。
2. `normalizeLabel` に空文字を渡す test を追加してください。
3. わざと `ScalaTourSuite` を registry から外し、実行件数が変わることを確認してください。
4. `Assert.close(Double.NaN, Double.NaN)` が失敗する理由を、比較処理から説明してください。

## 完了条件

- Arrange、Act、Assert を既存 test で指し示せる
- 浮動小数点数に許容誤差が必要な理由を説明できる
- 正常系と異常系の test を一つずつ追加できる
- `sbt check` が成功する
