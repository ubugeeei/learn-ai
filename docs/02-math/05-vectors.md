# 05 — ベクトル

## この章で作るもの

有限な `Double` を一次元に並べた immutable な `VectorD` を実装します。加算、減算、scalar 倍、
Hadamard 積、内積、norm を、外部の線形代数 library を使わずに作ります。

対象コードは `src/main/scala/learnai/math/VectorD.scala` です。

## scalar から vector へ

scalar は一つの数です。太字の小文字 \(\boldsymbol{x}\) で表す vector は、順序を持つ複数の
scalar です。

\[
\boldsymbol{x} =
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
\in \mathbb{R}^n
\]

`VectorD(2.0, -1.0, 3.0)` なら \(n=3\)、shape は `[3]` です。添字は数式では 1 から書く
慣習がありますが、Scala の index は 0 から始まります。

| 数式 | Scala | 値 |
| --- | --- | --- |
| \(x_1\) | `x(0)` | `2.0` |
| \(x_2\) | `x(1)` | `-1.0` |
| \(n\) | `x.size` | `3` |

LLM では一つの token を数百〜数千個の数で表します。その一列が embedding vector です。

## 要素ごとの演算

同じ size の vector は対応する要素ごとに足せます。

\[
\boldsymbol{x}+\boldsymbol{y} =
\begin{bmatrix}
x_1+y_1 \\ \vdots \\ x_n+y_n
\end{bmatrix}
\]

```scala
VectorD(1.0, 2.0) + VectorD(3.0, 4.0) // VectorD(4.0, 6.0)
```

異なる size では、どの要素を対応させるか定まりません。`VectorD` は先に shape を検査し、
`IllegalArgumentException` で停止します。暗黙に切り詰めたり、0 を補ったりしません。

scalar \(a\) との積は全要素へ同じ数を掛けます。

\[
a\boldsymbol{x} = [ax_1, ax_2, \ldots, ax_n]
\]

対応する要素を掛けた vector は Hadamard 積です。

\[
\boldsymbol{x}\odot\boldsymbol{y} = [x_1y_1, x_2y_2, \ldots, x_ny_n]
\]

これは次の内積とは出力の型が違います。

## 内積は二つの vector を一つの scalar へ縮約する

\[
\boldsymbol{x}\cdot\boldsymbol{y} = \sum_{i=1}^{n} x_i y_i
\]

```scala
VectorD(1.0, 2.0, 3.0).dot(VectorD(4.0, 5.0, 6.0))
// 1*4 + 2*5 + 3*6 = 32
```

shape は `[n] dot [n] -> scalar` です。内積は「二つの方向がどの程度一致するか」を大きさ込みで
表します。Attention では query と key の関連度を計算する中心的な演算になります。

## norm は vector の長さ

Euclidean norm（L2 norm）は、各要素を二乗した合計の平方根です。

\[
\lVert\boldsymbol{x}\rVert_2
= \sqrt{\sum_{i=1}^{n}x_i^2}
= \sqrt{\boldsymbol{x}\cdot\boldsymbol{x}}
\]

`VectorD(3.0, 4.0).norm` は直角三角形の三平方の定理と同じ計算で `5.0` になります。二つの
vector の cosine similarity は、内積を両者の norm で割った値です。

\[
\cos(\theta) =
\frac{\boldsymbol{x}\cdot\boldsymbol{y}}
{\lVert\boldsymbol{x}\rVert_2\lVert\boldsymbol{y}\rVert_2}
\]

後の vector search で使います。

## なぜ内部に `Array` を使い、外には見せないのか

JVM の `Array[Double]` は primitive の `double[]` となり、数値を連続して保持できます。しかし
mutable なので、同じ配列を外部と共有すると `VectorD` が知らない間に変わります。

`VectorD` は constructor を `private` にし、次を守ります。

- factory で全要素が有限か検査する
- 受け取った値から所有する配列を作る
- 配列そのものを返さない
- `updated` や演算は新しい `VectorD` を返す

この **representation invariant** により、「一度作られた vector は有限値を持ち、変化しない」
と仮定できます。

## `while` loop を使う理由

数値計算の内側では、要素ごとの一時 object を減らし、実際の index 操作を見えるようにするため
`while` を使います。`map` など利用側の API は高水準に保ちつつ、内部の計算量を追跡できます。

- vector 加算: 時間 \(O(n)\)、追加 memory \(O(n)\)
- 内積: 時間 \(O(n)\)、結果 memory \(O(1)\)
- index 参照: 時間 \(O(1)\)

`O(n)` は、要素数を定数倍するとおおむね処理回数も同じ定数倍になることを表す記法です。

## 確認

```console
$ nix develop -c sbt test
```

`VectorD` suite では値、immutability、shape error、非有限値、空 vector の reduction を調べます。

## 演習

1. `VectorD(1.0, 2.0).hadamard(VectorD(3.0, 4.0))` を手計算してください。
2. L1 norm \(\sum_i |x_i|\) を追加し、負の値を含む test を書いてください。
3. cosine similarity を実装し、同方向、直交、逆方向の vector で確認してください。
4. zero vector の cosine similarity が定義できない理由を説明し、戻り値の型を設計してください。
5. `map` の関数が `Double.PositiveInfinity` を返したときの動作を確認してください。

## 完了条件

- scalar、vector、shape を区別できる
- Hadamard 積と内積の出力 shape の違いを説明できる
- norm と内積の関係を説明できる
- immutable な wrapper が raw array より安全な理由を説明できる
- `VectorDSuite` が成功する
