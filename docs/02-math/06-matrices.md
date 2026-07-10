# 06 — 行列と shape

## この章で作るもの

`Double` を行と列に並べた immutable な `MatrixD` と、転置、matrix-vector 積、行列積を実装
します。対象コードは `src/main/scala/learnai/math/MatrixD.scala` です。

## matrix は二つの軸を持つ

\(m\) 行 \(n\) 列の matrix を次のように書きます。

\[
\boldsymbol{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\in \mathbb{R}^{m\times n}
\]

shape は `[m, n]`、Scala では `matrix.rows == m`、`matrix.columns == n` です。

```scala
val matrix = MatrixD.fromRows(
  Vector(
    VectorD(1.0, 2.0, 3.0),
    VectorD(4.0, 5.0, 6.0)
  )
)
```

この shape は `[2, 3]` です。`matrix(1, 2)` は 0-based index なので `6.0` です。

LLM では、語彙全体の embedding を `[vocabulary, channels]`、一つの weight を
`[outputChannels, inputChannels]` のような matrix で保持します。

## row-major layout

JVM の一次元 `Array[Double]` に、行ごとに連続して保存します。

```text
matrix:  [ [a, b, c],
           [d, e, f] ]
storage: [a, b, c, d, e, f]
offset(row, column) = row * columns + column
```

連続した memory は CPU cache を利用しやすく、二重の配列より object が少なくなります。
`MatrixD` は `rows` と `columns` を保持することで、一次元 storage を二次元として解釈します。

## 転置

転置 \(\boldsymbol{A}^\mathsf{T}\) は行と列を入れ替えます。

\[
(\boldsymbol{A}^\mathsf{T})_{ij}=A_{ji}
\]

shape は `[m, n] -> [n, m]` です。二回転置すると元へ戻ります。

\[
(\boldsymbol{A}^\mathsf{T})^\mathsf{T}=\boldsymbol{A}
\]

このような常に成り立つ性質は、個別の期待値だけでなく property としてテストできます。

## matrix-vector 積

`[m, n]` の matrix と `[n]` の vector の積は `[m]` の vector です。各出力は matrix の一行と
入力 vector の内積です。

\[
\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x},\quad
y_i=\sum_{j=1}^{n}A_{ij}x_j
\]

```text
[m, n] x [n] -> [m]
```

内側の dimension `n` が一致しなければ計算できません。neuron の観点では、各行が一つの出力
neuron の weight、入力 vector が前の層の activation です。

## 行列積

\(\boldsymbol{A}\) の shape を `[m, k]`、\(\boldsymbol{B}\) を `[k, n]` とすると、積の shape は
`[m, n]` です。

\[
\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B},\quad
C_{ij}=\sum_{r=1}^{k} A_{ir}B_{rj}
\]

```text
[m, k] x [k, n] -> [m, n]
        ^   ^
        inner dimensions must match
```

出力の一要素 \(C_{ij}\) は、左 matrix の行 \(i\) と右 matrix の列 \(j\) の内積です。実装の
三重 loop は、出力 row、出力 column、内積の index の順に走ります。

単純実装の計算量は時間 \(O(mkn)\)、出力 memory \(O(mn)\) です。実用 framework は block 化、
vector 命令、thread、GPU の多数の演算器を使いますが、計算する式は同じです。

## shape は型の一部か

現在の `MatrixD` の Scala type は、`[2, 3]` でも `[4, 8]` でも同じです。shape は runtime の
値なので、演算時に `require` で検査します。

型 level の自然数を使えば一部を compile 時に検査できますが、batch size や入力長は runtime に
決まるため、すべてを静的には表しにくいです。本教材では error message に両方の shape を含め、
失敗を観察可能にします。

## 0 を含む dimension

`MatrixD.zeros(0, 3)` と `MatrixD.zeros(2, 0)` は storage がどちらも空ですが、shape の意味は
違います。`MatrixD` は dimension を明示的に保持するため、この区別を失いません。

空 matrix を許すと一般的な計算規則を保てる一方、平均や最大値のように定義できない reduction も
あります。その場合は `Either` などで未定義を表します。

## 確認

```console
$ nix develop -c sbt test
```

`MatrixD` suite は indexing、transpose、matvec、matmul、shape error、immutability、空 dimension
を確認します。

## 演習

1. `[2, 3] x [3, 4]` の出力 shape と、一要素を計算する乗算回数を答えてください。
2. `MatrixD` に要素ごとの Hadamard 積を追加してください。
3. identity matrix \(\boldsymbol{I}\) の factory を実装し、`A.matmul(I) == A` をテストしてください。
4. 各列の合計を返す関数を実装し、出力 shape を先に書いてください。
5. 現在の loop 順序で右 matrix の memory access が連続しない理由を調べてください。

## 完了条件

- matrix の要素と row-major offset を対応させられる
- matvec と matmul の式を loop の index へ対応させられる
- 行列積で内側 dimension が一致する必要を説明できる
- `[m, k] x [k, n] -> [m, n]` を見ずに書ける
- `MatrixDSuite` が成功する
