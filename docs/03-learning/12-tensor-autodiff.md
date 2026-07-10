# 12 — Tensor と reverse-mode 自動微分

## この章で作るもの

任意 rank の dense row-major `Tensor`、shape/index 変換、要素ごとの演算、reduction、reshape、
transpose、行列積と、それぞれの backward rule を実装します。

- shape: `src/main/scala/learnai/tensor/Shape.scala`
- tensor: `src/main/scala/learnai/tensor/Tensor.scala`

## scalar graph の限界

前章の `Value` では、一つの数と演算ごとに JVM object を作りました。`[1024, 1024]` の行列積
なら出力だけで約 100 万 node、内部の乗算・加算はさらに多数です。

Tensor node は多数の数を一つの演算としてまとめます。

```text
scalar graph: value node x millions
Tensor graph: MatMul node -> Add node -> Tanh node
```

backward の数学は同じですが、loop を一つの backward rule 内で実行します。実用 framework では
この単位が CPU/GPU kernel になります。

## rank、shape、size

- rank 0: scalar、shape `[]`、size 1
- rank 1: vector、shape `[n]`
- rank 2: matrix、shape `[m,n]`
- rank 3: token batch など、shape `[batch,time,channels]`

shape の全 dimension を掛けた値が要素数です。

\[
\operatorname{size}([d_0,d_1,\ldots,d_{r-1}])=\prod_i d_i
\]

`Shape.scalar` の dimension は空ですが、空積を 1 と定義するため scalar は一要素です。

## stride と row-major offset

shape `[2,3,4]` の stride は `[12,4,1]` です。座標 `[i,j,k]` の flat offset は次です。

\[
\operatorname{offset}=12i+4j+k
\]

最後の軸が memory 上で連続します。`[1,2,3]` は `12 + 8 + 3 = 23`、24 要素中の最後です。

`Shape` は dimension、stride、size を一度計算し、すべての Tensor が同じ index 規則を使うように
します。dimension の積が `Int` を overflow する場合も構築時に拒否します。

## broadcasting をまだ実装しない

`[2,3] + [3]` のように dimension 1 や不足軸を暗黙に繰り返す規則を broadcasting と呼びます。
便利ですが、どの軸に gradient を合計して戻すかという追加規則が必要です。

この章の二項要素演算は shape の完全一致を要求します。

```text
[2,3] + [2,3] -> [2,3]  OK
[2,3] + [3]   -> error  implicit broadcasting is disabled
```

Transformer を実装する際に、必要な broadcasting を名前付き演算として追加し、forward と backward
の axis を明示します。

## 要素ごとの backward

\(\boldsymbol{C}=\boldsymbol{A}\odot\boldsymbol{B}\) なら各要素は独立です。

\[
\frac{\partial L}{\partial A_i}
=\frac{\partial L}{\partial C_i}B_i,qquad
\frac{\partial L}{\partial B_i}
=\frac{\partial L}{\partial C_i}A_i
\]

scalar `Value` の積の規則を array index ごとに適用しているだけです。同じ Tensor が複数経路で
使われるため、gradient array にも加算します。

## reduction の backward

sum は `[d0,...] -> []` と全要素を一つにします。

\[
s=\sum_i x_i,\qquad \frac{\partial s}{\partial x_i}=1
\]

したがって出力 scalar に届いた gradient を、入力の全要素へ同じように加えます。mean は sum を
要素数で割った合成です。

## reshape は値の順序を変えない

`reshape` は要素数が同じ別 shape へ解釈を変えます。

```text
[2,3] -> [6]   OK, both size 6
[2,3] -> [2,2] error
```

flat index の順序は変わらないため、backward も同じ flat index へ gradient を戻します。transpose
は要素順を入れ替えるので、backward でも逆の index mapping が必要です。

## 行列積の backward

\(\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B}\) の shape を次とします。

```text
A [m,k] x B [k,n] -> C [m,n]
```

matrix calculus では gradient を次の行列積で書けます。

\[
\frac{\partial L}{\partial \boldsymbol{A}}
=\frac{\partial L}{\partial \boldsymbol{C}}\boldsymbol{B}^\mathsf{T}
\]

\[
\frac{\partial L}{\partial \boldsymbol{B}}
=\boldsymbol{A}^\mathsf{T}\frac{\partial L}{\partial \boldsymbol{C}}
\]

実装では output の各 `[row,column]` に届いた gradient を、forward で寄与した全 inner index へ
配ります。これも scalar の積と加算の連鎖律を loop にまとめたものです。

## gradient check

`TensorSuite` は一つの matrix 要素だけを \(+h\)、\(-h\) へ動かした数値微分と、matmul → pow →
mean の backward を比較します。

大きな Tensor の全要素を数値微分すると遅いため、実用的には次を組み合わせます。

- 小 shape で全要素 check
- 大 shape でランダムな一部を check
- 演算の代数的 property
- reference implementation との比較

## memory の見積もり

`Double` は一要素 8 bytes です。`[B,T,C]` の data だけで
`8 * B * T * C` bytes、学習時には gradient、中間 activation、optimizer state も必要です。

現在の engine は各 node が data と同サイズの gradient array を最初から確保します。教育上単純
ですが memory 効率はよくありません。実用 engine は gradient が不要な node、inference mode、
buffer reuse、activation checkpointing を扱います。

## 演習

1. shape `[2,3,4,5]` の stride と座標 `[1,2,3,4]` の offset を計算してください。
2. `sigmoid` を Tensor 演算として追加し、数値微分で確認してください。
3. `transpose2D.transpose2D` の values と gradient が元に戻る property を追加してください。
4. 行列積 backward の式を、index notation から導出してください。
5. `[32,128,768]` の data、gradient を `Double` で保持する bytes を計算してください。
6. 明示的な `addRowVector` と backward の reduction axis を設計してください。

## 完了条件

- rank、shape、size、stride、offset を相互に変換できる
- scalar graph と Tensor graph の違いを説明できる
- sum、reshape、transpose の backward で gradient がどう移動するか説明できる
- matmul の両入力 gradient の shape と式を書ける
- broadcasting を暗黙に導入しない理由を説明できる
- `TensorSuite` が成功する
