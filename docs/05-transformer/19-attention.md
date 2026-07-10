# 19 — scaled dot-product causal self-attention

## この章で作るもの

query/key/value projection、scaled dot-product、causal mask、row softmax、multi-head split/concat、output
projection を実装します。対象コードは
`src/main/scala/learnai/transformer/Attention.scala` です。

テストでは shape だけでなく、未来位置の weight が 0、prefix 出力が未来 token 変更に不変、prefix
loss から未来 input への gradient が 0 であることを確認します。

## Attention が解く問題

各 token embedding は最初、自分の token と位置だけを表します。文脈依存の表現を作るには、過去の
どの位置から、どの程度情報を読むかを内容に応じて変える必要があります。

各位置の hidden vector \(\boldsymbol{x}_t\) から三つを作ります。

\[
\boldsymbol{q}_t=\boldsymbol{x}_t\boldsymbol{W}_Q
\]
\[
\boldsymbol{k}_t=\boldsymbol{x}_t\boldsymbol{W}_K
\]
\[
\boldsymbol{v}_t=\boldsymbol{x}_t\boldsymbol{W}_V
\]

- query: 今の位置が何を探しているか
- key: その位置が何を提供できるか
- value: 選ばれたとき実際に渡す情報

Q/K が「どこから読むか」、V が「何を読むか」を分離します。

## shape を追う

一 sequence、time \(T\)、channels \(C\) の single-head なら次です。

```text
X: [T,C]
Wq, Wk, Wv: [C,C]
Q, K, V: [T,C]
scores = Q K^T: [T,T]
weights = softmax(mask(scores)): [T,T]
headOutput = weights V: [T,C]
```

scores の row が「読む側の query position」、column が「読まれる key position」です。

## dot product similarity

query \(i\) と key \(j\) の score は内積です。

\[
s_{ij}=\boldsymbol{q}_i\cdot\boldsymbol{k}_j
\]

学習により、必要な関係で query/key の方向が揃い、score が高くなります。内積は norm にも依存する
ため、cosine similarity と完全には同じではありません。

## \(\sqrt{d}\) で scale する理由

head dimension \(d\) の各要素が独立で平均 0、分散 1 程度なら、内積の分散は \(d\) に比例します。
dimension が大きいほど logits の絶対値が増え、softmax が one-hot に近く飽和し、gradient が小さく
なります。

\[
s_{ij}=\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}
\]

標準偏差の規模を揃え、head dimension を変えても softmax の初期挙動を安定させます。

## causal mask

position \(i\) は未来 \(j>i\) を見てはいけません。training target 自体が input 内の次位置にある
ため、mask がなければ正解 token を直接読む data leakage になります。

```text
allowed key positions
query 0: [0, -, -]
query 1: [0, 1, -]
query 2: [0, 1, 2]
```

softmax 前に未来 score を非常に小さい値へ置換します。

\[
\tilde{s}_{ij}=\begin{cases}
s_{ij} & j\le i\\
-\infty & j>i
\end{cases}
\]

実装は finite invariant を保つため `-1e9` を使います。row 最大値を引いた `exp` で underflow し、
未来確率は `0.0` になります。backward も未来 score へ gradient を通しません。

## row softmax

query row ごとに key score を確率へします。

\[
a_{ij}=\frac{e^{\tilde{s}_{ij}}}{\sum_{r=1}^{T}e^{\tilde{s}_{ir}}}
\]

各 row は非負で合計 1 です。output は value の重み付き平均です。

\[
\boldsymbol{o}_i=\sum_j a_{ij}\boldsymbol{v}_j
\]

softmax backward は Jacobian を明示的な `[T,T]` matrix として作らず計算します。上流 gradient
\(g_i\) と output probability \(y_i\) に対し、

\[
\frac{\partial L}{\partial x_i}
=y_i\left(g_i-\sum_jg_jy_j\right)
\]

logit 全体へ同じ定数を足しても softmax は変わらないため、入力 gradient の row 合計は 0 です。

## multi-head

channels を \(H\) heads に分け、各 head dimension を \(d=C/H\) とします。

```text
projected Q/K/V: [T,C]
split H times:   H x [T,d]
attention:       H x [T,d]
concatenate:     [T,C]
output project:  [T,C]
```

異なる head が構文、参照、局所 pattern など異なる関係を表現する余地を作ります。ただし「head 1 は
必ず構文」のように人間が固定するのではなく、loss から学びます。channels は head count で割り切れる
必要があります。

この実装は Q/K/V を一度 `[T,C]` へ projection して column slice します。production では通常
`[T,H,d]` へ view/reshape し、batch/head を含む batched matmul kernel を使います。

## causality の三段階テスト

1. **weight:** upper triangle が 0
2. **forward:** future input を大きく変えても earlier output が同じ
3. **backward:** earlier position だけの loss から future input gradient が 0

一つだけでは mask の forward/backward bug を見逃し得ます。特に値がたまたま同じ、projection が 0、
mask 後の gradient が誤って流れる場合を分けて検査します。

## 計算量

score matrix と value 集約は時間 \(O(T^2C)\)、attention weights memory は \(O(T^2)\) per head/batch
です。context length を 2 倍にすると score 要素は 4 倍です。長文向けに flash attention、sliding
window、sparse/linear attention などが使われます。

Flash Attention は近似でなく、tiling と online softmax により巨大 score matrix を HBM に materialize
しない exact algorithm です。教材の式と結果を保ちながら memory traffic を減らします。

## 演習

1. `T=4` の causal mask を 0/1 matrix で書いてください。
2. query/key を手で決めた一 head attention を計算してください。
3. scale を外し、head dimension を増やした softmax entropy を測ってください。
4. mask を softmax 後に掛けるだけでは row sum が 1 でなくなることを示してください。
5. softmax backward 式を Jacobian から導出してください。
6. head count を変えた parameter 数が変わらない理由を Q/K/V shape から説明してください。
7. attention weight 可視化が「model の完全な説明」ではない理由を考えてください。

## 完了条件

- Q/K/V の役割と shape を説明できる
- score `[T,T]` の row/column の意味を説明できる
- \(1/\sqrt{d}\) scale の分散上の理由を説明できる
- causal mask を softmax 前に適用する理由を説明できる
- softmax row と weighted value sum を計算できる
- multi-head split/concat の shape を追える
- Attention/Tensor の causality と gradient test が成功する
