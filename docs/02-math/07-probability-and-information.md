# 07 — 確率と情報量

## この章で作るもの

検証済みの categorical distribution、数値的に安定した softmax と log-sum-exp、entropy、
cross entropy、sampling を実装します。対象コードは
`src/main/scala/learnai/math/Probability.scala` です。

## 確率は不確実性を数で表す

起こり得る結果の集合を sample space と呼びます。次 token 予測では、vocabulary 内の全 token が
候補です。各結果 \(i\) へ確率 \(p_i\) を割り当てます。

\[
p_i \geq 0, \qquad \sum_{i=1}^{V} p_i = 1
\]

\(V\) は vocabulary size です。この二条件を満たす vector が categorical distribution です。
`Categorical.from` は空、負の確率、合計が 1 でない入力を拒否します。一度構築できた
`Categorical` は、この不変条件を満たすと仮定できます。

## 確率変数、期待値、分散

確率変数 \(X\) は、各結果を数へ写す関数です。離散分布での期待値は、値を確率で重み付けした
平均です。

\[
\mathbb{E}[X] = \sum_i p_i x_i
\]

分散は、期待値からのずれの二乗の期待値です。

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]
\]

乱数を使った sampling の結果は一回ごとに変わりますが、十分な回数の平均は期待値へ近づきます。
この性質が mini-batch gradient の考えにつながります。

## model は確率になる前の score を出す

neural network の最後の層は各 token の実数 score \(z_i\) を出します。これを **logit** と呼び
ます。logit は負でも、合計が 1 でなくても構いません。softmax が確率へ変換します。

\[
p_i = \operatorname{softmax}(\boldsymbol{z})_i
= \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
\]

- `exp` により各値は正になる
- 全体の合計で割るため合計は 1 になる
- logit が大きい結果ほど確率が大きい

## 最大値を引く数値安定化

`exp(10000)` は `Double` の範囲を超えます。全 logit から同じ定数 \(c\) を引いても softmax は
変わりません。

\[
\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
=\frac{e^{z_i}/e^c}{\sum_j e^{z_j}/e^c}
=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

そこで \(c=\max_j z_j\) とします。最大の指数が `exp(0) == 1` となり、他は 1 以下になるため
overflow を避けられます。

`logSumExp` も同じ変形を使います。

\[
\log\sum_j e^{z_j}
= m + \log\sum_j e^{z_j-m},\quad m=\max_j z_j
\]

## 情報量と entropy

確率 \(p\) の出来事が起きたときの情報量を自然対数で次のように定義します。

\[
I(p)=-\log p
\]

確実な出来事 \(p=1\) の情報量は 0、珍しい出来事ほど大きくなります。分布から得る情報量の
期待値が entropy です。

\[
H(p)=-\sum_i p_i\log p_i
\]

確率 0 の項は極限により 0 と定義します。公平な coin の entropy は \(\log 2\)、必ず表が出る
coin の entropy は 0 です。

## cross entropy が言語モデルの loss になる

正解分布 \(q\) に対して model 分布 \(p\) が必要とする平均情報量が cross entropy です。

\[
H(q,p)=-\sum_i q_i\log p_i
\]

次 token が一つに決まる one-hot target なら、正解 index \(t\) の項だけが残ります。

\[
\mathcal{L}=-\log p_t
\]

softmax を展開すると、確率 vector を明示的に作らず logits から安定に計算できます。

\[
\mathcal{L}
=\log\sum_j e^{z_j} - z_t
\]

正解 logit が他より大きければ loss は 0 に近づき、正解へ低い確率を割り当てると loss は大きく
なります。

## sampling

確率分布から一つを選ぶため、`[0, 1)` の一様乱数 \(u\) を生成し、累積確率が \(u\) を初めて
超える index を返します。

```text
probabilities: [0.2, 0.5, 0.3]
cumulative:    [0.2, 0.7, 1.0]
u = 0.62  -> index 1
```

同じ seed から始めれば同じ乱数列になります。テストや比較実験では seed を固定し、生成の多様性を
観察するときだけ変えます。

## 確認

```console
$ nix develop -c sbt test
```

suite は分布の不変条件、巨大 logit、shift 不変性、cross entropy、entropy、決定的な sampling を
確認します。

## 演習

1. `softmax(VectorD(0.0, 0.0, 0.0))` を先に手計算してください。
2. `[0.9, 0.1]` と `[0.5, 0.5]` の entropy を比較し、どちらが不確実か説明してください。
3. 正解確率が `0.5`、`0.1`、`0.01` の cross entropy を計算してください。
4. 10000 回 sample した頻度を測り、元の確率との差を記録してください。
5. softmax で最大値を引かない版を一時的に作り、巨大 logit で何が起こるか観察してください。

## 完了条件

- logit、確率、one-hot target を区別できる
- softmax が非負かつ合計 1 になる理由を説明できる
- 最大値を引いても softmax が変わらない式を追える
- cross entropy と正解 token の負の対数確率を結び付けられる
- `ProbabilitySuite` が成功する
