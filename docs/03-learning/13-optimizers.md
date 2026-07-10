# 13 — 初期化、SGD、AdamW、gradient clipping

## この章で作るもの

Tensor parameter の更新境界、global gradient norm、clipping、SGD、AdamW、Xavier/He 初期化を
実装します。対象コードは `src/main/scala/learnai/optim/Optimizers.scala` です。

## optimizer の責務

自動微分は現在の parameter に対する gradient を求めます。optimizer は gradient と内部状態から、
次の parameter を決めます。

```text
forward -> loss -> backward -> gradients
                                |
                                v
                         optimizer state
                                |
                                v
                         updated parameters
```

`Tensor.updateParameter` だけが parameter data を変更する境界です。callback は flat index、現在値、
gradient を受け、次の値を返します。optimizer は graph の内部構造を知りません。

## SGD

最も単純な stochastic gradient descent は次です。

\[
\theta_t=\theta_{t-1}-\eta g_t
\]

\(g_t\) は step \(t\) の mini-batch から計算した gradient です。全 dataset でなく sample された
batch を使うため stochastic と呼ばれます。noise は不安定さの原因にも、探索を助ける要素にも
なります。

SGD は parameter ごとの追加 state を持たず、memory が少なく、挙動を理解しやすい baseline です。

## global gradient norm

全 parameter gradient を一つの長い vector と見なした L2 norm です。

\[
\lVert\boldsymbol{g}\rVert_2
=\sqrt{\sum_{p}\sum_i g_{p,i}^2}
\]

層ごとの値だけでなく model 全体で gradient が爆発しているかを一つの指標で観察できます。異なる
model size の単純比較には注意が必要ですが、同じ model の training 異常検知には有用です。

## gradient clipping

norm が上限 \(c\) を超えたとき、全 gradient に同じ scale を掛けます。

\[
\tilde{\boldsymbol{g}}
=\boldsymbol{g}\min\left(1,\frac{c}{\lVert\boldsymbol{g}\rVert_2}\right)
\]

全要素に同じ正の数を掛けるため方向は保ち、大きさだけを上限へ縮めます。`[3,4]` の norm は 5、
上限 1 なら scale は `0.2`、結果は `[0.6,0.8]` です。

clipping は gradient 爆発の影響を限定しますが、根本原因を隠す場合があります。clipped step の割合、
元の norm、loss、activation も記録します。

## momentum の考え方

谷の両側で gradient が振動するとき、過去の gradient の移動平均を使うと一貫した方向を強められ
ます。

\[
m_t=\beta m_{t-1}+(1-\beta)g_t
\]

\(\beta\) が 1 に近いほど長い履歴を滑らかに平均します。Adam はこの first moment に加え、
gradient 二乗の second moment も追跡します。

## Adam

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

初期値 0 の移動平均は、最初の step で 0 側へ偏ります。bias correction を行います。

\[
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
\]

update は次です。

\[
\theta_t=\theta_{t-1}
-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

second moment が大きい要素は update を小さく、安定して小さい gradient の要素は相対的に大きく
動かします。parameter 一要素につき `m` と `v` の二つの state が必要なため、model weight と
gradient 以外の memory が増えます。

## AdamW と decoupled weight decay

weight decay は parameter を 0 側へ縮めます。

\[
\theta\leftarrow(1-\eta\lambda)\theta
\]

AdamW はこの decay を adaptive gradient update から分離します。

\[
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

gradient が 0 でも decay は働きます。bias、normalization scale、embedding など、どの parameter に
decay を適用するかは実用 model で parameter group として分けます。

## 初期化と signal の分散

層が深いと、forward activation と backward gradient の大きさが層ごとに増幅・減衰し得ます。
weight の分散を fan-in/out に合わせます。

Xavier uniform:

\[
w\sim U\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}\right]
\]

tanh など両側を使う activation の出発点です。ReLU 向けの He uniform は fan-in を使います。

\[
w\sim U\left[-\sqrt{\frac{6}{n_{in}}},
\sqrt{\frac{6}{n_{in}}}\right]
\]

式を盲目的に選ばず、層ごとの activation mean/std、gradient norm を測ります。

## reproducibility

初期化は `SplittableRandom` と明示 seed を受け取ります。同じ seed、shape、順序なら同じ weight
になります。model 構造を変えて乱数の消費順が変われば、同じ seed でも対応する値は変わります。

実験記録には少なくとも次を含めます。

- seed
- optimizer と全 hyperparameter
- initialization
- batch/data 順序
- model 構造
- code revision

## 演習

1. gradient `[6,8]` を最大 norm `5` で clip した値を手計算してください。
2. SGD に momentum state を追加し、二次関数で軌跡を比較してください。
3. Adam の最初の一 step を scalar gradient `2.0` で手計算してください。
4. bias correction を外し、最初の 10 step の移動量を比較してください。
5. weight decay を適用する group としない group を表す API を設計してください。
6. Xavier/He 初期化で大きな layer の sample mean と variance を測ってください。

## 完了条件

- SGD と AdamW が保持する state の違いを説明できる
- Adam の first/second moment と bias correction を説明できる
- global norm clipping が gradient の方向を保つことを示せる
- decoupled weight decay が gradient 0 でも働くことを説明できる
- activation に応じた初期化を測定して選ぶ理由を説明できる
- `OptimizersSuite` が成功する
