# 09 — 勾配降下で「学習する」を作る

## この章で作るもの

一つの parameter を持つ関数を gradient descent で最小化し、全 step の parameter、loss、
gradient を記録します。対象コードは `src/main/scala/learnai/learning/GradientDescent.scala` です。

## 学習を最適化問題として書く

model の振る舞いを決める数を parameter \(\theta\)、悪さを測る関数を loss \(L(\theta)\) とします。
学習の目的は loss を最小にする parameter を探すことです。

\[
\theta^*=\operatorname*{arg\,min}_{\theta}L(\theta)
\]

`min` が最小値そのものを表すのに対し、`arg min` は最小値を与える引数を表します。

最初の例では、parameter を目標値 3 に近づけます。

\[
L(\theta)=(\theta-3)^2
\]

loss は 0 以上で、\(\theta=3\) のときだけ 0 です。導関数は次です。

\[
\frac{dL}{d\theta}=2(\theta-3)
\]

## update rule

gradient は loss が増える向きを示すため、その反対へ parameter を動かします。

\[
\theta_{t+1}=\theta_t-\eta\frac{dL}{d\theta_t}
\]

| 記号 | 意味 | コード |
| --- | --- | --- |
| \(t\) | 現在の step | `step` |
| \(\theta_t\) | 現在の parameter | `parameter` |
| \(L\) | loss function | `loss` |
| \(dL/d\theta_t\) | 現在位置の傾き | `gradient` |
| \(\eta\) | learning rate | `learningRate` |

learning rate は一回にどれだけ動くかを決める正の数です。

```scala
parameter -= learningRate * gradient
```

この一行は短いですが、順序が重要です。

1. 現在の parameter で forward 計算を行い loss を得る
2. loss の parameter に関する gradient を得る
3. parameter を update する
4. 新しい parameter で次の step を始める

## loss curve を観察する

```console
$ nix develop -c sbt 'runMain learnai.learning.runGradientDescentLab'
```

初期値 `-4.0` から parameter が 3 へ近づき、loss と gradient の絶対値が小さくなります。
`DescentObservation` を保存するのは、最終値だけで成功を判断しないためです。

見るべき pattern は次です。

- loss が一貫して下がるか
- gradient が 0 に近づくか
- parameter が同じ二点を往復していないか
- `NaN` や `Infinity` が出ていないか
- step あたりの改善量が小さくなり過ぎていないか

## learning rate の trade-off

- 小さ過ぎる: 安定するが、到達までの step が多い
- 適切: loss を安定して速く下げる
- 大き過ぎる: 最小点を飛び越え、振動・発散する

二次関数 \(L=(\theta-3)^2\) の update を展開すると、目標からの距離は毎回 \(1-2\eta\) 倍に
なります。`η = 0.1` なら `0.8` 倍、`η = 1.0` なら符号を反転して同じ距離、`η > 1.0` なら
距離が増えて発散します。

## local minimum と非凸関数

この例は一つだけ最小点を持つ convex な関数です。neural network の loss surface は高次元で、
鞍点、平らな領域、異なる経路があります。gradient descent は現在位置の局所的な傾きだけを使い、
大域的な最小値を保証しません。

それでも、多数の parameter、適切な初期化、mini-batch の noise、optimizer の工夫により、実用上
よい解を探索できます。成功を「理論上の最小値」だけでなく、未使用データでの性能で評価します。

## training と evaluation を分ける

training data の loss だけを下げると、例を暗記して未知データへ一般化しない **overfitting** が
起こります。データを少なくとも次へ分けます。

- training set: parameter update に使う
- validation set: hyperparameter 選択と途中評価に使う
- test set: 最終判断に一度使う

test set の結果を見て設計を変え続けると、test set にも間接的に適合するため注意します。

## 演習

1. learning rate を `0.01`、`0.5`、`1.0`、`1.1` に変え、loss curve を比較してください。
2. 初期値を 3 の左右へ置き、gradient の符号を説明してください。
3. loss を \((\theta+2)^2+1\) に変え、最小の parameter と loss を予想してください。
4. `history` を CSV 形式で表示し、任意の plot tool で曲線を描いてください。
5. 数値微分を `derivative` 引数へ渡し、解析的な導関数との到達誤差を比べてください。

## 完了条件

- loss、parameter、gradient、learning rate、step を区別できる
- update rule のマイナス符号を説明できる
- learning rate が小さ過ぎる場合と大き過ぎる場合を実験で示せる
- training loss だけでは一般化を判定できない理由を説明できる
- `GradientDescentSuite` が成功する
