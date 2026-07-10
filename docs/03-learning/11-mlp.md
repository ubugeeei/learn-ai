# 11 — neuron、layer、MLP を学習させる

## この章で作るもの

scalar 自動微分 engine の `Value` だけを使って neuron、layer、multi-layer perceptron（MLP）、
mean squared error、SGD を実装し、XOR を学習します。

- network: `src/main/scala/learnai/nn/ScalarNetwork.scala`
- experiment: `src/main/scala/learnai/nn/Xor.scala`

## 一つの neuron

入力 vector \(\boldsymbol{x}\in\mathbb{R}^n\)、weight \(\boldsymbol{w}\in\mathbb{R}^n\)、bias
\(b\in\mathbb{R}\) に対して、一つの neuron は次を計算します。

\[
z=\boldsymbol{w}\cdot\boldsymbol{x}+b
=\sum_{i=1}^{n}w_ix_i+b
\]

その後、activation function \(\phi\) を適用します。

\[
y=\phi(z)
\]

コードとの対応は次です。

```scala
val weightedInputs = weights.zip(inputs).map { case (weight, input) =>
  weight * input
}
activation(Value.sum(weightedInputs) + bias)
```

`weight` と `bias` は学習で変わる trainable parameter、入力と途中結果は通常 parameter では
ありません。

## activation が必要な理由

linear function を何層重ねても、一つの linear function にまとめられます。

\[
\boldsymbol{W}_2(\boldsymbol{W}_1\boldsymbol{x}+\boldsymbol{b}_1)+\boldsymbol{b}_2
=(\boldsymbol{W}_2\boldsymbol{W}_1)\boldsymbol{x}
+(\boldsymbol{W}_2\boldsymbol{b}_1+\boldsymbol{b}_2)
\]

曲がりを作る non-linear activation がないと、深くしても表現できる境界は増えません。

| activation | 式 | 特徴 |
| --- | --- | --- |
| Linear | \(x\) | 出力 score や回帰に使う |
| Tanh | \(\tanh x\) | `[-1, 1]`、滑らか、今回の XOR に使う |
| ReLU | \(\max(0,x)\) | 正側で勾配が一定、単純で計算しやすい |

LLM の feed-forward network では GELU や SwiGLU がよく使われます。原理は「linear 変換の間に
non-linearity を入れる」という同じ考えです。

## layer と MLP

一つの layer は複数 neuron を並列に計算します。入力 size が \(n\)、出力 size が \(m\) なら、
weight 全体は `[m, n]`、bias は `[m]` です。

\[
\boldsymbol{y}=\phi(\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b})
\]

複数 layer を順に合成したものが MLP です。

```text
input [2]
  -> hidden layer [4], tanh
  -> output layer [1], tanh
```

今回の parameter 数は次です。

- hidden: `4 * 2` weights + `4` biases = 12
- output: `1 * 4` weights + `1` bias = 5
- total: 17

parameter 数を数える習慣は、model memory と必要計算量を見積もる基礎になります。

## XOR は linear な境界一つでは解けない

XOR dataset は、二つの入力の符号が異なると `+1`、同じなら `-1` です。

```text
(-1,  1) +       + (1, -1)

(-1, -1) -       - (1,  1)
```

一本の直線で `+` と `-` を分けられません。hidden layer の neuron が複数の中間的な境界を作り、
activation で曲げ、output layer が組み合わせます。

## loss

prediction \(\hat{y}_i\) と target \(y_i\) の mean squared error（MSE）を使います。

\[
L=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
\]

誤差 0 で loss 0、外れるほど二乗で大きくなります。言語モデルでは categorical な次 token を
予測するため cross entropy を使いますが、training loop の構造は同じです。

## 一 step を追う

```scala
val predictions = dataset.map(example => model(example.inputs).head)
val loss = Loss.meanSquaredError(predictions, targets)
loss.backward()
Sgd.step(model.parameters, learningRate)
```

1. **forward:** 現在の全 parameter から 4 例を予測する
2. **loss:** 四つの誤差を一つの scalar にする
3. **backward:** 全 17 parameter の gradient を求める
4. **update:** \(\theta\leftarrow\theta-\eta\nabla_\theta L\)
5. 次の step で新しい computation graph を作る

全例を一度に使う full-batch training です。大規模データでは一部を mini-batch として sampling
します。

## 初期化

全 weight を 0 にすると、同じ layer の neuron が同じ gradient を受け取り続け、役割が分かれません。
そこで seed 付き乱数で異なる値を与えます。

この実装は入力・出力 size に基づく Xavier uniform の範囲を使います。

\[
w\sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},
\sqrt{\frac{6}{n_{in}+n_{out}}}\right)
\]

activation の分散が層を通して急激に消滅・増大しないことを狙います。適切な初期化は activation
に依存します。

## 実行

```console
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
initial loss: ...
final loss:   ...
input=[-1.0, -1.0] target=-1.0 prediction=-0....
...
```

seed を固定しているので同じ環境では同じ初期値と学習曲線になります。test は loss が十分に減り、
4 例すべての prediction の符号が正しいことを確認します。

## 今回の単純化

- scalar ごとに graph node を作るため、大きな layer には遅い
- XOR 全例を training に使い、generalization は評価していない
- MSE と単純 SGD だけを使う
- regularization、normalization、checkpoint がない
- parameter update は単一 thread

次に Tensor 単位の演算と optimizer を作り、同じ training loop を大きな配列へ拡張します。

## 演習

1. 17 parameter になる計算を自分で書いて確認してください。
2. hidden size を `1`、`2`、`8` に変え、複数 seed で成功率を比較してください。
3. hidden activation を ReLU に変え、loss curve を比較してください。
4. learning rate を変え、最初に loss が `0.02` を下回る step を記録してください。
5. 出力 activation を Linear に変え、prediction の範囲と収束を観察してください。
6. AND dataset を作り、hidden layer が不要か実験してください。

## 完了条件

- neuron の式を weight、input、bias、activation に分解できる
- non-linear activation がなければ深い linear network を一層にまとめられる理由を説明できる
- layer ごとの shape と parameter 数を計算できる
- training step の四段階をコード上で指し示せる
- seed を固定する理由を説明できる
- `ScalarNetworkSuite` が成功する
