# 17 — bigram 言語モデル

## この章で作るもの

現在の一 token だけから次 token の logits を出す trainable bigram model を Tensor 自動微分で実装
し、cross entropy で学習して sampling します。

- model: `src/main/scala/learnai/lm/BigramLanguageModel.scala`
- Tensor 演算: `gatherRows`、`crossEntropy`

この章で tokenizer → token pairs → logits → loss → backward → optimizer → generation の言語モデル
全体が初めてつながります。

## bigram の仮定

一般の autoregressive language model は過去 token 全体から次 token の確率を表します。

\[
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})
\]

bigram model は「次 token は現在の一 token だけに依存する」と単純化します。

\[
p(x_{t+1}\mid x_0,\ldots,x_t)\approx p(x_{t+1}\mid x_t)
\]

長期文脈、単語の意味、文法を十分に扱えませんが、next-token training の入出力と生成 loop を最小
構成で観察できます。

## transition logit table

vocabulary size を \(V\) とし、trainable matrix \(\boldsymbol{W}\in\mathbb{R}^{V\times V}\) を持ち
ます。現在 token ID \(i\) なら row \(i\) が次 token logits です。

```text
W shape: [current token, next token] = [V,V]

             next token
             0    1    2
current 0  [ ...  ...  ... ]
token   1  [ ...  ...  ... ]
        2  [ ...  ...  ... ]
```

input IDs `[N]` に `gatherRows` を使うと logits shape は `[N,V]` です。同じ ID が複数現れれば同じ
row を読み、backward ではその row へ全出現位置の gradient を合計します。これは embedding lookup
と同じ演算です。

parameter 数は \(V^2\) です。byte vocabulary 256 なら 65,536、50,000 vocabulary なら 25 億と
なり、単純 bigram table は vocabulary 拡大に対して非効率です。embedding と低次元 hidden state が
factorization の役割を果たします。

## fused cross entropy

logits row \(\boldsymbol{z}\) と target class \(t\) の loss は次です。

\[
L=\log\sum_j e^{z_j}-z_t
\]

gradient は特に単純です。

\[
\frac{\partial L}{\partial z_j}=p_j-\mathbb{1}[j=t]
\]

\(\mathbb{1}[j=t]\) は target class だけ 1 の indicator です。batch mean では example 数でも割り
ます。

`Tensor.crossEntropy` は stabilized log-sum-exp と、この gradient を一演算にまとめます。巨大な
one-hot Tensor や softmax graph を作らず、memory と計算を減らします。production kernel fusion の
小さな例です。

各 row の gradient 合計は 0 です。

\[
\sum_j(p_j-\mathbb{1}[j=t])=1-1=0
\]

test は巨大 logits で finite であること、この property、数値微分との一致を確認します。

## training pairs

token 列 `[t0,t1,t2,t3]` から bigram pair を作ります。

```text
inputs:  [t0,t1,t2]
targets: [t1,t2,t3]
```

model の context length は 1 なので window grouping は不要ですが、前章の causal target shift と同じ
です。full-batch の一 step は次です。

```scala
val loss = model.loss(inputs, targets)
loss.backward()
optimizer.step(model.parameters)
```

target transition が繰り返されるほど、その row の target logit が相対的に上がります。

## generation loop

学習時は正解の previous token を入力する teacher forcing です。生成時には、自分が sample した token
を次の入力に戻します。

```text
start token
  -> logits row
  -> softmax
  -> sample next token
  -> use sampled token as next input
  -> repeat
```

一度低確率の token を sample すると、training data に少ない遷移へ入り、error が連鎖する場合があり
ます。training と generation の input 分布が違う点を exposure bias と呼びます。

## temperature

softmax 前に logits を temperature \(\tau>0\) で割ります。

\[
p_i=\operatorname{softmax}(z_i/\tau)
\]

- \(\tau<1\): 差を拡大し、上位 token へ集中
- \(\tau=1\): 学習した分布そのまま
- \(\tau>1\): 平らにし、多様性を増やす
- \(\tau\to0\): argmax に近づくが、実装では 0 を許さない

同じ seed でも temperature や model state が変われば sample 列は変わります。

## UTF-8 生成の注意

byte/BPE token の任意の組み合わせが、途中または全体で有効な UTF-8 とは限りません。特に context 1
の bigram は multi-byte 文字の長い制約を覚えにくいです。demo は strict decode の失敗も結果として
表示します。壊れた生成を replacement で隠しません。

## 実行

```console
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
initial loss: ...
final loss:   ...
generated: ...
```

demo は小さな反復 corpus で BPE と bigram を学習します。品質を主張する benchmark ではなく、全
pipeline を trace するための実験です。

## bigram から Transformer へ

bigram の限界は、異なる文脈でも最後の token が同じなら完全に同じ分布を返すことです。

```text
"bank account" の後の token
"river bank"   の後の token
```

Transformer は各 token の表現を embedding にし、Attention で過去位置の情報を混ぜ、文脈依存の
hidden vector から logits を作ります。変わらない部分もあります。

- tokenization
- causal target shift
- logits と softmax
- cross entropy
- backward と optimizer
- autoregressive generation

## 演習

1. vocabulary 3 の transition table parameter 数と各 row の意味を説明してください。
2. 小 corpus の bigram count から経験確率を計算し、学習後 softmax と比べてください。
3. target class の logit を 1 増やしたとき loss がどう変わるか測ってください。
4. temperature `0.2`、`1.0`、`2.0` で token frequency を比較してください。
5. validation token pairs の loss を training と別に計算してください。
6. bigram が区別できない二文脈を corpus から探してください。
7. transition table を low-rank な二行列の積へ置き換え、parameter 数を比較してください。

## 完了条件

- chain rule による sequence probability を説明できる
- bigram の条件付き独立仮定と限界を説明できる
- `gatherRows` の forward/backward shape を書ける
- softmax cross entropy gradient \(p-oneHot\) を説明できる
- teacher forcing と autoregressive generation の違いを説明できる
- temperature が分布へ与える効果を説明できる
- `BigramLanguageModelSuite` と追加された Tensor test が成功する
