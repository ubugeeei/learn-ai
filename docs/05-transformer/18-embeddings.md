# 18 — embedding、位置、linear、RMSNorm

## この章で作るもの

Transformer を構成する前に、token/position embedding、channel 軸への linear layer、RMSNorm を
Tensor 自動微分上へ実装します。対象コードは
`src/main/scala/learnai/transformer/Layers.scala` です。

## discrete ID から continuous vector へ

token ID は単なる category 番号です。ID `100` が ID `10` の 10 倍の意味を持つわけではありません。
embedding table \(\boldsymbol{E}\in\mathbb{R}^{V\times C}\) の row を lookup し、各 token を \(C\)
次元 vector へ変換します。

```text
token IDs:       [time]
embedding table: [vocabulary, channels]
output:          [time, channels]
```

\[
\boldsymbol{x}_t=\boldsymbol{E}[\text{tokenId}_t]
\]

初期 vector に意味はありません。next-token loss から届く gradient により、似た文脈で役立つ token
の vector が使いやすい配置へ動きます。

## embedding backward は scatter-add

forward は row を集める gather、backward は各出力 row の gradient を元 table row へ戻す scatter
です。同じ token が一 sequence に複数回現れれば、一つの embedding row へ全 gradient を加えます。

```text
IDs [2, 0, 2]
forward: rows 2, 0, 2 を gather
backward: row 2 へ position 0 と 2 の gradient を加算
```

parameter sharing の具体例です。

## Attention だけでは順序を知らない

self-attention は入力 row の集合に対する演算だけでは、並び替えに同じように反応します。token
embedding に position embedding \(\boldsymbol{P}[t]\) を加えます。

\[
\boldsymbol{h}_t^{(0)}
=\boldsymbol{E}[x_t]+\boldsymbol{P}[t]
\]

```text
token embedding:    [time, channels]
position embedding: [time, channels]
sum:                [time, channels]
```

同じ token でも位置が違えば初期 hidden state が変わります。learned absolute position embedding は
`maximumContextLength` 行だけ持つため、それより長い sequence を拒否します。後で扱う RoPE は
query/key の各 head 内へ相対的な位置関係を回転として入れます。

## linear layer は最後の軸を変換する

token ごとに同じ affine transform を適用します。

\[
\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{W}+\boldsymbol{b}
\]

```text
X: [time, inputChannels]
W: [inputChannels, outputChannels]
b: [outputChannels]
Y: [time, outputChannels]
```

`bias` を全 row に加える `addRowVector` の backward では、全 time row の gradient を bias へ合計
します。この axis reduction を暗黙 broadcasting に隠さず、名前付き演算にしました。

Attention の query/key/value projection、head 結合後の projection、feed-forward network、logit head
はすべてこの linear 変換です。

## normalization が必要な理由

residual connection で層を重ねると、hidden activation の scale が変動します。極端な値は softmax
飽和、gradient 消失・爆発、浮動小数点問題を起こします。normalization は各 token row の scale を
揃えます。

RMSNorm は平均を引かず、root mean square で割り、学習可能な scale \(g_i\) を掛けます。

\[
\operatorname{RMS}(\boldsymbol{x})
=\sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2+\epsilon}
\]

\[
y_i=g_i\frac{x_i}{\operatorname{RMS}(\boldsymbol{x})}
\]

shape は `[time,channels] -> [time,channels]` で、time row ごとに独立です。\(\epsilon\) は zero vector
での 0 除算を防ぎます。

## RMSNorm backward

\(r=\operatorname{RMS}(x)\)、上流から scale 適用後の gradient を \(g_i\) と書くと、入力 gradient
は direct path と、\(r\) が全要素に依存する path の和です。

\[
\frac{\partial L}{\partial x_i}
=\frac{g_i}{r}
-\frac{x_i}{Cr^3}\sum_j g_jx_j
\]

scale gradient は各 row から合計します。

\[
\frac{\partial L}{\partial \gamma_i}
=\sum_{row}\frac{\partial L}{\partial y_{row,i}}
\frac{x_{row,i}}{r_{row}}
\]

test は input gradient を数値微分と比較し、scale gradient の有限性も確認します。

## parameter count

vocabulary \(V\)、maximum context \(T\)、channels \(C\) なら、

- token embedding: \(VC\)
- position embedding: \(TC\)
- linear: \(C_{in}C_{out}+C_{out}\)
- RMSNorm: \(C\)

token embedding は vocabulary が大きい model で大きな割合を占めます。出力 logit weight と同じ
matrix を転置して共有する weight tying により parameter を減らせます。

## 演習

1. `V=50,000`、`C=4,096` の token embedding parameter 数と fp16 bytes を計算してください。
2. 同じ token を三位置で lookup し、weight row gradient が三経路の和になる test を書いてください。
3. position embedding を外し、sequence を入れ替えた出力の関係を観察してください。
4. `addRowVector` の bias gradient が time 軸 sum になる式を導出してください。
5. LayerNorm と RMSNorm の forward 式を比較してください。
6. zero input に epsilon がない RMSNorm で何が起きるか説明してください。

## 完了条件

- token ID が ordinal 数値でなく category であることを説明できる
- embedding lookup の gather/scatter gradient を説明できる
- token と position embedding の shape を書ける
- linear が最後の channel 軸をどう変えるか説明できる
- RMSNorm の目的、forward、epsilon、learned scale を説明できる
- `LayersSuite` と追加 Tensor test が成功する
