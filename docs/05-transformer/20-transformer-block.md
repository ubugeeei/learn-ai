# 20 — Transformer block

## この章で作るもの

RMSNorm、causal self-attention、residual connection、position-wise feed-forward network を組み合わせ、
shape `[time,channels]` を保つ pre-norm Transformer block を実装します。対象コードは
`src/main/scala/learnai/transformer/TransformerBlock.scala` です。

## block の全体像

入力 \(\boldsymbol{X}\) に対して次の二段階を計算します。

\[
\boldsymbol{H}
=\boldsymbol{X}+\operatorname{Attention}(\operatorname{RMSNorm}(\boldsymbol{X}))
\]

\[
\boldsymbol{Y}
=\boldsymbol{H}+\operatorname{FFN}(\operatorname{RMSNorm}(\boldsymbol{H}))
\]

```text
X [T,C]
 |-------------------------------+
 -> RMSNorm -> causal attention -> + = H [T,C]
                                      |----------------------+
                                      -> RMSNorm -> FFN ------+ = Y [T,C]
```

block 入出力の shape を同じにするため、同じ構造を何層でも積めます。

## residual connection

sublayer \(F\) の出力だけで置き換えず、入力を足します。

\[
\boldsymbol{y}=\boldsymbol{x}+F(\boldsymbol{x})
\]

backward は次です。

\[
\frac{\partial L}{\partial \boldsymbol{x}}
=\frac{\partial L}{\partial \boldsymbol{y}}
\left(\boldsymbol{I}+\frac{\partial F}{\partial \boldsymbol{x}}\right)
\]

\(F\) の経路に加え、identity path を通して gradient が直接前層へ戻れます。深い network の最適化を
助け、sublayer は完全な新表現でなく入力への修正量を学べます。

加算する二 Tensor の shape は完全一致が必要です。attention と FFN の最終 projection が channels
\(C\) へ戻る理由です。

## pre-norm と post-norm

この実装は sublayer の前に normalize する pre-norm です。

```text
pre-norm:  x + Sublayer(Norm(x))
post-norm: Norm(x + Sublayer(x))
```

pre-norm は residual identity path が normalization を通らず、深い model で gradient flow を安定させ
やすい構成です。一方、表現 scale や最終 normalization の扱いが変わります。architecture の選択は
training curve、gradient norm、最終品質で評価します。

## position-wise feed-forward network

Attention は位置間で情報を混ぜます。FFN は各位置を独立に、channel 方向へ同じ nonlinear function
で変換します。

\[
\operatorname{FFN}(\boldsymbol{x})
=\boldsymbol{W}_2\operatorname{ReLU}(\boldsymbol{W}_1\boldsymbol{x}+\boldsymbol{b}_1)
+\boldsymbol{b}_2
\]

```text
[T,C] -> Linear -> [T,F] -> ReLU -> Linear -> [T,C]
```

hidden width \(F\) は通常 \(C\) より大きく、各 token 内で使える nonlinear capacity を増やします。
Attention が「どの token から読むか」、FFN が「集めた情報をどう変換するか」を担当すると見ることが
できます。

この章は原理が単純な ReLU を使います。現代的 model で使われる SwiGLU は後の章で、二 projection
の gated product として実装します。

## parameter count

channels \(C\)、FFN hidden \(F\) の一 block は、bias 込みで概ね次です。

- Q/K/V/output projection: \(4(C^2+C)\)
- two RMSNorm scales: \(2C\)
- FFN expansion: \(CF+F\)
- FFN projection: \(FC+C\)

合計:

\[
4C^2+2CF+7C+F
\]

大きな \(C,F\) では bias/norm の一次項より matrix weight の二次項が支配します。通常 \(F\approx4C\)
なら FFN weight は約 \(8C^2\)、attention projection は約 \(4C^2\) です。

## depth

block を \(L\) 層重ねると、一層目で集めた文脈表現を、次の層が再び query/key/value に変えて関係を
構築できます。単に Attention を一回広く見るだけでなく、複数段の計算を行えます。

ただし depth は次を増やします。

- parameter と FLOPs
- training activation memory
- sequential な layer latency
- optimization の難しさ

residual、normalization、initialization、learning-rate schedule が深い model を学習可能にします。

## dropout と regularization

教材 block は決定性を保ち backward を追いやすくするため dropout をまだ持ちません。dropout は
training 時に activation の一部を確率的に 0 にし、残りを scale します。

- training: mask を seed 付き RNG で生成し backward でも同じ mask を使う
- evaluation: dropout を無効にする

mode、RNG state、mask の graph 保存が必要です。大規模 pretraining では data/model size により
dropout を使わない構成もあります。常に必要な定石ではなく、validation gap を測って決めます。

## test する性質

- 入出力 shape が同じ
- parameter 数が構成式と一致
- parameter reference が重複していない
- scalar loss から input/parameter gradient が finite
- future token を変えても prefix output が同じ
- invalid hidden width を構築時に拒否

Attention 単体で causal でも、block 全体の別経路が time を混ぜれば causality は壊れます。FFN と
RMSNorm が row 単位であることを含め、block 全体の prefix property を再テストします。

## 演習

1. `C=768`、`F=3072` の一 block parameter 数を計算してください。
2. residual を外した block を深く積み、input gradient norm を比較してください。
3. pre-norm と post-norm を実装し、同じ seed で training curve を比較してください。
4. ReLU を tanh に変え、activation の zero 比率と gradient を測ってください。
5. FFN の各 time row が他 row に依存しない test を書いてください。
6. dropout protocol に必要な training mode と RNG state の型を設計してください。

## 完了条件

- block の二つの pre-norm residual 式を書ける
- residual identity path が backward を助ける理由を説明できる
- Attention と FFN の役割を区別できる
- FFN の expansion/nonlinearity/projection shape を追える
- block parameter 数を計算できる
- block 全体でも causal property を確認すべき理由を説明できる
- `TransformerBlockSuite` が成功する
