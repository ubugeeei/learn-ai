# 21 — MiniGPT を学習・生成する

## この章で作るもの

token/position embedding、複数 Transformer block、final RMSNorm、tied output projection、cross
entropy を一つの decoder-only language model に組み、実際に end-to-end で学習・生成します。

対象コードは `src/main/scala/learnai/transformer/MiniGpt.scala` です。

## architecture

token IDs \(x_0,\ldots,x_{T-1}\) から logits を得る全経路です。

```text
token IDs [T]
  -> token embedding + position embedding [T,C]
  -> Transformer block x L             [T,C]
  -> final RMSNorm                      [T,C]
  -> tied token embedding transpose     [T,V]
  -> cross entropy with targets [T]     scalar
```

`MiniGptConfig` の全値が parameter/activation shape を決めます。

| config | 意味 |
| --- | --- |
| `vocabularySize` \(V\) | tokenizer の token 種類数 |
| `maximumContextLength` \(T_{max}\) | learned position table と最大入力長 |
| `channels` \(C\) | 各 token hidden vector の幅 |
| `headCount` \(H\) | Attention head 数、\(C\) を割り切る |
| `hiddenChannels` \(F\) | FFN expansion 幅 |
| `layerCount` \(L\) | Transformer block 数 |

config と tokenizer/checkpoint は互換性契約です。一つでも違えば weight shape または token ID の意味が
変わります。

## final normalization

各 block が pre-norm のため、最後の residual stream はまだ block 後の normalization を通っていません。
logit projection 前に RMSNorm を適用し scale を安定させます。

\[
\boldsymbol{H}_{final}=\operatorname{RMSNorm}(\boldsymbol{H}_L)
\]

final norm の有無も architecture の一部で、checkpoint 互換性に影響します。

## weight tying

通常は hidden vector から vocabulary logits へ weight \(\boldsymbol{W}_{out}\in\mathbb{R}^{C\times V}\)
を掛けます。MiniGPT は token embedding \(\boldsymbol{E}\in\mathbb{R}^{V\times C}\) の転置を使います。

\[
\boldsymbol{Z}=\boldsymbol{H}_{final}\boldsymbol{E}^\mathsf{T}
\]

これを weight tying と呼びます。

- output weight \(CV\) parameter を追加せずに済む
- 入力 token 表現と出力 class 表現を同じ空間で共有する
- embedding は lookup と logit projection の両経路から gradient を受ける

`transpose2D` の backward が output path の gradient を元 embedding へ戻し、`gatherRows` の gradient
と加算します。`parameters` には embedding Tensor を一度だけ含め、optimizer の二重 update を防ぎ
ます。

## next-token loss

入力と target は一 token shift します。

```text
inputs:  [x0,x1,x2,x3]
targets: [x1,x2,x3,x4]
```

各 position logit は causal mask によりその位置以前だけに依存します。全 position の cross entropy
平均が一つの scalar loss になり、一回の `backward()` で embedding、全 block、final norm まで
gradient が流れます。

## end-to-end training step

```scala
val loss = model.loss(inputs, targets)
loss.backward()
optimizer.step(model.parameters)
```

短い三行の内部で、これまで実装したすべてが動きます。

1. UTF-8/BPE token IDs
2. embedding gather
3. position addition
4. RMSNorm
5. Q/K/V projection
6. causal attention
7. residual
8. FFN
9. tied vocabulary projection
10. stable cross entropy
11. Tensor reverse-mode
12. gradient clipping と AdamW

loss が下がるだけでは個別演算の正しさを保証できないため、各下位章の property/gradient test を残し
ます。end-to-end test は統合境界と「実際に学習可能」を確認します。

## autoregressive generation

prompt 全体を forward し、最後の row の logits だけを次 token 分布として使います。

```text
prompt -> logits for every position -> take final row -> sample token
  ^                                                     |
  +---------------- append sampled token ---------------+
```

`newTokenCount` 回 forward をやり直す単純実装です。長さ \(T\) の過去 Q/K/V も毎回再計算するため、
生成が進むほど無駄が増えます。後の KV cache 章で過去 key/value を保存し、新 token の一 row だけを
計算します。

## sliding context

生成列が `maximumContextLength` を超えると、左の古い token を捨てた window を使います。

```scala
val retained = context.takeRight(maximumContextLength)
```

この learned absolute position 実装は retained window の position を 0 から振り直します。training も
同じ window 規則で行う必要があります。長文を単純 crop すると、失われた情報へ model はアクセス
できません。

## generation quality と評価

demo は一つの短い sequence を繰り返し学習する overfit 実験です。これは次を確認します。

- graph がつながっている
- optimizer が parameter を更新する
- loss を下げられる capacity がある
- generation loop が動く

未知 text の品質、世界知識、指示追従を示しません。実用評価には train/validation split、大規模で
多様な data、複数 task の eval、contamination 検査、人間評価が必要です。

## 実行

```console
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
parameters:   ...
initial loss: ...
final loss:   ...
generated: ...
```

CPU と標準 library だけの教育 engine なので、小 config に限定します。model size を増やす前に profiler
で bottleneck と allocation を測ります。

## 現在の MiniGPT と frontier model の対応

同じ原理:

- autoregressive next-token objective
- embedding、multi-head causal attention、FFN、residual、normalization
- logits、cross entropy、AdamW
- temperature sampling

まだないもの:

- batch/GPU/distributed kernels
- mixed precision、loss scaling
- RoPE、GQA、SwiGLU、MoE
- dropout、learning-rate schedule
- checkpoint と resumable data/optimizer state
- KV cache、quantization、serving scheduler
- large-scale data curation と post-training

後続章は「新しい魔法」を追加するのではなく、この同じ計算を scale、stability、quality、serving の
制約へ合わせます。

## 演習

1. config から embedding、各 block、final norm の parameter 数を計算し、`parameterCount` と比べて
   ください。
2. weight tying を外した output Linear を作り、parameter 数と初期 loss を比較してください。
3. layer count `1,2,4` で loss、step time、gradient norm を記録してください。
4. training/validation windows を分け、overfitting curve を描いてください。
5. generation temperature を変え、entropy と重複率を測ってください。
6. prompt が context 上限を超えたとき、捨てられる token を表示してください。
7. 各 parameter label、shape、要素数を表として出力してください。

## 完了条件

- token IDs から logits まで全 shape を追える
- final norm と weight tying の役割を説明できる
- embedding に二経路から gradient が届くことを説明できる
- teacher-forced training と autoregressive generation をコードで指し示せる
- context crop の情報損失と position 再割当を説明できる
- 小さな model が loss を下げても実用品質を証明しない理由を説明できる
- `MiniGptSuite` を含む全テストが成功する
