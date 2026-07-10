# 16 — causal dataset、split、batch

## この章で作るもの

token 列から「ここまでを入力として、各位置の次 token を当てる」固定長 example を作り、境界を
またがない train/validation split と再現可能な batch sampling を実装します。対象コードは
`src/main/scala/learnai/data/CausalDataset.scala` です。

## causal language modeling

causal language model は未来 token を見ず、現在までの token から次を予測します。token 列

```text
[t0, t1, t2, t3, t4]
```

context length 3 の最初の example は次です。

```text
input:  [t0, t1, t2]
target: [t1, t2, t3]
```

各位置を揃えると、`input(i)` を処理した出力で `target(i)` を予測します。一 window から context
length 個の next-token training pair が得られます。

二つ目の sliding window は次です。

```text
input:  [t1, t2, t3]
target: [t2, t3, t4]
```

長さ \(N\)、context \(T\) の一 token 列から、full window は \(\max(0,N-T)\) 個できます。

## shape

一 example の shape:

```text
inputs:  [time]
targets: [time]
```

batch size \(B\) で積むと次です。

```text
inputs:  [batch, time]
targets: [batch, time]
```

`CausalBatch` は全 example の context length が同じことを constructor で検査します。不揃いな長さを
padding する方式は後で、padding mask と組にして導入します。

## split は window 化より先に行う

先に全 corpus を window 化してから example をランダムに split すると、隣接 window はほとんどの
token を共有します。

```text
train:      [t0,t1,t2,t3]
validation:    [t1,t2,t3,t4]
```

validation の内容が training にほぼ含まれ、generalization を過大評価します。この教材は raw token
列を一つの境界で先に分け、それぞれで独立に window を作ります。

```text
tokens -> [ training region | validation region ]
             -> windows       -> windows
```

境界付近で `contextLength` 分弱の example を失いますが、region をまたぐ leakage を防ぎます。複数
document の実データでは document ID や source 単位で split し、重複除去も行います。

## batch と stochastic gradient

全 training example の loss を毎 step 計算すると高価です。一部 \(B\) 個を選び、全体 gradient の
推定値として使います。

\[
\hat{L}=\frac{1}{B}\sum_{i\in\text{batch}}L_i
\]

- 小 batch: memory が少ない、gradient noise が大きい
- 大 batch: 安定した推定、memory と一 step の計算が大きい

batch size を変えると gradient variance、learning rate、training dynamics が変わります。単純に
速度だけの parameter ではありません。

## with-replacement sampling

`sampleBatch` は各要素を dataset 全体から独立に選ぶため、一 batch 内で同じ example が複数回出る
ことがあります。これを sampling with replacement と呼びます。

epoch ごとに shuffle して重複なしで一巡する方式も一般的です。streaming data や巨大 corpus では
epoch の定義自体が曖昧な場合があります。どの方式でも、random seed と sampler state を checkpoint
しないと途中再開後の data 順序は一致しません。

## deterministic evaluation

validation は sampling noise を避けるため `sequentialBatches` で一定順序に評価します。最後の短い
batch を保持するか `dropLast` するかを明示します。

- training: random sampling、parameter update あり
- validation: deterministic 順序、update なし
- generation/eval: model を inference mode にする

dropout や normalization に training/evaluation mode がある model では、mode の切り替えも必要です。

## data pipeline の不変条件

test で次を確認します。

- target は input のちょうど一 token 先
- partial window を example にしない
- split boundary をどの window もまたがない
- 同じ seed は同じ batch index 列を作る
- empty dataset の sampling は明示的に失敗する
- batch shape は一定

data bug は model が「学習しない」ように見える原因になり、loss が下がっても評価を汚染することが
あります。model code と同じ強さで test します。

## 演習

1. token 長 10、context 4 からできる example 数と最初・最後の window を書いてください。
2. epoch-based shuffle sampler を実装し、各 example が一度ずつ現れる test を書いてください。
3. 複数 document をまたぐ window を作らない `fromDocuments` を設計してください。
4. batch 内 token 数を一定にする dynamic batching を設計してください。
5. split 後の token/example 数を表示し、境界で失う example 数を説明してください。
6. random split による隣接 window leakage の割合を小 corpus で測ってください。

## 完了条件

- input/target を一 token shift する理由を説明できる
- `[batch,time]` の各軸を説明できる
- split を window 化より先に行う理由を leakage の例で示せる
- batch size と gradient noise の関係を説明できる
- training sampler と validation iterator の違いを説明できる
- `CausalDatasetSuite` が成功する
