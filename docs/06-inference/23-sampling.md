# 23 — temperature、top-k、top-p sampling

## この章で作るもの

model logits から temperature、top-k、nucleus top-p の順に候補を絞り、再正規化して seed 付き乱数で
sample する独立 policy を実装します。対象コードは
`src/main/scala/learnai/lm/Sampling.scala` です。

## decoding は model と別の意思決定

model は次 token の logits を出します。どの token を実際に選ぶかは decoding policy です。同じ
model/prompt でも policy により、決定性、多様性、破綻率、反復が変わります。

```text
logits
 -> temperature
 -> softmax
 -> top-k filter
 -> renormalize
 -> top-p filter
 -> renormalize
 -> sample
```

policy と seed は生成結果の再現条件なので、評価 log に残します。

## greedy decoding

最大 logit の token を常に選びます。

\[
x_{t+1}=\operatorname*{arg\,max}_i z_i
\]

再現可能で単純ですが、局所的な最大を選び続け、反復や不自然な高確率 path に入り得ます。beam
search は複数 sequence 仮説を保ちますが、open-ended generation では確率 sampling がよく使われ
ます。

top-k を 1 にすると greedy と同じ分布になります。

## temperature

\[
p_i=\operatorname{softmax}(z_i/\tau),\quad \tau>0
\]

低 temperature は logit 差を広げ entropy を下げます。高 temperature は分布を平らにし rare token
を選びやすくします。temperature 0 は除算できないため、greedy は明示的な argmax/top-k 1 として
表します。

## top-k

確率上位 \(k\) token だけを残し、他を 0 にして再正規化します。

\[
\tilde{p}_i=\begin{cases}
p_i/Z & i\in K\\
0 & \text{otherwise}
\end{cases}
\]

固定個数なので実装・latency が予測しやすい一方、分布が非常に確信している場合も曖昧な場合も同じ
候補数です。`k` が vocabulary size より大きければ全 token を残します。

同率 token の順序を collection iteration に任せると非決定的になり得ます。この実装は確率降順、
同率なら token ID 昇順にします。

## nucleus top-p

確率降順に並べ、累積確率が threshold \(p\) 以上になる最小集合を残します。

```text
probabilities: [0.60, 0.25, 0.10, 0.05]
top-p = 0.80: keep [0.60, 0.25], cumulative 0.85
```

model が確信していれば候補は少なく、分布が平らなら多くなります。最低一 token は必ず残します。

## filter の順序

この実装は temperature → top-k → top-p です。top-k 後に再正規化し、その条件付き分布へ top-p を
適用します。順序を変えると結果は変わるため、policy contract に含めます。

実際の API/serving engine と比較するときは、以下も確認します。

- top-p が top-k 前か後か
- boundary token を含むか
- logits bias/penalty の適用順
- repetition/frequency/presence penalty
- minimum tokens to keep
- random number generator と seed semantics

## evaluation での注意

model 自体の next-token 能力は、sampling された一回答だけでなく log-likelihood/perplexity でも測り
ます。生成 task は複数 seed で distribution として評価します。

- exact task: greedy/temperature low が適する場合
- creative task: moderate temperature/top-p が有効な場合
- safety-critical action: sampling だけでなく schema validation と policy gate が必要

「最良の temperature」は全 task 共通の定数ではありません。

## 演習

1. logits `[0,1,2]` を temperature `0.5,1,2` で確率と entropy にしてください。
2. top-k と top-p の順を逆にし、異なる結果になる例を作ってください。
3. 10,000 samples の frequency と理論確率を比較してください。
4. repetition penalty を logits へ適用する policy を設計し、順序を明記してください。
5. greedy、top-k、top-p を複数 prompt/seed で生成し、重複率と unique token 数を測ってください。

## 完了条件

- model logits と decoding policy を分離して説明できる
- temperature が entropy に与える効果を説明できる
- top-k と top-p の候補選択の違いを例で示せる
- filtering 後に再正規化が必要な理由を説明できる
- policy 順序と seed を再現条件として記録できる
- `SamplingSuite` が成功する
