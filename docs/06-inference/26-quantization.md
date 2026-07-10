# 26 — symmetric int8 weight quantization

## この章で作るもの

`Double` matrix の各 row を signed int8 と一つの scale に量子化し、dequantize せず matrix-vector
積を計算し、maximum/mean absolute error と RMSE を測ります。

対象コードは `src/main/scala/learnai/quantization/Int8Quantization.scala` です。

## quantization の目的

model weight を低 bit 表現にすると、主に次を減らせます。

- storage と download size
- memory capacity
- memory bandwidth
- 対応 hardware での演算 cost

ただし丸め誤差が入り、quantize/dequantize scale の計算や kernel support が必要です。file size が 4 倍
小さくても latency が必ず 4 倍速くなるわけではありません。bottleneck を測ります。

## symmetric int8

一 row の最大絶対値を \(a=\max_i|w_i|\) とします。int8 code は対称な `[-127,127]` を使います。

\[
s=\frac{a}{127}
\]

\[
q_i=\operatorname{clamp}\left(\operatorname{round}\left(\frac{w_i}{s}\right),-127,127\right)
\]

復元は次です。

\[
\hat{w}_i=sq_i
\]

`-128` を使わず、正負の最大 magnitude を同じにします。zero point は 0 なので symmetric quantization
です。

all-zero row は \(a=0\) で scale 0 になりますが、全 code が 0 なら scale は結果に影響しません。
representation invariant を単純に保つため `s=1` とします。

## per-tensor と per-row scale

全 matrix に一 scale を使う per-tensor quantization は metadata が少ない一方、小さな row が大きな
outlier row の scale に合わせられ、resolution を失います。

per-row は各 output channel に scale を持ちます。

```text
weights [outputChannels,inputChannels]
scales  [outputChannels]
```

linear layer で output row ごとに dynamic range が違う場合の誤差を減らします。scale metadata は
row あたり 8 bytes（教材では `Double`）増えます。production は fp16/fp32 scale、group-wise scale
などを使います。

## rounding error bound

clipping が起きず、通常の nearest rounding なら code の誤差は 0.5 以下なので、復元 absolute error
は概ね scale の半分以下です。

\[
|w_i-\hat{w}_i|\le\frac{s}{2}
\]

最大値から scale を作るため元 weight は範囲内で、通常 clipping は不要ですが、浮動小数点 rounding
に備えて clamp します。

## quantized matvec

\[
y_r=\sum_c w_{rc}x_c
\approx s_r\sum_c q_{rc}x_c
\]

row scale は sum の外へ出せます。`QuantizedInt8Matrix.matvec` は full `Double` weight matrix を作らず、
int8 code を読みながら accumulation し最後に scale を掛けます。

この教材は input と accumulator を `Double` にして weight quantization の誤差だけを観察します。
実用 kernel は int8 activation/int32 accumulator、weight-only int4/fp16 compute など hardware に合う方式を
選びます。

## error metrics

要素単位の reconstruction error:

- maximum absolute: 最悪の一要素
- mean absolute: 平均的なずれ
- RMSE: 大きな error を二乗で強く評価

しかし weight error が小さくても model quality が保たれる保証はありません。層ごとの activation
distribution、最終 logits、perplexity、task eval を quantized/non-quantized で比較します。

特に outlier channel、Attention softmax 前の score、rare token logits は小さな数値差に敏感な場合が
あります。

## memory estimate

`R x C` matrix の payload は、

- Double: \(8RC\) bytes
- per-row int8: \(RC+8R\) bytes

row が十分に広ければ約 8 分の 1 です。実際には object/header/alignment、packing、scale dtype、kernel
workspace も含めて測ります。

## PTQ と QAT

この章は学習後 weight を変換する post-training quantization（PTQ）です。

- PTQ: 速く、元 training をやり直さない。calibration data が必要な方式もある
- QAT: training 中に fake quantization error を入れ、model が誤差へ適応する

低 bit ほど QAT や fine-tuning が必要になりやすいですが、model/data/architecture に依存します。

## 実行

```console
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

payload bytes と実測 reconstruction error を表示します。performance claim には別途 warmup と反復を
含む benchmark が必要です。

## 演習

1. row `[-2,-1,0,1,2]` の scale、codes、復元値を手計算してください。
2. per-tensor quantization を実装し、magnitude が大きく違う二 row で error を比較してください。
3. scale を `Float` に変えた payload bytes と誤差を測ってください。
4. random weight matrix の matvec output cosine similarity を測ってください。
5. MiniGPT の各 2D weight を量子化し、固定 prompt logits の最大誤差を測ってください。
6. int4 packed representation に必要な bit 操作と scale group を設計してください。

## 完了条件

- scale、quantized code、dequantized value を手計算できる
- symmetric quantization が zero point 0 になる理由を説明できる
- per-row と per-tensor の metadata/error trade-off を説明できる
- weight error と end-task quality の両方を測る必要を説明できる
- storage reduction と latency improvement を区別できる
- `Int8QuantizationSuite` が成功する
