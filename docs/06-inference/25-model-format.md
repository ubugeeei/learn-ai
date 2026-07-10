# 25 — versioned model format と安全な checkpoint

## この章で作るもの

MiniGPT config、parameter label、shape、row-major `Double` values を versioned binary format に保存し、
SHA-256 検証後に新しい model へ復元します。保存は temporary file から atomic replace します。

対象コードは `src/main/scala/learnai/io/MiniGptCheckpoint.scala` です。

## checkpoint は単なる数値配列ではない

weight values だけでは各配列の意味を復元できません。最低限、次が必要です。

- format magic と version
- architecture config
- parameter の安定した名前
- rank、shape、element count
- dtype と byte order
- values
- integrity checksum

tokenizer vocabulary/merge table が違えば token ID の意味が変わるため、実際の配布 bundle では
tokenizer artifact と checksum も必要です。この章の inference checkpoint は tokenizer を意図的に
含めず、その限界を format contract に明記します。

## file layout

```text
magic "LAIGPT01"                 8 bytes
format version                   Int32
MiniGptConfig                    fixed fields
parameter tensor count          Int32
repeat parameter count:
  UTF-8 label length + bytes
  rank + dimensions
  scalar count
  Float64 values (big-endian)
SHA-256(payload)                 32 bytes
```

JDK `DataOutputStream` の integer/double は big-endian です。format は JVM memory layout や Java object
serialization に依存しません。

## parameter label と順序

`blocks.0.attention.query.weight` のような label は、model 内の意味を表す stable key です。loader は
新しい model を config から構築し、期待する順序・label・shape・count のすべてを照合します。

shape だけが同じ別 parameter を誤って入れ替える事故を label で検出します。save 前に label の重複も
拒否します。

## integrity と authenticity

SHA-256 は accidental corruption や不完全 upload を検出します。一 byte 変われば checksum mismatch
になります。ただし checksum は誰でも再計算できるため、攻撃者による改ざんを防ぐ署名ではあり
ません。

- integrity: SHA-256 checksum
- authenticity: trusted key による digital signature、配布 channel、provenance

実運用では artifact registry の署名、publisher identity、reviewed conversion pipeline を組み合わせ
ます。

## verify before parse

loader は file 全体の上限を確認し、payload checksum を検証してから config/shape/value を解釈します。
さらに次の上限を持ちます。

- file bytes
- config から推定した scalar parameter 数
- label bytes
- rank
- expected tensor count/shape/element count

untrusted length をそのまま array allocation に使うと memory exhaustion になります。期待 model shape と
照合してから values を読みます。

## all-or-nothing parameter assignment

`Tensor.assignParameterValues` は count と全値の finite check を先に済ませ、その後に data を変更
します。途中で `NaN` を見つけて半分だけ load された parameter を残しません。

loader は新しく構築した model だけを変更するため、失敗時に利用中 model が壊れることもありません。
service では新 model を完全検証し、smoke inference 後に pointer を切り替えます。

## atomic save

target file へ直接書くと、process crash や disk full で既存の正常 checkpoint まで壊れます。

1. 同じ directory に temporary file を作る
2. 完全な payload + checksum を書く
3. atomic move で target を置き換える
4. platform が atomic move 非対応なら replace move へ fallback

同じ filesystem 内の rename を使うことが重要です。より厳密な durability には file/directory `fsync`
も必要です。

## inference checkpoint と training checkpoint

この format は inference 再現に必要な model weight/config を持ちます。training を完全に再開するには
さらに次が必要です。

- AdamW first/second moments と step
- learning-rate scheduler state
- gradient scaler state
- RNG states
- data sampler position/shuffle state
- tokenizer/data revision
- code/build revision

weight だけから再開すると optimizer warmup が失われ、同じ training trajectory になりません。

## pickle/Java serialization を避ける理由

一般 object serialization は deserialization 時に任意 class constructor/hooks を実行し得て、untrusted
model file の code execution risk になります。また class 名や実装詳細へ強く結合します。

この format は primitive field を手続き的に読み、許可した model structure だけを構築します。
production では safetensors のように data と executable code を分離した format が使われます。

## test coverage

- training 後 model の config/label/value/logit bit-exact round-trip
- one-byte corruption の checksum rejection
- truncated file の rejection
- existing file の atomic replacement
- invalid assignment が元 parameter を変更しない all-or-nothing property

file I/O の成功例だけでなく、壊れた artifact を意図的に入力します。

## 演習

1. checkpoint を hex viewer で開き、magic と version を確認してください。
2. config field 順を変える format version 2 migration を設計してください。
3. tokenizer merge table と checksum を bundle manifest に追加してください。
4. optimizer state を別 section として追加し、inference-only load が skip できる設計にしてください。
5. Ed25519 signature と public-key verification を加える位置を設計してください。
6. save 後に load + fixed prompt logits を確認してから publish する workflow を書いてください。

## 完了条件

- checkpoint に config/label/shape が必要な理由を説明できる
- checksum と signature の保証範囲を区別できる
- untrusted length を検証前に allocate する危険を説明できる
- atomic replace が既存 checkpoint を守る理由を説明できる
- inference と resumable training checkpoint の state 差を列挙できる
- `MiniGptCheckpointSuite` が成功する
