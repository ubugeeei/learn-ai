# 15 — byte pair encoding（BPE）

## この章で作るもの

corpus 中で最も頻出する隣接 token pair を繰り返し結合し、vocabulary を学習する byte-level BPE
trainer と tokenizer を実装します。対象コードは
`src/main/scala/learnai/text/BpeTokenizer.scala` です。

## vocabulary size と sequence length の trade-off

byte tokenizer の vocabulary は 256 と小さい一方、よく出る文字列も毎回複数 token になります。
極端に、文書全体を一 token にすれば sequence は短いですが、あらゆる文書の token を用意できず、
未知入力へ一般化しません。

tokenizer は次の trade-off を選びます。

- 小さい vocabulary: embedding/output parameter が少ない、sequence は長い
- 大きい vocabulary: sequence は短い、parameter と sparse な token が増える

BPE は必ず表現できる byte token から始め、corpus で繰り返す列だけを追加 token にします。

## training algorithm

初期状態では各文書を UTF-8 byte ID 列にします。

```text
"banana" -> [98, 97, 110, 97, 110, 97]
```

次を target vocabulary size まで繰り返します。

1. 各 sequence の隣接 pair を数える
2. 最頻 pair を選ぶ
3. 新しい token ID を割り当てる
4. 全 sequence でその pair を非重複に置換する

仮に `(97,110)`、つまり byte 列 `an` が最頻なら token `256` にします。

```text
[98, 97, 110, 97, 110, 97]
 -> [98, 256, 256, 97]
```

次の merge は既存の merged token を参照できます。たとえば `(98,256) -> 257` なら `ban` を表し
ます。

## merge table は model artifact の一部

`BpeMerge(left, right, result)` の順序が encode 規則です。

```text
merge 0: (97, 110) -> 256
merge 1: (98, 256) -> 257
```

この実装は merge \(i\) の result ID を `256 + i` に固定し、left/right は byte または以前の merge
だけを参照できる DAG にします。そのため各 token を再帰的に元 bytes へ展開できます。

model weight が同じでも merge table が違えば token ID の意味が変わり、model は壊れます。
checkpoint には必ず tokenizer type、merge table、special tokens、normalization rule、version を含め
ます。

## encode

新しい text を bytes にした後、学習順に merge rule を適用します。

```text
UTF-8 bytes
 -> replace merge 0 everywhere
 -> replace merge 1 everywhere
 -> ...
 -> token IDs
```

実用 tokenizer は pair の priority queue や merge rank lookup で高速化します。この教材は規則を
追いやすいよう、merge table と sequence を順に走査します。

## decode

各 byte token `0..255` の展開は自分自身です。merge token の展開は left と right の bytes を連結
します。

```text
bytes(256) = bytes(97) ++ bytes(110)
           = [97, 110]
```

merge table が topological order なので、result token の展開を作る時点で子の展開は完成しています。
全 token を bytes へ戻した後、strict UTF-8 decoder を使います。

## tie-breaking と再現性

同じ頻度の pair が複数あると、iteration 順序に任せた実装は実行ごとに違う vocabulary を作り得ます。
この trainer は次の順で一意に選びます。

1. count が多い
2. left ID が小さい
3. right ID が小さい

tokenizer training の再現性は model training と同じくらい重要です。corpus の順序・内容、target
size、前処理も記録します。

## Unicode との関係

byte-level BPE の merge は code point 境界を知りません。UTF-8 の一文字の途中 bytes や、複数文字
をまとめる場合があります。最終的な全 token 列を連結すれば元 bytes へ戻るため round-trip は保て
ますが、単独 token の表示が常に有効な text とは限りません。

token 単位で UI に逐次表示するときは、UTF-8 decoder state を持ち、不完全な末尾 bytes を次 token
まで buffer します。

## 計算量と production 実装

教材実装は各 merge step で全 corpus を数え直し、置換します。corpus token 数 \(N\)、merge 数
\(M\) に対し概ね \(O(MN)\) です。巨大 corpus では pair occurrence の索引、incremental count、
parallel counting、external memory が必要です。

また production tokenizer では次も設計します。

- Unicode normalization
- pre-tokenization（空白や語境界）
- special token の衝突防止
- byte fallback
- encode/decode の streaming
- artifact の checksum と互換性

## 演習

1. `banana` の pair count を手で数え、最初の merge を予想してください。
2. target vocabulary size を変え、corpus の平均 token 数を表にしてください。
3. 日本語、英語、code の三 corpus で圧縮率を比較してください。
4. tie-break を外した場合に非決定性が生じる data structure を調べてください。
5. merge table の serialize/deserialize 形式を設計し、round-trip test を書いてください。
6. 一 token ごとの bytes を表示し、UTF-8 として単独 decode できない token を探してください。

## 完了条件

- vocabulary size と sequence length の trade-off を説明できる
- BPE training の count、select、merge を例で実行できる
- merge table の順序が encode と decode に必要な理由を説明できる
- byte-level BPE が Unicode 文字境界と一致しないことを説明できる
- tokenizer artifact と model weight を組で管理すべき理由を説明できる
- `BpeTokenizerSuite` が成功する
