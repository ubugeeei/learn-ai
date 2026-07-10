# 14 — Unicode、UTF-8、byte tokenizer

## この章で作るもの

Scala `String` を UTF-8 bytes の token ID 列へ変換し、壊れた byte 列を黙って置換せず検出する
tokenizer を作ります。対象コードは `src/main/scala/learnai/text/Utf8.scala` です。

## model は文字列を直接扱わない

neural network の入力と出力は数です。tokenizer は文字列と token ID 列の境界です。

```text
text --encode--> token IDs --model--> token IDs --decode--> text
```

tokenizer は model の外付け前処理ではありません。vocabulary size、sequence length、学習効率、
多言語性能、生成可能な文字列、model parameter 数に影響します。

## character とは何か

見た目の一文字を数えるだけでも複数の層があります。

- **grapheme cluster**: 人が一文字と認識するまとまり
- **Unicode code point**: `U+3042` のように割り当てられた番号
- **UTF-16 code unit**: JVM `String` 内部/API で現れる 16 bit 単位
- **UTF-8 byte**: file や network で広く使われる 8 bit 単位

emoji の一つの見た目が複数 code point からできる場合や、同じ見た目の文字列が異なる code point
列を持つ場合もあります。Scala の `String.length` をそのまま「文字数」と解釈できません。

## Unicode と encoding

Unicode は文字へ code point を割り当てます。UTF-8 は code point 列を bytes に変換する encoding
です。

UTF-8 は一 code point を 1〜4 bytes で表します。

| text | code point | UTF-8 bytes | byte tokens |
| --- | --- | --- | --- |
| `A` | `U+0041` | `41` | 1 |
| `あ` | `U+3042` | `E3 81 82` | 3 |
| `🚀` | `U+1F680` | `F0 9F 9A 80` | 4 |

`byte & 0xff` は JVM の signed `Byte` `[-128,127]` を、token ID に使いやすい unsigned 値
`[0,255]` へ変換します。

## byte tokenizer の利点

各 byte を 0〜255 の token ID にすれば、UTF-8 で表せる任意の text を未知 token なしで encode
できます。

- vocabulary が固定 256 と小さい
- unseen language や記号も必ず表現できる
- encoding/decoding の規則が単純で reversible

一方、日本語や emoji は一つの見た目に複数 token を使い、sequence が長くなります。Attention の
計算量は sequence length の二乗になるため、長さは大きな cost です。次章の BPE が頻出 byte 列を
一 token へまとめます。

## special token

通常の bytes とは別に、sequence の境界や protocol を表す token が必要です。

```text
0..255: byte values
256:    beginning of text
257:    end of text
```

special token は通常の text を encode して偶然現れてはいけません。ID 空間を分けます。実用 model
では role、tool call、padding などの token もあります。追加時には embedding/output layer の
vocabulary size と model checkpoint を一致させます。

## strict decoding

任意の byte 列が正しい UTF-8 とは限りません。たとえば `0x80` は先頭 byte なしの continuation
byte です。標準 decoder の既定動作で replacement character `�` に置き換えると、model が壊れた
列を生成した事実を失います。

この教材は `CodingErrorAction.REPORT` を指定し、`Either[String, String]` で失敗を返します。
生成系では次の選択を明示します。

- strict に失敗させる
- 不完全な末尾 byte を次 token まで buffer する
- UI 表示時だけ replacement を使い、raw token は log に残す

## round-trip invariant

正しい text に対して最重要 property は次です。

\[
\operatorname{decode}(\operatorname{encode}(text))=text
\]

ASCII、日本語、emoji、結合文字、改行、空文字で property をテストします。normalization を行う
tokenizer では完全一致しないため、どの normalization を仕様とするか明記します。今回の byte
tokenizer は normalization せず、元 bytes を保ちます。

## opaque type

`TokenId` は実行時には `Int` ですが、API 上は別の型です。

```scala
opaque type TokenId = Int
```

length、byte offset、class label など別の整数を誤って渡しにくくし、負の ID を constructor で拒否
します。必要な場所だけ `.value` で underlying `Int` を取り出します。

## 演習

1. 自分の名前を encode し、各 byte を 16 進数で表示してください。
2. 空文字、改行、NUL `\u0000` の round-trip test を追加してください。
3. `String.length`、code point 数、UTF-8 byte 数を複数の emoji で比較してください。
4. 不完全な三 byte 文字の先頭一 byte、二 bytes を decode し、error を観察してください。
5. special token を skip せず、構造を保持した decode result の型を設計してください。

## 完了条件

- grapheme、code point、UTF-8 byte、token を区別できる
- byte tokenizer に未知 token がない理由を説明できる
- 日本語で sequence が ASCII より長くなりやすい理由を説明できる
- strict decode が障害解析に有用な理由を説明できる
- round-trip invariant を複数言語で確認できる
- `Utf8Suite` が成功する
