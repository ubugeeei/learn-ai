# 設計原則

この文書は教材のコードと説明に共通する判断基準です。便利さのための抽象化より、観察可能性と
正しさを優先します。

## 1. 原理を隠す依存を置かない

本線の実装は Scala 3 と JDK の標準ライブラリだけを使います。BLAS、機械学習 framework、
tokenizer、JSON、HTTP client の外部ライブラリを最初から使いません。これにより、データ表現、
計算量、割り当て、誤差、失敗条件をコード上で追跡できます。

外部ライブラリを扱う場合は、原理を実装した後の比較対象として別 module に隔離します。

## 2. 不正な状態を型と constructor で拒否する

- ベクトル・行列の shape 不一致を即座に拒否する
- token ID、parameter、tool 名を可能な範囲で別の型にする
- 失敗し得る入出力を `Either` などで表し、例外を制御フローにしない
- mutable な状態は、学習 step や agent loop の境界に閉じ込める

数値計算内部では性能上の理由から配列を使いますが、所有権を外へ漏らしません。

## 3. 決定性を既定にする

乱数 seed、parameter 初期値、data 順序を明示します。同じ環境・同じ seed なら、テストと
教材の観察結果が再現されることを目指します。非決定的な並列計算を導入する章では、再現性との
trade-off を測定します。

## 4. 正しさを小さい入力で証明する

- 手計算できる unit test
- 複数の実装を比較する reference test
- 数値微分と自動微分を比較する gradient check
- cache/量子化など最適化前後を比較する equivalence test
- serialization の round-trip test

大きな end-to-end test だけで正しさを判定しません。

## 5. shape と単位を説明に含める

Tensor を扱うすべての式に shape を併記します。時間は duration、データ量は bytes、確率は
`[0, 1]` のように、値の意味と範囲も記録します。

例: token embedding lookup

```text
tokenIds: [batch, time]
embeddingTable: [vocabulary, channels]
output: [batch, time, channels]
```

## 6. 層の境界を一方向にする

```text
apps (CLI / chapter experiments)
  -> agent / lm / nn
  -> math
  -> Scala/JDK standard library
```

下位層は上位層を知りません。教材の章番号は説明の順序であり、core package の名前には含めません。
これにより「chapter 20 の完成コード」ではなく、再利用可能な概念として読めます。

## 7. 実験と製品上の判断を分ける

教育用の単純な実装であることを、実運用可能と誤認させません。各章で次を分けて記述します。

- 今回あえて単純化した点
- そのために失う性質
- 実用実装で一般に必要な対策
- どの測定値を見て採用を決めるか

## 8. エージェントの外部作用は capability として渡す

agent runtime は filesystem、shell、network へ暗黙にアクセスしません。許可された tool のみを
値として受け取り、引数を検証し、結果を観測として記録します。停止条件、予算、timeout、
idempotency key、監査 log を protocol の一部にします。

## 9. 変更単位は学習上の一概念にする

commit は conventional 形式にし、一つの概念・一つの確認可能な成果へ絞ります。コード変更と
対応する章本文・テストは、同じ commit に含めます。

例:

```text
feat(math): implement immutable vectors
feat(autodiff): add scalar reverse-mode differentiation
docs(attention): derive scaled dot-product attention
test(agent): cover malformed tool arguments
```
