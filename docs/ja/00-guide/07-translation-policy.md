# 翻訳ポリシー

`docs/` 配下の英語ドキュメントが正典です。日本語訳は `docs/ja/` 配下に
ファイル単位で英語版をミラーし、加えてリポジトリルートに `README.ja.md`
と `CONTRIBUTING.ja.md` を置きます。この分離は
`DocumentationLanguageSuite` の 3 つのテストガードで強制されます。すなわち、
正典ドキュメントは英語のままであること、すべての翻訳には対応する英語原文が
存在すること、そしてすべての翻訳が実際に日本語の文章を含むことです。数式
デリミタの検査は両方のツリーに適用されます。

## 用語ポリシー

この翻訳の目的は学習であり、この分野を学ぶことの一部は、その語彙を学ぶ
ことです。したがって用語は、優先順位の高い順に次の 3 つのルールに
従います。

1. **訳語を創作しない。** 定着した日本語の対応語がない用語に、独自の訳語を
   発明してはいけません。造語は、どこにも存在しない言葉を教えてしまいます。
2. **世界的に標準的な英語の用語をそのまま使うことを優先する。** 特に
   アーキテクチャや手法の名前は英語のままにします: attention、accounting、
   RoPE、GQA、MQA、LoRA、SFT、DPO、MoE、all-reduce、FLOPs。これにより
   日本語の本文が、論文、コード、そして広いコミュニティと整合します。
3. **定着した日本語の用語は維持する。** 本当に標準的な日本語訳がある言葉は
   翻訳します: 勾配 (gradient)、損失 (loss)、埋め込み (embedding)、
   量子化 (quantization)、推論 (inference)、訓練 (training)、
   検証 (validation)、断片化 (fragmentation)。

ファイル内で重要な用語が最初に登場するとき、訳語が日本語である場合は
英語の原語を括弧で添えます。例: 埋め込み (embedding)。

## 正典用語集

| 英語 | 表記 |
| --- | --- |
| attention（および複合語: causal/multi-head/self-） | attention（英語のまま。複合語は完全に英語） |
| accounting | accounting（動詞用法は 集計する） |
| RoPE / rotary position embedding | Rotary Position Embedding (RoPE) |
| grouped-query / multi-query attention | Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) |
| KV cache | KV キャッシュ |
| embedding | 埋め込み |
| gradient | 勾配 |
| loss / loss mask | 損失 / 損失マスク |
| training / validation / inference | 訓練 / 検証 / 推論 |
| optimizer / checkpoint / tokenizer | オプティマイザ / チェックポイント / トークナイザ |
| oracle | オラクル |
| quantization / fragmentation | 量子化 / 断片化 |
| speculative decoding | 投機的デコーディング (speculative decoding) |
| all-reduce / data parallelism | all-reduce / データ並列 |
| exact resume | 厳密再開 (exact resume) |
| fine-tuning / frozen / merge | ファインチューニング / 凍結 / マージ |

## 正典セクション見出し

| 英語 | 日本語 |
| --- | --- |
| What you will build | この章で作るもの |
| Prerequisites | 前提知識 |
| Run the experiment | 実験の実行 |
| Implementation walkthrough | 実装ウォークスルー |
| Reading the tests | テストの読み方 |
| Debugging checklist | デバッグチェックリスト |
| Failure modes to test | テストすべき失敗モード |
| Exercises | 演習 |
| Completion criteria | 完了基準 |
| Primary sources | 一次資料 |

## 書式ルール

- 本文は です・ます調 で書きます。
- コードブロック、インラインコード、コマンド、ファイルパス、識別子、URL、
  およびすべての数式（`$...$`、`$$...$$`）は英語原文とバイト単位で同一に
  保ちます。LaTeX のバックスラッシュ括弧形式のデリミタはテストで拒否
  されます。
- 論文や書籍のタイトルは英語のままにします。
- `docs/ja/` 内の相対リンクは変更しません（ミラー内で解決されます）。
  ルートの `README.ja.md` は `docs/ja/` へリンクします。
- 日本語の文章と埋め込まれた英単語の間には半角スペースを入れます。ただし
  句読点の前には入れません。
