# Translation policy

English documentation under `docs/` is canonical. Japanese translations
mirror it file-for-file under `docs/ja/`, plus `README.ja.md` and
`CONTRIBUTING.ja.md` at the repository root. Three test guards in
`DocumentationLanguageSuite` enforce the split: canonical documentation
stays English, every translation must have an existing English source,
and every translation must actually contain Japanese prose. The math
delimiter check applies to both trees.

## Terminology policy

The goal of the translation is learning, and part of learning this field
is learning its vocabulary. Terminology therefore follows three rules, in
priority order:

1. **Never coin a translation.** If a term has no established Japanese
   equivalent, do not invent one — a made-up Japanese term teaches a word
   that exists nowhere else.
2. **Prefer the globally standard English term as-is.** Architecture and
   technique names in particular stay in English: attention, accounting,
   RoPE, GQA, MQA, LoRA, SFT, DPO, MoE, all-reduce, FLOPs. This keeps the
   Japanese text aligned with papers, code, and the wider community.
3. **Keep established Japanese terms.** Words with a genuinely standard
   Japanese rendering are translated: 勾配 (gradient), 損失 (loss),
   埋め込み (embedding), 量子化 (quantization), 推論 (inference),
   訓練 (training), 検証 (validation), 断片化 (fragmentation).

On the first occurrence of an important term in a file, add the English
original in parentheses when the rendering is Japanese, e.g.
埋め込み (embedding).

## Canonical glossary

| English | Rendering |
| --- | --- |
| attention (and compounds: causal/multi-head/self-) | attention (English, compounds fully English) |
| accounting | accounting (verb use: 集計する) |
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

## Canonical section headings

| English | Japanese |
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

## Formatting rules

- Prose is です・ます調.
- Code blocks, inline code, commands, file paths, identifiers, URLs, and
  all math (`$...$`, `$$...$$`) stay byte-identical to the English
  source; LaTeX backslash-parenthesis and backslash-bracket delimiters
  are rejected by tests.
- Paper and book titles stay in English.
- Relative links are kept unchanged inside `docs/ja/` (they resolve
  within the mirror); the root `README.ja.md` links into `docs/ja/`.
- Insert a half-width space between Japanese text and embedded Latin
  words, but not before punctuation.
