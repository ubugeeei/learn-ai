# カリキュラム

## 状態の凡例

- ✅ 本文・実装・確認手順が揃っている
- 🚧 到達基準と位置づけは確定、本文または実装を作成中
- ⬜ 計画済み

各 Part は前の Part の成果物だけに依存します。横道の「発展」は飛ばしても、本線を最後まで
進められます。

## Part 0 — 学習環境とプログラムの読み方

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 00 | ✅ | 学習の進め方 | 実験ループと完了条件 |
| 01 | ✅ | Nix と開発環境 | 再現可能な Scala 3 shell |
| 02 | ✅ | Scala 3 最小入門 | 値、関数、型、制御構造、CLI |
| 03 | ✅ | テストとデバッグ | 依存なしのテスト runner |

## Part 1 — LLM のための数学と数値計算

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 04 | ✅ | 数、関数、誤差 | `Double` の性質を測る実験 |
| 05 | ✅ | ベクトル | immutable な `VectorD` |
| 06 | ✅ | 行列と shape | `MatrixD` と行列積 |
| 07 | ✅ | 確率と情報量 | 分布、期待値、entropy、cross entropy |
| 08 | ✅ | 微分と連鎖律 | 数値微分と手計算の比較 |

**Part 成果物:** 線形代数と確率の式を、shape を保つコードへ変換できる。

## Part 2 — 「学習する」をゼロから作る

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 09 | ✅ | 勾配降下 | 一変数の最適化と学習曲線 |
| 10 | ✅ | 自動微分 | scalar computation graph |
| 11 | ✅ | neuron と MLP | XOR を解くネットワーク |
| 12 | ✅ | Tensor 自動微分 | broadcasting のない明示的 Tensor |
| 13 | ✅ | optimizer と初期化 | SGD、AdamW、gradient clipping |

**Part 成果物:** forward、loss、backward、update の学習ループを自力で実装できる。

## Part 3 — テキストを学習データへ変える

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 14 | ✅ | Unicode と byte | UTF-8 tokenizer |
| 15 | ⬜ | BPE | train/encode/decode 可能な tokenizer |
| 16 | ⬜ | dataset と batch | causal language modeling dataset |
| 17 | ⬜ | bigram 言語モデル | 次 token 予測と文章生成 |

**Part 成果物:** 生テキストから training example を作り、確率的に続きを生成できる。

## Part 4 — Transformer と小さな GPT

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 18 | ⬜ | embedding と位置 | token/position embedding |
| 19 | ⬜ | Attention | causal self-attention |
| 20 | ⬜ | Transformer block | RMSNorm、residual、MLP |
| 21 | ⬜ | MiniGPT | 学習、checkpoint、生成 CLI |
| 22 | ⬜ | 正しさと性能 | gradient check、profile、benchmark |

**Part 成果物:** 小さな GPT を学習し、各 parameter が何のためにあるか説明できる。

## Part 5 — 推論エンジンとスケール

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 23 | ⬜ | sampling | temperature、top-k、top-p、seed |
| 24 | ⬜ | KV cache | cache あり/なしの同値性と速度比較 |
| 25 | ⬜ | model format | safetensors 風の安全な形式と loader |
| 26 | ⬜ | 量子化 | symmetric int8 と誤差測定 |
| 27 | ⬜ | 並列化 | data/tensor/pipeline parallel の simulation |

**Part 成果物:** 学習時と推論時の計算の違いを説明し、速度・メモリを測って改善できる。

## Part 6 — フロンティアモデルの学習技術

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 28 | ⬜ | 現代的 block | RoPE、GQA、SwiGLU、MoE |
| 29 | ⬜ | scaling | parameter/FLOPs/memory estimator |
| 30 | ⬜ | data engineering | dedup、filter、mixture、contamination |
| 31 | ⬜ | post-training | SFT、reward model、DPO の最小実装 |
| 32 | ⬜ | evaluation と safety | eval harness、red-team、model card |

**Part 成果物:** 公開技術レポートを読み、品質・計算資源・データ・運用上の trade-off を
定量的に議論できる。

## Part 7 — AI エージェント

| # | 状態 | 章 | 成果物 |
| --- | --- | --- | --- |
| 33 | ⬜ | model API 境界 | provider に依存しない typed protocol |
| 34 | ⬜ | tool calling | schema、validation、実行、結果の観測 |
| 35 | ⬜ | agent loop | stop 条件を持つ再現可能な loop |
| 36 | ⬜ | memory と検索 | chunk、embedding、vector search、引用 |
| 37 | ⬜ | planning | state machine、task graph、recovery |
| 38 | ⬜ | reliability | timeout、retry、idempotency、権限境界 |
| 39 | ⬜ | evaluation | trajectory、fake model、品質/コスト評価 |

**Part 成果物:** 外部操作を安全に行い、失敗理由を追跡できるエージェント runtime を作る。

## Capstone — 実用システム

ローカルの文書を索引化し、根拠を引用して質問へ答え、許可された操作だけを tool として
実行するエージェントを構築します。以下をすべて満たしたら修了です。

- model provider を fake/local/remote の間で差し替えられる
- すべての外部作用に schema 検証、timeout、監査 log がある
- 固定 eval set に対する品質、遅延、token 数を再現できる
- prompt injection と tool 出力の汚染を threat model に含めている
- 設計判断、既知の限界、再現手順を model/system card に記録している

## 本線の依存関係

```text
environment
  -> Scala/type/test
  -> vector/matrix/probability/calculus
  -> autodiff/MLP/Tensor
  -> tokenizer/dataset/bigram
  -> attention/Transformer/MiniGPT
  -> inference/scaling/post-training
  -> tools/agent/memory/evaluation
  -> Capstone
```
