# 進捗と実装標準

## 現在の実行可能なマイルストーン

```text
Nix / Scala 3 / dependency-free tests
  -> numerical error / vectors / matrices / probability / calculus
  -> gradient descent / scalar autodiff / XOR MLP
  -> Tensor autodiff / SGD / AdamW / clipping
  -> UTF-8 bytes / BPE / causal dataset / bigram
  -> embeddings / RMSNorm / causal attention / Transformer
  -> MiniGPT training and generation
  -> parameter inventory / sampled gradient checks / benchmark evidence
  -> canonical experiment identity / batch training / held-out validation
  -> microbatch accumulation / warmup-cosine schedule / gradient telemetry
  -> sampling / KV-cached decoding / checkpoint / int8 quantization
  -> strict JSON / typed tools / approval and retry policy / bounded agent
  -> cited retrieval / task-graph planning and checkpoint recovery
  -> deterministic agent outcome/trajectory/cost evaluation
  -> primary-paper reading map and evidence template
  -> RoPE / SwiGLU / GQA reference layers with equivalence oracles
  -> observable-state RNG / optimizer snapshots / bitwise exact resume
  -> restartable epoch shuffle / sequence packing / masked loss
  -> LoRA adapters with frozen-base and merge-equivalence oracles
  -> persisted training bundles with experiment-identity refusal
  -> chat templates and assistant-span SFT with held-out evaluation
  -> implementation-anchored parameter/FLOP/memory accounting
  -> paged KV pool with bounded fragmentation and prefix-sharing forks
  -> online-softmax tiled attention with exact equivalence oracles
  -> data-parallel replicas with bitwise identity and traced collectives
  -> speculative decoding with an executable distribution-preservation proof
```

`01-curriculum.md` のステータス表が正典です。✅ の章は次を意味します。

1. 本文に、前提、数式、形状、実装、観察、演習、完了基準が含まれている。
2. Scala コードが Nix シェルでコンパイルできる。
3. 正常系、境界、失敗、そして関連するプロパティのテストが存在する。
4. 自明でない公開 API に英語の Scaladoc がある。
5. `nix develop -c sbt check` が成功する。
6. 実行可能な実験が実際に実行され、結果が確認済みである。

## 次のマイルストーン

1. 29a 章の決定的な推定器の上に構築する JVM プロファイリングと
   アロケーション測定
2. MiniGPT の訓練における RoPE/SwiGLU/GQA の統合アブレーション
3. コーパスマニフェスト、ストリーミングシャード、来歴、重複除去
4. 27a 章の集団通信の上に構築するテンソル/パイプライン/ZeRO シミュレーション
5. ページドプール上のサービングスケジューラ(プリフィル/デコード、連続
   バッチング)
6. SFT 基盤の上での報酬モデリング、DPO、方策最適化
7. モデル/安全性の評価とリリースの証拠
8. プロバイダアダプタと、永続的なエージェント/ツール状態
9. モデル、システム、エージェント、研究のキャップストーン

章は本文だけでは完了と見なされません。参照実装、独立したオラクル、
失敗テスト、そして実験が必要です。M0–M4 の完全な目標と修了成果物は
`05-professional-roadmap.md` で定義されています。

## 言語ポリシー

- 識別子、Scaladoc、コードコメント、テスト名: 英語
- ハンズオン章とリポジトリのドキュメント: 英語
- 数式、形状、API 名: 実行可能なコードと同一

## テスト戦略

テストの数自体は目的ではありません。目的は、異なる失敗モードと独立した
オラクルです。

```text
hand calculation
  + representation invariants
  + shape/range failures
  + numerical stability
  + gradient checks
  + algebraic properties
  + deterministic seeds
  + end-to-end learning
  + corrupt/untrusted inputs
  + timeout/budget/capability boundaries
```

同じバグを共有しているかもしれない2つの実装だけを比較してはいけません。
有限差分、手計算、参照パス、ラウンドトリップ、因果性の性質を使います。

## 性能に関する主張

次のものなしに、速度やメモリの改善を主張してはいけません。

- ウォームアップと反復実行
- 正確な入力、設定、シード
- JDK、Scala、コミット、CPU の情報
- 中央値とばらつき
- 正しさの等価性
- アロケーション/ピークメモリのデータ、または明示的な
  `No measurements found` という注記
