# learn-ai

> 本書は英語版 README の日本語訳です。英語版が正典であり、翻訳は更新が遅れることがあります。日本語版ドキュメント全体は docs/ja/ にあります。

Scala 3 で、大規模言語モデルと AI エージェントをゼロから手を動かして学ぶ
ためのパスです。

このリポジトリは、すべての重要な数式を実行可能なコードに対応付けます。
ベクトル、自動微分、トークナイゼーション、attention、訓練、
ツール実行を、機械学習フレームワークの背後に隠しません。メインパスは
Scala と JDK の標準ライブラリのみを使います。

小さなモデルは、あらゆる仕組みを観察可能にします。後半の章では、それらと
同じ仕組みを、フロンティアモデルで使われるシステム、データ、スケーリング
の意思決定へと接続します。

## 学習成果

カリキュラムを修了すると、次のことができるようになるはずです。

- 微分と勾配降下法を使って、訓練が何を最小化するのかを説明する
- テンソルと逆方向モードの自動微分を実装する
- ニューラルネットワークと次トークン言語モデルを訓練する
- バイト/BPE トークナイザ (tokenizer) と因果的データセットパイプラインを
  構築する
- causal multi-head attention と小さな GPT を導出し、実装する
- チェックポイント (checkpoint)、サンプリング、KV キャッシュ、量子化、
  並列化を説明する
- SFT、選好最適化、評価、安全性について推論する
- モデル、ツール、メモリ、プランニング、ポリシーの境界からエージェントを
  実装する
- 品質、レイテンシ、メモリ、コスト、再現性、失敗モードを測定する
- フロンティアモデルの技術レポートを読み、その設計判断を批評する

> フロンティアモデルを理解することは、その訓練の全体を個人のコンピュータで
> 再現することを意味しません。このコースは、同じ計算の最小限の実装と、
> それを動かすために必要な大規模システム設計を組み合わせます。

## ここから始める

1. [学び方](docs/ja/00-guide/00-how-to-learn.md)を読みます。
2. [完全な実装章の構造](docs/ja/00-guide/04-chapter-anatomy.md)を読みます。
3. 小さな参照実装をゴールと誤解しないよう、
   [プロフェッショナル能力ロードマップ](docs/ja/00-guide/05-professional-roadmap.md)
   を読みます。
4. [カリキュラム](docs/ja/00-guide/01-curriculum.md)を順に進めます。
5. 完了した各章の検証コマンドを実行します。
6. 答えを読んだり実装したりする前に、演習を解きます。
7. 各パートの成果物を自分の言葉で説明します。
8. 各実装マイルストーンの後に、
   [一次論文リーディングマップ](docs/ja/09-papers/40-primary-reading-map.md)
   を使います。

コードは使い捨てではなく累積的です。各章は、後の章が再利用するテスト済みの
コンポーネントを追加します。アーキテクチャ上の選択は
[設計原則](docs/ja/00-guide/02-design.md)に記録されています。

## 再現可能な環境

初回の実行には、ガイド付きラッパーを使います。

```console
$ ./learn-ai help
$ ./learn-ai model
$ ./learn-ai training
```

このラッパーは、各ラボが何を実行するのか、どの出力が重要なのか、その結果が
何を証明しないのかを説明します。Nix、sbt、`runMain`、そして各ラボの前提
知識ゼロの解説については、
[迷わずラボを実行する](docs/ja/00-guide/06-running-the-labs.md)を参照して
ください。

より低レベルの環境コマンドは次のとおりです。

```console
$ nix develop
$ sbt check
```

このリポジトリは、JDK 21、sbt、Scala 3、Nix のパッケージリビジョンを
ピン留めしています。

## 現在の実装

実行可能なパスは、現在ここまで到達しています。

```text
math -> autodiff -> neural networks -> tokenization -> language modeling
  -> shuffled/packed datasets with loss masks
  -> causal Transformer -> MiniGPT -> gradient/benchmark diagnostics
  -> canonical experiment records -> batch training and held-out validation
  -> bitwise exact mid-run resume (RNG, optimizer, scheduler, data cursor)
  -> KV-cached decoding -> inference artifacts
  -> paged KV pool / speculative decoding / traced data parallelism
  -> RoPE / SwiGLU / grouped-query / tiled online-softmax attention
  -> model accounting anchored to the implementation
  -> LoRA adapters and chat-template SFT with assistant-span masks
  -> typed tools -> approval and retry policy -> bounded agent runtime
  -> cited retrieval -> task-graph planning and recovery -> agent evaluation
```

有用なエントリポイント:

```console
$ nix develop -c sbt check
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
$ nix develop -c sbt 'runMain learnai.transformer.runMiniGptDiagnostics'
$ nix develop -c sbt 'runMain learnai.training.runMiniGptTrainingLab'
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

テンソル/パイプライン/ZeRO のシミュレーション、サービングスケジューラ、
大規模なコーパスキュレーション、選好最適化、プロダクション向けエージェント
アダプタは、明示的に今後のマイルストーンとして残っています。正確な章の
ステータスは[カリキュラム](docs/ja/00-guide/01-curriculum.md)を、能力目標の
全体は[プロフェッショナルロードマップ](docs/ja/00-guide/05-professional-roadmap.md)
を、完了の定義は[進捗と品質基準](docs/ja/00-guide/03-progress.md)を参照して
ください。

## コントリビューション基準

公開 API には英語の Scaladoc を使います。すべての実装章には、関連する
正常系、境界、失敗、プロパティ、数値、勾配 (gradient) のテストが含まれます。
[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md) を参照してください。

## ライセンス

これは学習用リポジトリです。再配布の条件はまだ選定されていません。
