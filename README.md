# learn-ai

Scala 3 で、数式と実装を一行ずつ対応させながら、大規模言語モデル（LLM）と
AI エージェントをゼロから学ぶハンズオンです。

この教材では、完成済みの機械学習フレームワークを呼び出して終わりにはしません。
ベクトル・行列、自動微分、ニューラルネットワーク、トークナイザ、Transformer、
学習と推論、ツール実行型エージェントを、原則として Scala/JDK の標準機能だけで
実装します。小さなモデルで原理を実験し、同じ原理がフロンティアモデルではどのように
大規模化・分散化されているかまでつなげます。

## 到達目標

通読と演習を終えると、次のことを自分の言葉とコードで説明できる状態を目指します。

- 「学習する」とは何を最小化しているのかを、微分と勾配降下から説明する
- Tensor と自動微分エンジンを実装し、ニューラルネットワークを学習させる
- テキストを token ID に変換し、次 token 予測モデルを学習させる
- Attention と Transformer を数式から実装し、小さな GPT を動かす
- checkpoint、sampling、KV cache、量子化、並列学習の役割を説明する
- SFT、選好学習、評価、安全性を含むモデル開発工程を説明する
- モデル、tool、memory、planning、guardrail からなるエージェントを実装する
- 品質、遅延、コスト、再現性、安全性を測り、実用システムとして改善する
- 公開されたフロンティアモデルの技術資料を読み、設計上の選択を批評する

> 「フロンティアモデルを理解する」は、個人の計算機で同じ規模を再学習することでは
> ありません。本教材では、同じ計算原理を観察できる最小実装と、巨大規模で必要になる
> システム設計の両方を扱います。

## 学び方

1. [学習の進め方](docs/00-guide/00-how-to-learn.md)を読む
2. [カリキュラム](docs/00-guide/01-curriculum.md)を上から順に進める
3. 各章の `確認` コマンドを実行する
4. 章末問題を、先に解答を見ずに実装する
5. 各 Part の成果物を自分の言葉で説明し直す

コードは「後で完成形に置き換える使い捨て」ではなく、後続の章で再利用する小さな部品に
分けます。設計判断は [設計原則](docs/00-guide/02-design.md) に記録します。

## 現在の実装範囲

現在は、環境構築から MiniGPT の end-to-end 学習、checkpoint、int8 量子化、tool 実行型 agent、
引用付き retrieval までが実行可能です。実装済みの各章には本文、Scala code、境界/異常系を含む
test があります。

```console
$ nix develop -c sbt check
$ nix develop -c sbt 'runMain learnai.nn.trainXor'
$ nix develop -c sbt 'runMain learnai.lm.trainBigram'
$ nix develop -c sbt 'runMain learnai.transformer.trainMiniGpt'
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

KV cache、分散学習、現代的 block、post-training、agent planning/evaluation は次の実装 milestone です。
章単位の正確な状態と依存順は[カリキュラム](docs/00-guide/01-curriculum.md)、完成基準と次の作業は
[進捗と実装規約](docs/00-guide/03-progress.md)で確認できます。未実装の章にも到達基準を先に定義し、
学習経路が途中で分岐しないようにしています。

## 開発規約

公開 API の Scaladoc、数式と shape の対応、正常/境界/異常/property/gradient test を実装と同じ
commit に含めます。詳しくは [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## ライセンス

学習目的のリポジトリです。ライセンスを確定するまでは、再配布条件は未指定です。
