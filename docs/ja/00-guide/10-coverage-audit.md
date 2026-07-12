# 説明と実装の網羅性監査

## 結論

このリポジトリは、宣言したカリキュラム全体については**まだ網羅的ではありません**。
2026-07-12 時点で、77項目中58項目（75%）が実装済み、19項目（25%）が計画中です。

ただし「実装済み」は単にソースファイルがあるという意味ではありません。本文、実装、
コロケーションされた宣言的テスト、実行または観察経路、限界の説明がそろうことを要求
します。`RepositoryCoverageSuite` は、この証拠が構造的に失われていないか検査します。

## 何を数えたか

| 証拠 | 現在値 | 意味 |
| --- | ---: | --- |
| カリキュラム項目 | 77 | このコース自身が掲げる学習範囲 |
| 実装済み | 58 | 章・コード・検証経路が存在する項目 |
| 計画中 | 19 | 説明または実装が未完成の項目 |
| production Scala | 約16,000行 | 標準ライブラリ中心の実装 |
| 宣言的テスト | 390 case | 正常・境界・失敗・数値propertyの証拠 |
| 英語正典 | 約80,000語 | 日本語版の翻訳元 |

行数は品質を証明しません。抜けを隠さないための規模の参考値です。

## 実装済み範囲の監査基準

```mermaid
flowchart LR
    A["平易な問題説明"] --> B["手計算できる例"]
    B --> C["型・shape・数式"]
    C --> D["production実装"]
    D --> E["正常・境界・失敗test"]
    E --> F["実行して観察"]
    F --> G["限界と次の設計"]
```

✅ に必要なもの:

1. 専門名より先に、解く問題を平易な言葉で説明する。
2. 入力・出力・shape・失敗条件を示す。
3. 完成コードだけでなく、実行順のwalkthroughを示す。
4. 実装の隣に `*Suite.scala` を置き、`specify(...)` で性質を宣言する。
5. 数値処理は独立な参照値、有限差分、またはpropertyで検証する。
6. 実行結果が証明することと証明しないことを分ける。
7. 小規模な教材実装とproduction systemの差を明記する。

## まだ網羅していない19項目

| 領域 | 未実装項目 | 網羅に必要な成果物 |
| --- | --- | --- |
| 分散/serving | tensor/pipeline parallel、ZeRO、scheduler | partition、schedule、coordinated recovery、overload test |
| architecture | MoE | routing、capacity、load balance、drop test |
| scaling/data quality | scaling則、重複除去、filter、mixture/contamination | uncertainty、precision/recall、lineage、eval overlap |
| post-training | reward model、DPO、policy optimization | preference data、reference比、KL/reward-hacking test |
| evaluation/release | LM評価、安全性評価、release evidence | slice/uncertainty、adversarial suite、model/data/system card |
| agent運用 | provider adapter、durability、sandbox、hybrid retrieval | contract/failover、event store、quota/isolation、fusion/rerank |

これらは用語だけを追記して✅にはしません。各行には実装、test、章、実行可能な観察が
必要です。

## 現在の保証と保証しないこと

現在の58項目については、最小機構をコードから再構築し、failure pathをtestで観察する
学習経路を目標にしています。`sbt check` はformat、compile、390 case、英日対応、章構造、
実装/testのコロケーション、Scaladocのfile-level存在を検証します。

これは次を保証しません。

- GPU clusterでfrontier modelを訓練できること
- production SLA、security certification、法令適合
- 19の計画項目が完成していること
- testされていないすべての入力で正しいこと

したがって、正確な評価は「現在の✅範囲は証拠付きで広いが、カリキュラム全体は75%」
です。進捗率は[カリキュラム](01-curriculum.md)を正典とし、この監査と自動testが誇張を
防ぎます。
