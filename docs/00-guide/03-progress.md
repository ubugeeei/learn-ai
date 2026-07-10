# 進捗と実装規約

## 現在の runnable milestone

現時点の repository は次の一本の経路を実行できます。

```text
Nix / Scala 3 / dependency-free tests
  -> numeric error / vector / matrix / probability / calculus
  -> gradient descent / scalar autodiff / XOR MLP
  -> Tensor autodiff / SGD / AdamW / clipping
  -> UTF-8 bytes / BPE / causal dataset / bigram
  -> embedding / RMSNorm / causal multi-head attention / Transformer block
  -> MiniGPT training and generation
  -> sampling / checksummed checkpoint / int8 quantization
  -> strict JSON / typed tools / bounded agent loop / cited retrieval
```

正確な章 status は `01-curriculum.md` の ✅ / 🚧 / ⬜ を source of truth とします。✅ は次をすべて
満たす意味です。

1. hands-on 本文が前提知識、数式、shape、実装、観察、演習、完了条件を持つ
2. Scala code が Nix shell 内で compile できる
3. 正常系だけでなく concept に対応する境界・異常・property test がある
4. 公開される非自明な API に英語 Scaladoc がある
5. `nix develop -c sbt check` が成功する
6. runnable experiment は実際に実行して出力を確認している

## 次の milestone

未実装章は次の順で進めます。

1. correctness/performance harness と KV cache equivalence
2. data/tensor/pipeline parallel simulation
3. RoPE、GQA、SwiGLU、MoE と scaling estimator
4. data dedup/mixture/contamination checks
5. SFT、reward model、DPO の小規模実装
6. model/safety evaluation harness と model card
7. agent task graph、durable state、approval、recovery
8. trajectory/cost/safety eval と Capstone

各項目は「説明だけ」を完成扱いにせず、小さい reference implementation、独立した oracle、失敗 test、
実験 command を揃えてから ✅ にします。

## code と説明の言語

- Scala identifier、Scaladoc、code comment、test name: English
- hands-on 本文: 現在は日本語を主にし、technical term は英語を併記
- 数式記号、shape、API 名: code と完全に対応させる

英語版本文を追加する場合も code block を複製せず、章と runnable source の対応を一意に保ちます。

## test strategy

test 件数そのものを目的にしません。異なる failure mode を独立に検出できることを重視します。

```text
hand calculation
  + representation invariant
  + shape/range failures
  + numerical stability
  + gradient check
  + algebraic property
  + deterministic seed
  + end-to-end learning
  + corrupted/untrusted input
  + timeout/budget/capability boundary
```

同じ bug を共有する実装同士だけを比較しません。数値微分、手計算、reference path、round-trip、prefix
causality など独立した oracle を選びます。

## performance claims

教材の単純 CPU implementation は原理観察用です。「高速」「省 memory」の claim は以下が揃うまで
書きません。

- warmup と複数 iteration
- input/config/seed
- JDK/Scala/commit/CPU
- median とばらつき
- correctness equivalence
- allocation/peak memory または明示的に未測定

測定がない場合は `No measurements found` と記録します。
