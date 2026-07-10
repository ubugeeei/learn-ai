# 33 — provider-neutral model boundary

## この章で作るもの

特定 vendor SDK や wire JSON を agent logic から分離する `LanguageModel` protocol を定義します。
model は typed history と許可された tool definitions を受け、final answer または tool calls を返します。

対象コードは `src/main/scala/learnai/agent/Protocol.scala` です。

## model と agent は別の component

language model は入力 token から次 token を予測します。agent は model 出力を解釈し、tool を実行し、
結果を次の model input へ戻す host program です。

```text
agent runtime
  -> ModelRequest
  -> LanguageModel adapter
  -> local/remote model
  <- ModelDecision
```

model 自体に filesystem/network 権限があるのではありません。host が tool capability を渡したときだけ
外部作用が可能です。

## typed conversation

history は文字列一つでなく、意味の異なる item を区別します。

- `TextMessage(System|User|Assistant, content)`
- `AssistantToolCalls(calls)`
- `ToolObservation(callId, toolName, outcome)`

tool result は必ず元 call ID と対応します。複数 call がある場合も、どの observation がどれへの応答
かを順序だけに依存せず追跡できます。

## model decision

model adapter は provider 固有 response を次のどちらかへ変換します。

```scala
FinalAnswer(text, usage)
RequestTools(calls, usage)
```

text と tool call が同時に返る provider もあります。その扱いを adapter contract で明示し、runtime
内部に provider 条件分岐を散らしません。

## errors are data

network/認証/rate limit/response parse error は `ModelError(code,message,retryable)` です。tool の失敗は
`ToolError` として observation にできますが、model 自体が応答しない場合は次の意思決定ができないため
run を terminal failure にします。retry policy を入れる場合は最大回数、backoff、deadline を runtime
policy として追加します。

## usage accounting

`ModelUsage(inputTokens,outputTokens)` を各 decision に含め、run 全体で合計します。token 数は cost、
latency、context limit、異常 loop の指標です。provider の cached/reasoning token など詳細 category は
adapter の拡張 field または別 metrics にします。

## fake model first

tests は remote model を呼ばず `ScriptedModel` を使います。

- decision sequence が固定
- request history/tool definitions を検査できる
- timeout/rate limit を待たない
- invalid tool call を意図的に生成できる

model quality eval と runtime correctness test を分けます。non-deterministic model を unit test の oracle
にしません。

## adapter の責務

実際の provider adapter は次を担当します。

- internal history ↔ provider message format
- `ToolSchema` ↔ provider JSON schema subset
- authentication、HTTP timeout、status mapping
- streaming chunks の組み立て
- usage mapping
- response size/depth limit と strict JSON parse

API key を prompt/history/log に入れず、secret provider から HTTP header へ直接渡します。

## 演習

1. scripted model へ渡る二 step 目 history を図にしてください。
2. streaming model event と最終 `ModelDecision` の型を設計してください。
3. rate-limit error の bounded retry policy を設計してください。
4. provider A/B adapter が同じ runtime test suite を通る contract test を書いてください。

## 完了条件

- model と agent runtime の責務を分けて説明できる
- text/tool call/tool observation の対応を説明できる
- provider 固有形式を adapter に閉じ込める理由を説明できる
- fake model で deterministic test を書ける
- usage と terminal model error が trace に残ることを確認できる
