# 35 — bounded agent loop と audit trace

## この章で作るもの

model → tool → observation → model を繰り返し、final answer または明示的 limit/error で必ず停止する
`AgentRuntime` を実装します。対象コードは
`src/main/scala/learnai/agent/AgentRuntime.scala` です。

## loop as a state machine

```text
InvokeModel
  | FinalAnswer ----------------------> Completed
  | RequestTools
  v
ValidateAndExecute
  | append observations
  +-----------------------------------> InvokeModel

Any state -> model error / step limit / tool limit -> terminal failure
```

`while true` と曖昧な文字列 history ではなく、state、counter、terminal reason を型と trace に残します。

## one iteration

1. current history と granted tool definitions で `ModelRequest`
2. usage を累積
3. final answer なら assistant text を append して完了
4. tool calls なら assistant call item を append
5. 各 call の capability/schema/budget を検査
6. tool を timeout 内で実行
7. observation を append
8. 次 model step

model は tool の実際の副作用を直接見ず、structured observation だけを見ます。

## hard limits

`AgentConfig` は少なくとも次を持ちます。

- `maximumModelSteps`
- `maximumToolCalls`
- `toolTimeoutMillis`

これらは品質 tuning でなく safety/liveness invariant です。model が同じ call を繰り返す、tool が hang
する、invalid call を無限に修正する場合も terminal state へ到達します。外側には run 全体 deadline と
token/cost budget も追加します。

## timeout と cancellation

tool は JDK virtual thread で実行し、`Future.get(timeout)` を使います。timeout 時は task を interrupt
し、retryable `tool_timeout` observation を返します。

interrupt は強制停止保証ではありません。tool implementation は interrupt/deadline に協力し、network
client 自身にも connect/read timeout を設定する必要があります。外部 side effect が timeout 直前に
成功した可能性もあるため、盲目的 retry は危険です。

## idempotency

call ID を idempotency key として outcome を cache します。

- 同じ ID・同じ内容: outcome を再利用し side effect を繰り返さない
- 同じ ID・違う内容: `conflicting_call_id` として拒否

message send/payment/create のような tool は、runtime cache だけでなく外部 service にも idempotency key
を渡します。process crash 後も重複を防ぐには durable storage が必要です。

## audit events

run は success/failure に関係なく event sequence を返します。

- model invoked/returned と usage
- tool started/finished/reused
- terminal status/reason

event は deterministic な step と stable ID を持ちます。production では monotonic duration、trace ID、
principal、policy decision、redacted argument hash を追加します。secret/full personal data はそのまま log
しません。

## stopping is not failure hiding

terminal status を区別します。

- `Completed`
- `ModelFailed`
- `ModelStepLimitExceeded`
- `ToolCallLimitExceeded`

limit 到達を成功回答に変換しません。UI/orchestrator が retry、追加 budget、ユーザー確認、manual handoff
を選べるようにします。

## tests as trajectory evaluation

unit tests は final text だけでなく全 trajectory を検査します。

- valid call が次 request の observation になる
- schema error では tool invocation count が 0
- unknown capability は実行されない
- identical ID の side effect は一回
- conflict/timeout が typed error
- budget の次の call は実行されない
- model/step failure に完全 trace が残る

実 model の agent evaluation でも、final correctness、tool selection、argument accuracy、step count、cost、
latency、unsafe attempt を別 metric にします。

## 演習

1. run 全体 deadline と token budget を追加してください。
2. retryable model error に exponential backoff + jitter を追加し、fake clock で test してください。
3. parallel read-only tool calls と ordered observations を設計してください。
4. write tool に human approval state を追加してください。
5. event trace から success rate、p95 steps、tool error rate を集計してください。
6. crash 後に idempotency cache を復元する append-only store を設計してください。

## 完了条件

- agent loop を有限 state machine として説明できる
- step/tool/timeout limit が liveness を保証する理由を説明できる
- timeout と side-effect 成否が別問題であることを説明できる
- idempotency key を runtime と外部 service の両方で使う理由を説明できる
- final answer 以外の trajectory metric を列挙できる
- `AgentRuntimeSuite` の正常/異常/limit tests が成功する
