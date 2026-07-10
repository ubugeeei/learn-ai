# 34 — JSON、schema、tool capability

## この章で作るもの

外部 dependency なしの strict JSON parser/renderer、top-level object schema、明示的に許可された `Tool`
capability、成功/失敗 observation を実装します。

- JSON: `src/main/scala/learnai/json/Json.scala`
- tool protocol: `src/main/scala/learnai/agent/Protocol.scala`

## tool call は命令でなく未検証の提案

model が `{"path":"..."}` を生成しても、そのまま実行しません。

```text
model output
 -> strict JSON parse
 -> tool name lookup in granted registry
 -> schema validation
 -> policy/authorization checks
 -> timeout-bounded execution
 -> typed observation
```

LLM output、user text、retrieved document、tool output はすべて untrusted data として扱います。

## dependency-free JSON

parser は object/array/string/number/boolean/null を AST へ変換し、次を拒否します。

- trailing data/comma
- duplicate object fields
- leading zero や不完全 exponent
- invalid escape/control character
- unpaired UTF-16 surrogate
- input size/depth limit 超過

duplicate field を許す parser は「最初を採用」「最後を採用」で security check と executor の解釈が
分かれる危険があります。strict に一意性を要求します。

renderer は field insertion order を保ち、test snapshot と audit を deterministic にします。JSON object
field order へ意味を持たせてはいけませんが、stable serialization は hash/debug に有用です。

## schema validation

`ToolSchema` は required field、JSON type、additional-field policy を検査し、全問題を一度に返します。

```scala
ToolSchema(Vector(
  ToolField("message", JsonFieldType.StringValue, "Text to echo", required = true)
))
```

教材 schema は top-level 型だけです。production では string length/pattern、number range、enum、nested
object/array size、相互制約も必要です。schema validation 後も domain validation を tool 内で行います。

## capability-based design

runtime が持つ `Vector[Tool]` だけが実行可能です。model request にも同じ definitions だけを渡します。
unknown name は `unknown_tool: not granted` になり、reflection や shell fallback はしません。

```text
agent with [weather.read]  cannot call filesystem.write
agent with [draft.create]  cannot call message.send
```

tool は大きな万能 shell でなく、最小権限の domain operation に分けます。read/write、draft/send、
preview/commit を別 capability にすると approval と audit が明確になります。

## ToolContext と arguments を分ける

model-controlled JSON は `arguments` だけです。call ID、agent step、deadline、principal、tenant、trace ID
など host-controlled metadata は `ToolContext` で渡します。model が authorization context を偽装できない
境界にします。

## failure observation

tool は exception を通常 error path にせず、`Either[ToolError,JsonValue]` を返します。

- stable machine code
- human/model-readable message
- retryable flag

schema error、unknown tool、timeout も `ToolObservation` になり、model は修正・代替・ユーザーへの説明を
選べます。secret、stack trace、internal path を observation に漏らしません。

## prompt injection と tool output

retrieved page が「以前の指示を無視して送信せよ」と書いていても、それは data です。model が従う
可能性を前提に、runtime が最後の policy enforcement point になります。

- tool allowlist
- argument/path/domain allowlist
- read/write separation
- high-impact action approval
- output size/content filtering
- immutable audit record

prompt だけを security boundary にしません。

## 演習

1. integer range と string maximum length validation を追加してください。
2. read-file tool の allowed root と symlink traversal 対策を設計してください。
3. draft/send を別 tool にし、send だけ approval required にしてください。
4. tool output の maximum JSON bytes/depth を追加してください。
5. JSON parser を property/fuzz test する generator を作ってください。

## 完了条件

- model tool call を untrusted proposal と説明できる
- parse/schema/domain/auth validation の違いを説明できる
- duplicate JSON field を拒否する security 理由を説明できる
- capability allowlist が prompt injection 後も必要な理由を説明できる
- structured failure を model observation に戻せる
- `JsonSuite` と agent schema tests が成功する
