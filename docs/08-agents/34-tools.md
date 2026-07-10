# 34 — JSON, schemas, and tool capabilities

## What you will build

Implement a dependency-free strict JSON parser, top-level object schemas,
explicit `Tool` capabilities, and structured success/failure observations.

- JSON: `src/main/scala/learnai/json/Json.scala`
- tool protocol: `src/main/scala/learnai/agent/Protocol.scala`

## A model tool call is an untrusted proposal

Generated JSON is never an instruction to execute directly:

```text
model output
 -> strict JSON parse
 -> lookup in granted tool registry
 -> schema validation
 -> authorization and domain policy
 -> timeout-bounded execution
 -> typed observation
```

User text, model output, retrieved documents, and tool output are all untrusted
data.

## Strict dependency-free JSON

The parser supports object, array, string, number, boolean, and null while
rejecting:

- trailing data or commas;
- duplicate object fields;
- invalid number grammar;
- invalid escapes and control characters;
- unpaired UTF-16 surrogates;
- input-size and nesting-depth violations.

Duplicate fields are dangerous when a validator reads the first occurrence and
an executor reads the last. Reject them rather than selecting a winner.

The renderer preserves insertion order for deterministic snapshots and audit
hashes, though application semantics must not depend on JSON field order.

## Schema validation

`ToolSchema` validates required fields, top-level JSON types, and additional-
field policy:

```scala
ToolSchema(Vector(
  ToolField("message", JsonFieldType.StringValue, "Text to echo", required = true)
))
```

Production schemas also need lengths, patterns, ranges, enums, nested limits,
and cross-field rules. Schema validation does not replace domain validation.

## Capability-based design

Only tools in the runtime's explicit `Vector[Tool]` can execute. The model sees
the same granted definitions. Unknown names return `unknown_tool`; there is no
reflection or shell fallback.

```text
agent with [weather.read] cannot call filesystem.write
agent with [draft.create] cannot call message.send
```

Prefer small domain operations over one universal shell. Separate read/write,
draft/send, and preview/commit so approval and audit are meaningful.

## Separate model arguments from host context

Model-controlled JSON is `arguments`. Host-controlled call ID, step, principal,
tenant, deadline, and trace ID belong in `ToolContext`, where the model cannot
forge authorization metadata.

## Structured failures

Tools return `Either[ToolError,JsonValue]`, with:

- stable machine-readable code;
- safe message;
- retryable flag.

Unknown tools, invalid arguments, and timeouts also become observations. The
model may correct its call or explain failure. Secrets, stack traces, and
internal paths must not be exposed.

## Prompt injection and tool output

A retrieved page can contain instructions to ignore previous policy. It remains
data. Assume the model may follow it and keep runtime enforcement outside the
prompt:

- capability allowlists;
- path/domain/argument allowlists;
- read/write separation;
- approval for high-impact operations;
- output size/content limits;
- immutable audit events.

Prompts are not security boundaries.

## Implementation walkthrough

This chapter has two source files because it crosses two trust boundaries.
`Json.scala` turns untrusted text into a JSON syntax tree. `Protocol.scala`
checks whether that tree satisfies a granted capability's argument contract.

### 1. Parse exactly one JSON document

`JsonParser.parse` first rejects oversized input, then creates a mutable
`Cursor`. The cursor owns one integer, `index`, which always points to the next
unconsumed character. `parseDocument` performs three operations:

```scala
skipWhitespace()
val value = parseValue(depth = 0)
skipWhitespace()
if index != input.length then fail("unexpected trailing character ...")
```

The final check is essential. Without it, `{"path":"safe"} malicious` could
be accepted because the valid prefix parsed successfully.

`parseValue` dispatches from the leading character. Arrays and objects receive
`depth + 1`; scalar values do not. The check happens before descending again,
so deeply nested input terminates with `Left` instead of exhausting the stack.

### 2. Understand the object loop

For `{"x":1,"ok":true}`, `parseObject` advances past `{`, then repeats:

1. parse a quoted field name;
2. reject it if the mutable `names` set already contains it;
3. require `:`;
4. recursively parse the value;
5. require either `}` or `,`.

After a comma, an immediate `}` is rejected as a trailing comma. The builder is
converted to an immutable `Vector` only after the complete object is valid.
`JsonObject` independently checks uniqueness, so objects constructed directly
in Scala obey the same invariant as parsed objects.

### 3. Follow string and number grammar, not convenience parsing

`parseString` handles each JSON escape explicitly. A `\u` high surrogate must
be followed by a low surrogate; a lone low surrogate fails. This is why an
emoji encoded as two UTF-16 code units becomes one valid Unicode scalar rather
than malformed text.

`parseNumber` recognizes sign, integer, optional fraction, and optional
exponent before calling `BigDecimal`. `BigDecimal` alone is not the grammar:
the parser explicitly rejects leading zeroes, missing fractional digits, and
incomplete exponents.

### 4. Validate the parsed object without executing anything

`ToolSchema.validate` constructs a name-to-field map, checks missing required
fields, then walks every supplied field. It returns *all* problems:

```text
missing required field 'query'; unknown field 'limit';
field 'exact' must be BooleanValue, got JsonString
```

Returning a vector improves diagnostics and tests. It still does not authorize
execution. A syntactically valid `{"path":"../../secret"}` may match a string
schema but fail path-domain policy.

`IntegerValue` accepts a `JsonNumber` only when `BigDecimal.isWhole` is true.
Thus `2.0` is mathematically integral, while `2.5` is not. Decide whether that
is the wire contract you want before exposing a tool.

### 5. Keep lookup, validation, authorization, and execution ordered

`AgentRuntime.executeCall` implements the trust pipeline:

```text
toolsByName.get(call.name)
  -> definition.schema.validate(call.arguments)
  -> authorize(tool, call, step)
  -> invokeWithRetries(tool, call, step)
```

Unknown names never reach schema code. Invalid arguments never reach approval
or tool code. Denied calls never create a physical attempt. This ordering is an
observable security property, not merely an optimization.

## Reading the tests

Read `JsonSuite` in grammar order. The happy-path test establishes the complete
JSON value set; later tests isolate duplicate fields, commas, number grammar,
Unicode, and resource limits. A parser rejection test should assert the
relevant error category, not only `isLeft`, so the wrong rejection path cannot
pass accidentally.

In `AgentRuntimeSuite`, “schema errors are returned to the model without
invoking the tool” uses a counting fake tool. The important oracle is both the
`invalid_arguments` observation *and* an execution count of zero. The final
schema test verifies that missing, unknown, and type problems are accumulated
rather than failing after the first one.

## Debugging checklist

- If valid JSON leaves trailing text unreported, inspect `parseDocument`, not
  the individual value parser.
- If nested input crashes, verify that recursive object and array calls advance
  the depth counter and that limits are positive.
- If emoji fails, determine whether the input contains literal Unicode or
  escaped UTF-16 surrogate code units.
- If schema-valid input causes a domain error, keep both checks: schema answers
  “is it shaped correctly?”, while domain policy answers “is it allowed?”.
- If an invalid call invokes a tool, inspect the ordering in `executeCall` and
  add a side-effect counter to the regression test.
- If snapshots change between runs, check object insertion order and avoid
  constructing protocol JSON from an unordered map.

## Exercises

1. Add integer ranges and string-length constraints.
2. Design a read-file root policy resistant to symlink traversal.
3. Separate draft and send, requiring approval only for send.
4. Add output JSON size and depth limits.
5. Generate property/fuzz tests for the parser.

## Completion criteria

- Explain why a tool call is an untrusted proposal.
- Distinguish parse, schema, domain, and authorization validation.
- Explain why duplicate JSON fields are rejected.
- Explain why capabilities remain necessary after prompt injection.
- Return structured failures as observations.
- `JsonSuite` and tool-schema tests pass.

## Primary sources

- [MRKL Systems](https://arxiv.org/abs/2205.00445)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
