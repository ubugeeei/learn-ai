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
