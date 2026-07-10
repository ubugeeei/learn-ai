# 33 — A provider-neutral model boundary

## What you will build

Define a `LanguageModel` protocol that separates vendor SDK and wire formats
from agent logic. A model receives typed history and granted tool definitions,
then returns a final answer or tool calls.

Source: `src/main/scala/learnai/agent/Protocol.scala`.

## Model and agent are different components

A language model predicts tokens. An agent is a host program that interprets
model output, executes tools, and returns observations to the model.

```text
agent runtime
  -> ModelRequest
  -> provider adapter
  -> local or remote model
  <- ModelDecision
```

The model has no inherent filesystem or network permission. External effects
exist only through capabilities supplied by the host.

## Typed conversation history

History distinguishes:

- `TextMessage(System|User|Assistant, content)`;
- `AssistantToolCalls(calls)`;
- `ToolObservation(callId, toolName, outcome)`.

Every tool result references its call ID and name. Multiple calls do not depend
on positional ordering alone.

## Model decisions

Provider adapters normalize responses into:

```scala
FinalAnswer(text, usage)
RequestTools(calls, usage)
```

Some APIs can mix text and calls. The adapter must define that mapping rather
than spreading provider-specific conditions through the runtime.

## Errors are typed data

Network, authentication, rate-limit, and response-parse failures become:

```scala
ModelError(code, message, retryable)
```

A tool failure can be returned as an observation so the model can choose a
recovery. A model failure prevents the next decision, so this runtime ends the
run. A future retry policy must define count, backoff, jitter, and deadline.

## Usage accounting

Each decision reports input and output token counts. The run accumulates them
for cost, latency, context-limit, and runaway-loop analysis. Provider-specific
cached or reasoning-token categories may be exposed as additional metrics.

## Test with a fake model

`ScriptedModel` avoids remote calls in unit tests:

- decisions are fixed;
- received history and tool definitions are inspectable;
- invalid calls are easy to generate;
- tests do not depend on network, rate limits, or model randomness.

Runtime correctness and model quality are separate evaluation layers.

## Adapter responsibilities

A real provider adapter handles:

- internal history to provider messages;
- tool schemas to the provider-supported JSON Schema subset;
- authentication and HTTP timeout;
- streaming chunk assembly;
- error and usage mapping;
- response size/depth limits and strict parsing.

Secrets flow from a secret provider directly to transport headers, never into
prompts, history, or ordinary logs.

## Implementation walkthrough

Open `Protocol.scala` and read it from top to bottom. The declarations are in
the same order as data moves through one agent step.

### 1. Represent history as a sum type

`ConversationItem` is a `sealed trait`. Its three implementations represent
three events that must not be confused:

```scala
TextMessage(MessageRole.User, "What is in the index?")
AssistantToolCalls(Vector(ToolCall("call-1", "search_documents", arguments)))
ToolObservation("call-1", "search_documents", ToolSucceeded(result))
```

Because the hierarchy is sealed, a provider adapter can pattern-match on every
case and the compiler warns when a new case is not handled. A plain
`Vector[String]` would lose the role, the call-to-result relationship, and the
difference between data and control messages.

The call ID is the join key. Suppose a model requests two tools in one turn and
they finish in the opposite order. Positional matching could attach the wrong
result. `callId` lets the runtime preserve the logical relationship regardless
of execution order.

### 2. Separate a proposal from an outcome

`ToolCall` contains model-controlled data: a name and JSON arguments.
`ToolObservation` contains host-controlled facts about what happened. The model
may propose `delete_file`, but only the host can create a successful
observation for it. This distinction becomes a security boundary in Chapters
34 and 38.

`ToolOutcome` is another sum type:

```scala
ToolSucceeded(JsonString("result"))
ToolFailed(ToolError("approval_denied", "...", retryable = false))
```

Failures are ordinary values rather than thrown exceptions. That makes them
serializable, testable, and safe to return to a model after redaction.

### 3. Expose only model-visible tool metadata

`ToolDefinition` contains a name, description, schema, and effect class. A
`ModelRequest` receives only definitions, never executable `Tool` objects. The
provider can therefore advertise capabilities without receiving Scala
references that perform the operations.

The host keeps the executable implementation separately:

```scala
trait Tool:
  def definition: ToolDefinition
  def execute(arguments: JsonObject, context: ToolContext)
      : Either[ToolError, JsonValue]
```

`ToolContext` is created by the runtime. A model cannot change the agent step
or attempt number by adding fields to its JSON.

### 4. Normalize one model turn

`LanguageModel.complete` has one provider-neutral signature:

```scala
def complete(request: ModelRequest): Either[ModelError, ModelDecision]
```

The outer `Either` answers whether a decision could be obtained. The inner
`ModelDecision` answers what the model decided. These are different axes: an
HTTP timeout is `Left(ModelError(...))`; a valid request for a failing tool is
still `Right(RequestTools(...))`.

`ModelUsage.+` performs component-wise addition. If step 0 reports `(120, 15)`
tokens and step 1 reports `(180, 25)`, the run total is `(300, 40)`, and
`totalTokens` is `340`. Keeping input and output separate matters because
providers often price them differently.

### 5. Implement a real adapter outside the protocol

A provider adapter should follow this order:

1. convert typed history and tool definitions into the provider request;
2. send it with a transport-level deadline;
3. parse the response under size and nesting limits;
4. map provider stop reasons into `FinalAnswer` or `RequestTools`;
5. map token counters into `ModelUsage`;
6. map failures into stable `ModelError` codes.

Do not add provider response classes to `AgentRuntime`. If switching providers
requires editing the loop, the boundary has leaked.

## Reading the tests

`AgentRuntimeSuite` defines scripted models whose decisions are fixed in
advance. Start with “a final model answer completes without executing tools.”
It is the smallest protocol trace: one request, one decision, one terminal
event. Then read “a valid tool call becomes an observation for the next model
step.” That test inspects the second request and proves that the typed call and
observation survived the round trip.

When adding an adapter, create a contract suite that feeds recorded provider
payloads into the adapter and asserts the resulting `ModelDecision`. Keep those
tests separate from runtime tests so a wire-format change cannot be mistaken
for an agent-loop regression.

## Debugging checklist

- If a tool result reaches the wrong call, compare `callId` values rather than
  vector positions.
- If history cannot be serialized, pattern-match every `ConversationItem` and
  enable exhaustive-match warnings.
- If a model failure appears as a tool failure, inspect which side of
  `Either[ModelError, ModelDecision]` the adapter returned.
- If token totals are wrong, log per-decision `ModelUsage` and add them by
  component; do not infer usage from string length.
- If provider-specific conditions appear in `AgentRuntime`, move them back into
  the adapter and add a protocol-level test.

## Exercises

1. Draw the complete history sent on a second model step.
2. Design streaming events and final decision assembly.
3. Design bounded retry for rate-limit errors.
4. Write one contract suite shared by two provider adapters.

## Completion criteria

- Separate model and agent responsibilities.
- Explain text/call/observation correspondence.
- Explain why provider details live in adapters.
- Write deterministic tests with a fake model.
- Verify usage and terminal model errors appear in the trace.
