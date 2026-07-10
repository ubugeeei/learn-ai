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
