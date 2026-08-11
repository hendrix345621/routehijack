# Reasoning ("thinking") models: technical incompatibilities with boundary-token analysis

A neutral survey of why analyses that localize a model's behavioral decision to the **first
generated token** break on reasoning models, and what is technically required to make them
work in thinking mode. Framed around DeepSeek-V4-class models, which expose explicit
reasoning modes; the issues generalize to any chain-of-thought model. Concrete details are
taken from the DeepSeek-V4 paper (reasoning modes, `<think>…</think>` protocol, interleaved
thinking) and released configuration.

## Background: the assumption that breaks

A large class of interpretability and behavioral methods assumes the **first generated
token** — call it the boundary token `t*` — is where the model commits to a behavior. They
localize state, apply objectives, and measure quantities *at that position*. Reasoning
models violate the assumption directly: with chain-of-thought enabled, `t*` is the start of
a **thinking preamble**, and the behavioral decision is only expressed much later, after the
`</think>` delimiter. Everything anchored to `t*` then measures the wrong position.

DeepSeek-V4 makes this explicit. It ships three reasoning modes — **Non-think**, **Think
High**, **Think Max** — demarcated by `<think>` / `</think>` tokens, where Non-think emits
`</think>` immediately (empty thinking) and the Think modes emit a variable-length reasoning
trace first. Think Max additionally **injects a system-prompt instruction** that lengthens
and restructures the trace. So the decision-relevant token position is not fixed: it moves
by hundreds to thousands of tokens depending on mode and prompt.

---

## 1. The decision-relevant token position is not the first generated token

**Blocker.** Under chain-of-thought, the model's committing token lives at the **answer
onset (post-`</think>`)**, deep in the generation, not at `t*`. Any method that inspects
router state, hidden states, or output distribution at the first generated token inspects
the *thinking preamble*, which does not carry the decision.

**What is needed.**
- **Re-anchoring:** detect the `</think>` delimiter in the generated stream and relocate the
  analysis position to the first answer token after it. This requires generating (or being
  given) the thinking trace first, then locating the boundary — it is no longer a
  single-forward, zero-generation operation.
- Robust delimiter handling for models/templates that emit malformed, missing, or nested
  `</think>` tags.

## 2. Per-token localization counts thinking tokens as if they were answer tokens

**Blocker.** Methods that aggregate per-token state to localize *where/which* components
drive a behavior (e.g. averaging router or activation state over generated positions) will
pool the long thinking trace into the estimate. The thinking tokens dominate by count and
dilute or corrupt the signal from the (few) answer tokens that actually express the
decision.

**What is needed.**
- A **segmentation mask** that partitions the generation into thinking vs answer spans
  (via the `<think>…</think>` boundaries) and restricts localization to answer-span tokens.
- A defined policy for whether thinking tokens are ever included, held constant across runs
  so results are comparable.

## 3. Decision detectors misfire on the thinking preamble

**Blocker.** Any classifier or heuristic that inspects the early output to categorize the
model's behavior will read the thinking preamble, not the answer. A trace that opens with a
neutral deliberation ("Here's how I'll approach this…") can be mis-scored, and a truncated
generation that never reaches `</think>` has no answer to score at all — systematically
biasing any metric computed from early tokens.

**What is needed.**
- Detectors that run on the **answer span only**, after `</think>`.
- A generation budget large enough to reach the answer under the active mode (Think Max
  traces can be very long), and explicit handling of truncated-before-answer cases rather
  than scoring the preamble as if it were the answer.

## 4. Variable trace length makes per-candidate evaluation expensive

**Blocker.** Boundary-token methods are cheap because they need a single forward pass — no
generation. In thinking mode, reaching the decision position requires **rolling out the
thinking trace**, whose length varies per prompt and per mode. If a method evaluates many
candidates (prompts, interventions, configurations), a full per-candidate rollout multiplies
cost by the trace length and makes the analysis intractable at scale.

**What is needed.**
- A **frozen-prefix approximation:** generate one clean thinking trace, freeze it, and
  evaluate at the answer onset conditioned on that fixed prefix — no per-candidate rollout.
  Cheap, but assumes the trace itself is insensitive to the manipulation under study.
- A **faithful rollout** path (regenerate thinking per candidate) for cases where the trace
  is not insensitive — far more expensive, and the correctness reference for the frozen-prefix
  approximation.
- A validation step measuring how much the two paths disagree, so the cheap path is only
  trusted where fidelity is established.

## 5. Templates and modes silently change what is generated

**Blocker.** The reasoning mode is controlled by chat-template flags and, for the strongest
mode, an injected system prompt. Some custom `trust_remote_code` templates ignore a
`enable_thinking`-style keyword and require an in-prompt switch instead. If the intended mode
is not actually applied, the generation silently contains (or omits) a thinking trace and
every position-dependent measurement is invalid without any error being raised.

**What is needed.**
- A post-hoc **sanity check** that inspects the actual generation for the presence/absence
  of a thinking preamble and confirms it matches the requested mode.
- Per-model handling of the template/switch mechanism (keyword flag vs in-prompt token),
  and a record of which mode each run actually evaluated.

## 6. Interleaved / retained thinking across turns changes the context

**Blocker.** DeepSeek-V4 retains reasoning traces across rounds in tool-calling scenarios
(the full thinking history persists across user-message boundaries), while discarding them
on a new user message in general conversation. Whether prior thinking is in-context changes
the state at the decision position from turn to turn, so a single-turn analysis does not
describe multi-turn behavior, and the context-management path (tool-calling vs conversational)
determines what the model conditions on.

**What is needed.**
- Explicit control of the context-management regime (retain vs discard) and a note of which
  one a given measurement assumes.
- Multi-turn evaluation for any claim about multi-turn behavior, since retained thinking
  makes each turn's decision position depend on the accumulated trace.

## 7. Thinking mode may itself change the behavior under study (open question)

**Blocker.** Beyond the mechanical relocation of the decision, deliberating over many tokens
may change *what* the model decides relative to a snap decision at `t*`. A boundary-token
analysis of a reasoning model therefore characterizes at best its **non-thinking mode**,
which is a different operating point — not a measurement of full reasoning-mode behavior.
Whether reasoning changes robustness or the behavior itself is not settled.

**What is needed.**
- A deliberate scoping statement: whether the study targets non-thinking mode (boundary-token
  consistent, cheaper) or thinking mode (requires items 1–4), and no conflation of the two.
- If the question is whether reasoning *itself* alters the behavior, a matched comparison of
  the same model across modes — a research result in its own right, not a configuration flag.

---

## Summary of the critical path
The mechanical blockers (items 1–3, 5) are addressable with **`<think>`/`</think>`-aware
re-anchoring, answer-span masking, and a mode sanity check** — necessary and sufficient to
make a boundary-token method *correct* in thinking mode. The hard part is **cost** (item 4):
reaching the decision position requires rolling out variable-length traces, so a
frozen-prefix approximation with a validated fidelity check is the practical enabler for any
at-scale analysis. Items 6–7 (interleaved thinking, and whether reasoning changes the
behavior) are genuine research extensions rather than engineering fixes. The cleanest,
lowest-cost scoping is to evaluate the **non-thinking mode** explicitly and treat thinking-mode
analysis as a separate, rollout-based track.
