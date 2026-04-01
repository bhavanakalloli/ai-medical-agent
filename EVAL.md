# EVAL.md

## 1) Agent Persona

### Problem Solved
This system is a medical triage assistant that converts free-text user symptoms into a safe, structured response by coordinating specialized agents. It is not a diagnostic engine. It focuses on:

I wanted to work on a healthcare agent for a long time, and this internship opportunity was a good way to finally start building one in a structured, production-oriented way. I also want to explore this area further beyond this assignment.

- Sleep-related concerns
- Cardiovascular risk signals
- Medication safety and side-effect context
- Final patient-facing synthesis with safety guidance

### Why These Agents Were Chosen

- Supervisor: decides the next best specialist and controls loops/termination.
- Sleep agent: handles insomnia and sleep-pattern contributors.
- Cardiovascular agent: handles chest/BP/risk context grounded in mock cardio data.
- Drug agent: handles medication safety using symptom extraction plus Wikipedia lookup.
- Writing agent: produces concise final report sections (Summary, Key concerns, Suggested next steps, Safety note).

This split keeps each agent narrow, improves interpretability, and allows Supervisor-level policy controls.

## 2) Architecture and Delegation Check

### Is Delegation Real (not hard-coded chain)?
Yes. Delegation is dynamic and policy-guided.

- Supervisor chooses next_agent from state and context.
- Policy layer enforces missing specialist coverage for multi-intent queries.
- Repeated-specialist guard can force writing to avoid no-answer outcomes.
- Out-of-scope guardrail routes to writing for scope message, then finishes.

Graph flow is Supervisor -> Specialist -> Supervisor until Writing -> Finish.

### Shared State Used by Supervisor

- messages
- route_history
- loop_count
- execution_trace
- final_response
- conversation_summary / conversation_turn_count

## 3) Tool Usage Quality

### Tools Integrated

- cardio_mock_lookup (custom mock API-style tool)
- extract_symptoms_from_conversation (custom extraction tool)
- wikipedia_drug_lookup (Wikipedia-backed tool with fallback)

### Autonomous Tool Use
Cardiovascular and Drug workers attempt bind_tools first. If model/tool-calling is unavailable, they execute fallback tool logic and still append tool traces so the Supervisor can reason over outputs.

### Tool Interpretation Quality

- Cardio agent receives structured vitals/labs/history and reports risk-oriented signals.
- Drug agent combines extracted symptom/drug terms with Wikipedia snippet/fallback.
- Writing agent synthesizes specialist outputs and applies safety framing.

## 4) Trace Logging, Tokens, and Persistence

### Trace Logging
Execution trace captures per-step records:

- agent name
- input
- output
- note (including bind_tools_call markers and policy reasons)

### Token/Cost Tracking
Token usage is estimated per run and per trace event, with by_agent rollups and estimated_cost_usd returned by API.

### Persistence Robustness

- SQLite stores sessions, turns, and message history.
- Session context is loaded by session_id across restarts.
- Similar-query retrieval can reuse prior responses within a session.
- Memory summarization trims context every 20 turns for long conversations.

## 5) Stress Test Suite (6 Cases)

Below are six evaluation cases covering full routing, hybrid routing, skipped-agent routing, and ambiguity.

### Case 1: Full Multi-Intent, All Specialists Used
Query:
I have not slept for 3 nights, I feel dizzy and uneasy, my blood pressure seems high, and I started ibuprofen plus a new BP tablet yesterday.

Expected route:
sleep -> cardiovascular -> drug -> writing -> finish

Observed behavior:

- Sleep evaluates deprivation and behavioral contributors.
- Cardio evaluates BP/chest/dizziness signals using mock cardio context.
- Drug evaluates medication side-effect/interaction risk from extraction + Wikipedia.
- Writing returns structured final synthesis with safety note.

Outcome:
All specialist agents are used correctly with final synthesis.

### Case 2: Drug-Only Scenario (Agents Skipped)
Query:
I started amlodipine and ibuprofen together and now have nausea. Any side effects or interaction concerns?

Expected route:
drug -> writing -> finish

Observed behavior:

- Sleep skipped.
- Cardiovascular skipped unless explicit BP/chest risk terms are present.
- Drug tool chain runs and writing summarizes.

Outcome:
Correct selective routing, no unnecessary agents.

### Case 3: Cardiovascular-Only Scenario (Agents Skipped)
Query:
I have chest tightness and high blood pressure today. What immediate concerns should I watch?

Expected route:
cardiovascular -> writing -> finish

Observed behavior:

- Sleep skipped.
- Drug skipped unless medication terms are introduced.
- Cardio context is used and writing provides safety-focused summary.

Outcome:
Correct single-specialist routing with final writing output.

### Case 4: Sleep + Cardiovascular Hybrid
Query:
I have not slept for 2 days and I feel uneasy with possible BP fluctuation.

Expected route:
sleep -> cardiovascular -> writing -> finish

Observed behavior:

- Sleep path captures deprivation factors.
- Cardio path captures BP/uneasy risk framing.
- Drug skipped as no medication concern is present.

Outcome:
Correct hybrid routing without over-calling drug agent.

### Case 5: Ambiguous Query (Potential Agent Skip)
Query:
I feel off and not sure what is happening.

Expected route:
Model-guided route (often sleep first) -> writing -> finish

Observed behavior:

- Supervisor may choose one specialist then writing.
- Full specialist set is not forced due to low-signal ambiguity.

Outcome:
System avoids over-routing and still returns a coherent final answer.

### Case 6: Repeated Specialist Loop Risk (Recovered)
Query:
I have cough and bloating issues.

Expected route:
Specialist may repeat initially -> writing forced by policy -> finish

Observed behavior:

- Previous issue: repeated cardiovascular routing could end with weak/no final answer.
- Current fix: repeated-specialist guard forces writing before finish.

Outcome:
No-answer failure mode mitigated.

## 6) Routing Logic Analysis

### Did Supervisor Skip Agents?
Yes, by design, and this is desirable when intent is narrow.

- Drug-only prompts skip sleep/cardio.
- Cardio-only prompts skip sleep/drug.
- Ambiguous low-signal prompts may use one specialist then writing.

### Did Supervisor Get Stuck in Loops?
Historically, yes in some low-signal symptom prompts.

Mitigations now in place:

- MAX_SUPERVISOR_LOOPS guard
- Repeated-specialist detection that forces writing
- Policy-based finish after writing

### Why Loops Happened

- Overlap in symptom language (for example dizziness/uneasy)
- Sparse signal for unresolved pending domains
- LLM routing prior to explicit anti-repeat policy

## 7) Optimization Proposal for Ambiguous Requests

### Prompt Changes Recommended for Supervisor

1. Add confidence-aware routing fields:
- next_agent
- confidence_score (0-1)
- missing_information

2. Add explicit ambiguity strategy:
- If confidence < 0.6 and at least one specialist already ran, route to writing with uncertainty statement.

3. Add anti-repeat hard rule:
- Never route same specialist more than twice consecutively.

4. Add triage risk precedence:
- chest pain/breathing issues/neurologic red flags -> cardiovascular first.

5. Add missing-slot completion check:
- After each specialist, explicitly list unresolved slots (sleep/cardio/drug). If none unresolved, route to writing.

### Example Supervisor Prompt Patch (Conceptual)

- If the user intent is ambiguous, choose one high-yield specialist.
- If the same specialist was selected twice and uncertainty remains, route to writing now.
- Writing must include: what is known, what is uncertain, and safety escalation criteria.

## 8) Final Rubric Readout

- Architectural Correctness: Pass (dynamic supervisor delegation with policy constraints)
- Feature Robustness: Pass (session persistence, retrieval, memory summarization, restart-safe SQLite)
- Tool Usage: Pass (bind_tools + fallback, tool outputs traced and consumed)
- Critical Thinking: Pass with improvements documented (loop causes, ambiguity handling, prompt optimization)

## 9) Notes and Limitations

- Token/cost are heuristic estimates, not provider-billed exact values.
- Wikipedia tool may fallback when live calls fail; this is intentional for reliability.
- Medical outputs are non-diagnostic and include safety escalation language.