# Medical Supervisor Agent - Task Checklist

Scope: Build a LangGraph-based multi-agent medical system using Gemini, in iterative phases.

## Master Plan

- [x] Phase 1: LangGraph core structure (supervisor + agents + shared state), no database, no API
- [x] Phase 2: Add mock cardiovascular data flow for cardio agent
- [x] Phase 3: Validate hybrid collaboration (cardio + drug agent together)
- [x] Phase 4: Edge-case testing and routing robustness checks
- [x] Phase 5: Conversation context management + summarization every 20 turns
- [x] Phase 6: SQLite integration and persistence layer
- [x] Phase 7: Production FastAPI endpoint + telemetry (trace + token/cost tracking)

---

## Phase 1 - LangGraph Structure Only

Goal: Build only graph, states, and agents (supervisor, cardiovascular, sleep, drug, writing) using Gemini model wiring.

Deliverables:
- [x] Define project module layout for graph, nodes, prompts, and state
- [x] Define typed shared state (messages, routing metadata, intermediate outputs)
- [x] Implement supervisor node that decides next agent
- [ ] Implement agent nodes:
  - [x] Cardiovascular agent (logic scaffold only)
  - [x] Sleep agent
  - [x] Drug agent
  - [x] Writing agent
- [x] Add router/edges and termination conditions in LangGraph
- [x] Add loop guard to prevent infinite routing
- [x] Add minimal run script for local graph execution

Out of scope in Phase 1:
- [x] No database
- [x] No mock data fetch tool
- [x] No context summarization
- [x] No API endpoint
- [x] No telemetry persistence

Exit criteria:
- [x] Supervisor can route at least one user query to each agent type in controlled tests
- [x] Graph completes without loop/stall for standard prompts

---

## Phase 2 - Mock Cardiovascular Data

Goal: Add mock data retrieval tool usage inside cardiovascular agent.

Deliverables:
- [x] Create mock cardiovascular dataset source
- [x] Add cardio tool function to fetch structured mock records
- [x] Ensure cardiovascular agent can call tool and reason over returned structure
- [x] Include basic validation for missing/mock patient fields

Exit criteria:
- [x] Cardio agent responses include structured insights derived from tool output

---

## Phase 3 - Hybridness Test (Cardio + Drug)

Goal: Verify multi-agent cooperation when medication and cardiovascular reasoning are both needed.

Deliverables:
- [x] Create test prompts that require cardio + drug collaboration
- [x] Confirm supervisor invokes both agents when appropriate
- [x] Validate merged output consistency (no contradictory recommendations)

Exit criteria:
- [x] At least one test demonstrates correct sequential or iterative use of both agents

---

## Phase 4 - Edge Cases and Full Utilization

Goal: Stress routing logic and coverage across all agents.

Deliverables:
- [x] Add ambiguous/multi-intent prompts
- [x] Add safety and uncertainty handling cases
- [x] Check for skipped agents, dead ends, and loop risks
- [x] Validate writing agent usage for final formatting/synthesis

Exit criteria:
- [x] Supervisor handles edge prompts without crashing or infinite loops

---

## Phase 5 - Context and Summarization

Goal: Introduce conversation memory policy.

Deliverables:
- [x] Track conversation turns in state
- [x] Trigger summary function every 20 conversations
- [x] Replace raw old messages with summary context injection
- [x] Keep recent window + summary for prompt efficiency

Exit criteria:
- [x] Context remains coherent over long chats with bounded prompt growth

---

## Phase 6 - SQLite Database Integration

Goal: Move persistence to SQLite.

Deliverables:
- [x] Define SQLite schema (sessions, messages, traces, summaries)
- [x] Implement DB layer and repository helpers
- [x] Persist conversation state and summaries
- [x] Wire cardio data path to SQLite-backed storage (where applicable)

Exit criteria:
- [x] Session data and summaries survive process restarts

---

## Phase 7 - FastAPI + Agentic Health Telemetry

Goal: Production-ready API and observability.

Deliverables:
- [x] Implement `POST /v1/execute`
- [x] Define request/response schema
- [x] Add full execution trace logging:
  - [x] Agent called
  - [x] Input to agent/tool
  - [x] Tool outputs
  - [x] Route transitions
- [x] Add token usage estimation across supervisor loop
- [x] Add cost estimation using configurable pricing map
- [x] Return telemetry block in API response

Exit criteria:
- [x] API returns answer + trace + token/cost metrics in one response

---

## Evaluation Work (After Core Build)

- [ ] Run 5 complex stress-test queries that force supervisor to utilize all agents
- [ ] Write analysis in `EVAL.md`:
  - [ ] Agent Persona: problem solved + why these agents
  - [ ] Routing Logic: skips/loops analysis and causes
  - [ ] Optimization: improved supervisor prompt for ambiguous requests
