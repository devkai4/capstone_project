# SPEC v0 – Ticket Triage Agent (Option A)

## Problem
Support engineers spend time routing tickets; misroutes delay answers.

## Users
A support engineer reviews the agent's suggestion in a queue.

## Inputs and outputs
- In: ticket text (subject + body)
- Out: JSON {queue, priority, type, answer_or_escalate, citations}

## Success metrics (targets, revisit in Week 3+)
- Queue accuracy >= 80%
- Cost per ticket <= $0.01
- p95 latency <= 8 s
- Escalation rate 10–25%
- 0 duplicate actions

## Non-goals
- No auto-reply to customers
- No access to real customer data

## Data
- HF customer-support-tickets (CC BY-NC 4.0), English only
- 30 tickets held out for evals

## Risks and unknowns
- Labels may be noisy → hand-check 20 tickets in Week 3

## Phases
| Phase | Weeks | Dates (approx.) |
|---|---|---|
| 1. Foundations | 0–2 | Oct 13 – Nov 2 |
| 2. Model core | 3–4 | Nov 3 – Nov 16 |
| 3. Agent | 5–6 | Nov 17 – Nov 30 |
| 4. Architecture choice | 7 | Dec 1 – Dec 7 |
| 5. Production hardening | 8–9 | Dec 8 – Dec 21 |
| 6. Secure and ship | 10 | Dec 22 – Dec 28 |