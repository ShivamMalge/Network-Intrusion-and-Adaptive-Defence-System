---
trigger: always_on
---

# Rules & Architectural Invariants
Multi-Agent Network Intrusion Simulation & Adaptive Defense System

Version: 2.0  
Status: Architecture Frozen for Phase 1 Implementation  

---

# 1. Formal System Definition (Research-Grade)

## 1.1 Game-Theoretic Model

The system is formally defined as a Partially Observable Stochastic Markov Game:

G = (S, A_att, A_def, T, Ω_att, Ω_def, R_att, R_def, γ)

Where:

S          = Global state space  
A_att      = Attacker action space  
A_def      = Defender action space  
T          = State transition kernel  
Ω_att      = Attacker observation function  
Ω_def      = Defender observation function  
R_att      = Attacker reward function  
R_def      = Defender reward function  
γ          = Discount factor  

---

## 1.2 Global State Definition

Let:

S_t = (G_t, V_t, P_t, B_t, E_t)

Where:

G_t = (N_t, E_t) → Network graph at time t  
V_t = Vulnerability assignments (known + hidden)  
P_t = Privilege distribution across nodes  
B_t = Defender budget state  
E_t = Scheduled event queue  

State is fully observable ONLY to the EnvironmentEngine.

Agents never access S_t directly.

---

## 1.3 Transition Function

T : S × A_att × A_def → Δ(S)

State transition is stochastic due to:

- Exploit success probability
- IDS probabilistic detection
- Zero-day revelation events
- Random topology events

All randomness must originate from a centralized seeded RNG.

---

## 1.4 Observation Functions

Ω_att(S_t) → O_att_t  
Ω_def(S_t) → O_def_t  

### Attacker Observation Includes:
- Nodes scanned
- Nodes compromised
- Discovered vulnerabilities
- Local neighborhood topology

### Defender Observation Includes:
- Known topology
- Patch status
- Honeypot triggers
- IDS alerts

Defender must NOT observe attacker position unless detection event occurs.

Observation operators must be deterministic projections of S_t.

---

## 1.5 Reward Formalization

Rewards must be defined as measurable functions over (S_t, A_t, S_{t+1}).

R_att(S_t, A_t, S_{t+1}) =
    + α1 * critical_asset_compromise
    + α2 * lateral_movement
    - α3 * detection_event
    - α4 * failed_exploit

R_def(S_t, A_t, S_{t+1}) =
    + β1 * asset_survival
    + β2 * successful_detection
    - β3 * node_compromise
    - β4 * patch_cost

Reward functions must be:

- Deterministic under fixed seed
- Stateless (no hidden memory)
- Independent of agent internal model

---

# 2. State Invariants (Non-Negotiable)

The following must always hold:

1. Graph consistency:
   - No orphan edges
   - No duplicate node IDs
   - All edge endpoints must exist

2. Privilege consistency:
   - Privilege cannot exceed node’s maximum privilege level
   - Privilege downgrade must be monotonic

3. Budget consistency:
   - Defender budget cannot be negative

4. Zero-day secrecy:
   - Hidden vulnerabilities cannot appear in defender observation unless revealed

Violations must raise system-level exceptions.

---

# 3. Deterministic Step Pipeline

Each step must execute in this exact order:

1. Process scheduled events
2. Validate attacker action
3. Validate defender action
4. Apply attacker action
5. Apply defender action
6. Resolve exploit outcomes
7. Run detection logic
8. Update topology state
9. Compute rewards
10. Generate observations
11. Log step

Order is invariant.

---

# 4. Action Space Constraints

## 4.1 Atomicity

Each action must:
- Be fully resolved within a single step
- Not produce partial state mutations

## 4.2 Validation Requirements

Before application, each action must satisfy:

- Privilege requirements
- Topology feasibility
- Budget availability
- Node existence

Invalid actions:
- Produce penalty
- Do not mutate state

---

# 5. Reproducibility & Experimental Integrity

- Global RNG instance required
- Seed must be externally configurable
- Entire episode must be reproducible with same seed
- Logs must allow full replay reconstruction

---

# 6. Logging & Audit Guarantees

Each step must log:

- Step index
- State hash
- Attacker action
- Defender action
- Exploit success/failure
- Detection outcome
- Reward values
- Topology changes

Replay must reconstruct S_t exactly.

---

# 7. Enterprise-Grade Engineering Rules

## 7.1 Strict State Ownership

Only EnvironmentEngine may mutate global state.

Modules must operate on:
- Copies
- Read-only views
- Interface contracts

---

## 7.2 Interface Contracts

Every module must expose:

- Explicit input types
- Explicit output types
- No side effects outside declared scope

Implicit state mutation is forbidden.

---

## 7.3 Performance Guarantees

- Graph operations: target O(E log V)
- No full graph deep copy per step
- Observations must be projections, not clones
- Logging must be asynchronous or buffered

---

## 7.4 Scalability Requirements

Architecture must support:

- Multiple attacker agents
- Graph size scaling to O(10^4) nodes
- Plug-in RL algorithms (DQL, PPO, etc.)
- Optional GNN-based policy integration

No module may assume single-agent constraint.

---

## 7.5 Extensibility Constraints

Allowed future modifications:

- Replace learning algorithm
- Add stochastic topology generator
- Introduce new defense mechanisms

Forbidden modifications:

- Breaking step order
- Granting omniscient access to agents
- Embedding policy logic inside environment

---

# 8. Mathematical Integrity Constraints

System must remain:

- Partially Observable
- Multi-Agent
- Stochastic
- Adversarial

Any change violating these properties requires architectural version increment.

---

# Architecture Freeze Declaration

This document defines invariant system properties.

No implementation may violate these constraints without:
- Version increment
- Architectural review
- Updated formal specification