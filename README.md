# Adaptive Risk Governor

**An adaptive decision-time risk system designed to defend against attackers that learn from the outcomes of previous interactions.**

Adaptive abuse is fundamentally different from static fraud.

A fixed fraud detector can identify known patterns. But when an attacker can observe whether a request was approved, challenged, or escalated, they can change their behavior and search for weaknesses in the defense.

This project studies that problem through a **closed-loop simulation of an adaptive attacker, a support agent, and a decision-time Risk Governor**.

> **Can a decision-time Risk Governor reduce abusive outcomes when the attacker itself adapts to the system's observable behavior — without needing to know whether the requester is human or AI?**

---

# The Core Idea

The system models three interacting decision-makers:

* **Agent B — Adaptive Attacker:** learns from observable outcomes and changes its strategy.
* **Agent A — Support Agent:** makes the underlying business decision for a request.
* **Risk Governor:** sits between the decision and its effective outcome and can intervene when the decision appears risky.

The Governor does **not** need to determine whether the requester is human or AI.

Instead, it evaluates the **observable behavior and risk of the interaction**.

This creates a feedback loop in which the attacker can learn from the defense itself.

```text
                    ┌─────────────────────┐
                    │   Incoming Request  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Observable Request  │
                    │ + Customer / Order  │
                    │       State         │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │          Risk Signals            │
              │                                 │
              │ Behavioral / Temporal           │
              │ Semantic / Network / GNN         │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Risk Fusion      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Governor     │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
          ALLOW          REQUEST EVIDENCE    HUMAN REVIEW
             │                 │                  │
             └─────────────────┴──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Effective Outcome   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Business Consequence│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Observable Feedback │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Adaptive Agent B   │
                    │   updates policy    │
                    └──────────┬──────────┘
                               │
                               ▼
                         Next Request
```

The important property is that **the loop does not end after one prediction**.

The attacker's behavior changes as a consequence of the defense, so the defense must be evaluated against an attacker that is capable of adapting.

---

# 1. Why This Problem Matters

Traditional fraud and risk systems often evaluate a request as an isolated event:

```text
Input → Model → Prediction → Metric
```

That assumption becomes weaker when an attacker can learn from repeated interactions.

For example:

```text
Request → APPROVED
             ↓
      Attacker observes outcome
             ↓
      Attacker changes strategy
             ↓
      New request
             ↓
      Different behavior
```

The attacker is therefore part of the system dynamics.

The objective of this project is not simply to build a classifier with a strong offline metric.

The objective is to evaluate whether a **decision-time control layer remains effective inside an adversarial feedback loop**.

---

# 2. System Components

## Agent A — Support Agent

Agent A represents the business-facing decision maker.

It receives the available request, customer, and order information and can make decisions such as:

```text
APPROVE
REQUEST_EVIDENCE
ESCALATE
DENY
```

The Governor does not replace Agent A.

Instead, it provides an additional safety boundary around the business decision.

---

## Agent B — Adaptive Attacker

Agent B represents a requester attempting to obtain abusive outcomes.

The important property is that Agent B is **adaptive**.

After each interaction, it receives feedback that is observable to the attacker and updates its policy.

Conceptually:

```text
Previous strategy
       ↓
Observed outcome
       ↓
Policy update
       ↓
New strategy
       ↓
Next request
```

This allows the system to test the Governor against a **moving target**, rather than a static collection of fraudulent examples.

The attacker is not assumed to follow one fixed attack pattern throughout the experiment.

---

## Risk Governor

The Governor is a **decision-time control layer**.

Its action space is intentionally constrained to:

```text
ALLOW_AGENT_A_DECISION
REQUEST_ADDITIONAL_EVIDENCE
ESCALATE_TO_HUMAN_REVIEW
```

Importantly, the Governor does **not** directly contain a `DENY` action.

A denial remains a possible **Support Agent decision**.

This keeps the responsibilities of the two components separate:

```text
Agent A  → Business decision

Governor → Safety intervention
```

The Governor therefore acts as a **policy enforcement layer**, rather than simply becoming another fraud classifier.

---

# 3. Risk Intelligence

The Governor combines multiple sources of evidence rather than relying on a single signal.

## Behavioral Signals

Behavioral features capture patterns such as:

* request velocity
* amount acceleration
* decision anomalies
* strategic behavior

These signals help identify behavior that becomes suspicious across repeated interactions.

## Temporal Signals

Temporal patterns capture how behavior evolves over time.

A request that appears normal in isolation can become suspicious when considered alongside the frequency, timing, or progression of previous interactions.

## Semantic Signals

The project uses **Sentence-Transformers with a MiniLM-based embedding model** to represent request/claim text semantically.

Rather than relying only on exact wording, the semantic pipeline can identify relationships between claims and changes in how a requester describes a situation.

These representations contribute to signals such as:

* semantic paraphrase similarity
* claim switching
* claim consistency
* semantic behavior changes

This allows the system to reason about changes in meaning even when the wording itself changes.

## Network Signals

Relationships between entities such as:

* customers
* devices
* addresses
* payment instruments

can reveal coordinated or shared behavior that may not be visible from an individual request.

## GNN Representation

A Graph Neural Network provides a learned representation of the underlying interaction/network structure.

The purpose is to combine complementary evidence:

```text
Behavioral evidence
        +
Temporal evidence
        +
Semantic evidence
        +
Network / GNN evidence
        ↓
    Risk Fusion
        ↓
      Governor
```

---

# 4. Closed-Loop Evaluation

The most important part of the project is the evaluation loop.

A conventional ML evaluation looks approximately like:

```text
Input → Model → Prediction → Metric
```

This project evaluates the complete interaction:

```text
Request
  ↓
Risk assessment
  ↓
Governor intervention
  ↓
Effective outcome
  ↓
Attacker observes outcome
  ↓
Attacker adapts
  ↓
New request
  ↓
Repeat
```

This makes the attacker part of the evaluation environment.

The key question becomes:

> **Does the Governor continue to defend the system after the attacker has had opportunities to learn from it?**

This distinction is important because a defense can perform well against a static dataset while becoming weaker once the adversary begins adapting to its behavior.

---

# 5. Engineering Problem Solving

Building the system exposed an important integration problem in the evaluation pipeline.

## The Problem: An Assumed Evaluation Interface

A downstream evaluation phase initially assumed that Phase 18 returned a particular structured result schema.

That assumption was incorrect.

Phase 18 actually returned a **flat result object**, and some population-level metrics expected by the downstream phase were not exposed by that interface.

This created a serious evaluation risk.

The downstream code could have:

* inferred missing values,
* parsed console output,
* or forced the previous phase into the expected schema.

Those approaches might have produced numbers, but they would not necessarily have represented measurements actually produced by the experiment.

## The Investigation

Instead of treating the expected schema as authoritative, the actual Phase 18 implementation and return values were inspected.

The evaluation boundary was treated as an interface that needed to be verified.

This revealed the mismatch between:

```text
Expected interface
        ≠
Actual interface
```

## The Solution

The downstream evaluation was changed to consume the **actual Python return values produced by Phase 18**.

Missing metrics were not fabricated.

Console output was not treated as a substitute for a structured experimental result.

The evaluation therefore became dependent on the real output contract of the experiment rather than an assumed one.

## Why This Matters

This was more than a debugging fix.

In an experimental system, an incorrect evaluation layer can make an otherwise valid experiment produce misleading conclusions.

The resulting principle is:

> **Evaluation code must follow the actual contract of the experiment, not an assumed contract.**

This is particularly important for an adaptive-defense system, because claims about whether the Governor works must be based on measurements that can be traced back to the actual simulation.

---

# 6. What This Demonstrated

The debugging process reinforced several engineering principles.

### Interface Validation

Each evaluation stage should consume the real output contract of the preceding stage.

### No Fabricated Metrics

If an experiment does not expose a metric, the correct response is to identify that limitation rather than reconstructing a number from unrelated output.

### Separation of Concerns

Simulation, risk estimation, intervention, attacker adaptation, and evaluation remain conceptually separate components.

### Tests as Architectural Checks

The project uses automated tests to catch regressions in behavior and interfaces rather than relying only on successful execution.

### Evaluation Integrity

A sophisticated model is not useful if the evaluation pipeline itself is making unsupported assumptions.

---

# 7. Project Architecture

At a high level, the complete system can be viewed as:

```text
                    ┌─────────────────────┐
                    │   Environment       │
                    │ Customer / Orders   │
                    │ Refunds / Network   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Agent B        │
                    │ Adaptive Attacker   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Agent A        │
                    │ Support Decision    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Risk Intelligence   │
                    │                     │
                    │ Behavioral          │
                    │ Temporal            │
                    │ Semantic + MiniLM   │
                    │ Network             │
                    │ GNN                 │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Risk Fusion      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Governor     │
                    └──────────┬──────────┘
                               │
                    Intervention / Allow
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Effective Outcome   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Economic / Business │
                    │    Consequence      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Observable Feedback │
                    └──────────┬──────────┘
                               │
                               └──────────► Agent B
```

---

# 8. Repository Structure

The repository currently follows this structure:

```text
adaptive-risk-governor/
│
├── app/                    # Interactive demonstration / UI
├── data/                   # Synthetic data and experiment data
├── src/                    # Core project implementation
├── tests/                  # Automated tests
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Files excluded from version control
```

Local development artifacts such as `.venv/` and `.pytest_cache/` are intentionally excluded from version control.

---

# 9. Technology Stack

* **Python**
* **React JS**
* **Machine Learning**
* **Graph Neural Networks**
* **Behavioral Risk Modeling**
* **Network Analysis**
* **Agent-based Simulation**
* **Adaptive Policy Learning**
* **Sentence-Transformers**
* **MiniLM**
* **Semantic Similarity Modeling**
* **Automated Testing**
* **Interactive Web Interface**

---

# 10. Key Design Principles

### 1. Defend Behavior, Not Identity

The Governor does not need to determine whether the requester is human or AI.

It evaluates observable behavior and risk.

### 2. Assume Adaptation

The attacker is allowed to learn from observable outcomes.

### 3. Intervene at Decision Time

The Governor sits directly on the path to the effective business outcome.

### 4. Preserve Decision Boundaries

The Support Agent and Governor have different responsibilities and action spaces.

### 5. Evaluate the System Dynamically

Performance is measured inside an interaction loop rather than only through static offline predictions.

### 6. Treat Evaluation as Engineering

Experiment interfaces, metrics, and data flow must be validated just as carefully as the model itself.

---

# 11. Running the Project

Clone the repository:

```bash
git clone https://github.com/<your-username>/adaptive-risk-governor.git
cd adaptive-risk-governor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository contains the application, source implementation, and evaluation components required to run the project.

---

# 12. Current Status

The implementation includes:

* Synthetic environment construction
* Human and adaptive-agent populations
* Support Agent decision pipeline
* Behavioral risk signals
* Temporal risk signals
* Semantic risk signals using MiniLM embeddings
* Network risk representation
* GNN-based representation
* Risk fusion
* Decision-time Risk Governor
* Adaptive attacker feedback loop
* Closed-loop evaluation
* Economic evaluation
* Robust ML evaluation
* Integration testing
* React based interactive demonstration interface

The system is designed to evaluate the **interaction between attack adaptation and defensive intervention**, rather than evaluating each component only in isolation.

---

# 13. Why This Approach

The central idea is simple:

> **A risk system should not only ask whether a request looks risky. It should ask whether its defense remains effective after the requester learns how the system behaves.**

That shifts the problem from static classification toward **adaptive risk management**.

The Risk Governor is therefore evaluated not just as a predictor, but as a **defensive control mechanism inside an adversarial feedback loop**.
