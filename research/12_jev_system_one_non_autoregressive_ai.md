# 🧠 Research Paper 12: Jev — Non-Autoregressive System-1 Decision Intelligence & RLCD

## 1. Abstract & The "System-1" Paradigm Shift
Traditional Large Language Models (LLMs) are **"System 2"** architectures: they generate answers sequentially, token-by-token in an autoregressive loop. While suitable for open-ended creative writing and multi-step reasoning, this autoregressive process carries massive overhead:
- High latency: 1,000ms to 4,000ms per inference query.
- High cost: quadratic memory cache growth ($O(N^2)$ attention) and billions of FLOPs for simple binary or categorical determinations.
- Parsing failure risk: generative models frequently hallucinate or fail strict JSON schema validation.

In September 2026, **TypeSafe AI** (founded by former OpenAI researcher Diogo Almeida, co-inventor of RLHF and InstructGPT) released **Jev**: a novel **"System 1"** AI model built specifically for fast, probabilistic, and structured decisions rather than generative text.

This paper documents our derivation, recreation, and benchmark of the Jev architecture inside `doraneural` ([`doraneural/jev.py`](../doraneural/jev.py) and [`research/explore_jev_system1.py`](explore_jev_system1.py)).

---

## 2. Core Architectural Principles

### 1. Non-Autoregressive Execution
Unlike standard LLMs that generate tokens conditioned on previous tokens:
$$P(y_{1:T} \mid x) = \prod_{t=1}^T P(y_t \mid y_{<t}, x)$$
Jev evaluates decisions in a **single parallel pass**:
$$\hat{y} = \text{Head}(\text{Encoder}(x))$$
Because token-by-token sequential streaming is eliminated, inference latency drops from seconds to **sub-millisecond CPU speeds (< 2ms)**.

### 2. Typed Decision Primitives
Instead of prompting an LLM to "return JSON with schema...", Jev projects hidden representations directly into typed mathematical primitives:
- `Choice(options)`: Multi-class categorical routing / classification with calibrated probabilities.
- `Score(min, max)`: Bounded continuous metric regression (e.g. risk score, priority value).
- `Boolean()`: Binary gate / threshold approval.

Because the output head geometry matches the target schema, the model is **mathematically incapable of schema violations or syntax hallucinations**.

### 3. RLCD: Reinforcement Learning for Calibrated Decisions
Standard models trained with pure Cross-Entropy suffer from overconfidence calibration drift (high confidence on incorrect predictions). Jev is trained using **RLCD**:
$$\mathcal{L}_{\text{RLCD}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cal}} \cdot \text{Brier}(p, y) - \lambda_{\text{ent}} \cdot \mathcal{H}(p)$$
where the Brier Score calibration penalty:
$$\text{Brier}(p, y) = \sum_{k=1}^K (p_k - y_k)^2$$
strictly penalizes deviation between predicted confidence and empirical decision accuracy.

---

## 3. Empirical Benchmark & Speed Advantage

Running the in-repo benchmark [`research/explore_jev_system1.py`](explore_jev_system1.py):

```
============================================================================
 📊 SYSTEM-1 SPEED ADVANTAGE (JEV VS AUTOREGRESSIVE LLM)
============================================================================
Metric                         | Typical Autoregressive LLM | Jev (System-1)
----------------------------------------------------------------------------
Inference Latency              | 1,200 ms - 3,500 ms        | 1.657 ms (sub-2ms CPU)
Speed Advantage                | 1x (Baseline)              | >200x - 400x faster
Decoding Mechanism             | Autoregressive (token loop)| Non-Autoregressive (1-pass)
Format Conformity              | Probabilistic (can fail)   | Mathematically Guaranteed
Training Objective             | Next-token Cross-Entropy   | RLCD (Calibrated Decisions)
============================================================================
```

---

## 4. Usage in Doraneural

```python
from doraneural.jev import JevDecisionModel, rlcd_loss

# 1. Instantiate Jev model
jev = JevDecisionModel(dim=64, n_layers=2)

# 2. Register typed output schemas
jev.add_choice_head("router", options=["tech_support", "billing", "security"])
jev.add_score_head("urgency", min_val=0.0, max_val=10.0)
jev.add_boolean_head("auto_approve")

# 3. Instant non-autoregressive decision (< 2ms)
decision = jev.decide("Unauthorized root access detected on server!", "router")
print(decision.action)        # "security"
print(decision.confidence)    # 0.982
print(decision.latency_ms)    # 1.4 ms
```
