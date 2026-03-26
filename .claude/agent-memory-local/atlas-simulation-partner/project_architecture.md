---
name: DualMirakl Architecture
description: Multi-agent simulation framework with dual GPU inference, stratified tick loop, pluggable scoring, and FLAME population dynamics. Key design decisions and identified failure modes.
type: project
---

DualMirakl is a general-purpose multi-agent simulation framework running on dual Blackwell GPUs.

**Architecture:**
- GPU 0 (Authority): Mistral Nemo 12B FP8, port 8000 -- environment + observers
- GPU 1 (Swarm): Nemotron Nano 30B NVFP4, port 8001 -- participants
- CPU Gateway: e5-small-v2 embeddings + FastAPI proxy, port 9000
- Optional GPU 2: FLAME population dynamics

**Tick Loop (6 phases):** Pre(persona) -> A(batch stimuli) -> B(concurrent responses) -> C(embedding+score) -> D(observer analysis+intervention) -> E(event stream+graph) -> F(FLAME optional)

**Scoring:** EMA or Logistic mode. Heterogeneous agents with susceptibility~Beta(2,3) and resilience~Beta(2,5). Coupling kappa for peer influence.

**Key design tensions identified (2026-03-26 review):**
- Score signal derived from cosine similarity to engagement anchor phrases -- measures lexical surface, not semantic content
- EMA alpha=0.15 with only 12 ticks means agents can only traverse ~60% of score space
- Intervention extraction via cosine codebook matching is fragile
- No coupling (kappa=0.0) in default config makes information flow metrics misleading

**Why:** Understanding this architecture is essential for all future review work on this project.
**How to apply:** Ground all analysis in actual code behavior, not documentation claims. The scoring model has known limitations worth revisiting.
