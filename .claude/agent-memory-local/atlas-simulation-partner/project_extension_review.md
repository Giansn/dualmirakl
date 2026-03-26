---
name: Extension Review - ml/parallel/stats Proposals
description: Assessment of proposed ml/, parallel/, stats/ extensions against existing codebase. Identified 60% duplication, key integration gaps, and recommended build order.
type: project
---

Reviewed proposed extensions (ml/, parallel/, stats/, sim_loop_v4.py) against existing codebase on 2026-03-26.

Key findings:
- stats/chaos.py and stats/emergence.py are 100% duplicates of dynamics.py (Lyapunov, TE, MI, Sobol, bifurcation, attractor basins, stochastic resonance all already exist)
- parallel/batch_inference.py duplicates orchestrator.py + asyncio.gather in Phase B
- ml/bandit.py already exists and is clean; integration gap is the reward signal definition
- ml/bayesian_beliefs.py (NumPyro) is genuinely new but JAX/GPU coexistence with vLLM needs verification
- ml/evolutionary.py conflicts with transitions.py registry system; should be outer loop, not inner
- sim_loop_v4.py should be incremental extension of v3, not a fork

**Why:** Prevents wasted effort rebuilding existing functionality and flags integration risks early.
**How to apply:** When this user proposes new modules, always check dynamics.py and sim_loop.py first -- these files contain ~1600 and ~2300 lines respectively with extensive analytics already built.
