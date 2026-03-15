"""
causal_model.py — Bayesian causal DAG for media addiction simulation.

Theoretical framework: Griffiths (2005) six-component addiction model +
DSM-5 IGD criteria + Bergen Social Media Addiction Scale (Andreassen 2012).

The DAG encodes the causal theory BEFORE seeing simulation results.
This is methodologically important: the structure should be committed
to based on theory, not data-dredged from simulation outputs.

Usage:
    from simulation.causal_model import sample_agent_params, DAG_EDGES

    # Sample heterogeneous agent parameters from the prior
    params = sample_agent_params(n_agents=10, seed=42)

    # Inspect the causal structure
    for parent, child in DAG_EDGES:
        print(f"  {parent} -> {child}")

Future integration:
    1. Replace sim_loop's Beta(2,3)/Beta(2,5) with posteriors from this model
    2. Fit conditional probability tables from qualitative interview data
    3. Use SBI to calibrate branching parameters against simulation outputs
    4. When FLAME GPU is ready, export posteriors as transition function params
"""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    import jax
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL DAG — Griffiths (2005) + DSM-5 IGD
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each node is a latent construct. Edges encode theoretical causal pathways.
# Edge weights will be learned from data via NumPyro posterior inference.
#
#   peer_influence ──────┐
#                        ▼
#   platform_algorithm → usage_frequency → salience (Griffiths 1)
#                                              │
#                                              ▼
#                                          conflict (Griffiths 4)
#                                              │
#                                              ▼
#                                     mood_modification (Griffiths 2)
#                                              │
#                                              ▼
#                                         tolerance (Griffiths 3)
#                                              │
#                                              ▼
#                                        withdrawal (Griffiths 5)
#                                              │
#                                              ▼
#                                          relapse (Griffiths 6)
#
#   self_regulation ──→ [dampens conflict, mood_modification, tolerance]
#
# Feedback loop: relapse → usage_frequency (re-entry)
#
# ═══════════════════════════════════════════════════════════════════════════════

DAG_EDGES: list[tuple[str, str]] = [
    # Exogenous → usage
    ("peer_influence",      "usage_frequency"),
    ("platform_algorithm",  "usage_frequency"),

    # Griffiths (2005) six-component addiction cascade
    ("usage_frequency",     "salience"),          # criterion 1: preoccupation
    ("salience",            "conflict"),           # criterion 4: interpersonal/intrapersonal
    ("conflict",            "mood_modification"),  # criterion 2: using to cope
    ("mood_modification",   "tolerance"),          # criterion 3: needing more
    ("tolerance",           "withdrawal"),         # criterion 5: discomfort when stopping
    ("withdrawal",          "relapse"),            # criterion 6: returning after quit attempt

    # Feedback loop (relapse drives re-engagement)
    ("relapse",             "usage_frequency"),

    # Protective factor (dampens multiple nodes)
    ("self_regulation",     "conflict"),
    ("self_regulation",     "mood_modification"),
    ("self_regulation",     "tolerance"),
]

DAG_NODES: list[str] = [
    "peer_influence",
    "platform_algorithm",
    "usage_frequency",
    "salience",
    "conflict",
    "mood_modification",
    "tolerance",
    "withdrawal",
    "relapse",
    "self_regulation",
]

# Maps DAG nodes to sim_loop agent parameters
NODE_TO_PARAM: dict[str, str] = {
    "usage_frequency": "susceptibility",   # high usage → high susceptibility
    "self_regulation": "resilience",       # high self-regulation → high resilience
}


# ═══════════════════════════════════════════════════════════════════════════════
# PRIOR DISTRIBUTIONS — used before empirical data is available
# ═══════════════════════════════════════════════════════════════════════════════

# Beta distribution parameters for each construct.
# These are theoretical priors — update from qualitative data when available.
PRIORS: dict[str, tuple[float, float]] = {
    # Exogenous factors (population-level variation)
    "peer_influence":      (2.0, 3.0),   # mode ≈ 0.33 — moderate peer effects
    "platform_algorithm":  (3.0, 2.0),   # mode ≈ 0.67 — algorithms are effective
    "self_regulation":     (2.0, 5.0),   # mode ≈ 0.20 — adolescents have limited SR

    # Endogenous (derived from parents, but prior represents initial state)
    "usage_frequency":     (2.0, 3.0),   # mode ≈ 0.33
    "salience":            (2.0, 5.0),   # mode ≈ 0.20 — most start low
    "conflict":            (2.0, 5.0),   # mode ≈ 0.20
    "mood_modification":   (2.0, 4.0),   # mode ≈ 0.25
    "tolerance":           (2.0, 5.0),   # mode ≈ 0.20
    "withdrawal":          (2.0, 6.0),   # mode ≈ 0.14 — rare initially
    "relapse":             (2.0, 7.0),   # mode ≈ 0.11 — very rare initially
}


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING — generate heterogeneous agent parameters
# ═══════════════════════════════════════════════════════════════════════════════

def sample_agent_params(
    n_agents: int = 4,
    seed: int = 42,
) -> list[dict[str, float]]:
    """
    Sample agent parameters from the causal DAG prior.

    Returns list of dicts, one per agent, with all DAG node values
    plus the mapped sim_loop parameters (susceptibility, resilience).

    Uses numpy (no JAX required). For posterior sampling, use
    sample_agent_params_posterior() instead.
    """
    rng = np.random.RandomState(seed)
    agents = []

    for _ in range(n_agents):
        params = {}

        # Sample exogenous nodes from priors
        for node in ["peer_influence", "platform_algorithm", "self_regulation"]:
            a, b = PRIORS[node]
            params[node] = float(rng.beta(a, b))

        # Derive endogenous nodes via conditional sampling
        # usage_frequency depends on peer_influence + platform_algorithm
        uf_base_a, uf_base_b = PRIORS["usage_frequency"]
        uf_shift = 0.5 * params["peer_influence"] + 0.5 * params["platform_algorithm"]
        params["usage_frequency"] = float(rng.beta(
            uf_base_a + 2 * uf_shift, uf_base_b + 2 * (1 - uf_shift)
        ))

        # Cascade: each node influenced by parent + self_regulation dampening
        sr = params["self_regulation"]
        prev = params["usage_frequency"]
        for node in ["salience", "conflict", "mood_modification",
                      "tolerance", "withdrawal", "relapse"]:
            a, b = PRIORS[node]
            # Parent influence increases alpha (pushes toward higher values)
            parent_boost = 2.0 * prev
            # Self-regulation dampens (increases beta for conflict/mood/tolerance)
            sr_damp = 3.0 * sr if node in ("conflict", "mood_modification", "tolerance") else 0.0
            params[node] = float(rng.beta(a + parent_boost, b + sr_damp))
            prev = params[node]

        # Map to sim_loop parameters
        params["susceptibility"] = params["usage_frequency"]
        params["resilience"] = params["self_regulation"]

        agents.append(params)

    return agents


def sample_agent_params_posterior(
    n_agents: int = 4,
    observed_prevalence: float = 0.20,
    n_warmup: int = 500,
    n_samples: int = 1000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """
    Sample agent parameters from the NumPyro posterior, conditioned on
    an observed prevalence rate (fraction of population with score > 0.7).

    Requires JAX + NumPyro. Falls back to prior sampling if unavailable.

    This is the integration point for empirical data: replace
    observed_prevalence with actual prevalence from your study population.
    """
    if not HAS_NUMPYRO:
        import warnings
        warnings.warn("NumPyro not available, falling back to prior sampling")
        return sample_agent_params(n_agents, seed)

    def model(observed_prev=None):
        # Population-level hyperpriors
        peer_mu = numpyro.sample("peer_mu", dist.Beta(2.0, 3.0))
        platform_mu = numpyro.sample("platform_mu", dist.Beta(3.0, 2.0))
        sr_mu = numpyro.sample("sr_mu", dist.Beta(2.0, 5.0))

        # Usage frequency as a function of peer + platform
        uf_logit = numpyro.sample("uf_logit_offset", dist.Normal(0, 1))
        uf_prob = jax.nn.sigmoid(
            jnp.log(peer_mu / (1 - peer_mu + 1e-8)) * 0.5 +
            jnp.log(platform_mu / (1 - platform_mu + 1e-8)) * 0.5 +
            uf_logit
        )

        # Cascade probability: simplified single-path
        cascade_rate = numpyro.sample("cascade_rate", dist.Beta(2.0, 5.0))
        addiction_prob = uf_prob * cascade_rate * (1.0 - sr_mu * 0.5)

        # Observation: prevalence ~ Beta(concentration * prob, concentration * (1-prob))
        concentration = numpyro.sample("concentration", dist.Gamma(10.0, 1.0))
        numpyro.sample(
            "prevalence",
            dist.Beta(
                concentration * addiction_prob + 1e-4,
                concentration * (1 - addiction_prob) + 1e-4,
            ),
            obs=observed_prev,
        )

        # Store derived params for extraction
        numpyro.deterministic("susceptibility_pop", uf_prob)
        numpyro.deterministic("resilience_pop", sr_mu)

    # Run MCMC
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(rng_key, observed_prev=observed_prevalence)
    samples = mcmc.get_samples()

    # Sample n_agents from the posterior
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_samples, size=n_agents, replace=False)

    agents = []
    for idx in indices:
        sus_pop = float(samples["susceptibility_pop"][idx])
        res_pop = float(samples["resilience_pop"][idx])
        # Add individual variation around the population mean
        sus = float(np.clip(rng.normal(sus_pop, 0.1), 0.05, 0.95))
        res = float(np.clip(rng.normal(res_pop, 0.1), 0.05, 0.95))
        agents.append({
            "susceptibility": sus,
            "resilience": res,
            "peer_mu": float(samples["peer_mu"][idx]),
            "platform_mu": float(samples["platform_mu"][idx]),
            "sr_mu": float(samples["sr_mu"][idx]),
            "cascade_rate": float(samples["cascade_rate"][idx]),
        })

    return agents


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — inspect DAG and sample params
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── Causal DAG (Griffiths 2005 + DSM-5 IGD) ──\n")
    for parent, child in DAG_EDGES:
        print(f"  {parent:>22s} → {child}")

    print(f"\n── Prior Distributions ──\n")
    for node, (a, b) in PRIORS.items():
        mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else a / (a + b)
        print(f"  {node:>22s} ~ Beta({a:.1f}, {b:.1f})  mode={mode:.2f}")

    print(f"\n── Sample Agent Parameters (n=6, prior) ──\n")
    agents = sample_agent_params(n_agents=6, seed=42)
    for i, ag in enumerate(agents):
        print(f"  agent_{i}: sus={ag['susceptibility']:.3f}  res={ag['resilience']:.3f}  "
              f"usage={ag['usage_frequency']:.3f}  salience={ag['salience']:.3f}  "
              f"conflict={ag['conflict']:.3f}  relapse={ag['relapse']:.3f}")

    if HAS_NUMPYRO:
        print(f"\n── Sample Agent Parameters (n=6, posterior, prevalence=0.20) ──\n")
        agents_post = sample_agent_params_posterior(n_agents=6, observed_prevalence=0.20)
        for i, ag in enumerate(agents_post):
            print(f"  agent_{i}: sus={ag['susceptibility']:.3f}  res={ag['resilience']:.3f}  "
                  f"cascade={ag['cascade_rate']:.3f}")
    else:
        print("\n  [NumPyro not available — posterior sampling skipped]")
