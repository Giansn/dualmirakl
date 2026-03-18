#!/usr/bin/env python3
"""
dualmirakl -- Multi-Agent Simulation Framework
Interactive Walkthrough

This script walks you through the core concepts step by step.
No GPU required for steps 1-4. Steps 5+ need running vLLM.

Run:  python examples/walkthrough.py
Or:   python examples/walkthrough.py --step 3

dualmirakl is a domain-agnostic multi-agent simulation framework.
It runs on a dual-GPU vLLM stack with asyncio orchestration and
JSON data export. Agents are LLM-driven -- their behavior comes
from prompts and context injection, not hardcoded rules.

The framework doesn't care whether you're simulating social dynamics,
network infrastructure, market ecosystems, or anything else. The
domain lives in a scenario.yaml file. The engine stays the same.

Architecture at a glance:

    GPU 0 -- authority :8000    (environment + observer agents)
    GPU 1 -- swarm     :8001    (participant agents)
    CPU   -- gateway   :9000    (embeddings + proxy + document store)

    asyncio drives the tick loop.
    JSON files store trajectories and events.
    FLAME GPU 2 (optional) amplifies to population scale.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
MINIMAL_SCENARIO = SCENARIOS_DIR / "minimal.yaml"
CONTEXT_DIR = Path(__file__).parent.parent / "context"
WORLD_CONTEXT_PATH = CONTEXT_DIR / "world_context.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(number: int, title: str, needs_gpu: bool = False):
    """Print a section header."""
    gpu_tag = " [needs running vLLM]" if needs_gpu else " [no GPU needed]"
    print(f"\n{'=' * 64}")
    print(f"  Step {number}: {title}{gpu_tag}")
    print(f"{'=' * 64}\n")


def pause(msg: str = "Press Enter to continue..."):
    """Wait for user input between steps."""
    try:
        input(f"\n  >> {msg}")
    except (EOFError, KeyboardInterrupt):
        print("\n  Exiting walkthrough.")
        sys.exit(0)


def check_file(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    exists = path.exists()
    status = "[ok]" if exists else "[--]"
    print(f"  {status}  {description}")
    print(f"         {path}")
    return exists


# ===================================================================
# GREETING
# ===================================================================

def hello_world():
    print("""
    Hello World!

    Welcome to dualmirakl.

    This walkthrough will guide you through the framework in 9 steps.
    Steps 1-4 run entirely on CPU -- no GPU, no vLLM, no RunPod.
    Steps 5-9 need a running dual-GPU vLLM stack.

    What you'll learn:
      Step 1  Scenario loading and validation
      Step 2  The document-to-simulation bridge
      Step 3  Agent architecture and archetype profiles
      Step 4  The tick loop and score dynamics
      Step 5  Infrastructure preflight check
      Step 6  Running a single tick manually
      Step 7  Using the document bridge in practice
      Step 8  Post-run dynamics analysis
      Step 9  What's next

    Let's go.
    """)
    pause("Press Enter to begin...")


# ===================================================================
# STEP 1: Scenario Loading + Validation
# ===================================================================

def step_1():
    section(1, "Scenario loading and validation")

    print("""\
  Everything domain-specific lives in a scenario.yaml file.
  The engine reads it, builds agents, configures scoring, and runs.

  A scenario defines:
    - agents       -- roles, slots (authority/swarm), prompt templates
    - archetypes   -- named profiles for participant agents
    - actions      -- what decisions agents can make per tick
    - scoring      -- how agent scores evolve (EMA, logistic, custom)
    - transitions  -- when agents change archetype (Python functions)
    - memory       -- per-agent memory settings
    - safety       -- action allowlists, fallback behavior
    - flame        -- optional population amplification
    - environment  -- tick count, initial state variables

  Before running a simulation, always validate first:

    python -m simulation.scenario validate scenarios/minimal.yaml

  This catches YAML errors, missing prompt variables, invalid
  archetype references, and unknown transition functions -- without
  touching vLLM. Saves GPU hours.\
""")

    print("\n  Checking for scenario files...\n")
    check_file(MINIMAL_SCENARIO, "minimal.yaml (hello world)")
    check_file(SCENARIOS_DIR / "social_dynamics.yaml", "social_dynamics.yaml (full example)")
    check_file(SCENARIOS_DIR / "network_resilience.yaml", "network_resilience.yaml (infrastructure)")
    check_file(SCENARIOS_DIR / "market_ecosystem.yaml", "market_ecosystem.yaml (economic)")
    check_file(SCENARIOS_DIR / "_template.yaml", "_template.yaml (annotated reference)")

    # Try loading if available
    try:
        from simulation.scenario import ScenarioConfig
        if MINIMAL_SCENARIO.exists():
            config = ScenarioConfig.load(str(MINIMAL_SCENARIO))
            print(f"\n  Loaded minimal.yaml via ScenarioConfig:")
            print(f"  Name:        {config.meta.name}")
            print(f"  Agents:      {len(config.agents.roles)} roles")
            print(f"  Participants:{config.participant_count()}")
            print(f"  Ticks:       {config.environment.tick_count}")
            print(f"  Scoring:     {config.scoring.mode}")
            print(f"  FLAME:       {'enabled' if config.flame.enabled else 'disabled'}")
            print(f"  Memory:      {'enabled' if config.memory.enabled else 'disabled'}")

            report = config.validate_scenario(strict=False)
            if report["valid"]:
                print(f"\n  Validation: PASS (0 errors, {len(report['warnings'])} warnings)")
            else:
                print(f"\n  Validation: FAIL ({len(report['errors'])} errors)")
                for e in report["errors"]:
                    print(f"    [X] {e}")
    except ImportError:
        print("\n  (Install dependencies to enable ScenarioConfig loading)")
    except Exception as e:
        print(f"\n  Could not load scenario: {e}")

    pause()


# ===================================================================
# STEP 2: The Document -> Simulation Bridge
# ===================================================================

def step_2():
    section(2, "Document -> simulation bridge (PDF + world context)")

    print("""\
  dualmirakl can ground simulations in real-world documents.
  This is the document bridge -- one of the features that sets
  dualmirakl apart from classical ABM frameworks.

  How it works:

    1. Upload a PDF via the gateway API:
       POST /v1/documents  (multipart/form-data)

    2. The gateway extracts text and embeds it with e5-small-v2
       (runs on CPU -- no GPU needed for this step).

    3. Extracted context is stored in:
       context/world_context.json

    4. On each tick, the environment and observer agents receive
       the relevant context via {domain_context} in their prompts.
       Participant agents stay domain-blind -- they only see their
       archetype profile and the current stimulus. This separation
       is intentional: participants react to situations, not to
       meta-knowledge about the simulation setup.

  These categories are configurable per scenario (see the
  context_categories: block in your scenario.yaml). The gateway's
  detect_missing_context() function checks which categories are
  present and warns about missing required ones.

  You can also write world_context.json manually -- no PDF needed.
  This is useful for quick experiments or when your domain knowledge
  isn't in document form.\
""")

    print("\n  Checking for existing world context...\n")
    if check_file(WORLD_CONTEXT_PATH, "world_context.json"):
        try:
            with open(WORLD_CONTEXT_PATH) as f:
                ctx = json.load(f)
            print(f"\n  Found {len(ctx)} context categories:")
            for key, value in ctx.items():
                preview = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                print(f"    - {key}: {preview}")
        except Exception as e:
            print(f"\n  Could not read world context: {e}")
    else:
        print("""
  No world_context.json found. That's fine -- simulations run
  without it. To create one:

    Option A: Upload a PDF via POST /v1/documents
    Option B: Create context/world_context.json manually
    Option C: Use the detect endpoint: GET /simulation/detect\
""")

    pause()


# ===================================================================
# STEP 3: Agent Architecture
# ===================================================================

def step_3():
    section(3, "Agent architecture and archetype profiles")

    print("""\
  dualmirakl has three agent types, split across two GPU slots:

  +-----------------------------------------------------------+
  |  AUTHORITY slot (GPU 0, port 8000)                        |
  |                                                           |
  |  - environment agent -- generates stimuli each tick       |
  |  - observer agents   -- analyze trends, flag anomalies    |
  |                                                           |
  |  These agents see the world_context (document bridge).    |
  |  They have the "big picture."                             |
  +-----------------------------------------------------------+
  |  SWARM slot (GPU 1, port 8001)                            |
  |                                                           |
  |  - participant agents -- make decisions based on           |
  |    their archetype profile + current stimulus              |
  |                                                           |
  |  These agents are domain-blind. They don't see the        |
  |  world_context. They react to what's in front of them.    |
  +-----------------------------------------------------------+

  Archetype profiles define participant diversity. Instead of
  giving every agent the same personality, you define named
  profiles with distinct properties:

    profiles:
      - id: "K1"
        label: "High-risk"
        properties:
          responsiveness: "high"
          stability: "fragile"
      - id: "K2"
        label: "Resilient"
        properties:
          responsiveness: "low"
          stability: "stable"

    distribution:
      K1: 0.3
      K2: 0.7

  The engine assigns participants to profiles based on the
  distribution. Properties are injected into prompts via
  {archetype_profile}. The LLM decides how to interpret them.

  Archetype transitions: agents can change profile over time.
  Transitions are Python functions (not YAML expressions):

    # simulation/transitions.py
    @register_transition("escalation_sustained")
    def escalation_sustained(agent_state, tick_history, **params):
        threshold = params.get("threshold", 0.8)
        n = params.get("consecutive_ticks", 3)
        recent = [t["score"] for t in tick_history[-n:]]
        return len(recent) == n and all(s > threshold for s in recent)

  The scenario.yaml references these by name:

    transitions:
      - from: "K2"
        to: "K1"
        function: "escalation_sustained"
        params:
          threshold: 0.8
          consecutive_ticks: 3\
""")

    pause()


# ===================================================================
# STEP 4: Tick Loop and Score Dynamics
# ===================================================================

def step_4():
    section(4, "Tick loop and score dynamics")

    print("""\
  Each simulation tick follows a fixed phase sequence:

    Pre   ->  Persona prompts are prepared for all participants.
              Archetype profiles are injected into templates.

    A     ->  Environment agent generates a stimulus (1 LLM call).
              Uses batch_decide() for all participants at once.

    B     ->  All participants respond concurrently (asyncio.gather).
              Actions are parsed via action_schema.py.
              Structured JSON output with free-text fallback.

    C+D   ->  Overlap phase (CPU + GPU in parallel):
              - CPU: responses are embedded (e5-small-v2)
              - GPU: observer agents analyze the round
              - Scores are updated per agent

    F     ->  (Optional) FLAME GPU 2 population step.
              LLM participants seed behavior into N population
              agents via spatial messaging.

  Score dynamics track each agent's state over time:

    EMA mode:      score += alpha * (signal - score) * dampening
    Logistic mode:  sigmoid saturation at extremes

  Each agent has heterogeneous parameters drawn from Beta
  distributions (susceptibility, resilience). Peer influence
  is controlled by coupling_kappa.

  All data is exported to data/{run_id}/ as JSON:
    - config.json         -- frozen scenario config + run metadata
    - trajectories.json   -- per-agent scores over time
    - observations.json   -- per-tick stimuli and responses
    - event_stream.json   -- unified event log (all phases)
    - agent_memories.json -- what agents remembered across ticks
    - compliance.json     -- any safety/compliance violations
    - interventions.json  -- observer-triggered interventions

  The dynamics.py toolkit operates on these trajectories:

    A: Coupled ODE        B: Bifurcation sweep
    C: Lyapunov exponent  D: Sobol S1+S2
    E: Transfer entropy   F: Emergence index
    G: Attractor basins   H: WorldState bridge\
""")

    pause()


# ===================================================================
# STEP 5: Infrastructure Preflight
# ===================================================================

def step_5():
    section(5, "Infrastructure preflight check", needs_gpu=True)

    print("""\
  Before running a simulation, check that everything is up:

    bash start_all.sh

  Or individually:

    bash start_authority.sh   # GPU 0
    bash start_swarm.sh       # GPU 1
    bash start_gateway.sh     # CPU -- embeddings + proxy

  The gateway exposes a preflight endpoint:

    GET /simulation/preflight

  This checks:
    - vLLM backends reachable
    - Models loaded and responding
    - Embedding model available
    - World context present (optional but reported)
    - Disk space for data export
""")

    print("  Attempting health check...\n")

    async def _check():
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from orchestrator import health_check, close_client
            status = await health_check()
            for name, state in status.items():
                icon = "[ok]" if state == "up" else "[--]"
                print(f"  {icon}  {name}: {state}")
            await close_client()
            return all(v == "up" for v in status.values())
        except Exception as e:
            print(f"  Could not reach orchestrator: {e}")
            print("  Make sure vLLM servers are running (bash start_all.sh)")
            return False

    all_up = asyncio.run(_check())

    if all_up:
        print("\n  All systems go. Ready for simulation.")
    else:
        print("\n  Some backends are down. Steps 6-8 require running vLLM.")
        print("  You can still explore steps 1-4 without GPU.")

    pause()


# ===================================================================
# STEP 6: Run a Simulation
# ===================================================================

def step_6():
    section(6, "Running a simulation", needs_gpu=True)

    print("""\
  The simplest way to run a simulation:

    python -m simulation.sim_loop --scenario scenarios/minimal.yaml

  This will:
    1. Load and validate the scenario config
    2. Create agents based on archetype profiles
    3. Run the tick loop (3 ticks for minimal.yaml)
    4. Export results to data/{run_id}/

  For the minimal scenario (2 participants, 3 ticks), a full
  run takes about 10-15 seconds on the swarm GPU.

  To run with the full social dynamics scenario:

    python -m simulation.sim_loop --scenario scenarios/social_dynamics.yaml

  Or without --scenario for the legacy interactive CLI:

    python -m simulation.sim_loop

  The interactive CLI lets you override individual parameters
  (n_ticks, n_participants, alpha, etc.) at the prompt.

  After the run, inspect the output:

    data/{run_id}/
      config.json         -- what was configured
      trajectories.json   -- score trajectories per agent
      observations.json   -- stimuli + responses per tick
      event_stream.json   -- full event log
      agent_memories.json -- what agents remembered
      compliance.json     -- safety violations (if any)
      interventions.json  -- observer interventions (if any)\
""")

    pause()


# ===================================================================
# STEP 7: Document Bridge in Practice
# ===================================================================

def step_7():
    section(7, "Using the document bridge", needs_gpu=True)

    print("""\
  Let's ground a simulation in a real document.

  Option A -- Upload via API:

    curl -X POST http://localhost:9000/v1/documents \\
      -F "file=@my_report.pdf" \\
      -F "name=research_context"

    The gateway will:
    1. Extract text from the PDF
    2. Embed it with e5-small-v2
    3. Store structured context in context/world_context.json
    4. Return a summary of detected categories

  Option B -- Check what's missing:

    curl http://localhost:9000/simulation/detect

    Returns which context categories are present/missing
    based on your scenario's context_categories: block.

  Option C -- Write manually:

    echo '{
      "summary": "Network resilience under load with nodes of varying capacity"
    }' > context/world_context.json

  Once world_context.json exists, the environment and observer
  agents automatically receive it via {domain_context} in their
  prompts. No code changes needed.\
""")

    pause()


# ===================================================================
# STEP 8: Dynamics Analysis
# ===================================================================

def step_8():
    section(8, "Dynamics analysis toolkit")

    print("""\
  After a run, the dynamics toolkit analyzes your trajectories:

    python simulation/dynamics.py

  Or from code:

    from simulation.dynamics import analyze_from_json
    results = analyze_from_json("data/{run_id}/")

  Available analyses:

    A  Coupled ODE        -- continuous approximation with coupling
    B  Bifurcation sweep  -- vary a parameter, find regime transitions
    C  Lyapunov exponent  -- is the system chaotic or stable?
    D  Sobol S1 + S2      -- which parameters matter? which interact?
    E  Transfer entropy   -- who influences whom? (causal direction)
    F  Emergence index    -- is collective behavior > sum of parts?
    G  Attractor basins   -- where does the system settle?
       Stochastic res.    -- does noise actually help?
    H  WorldState bridge  -- connect analysis back to simulation state

  The sensitivity analysis pipeline runs in order:

    Morris screening -> History Matching (NROY) -> Sobol S1+S2

  This narrows the parameter space progressively:
    Morris:  cheap, identifies non-influential parameters
    NROY:    removes implausible parameter regions
    Sobol:   expensive, quantifies interactions on reduced space

  7 tunable parameters:
    alpha, K, threshold, dampening, susceptibility, resilience, logistic_k

  With Optuna (pip install optuna), you can optimize these:

    python -m simulation.optimize --mode fast --trials 100

  Two optimization modes:
    fast:  surrogate objective, no vLLM, 100+ trials/min
    full:  live simulation with LLM, captures emergent dynamics\
""")

    pause()


# ===================================================================
# STEP 9: What's Next
# ===================================================================

def step_9():
    section(9, "Next steps")

    print("""\
  You've seen the core concepts. Here's where to go next:

  Build your first scenario:
    1. Copy scenarios/_template.yaml -> scenarios/my_scenario.yaml
    2. Edit agent roles, archetypes, actions
    3. Validate:  python -m simulation.scenario validate scenarios/my_scenario.yaml
    4. Run:       python -m simulation.sim_loop --scenario scenarios/my_scenario.yaml

  Ground it in data:
    1. Upload a PDF:   curl -X POST localhost:9000/v1/documents -F "file=@doc.pdf"
    2. Check context:  curl localhost:9000/simulation/detect
    3. Fill gaps in context/world_context.json if needed

  Scale with FLAME:
    1. Set flame.enabled: true in your scenario
    2. Set FLAME_GPU and FLAME_N_POPULATION in .env
    3. Your LLM agents become influencers seeding behavior
       into a population of thousands

  Optimize parameters:
    1. pip install optuna wandb
    2. python -m simulation.optimize --mode fast --trials 100
    3. Review results in W&B dashboard

  Key files:
    scenarios/              -- all scenario configurations
    simulation/sim_loop.py  -- the tick engine
    simulation/scenario.py  -- config loader + validator
    simulation/dynamics.py  -- post-run analysis
    orchestrator.py         -- dual-backend LLM client
    gateway.py              -- API, embeddings, document bridge
    context/                -- world context storage

  Key endpoints:
    GET  /health                   -- system status
    GET  /simulation/preflight     -- full infrastructure check
    POST /simulation/start         -- start a run
    GET  /simulation/status        -- progress, scores, events
    GET  /simulation/results       -- final data
    POST /v1/documents             -- upload PDF for context
    GET  /simulation/detect        -- check context completeness

  Documentation:
    CLAUDE.md                -- for Claude Code sessions
    README.md                -- project overview\
""")

    print("  - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("  Walkthrough complete. Happy simulating.")
    print()


# ===================================================================
# Main
# ===================================================================

STEPS = [step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8, step_9]

def main():
    print(__doc__)

    # Parse --step argument
    start_step = 1
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        if idx + 1 < len(sys.argv):
            try:
                start_step = int(sys.argv[idx + 1])
            except ValueError:
                print(f"  Invalid step number: {sys.argv[idx + 1]}")
                sys.exit(1)

    hello_world()

    print(f"  Starting from step {start_step} of {len(STEPS)}.")
    print(f"  Steps 1-4 need no GPU. Steps 5+ need running vLLM.")
    print(f"  Press Ctrl+C at any time to exit.\n")

    for i, step_fn in enumerate(STEPS, 1):
        if i < start_step:
            continue
        step_fn()

if __name__ == "__main__":
    main()
