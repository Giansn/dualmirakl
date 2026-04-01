"""
FLAME GPU 2 model definitions for dualmirakl population dynamics.

Architecture:
    GPU 2 runs a FLAME GPU 2 simulation with two agent populations:
    - Influencer agents (N_llm, mapped 1:1 from dualmirakl LLM participants)
    - Population agents (N_pop, thousands to millions of reactive agents)

    Influencer agents broadcast their scores via spatial messaging.
    Population agents receive nearby influencer signals and update their
    own scores using the same EMA/logistic dynamics as dualmirakl,
    plus spatial peer coupling (κ parameter).

All agent functions use RTC (runtime-compiled CUDA C++) for performance.
GPU constraints: no Python stdlib, no file I/O, printf only for debug.
"""

# ---------------------------------------------------------------------------
# RTC source: Influencer agents broadcast their position + score
# ---------------------------------------------------------------------------
INFLUENCER_OUTPUT_SRC = r"""
FLAMEGPU_AGENT_FUNCTION(influencer_output, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("score", FLAMEGPU->getVariable<float>("score"));
    FLAMEGPU->message_out.setVariable<int>("llm_index", FLAMEGPU->getVariable<int>("llm_index"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
"""

# ---------------------------------------------------------------------------
# RTC source: Population agents broadcast their position + score
# ---------------------------------------------------------------------------
POPULATION_OUTPUT_SRC = r"""
FLAMEGPU_AGENT_FUNCTION(population_output, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("score", FLAMEGPU->getVariable<float>("score"));
    FLAMEGPU->message_out.setVariable<int>("llm_index", -1);
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
"""

# ---------------------------------------------------------------------------
# RTC source: Population agents receive nearby messages and update score
#
# Score dynamics (mirrors dualmirakl sim_loop.py):
#   effective_signal = score + susceptibility * (neighbor_mean - score)
#   delta = alpha * (effective_signal - score)
#   new_score = clamp(score + delta * dampening * (1 - resilience), 0, 1)
#
# Coupling (mirrors dynamics.py coupled_score_update):
#   peer_mean = mean of nearby agent scores (spatial radius)
#   coupling_term = kappa * (peer_mean - score)
#   score += coupling_term
#
# Influencer weighting:
#   influencer messages have weight = influencer_weight (env property)
#   population messages have weight = 1.0
# ---------------------------------------------------------------------------
POPULATION_UPDATE_SRC = r"""
FLAMEGPU_AGENT_FUNCTION(population_update, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const float my_x = FLAMEGPU->getVariable<float>("x");
    const float my_y = FLAMEGPU->getVariable<float>("y");
    const float my_score = FLAMEGPU->getVariable<float>("score");
    const float susceptibility = FLAMEGPU->getVariable<float>("susceptibility");
    const float resilience = FLAMEGPU->getVariable<float>("resilience");

    // Environment parameters (read-only from agent functions)
    const float alpha = FLAMEGPU->environment.getProperty<float>("alpha");
    const float kappa = FLAMEGPU->environment.getProperty<float>("kappa");
    const float dampening = FLAMEGPU->environment.getProperty<float>("dampening");
    const float influencer_weight = FLAMEGPU->environment.getProperty<float>("influencer_weight");
    const int score_mode = FLAMEGPU->environment.getProperty<int>("score_mode");
    const float logistic_k = FLAMEGPU->environment.getProperty<float>("logistic_k");
    const float drift_sigma = FLAMEGPU->environment.getProperty<float>("drift_sigma");

    // Accumulate weighted neighbor scores
    float weighted_sum = 0.0f;
    float total_weight = 0.0f;
    int count = 0;

    for (const auto& msg : FLAMEGPU->message_in(my_x, my_y)) {
        if (msg.getVariable<flamegpu::id_t>("id") != FLAMEGPU->getID()) {
            float neighbor_score = msg.getVariable<float>("score");
            int llm_idx = msg.getVariable<int>("llm_index");
            float w = (llm_idx >= 0) ? influencer_weight : 1.0f;
            weighted_sum += neighbor_score * w;
            total_weight += w;
            count++;
        }
    }

    float new_score = my_score;

    if (count > 0) {
        float neighbor_mean = weighted_sum / total_weight;

        // Coupling term (peer influence, kappa)
        float coupling = kappa * (neighbor_mean - my_score);

        // EMA or logistic score update
        float effective_signal = my_score + susceptibility * (neighbor_mean - my_score);
        float effective_dampening = dampening * (1.0f - resilience);

        if (score_mode == 0) {
            // EMA mode
            float delta = alpha * (effective_signal - my_score);
            new_score = my_score + delta * effective_dampening + coupling;
        } else {
            // Logistic mode (saturation at extremes)
            float centered = effective_signal - 0.5f;
            float logistic_val = 1.0f / (1.0f + expf(-logistic_k * centered));
            float delta = alpha * (logistic_val - my_score);
            new_score = my_score + delta * effective_dampening + coupling;
        }
    }

    // Stochastic drift (brownian noise for stochastic resonance)
    if (drift_sigma > 0.0f) {
        // Simple hash-based pseudo-random (deterministic per agent per step)
        unsigned int hash = FLAMEGPU->getID() * 2654435761u;
        hash ^= FLAMEGPU->environment.getProperty<unsigned int>("step_count");
        hash *= 2246822519u;
        // Box-Muller-ish approximation: map to [-1, 1]
        float rand_val = ((float)(hash & 0xFFFF) / 32768.0f) - 1.0f;
        new_score += drift_sigma * rand_val;
    }

    // Clamp to [0, 1]
    new_score = fminf(1.0f, fmaxf(0.0f, new_score));

    FLAMEGPU->setVariable<float>("score", new_score);

    // Track score delta for analysis
    FLAMEGPU->setVariable<float>("score_delta", new_score - my_score);

    return flamegpu::ALIVE;
}
"""

# ---------------------------------------------------------------------------
# RTC source: Population agents drift in social space
# Agents move toward nearby influencers (gravity) with random walk
# ---------------------------------------------------------------------------
POPULATION_MOVE_SRC = r"""
FLAMEGPU_AGENT_FUNCTION(population_move, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    float my_x = FLAMEGPU->getVariable<float>("x");
    float my_y = FLAMEGPU->getVariable<float>("y");
    const float move_speed = FLAMEGPU->environment.getProperty<float>("move_speed");

    // Drift toward nearby influencers
    float fx = 0.0f, fy = 0.0f;
    int inf_count = 0;
    for (const auto& msg : FLAMEGPU->message_in(my_x, my_y)) {
        int llm_idx = msg.getVariable<int>("llm_index");
        if (llm_idx >= 0) {
            float dx = msg.getVirtualX() - my_x;
            float dy = msg.getVirtualY() - my_y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > 0.01f) {
                fx += dx / dist;
                fy += dy / dist;
                inf_count++;
            }
        }
    }

    if (inf_count > 0) {
        my_x += (fx / inf_count) * move_speed;
        my_y += (fy / inf_count) * move_speed;
    }

    // Random walk (hash-based pseudo-random)
    unsigned int hash = FLAMEGPU->getID() * 1664525u + 1013904223u;
    hash ^= FLAMEGPU->environment.getProperty<unsigned int>("step_count");
    float rx = ((float)((hash >> 0) & 0xFF) / 128.0f) - 1.0f;
    float ry = ((float)((hash >> 8) & 0xFF) / 128.0f) - 1.0f;
    my_x += rx * move_speed * 0.5f;
    my_y += ry * move_speed * 0.5f;

    // Wrap boundaries
    const float env_max = FLAMEGPU->environment.getProperty<float>("space_size");
    my_x = fmodf(my_x + env_max, env_max);
    my_y = fmodf(my_y + env_max, env_max);

    FLAMEGPU->setVariable<float>("x", my_x);
    FLAMEGPU->setVariable<float>("y", my_y);

    return flamegpu::ALIVE;
}
"""


def build_model_description(config: dict):
    """
    Build the FLAME GPU 2 ModelDescription from a config dict.

    Config keys:
        n_population (int):     Number of population agents (default 10000)
        n_influencers (int):    Number of influencer slots (matches n_participants)
        space_size (float):     Social space dimension (default 100.0)
        interaction_radius (float): Spatial messaging radius (default 10.0)
        alpha (float):          EMA learning rate (default 0.15)
        kappa (float):          Coupling strength (default 0.1)
        dampening (float):      Score dampening (default 1.0)
        influencer_weight (float): Influencer message weight (default 5.0)
        score_mode (str):       "ema" or "logistic" (default "ema")
        logistic_k (float):     Logistic steepness (default 6.0)
        drift_sigma (float):    Stochastic noise amplitude (default 0.01)
        mobility (float):       Movement as fraction of interaction_radius (default 0.1)
        sub_steps (int):        FLAME steps per dualmirakl tick (default 10)

    Returns:
        pyflamegpu.ModelDescription
    """
    import pyflamegpu

    n_pop = config.get("n_population", 10000)
    n_inf = config.get("n_influencers", 4)
    space = config.get("space_size", 100.0)
    radius = config.get("interaction_radius", 10.0)

    model = pyflamegpu.ModelDescription("DualMiraklPopulation")

    # --- Environment properties ---
    env = model.Environment()
    env.newPropertyFloat("alpha", config.get("alpha", 0.15))
    env.newPropertyFloat("kappa", config.get("kappa", 0.1))
    env.newPropertyFloat("dampening", config.get("dampening", 1.0))
    env.newPropertyFloat("influencer_weight", config.get("influencer_weight", 5.0))
    env.newPropertyInt("score_mode", 0 if config.get("score_mode", "ema") == "ema" else 1)
    env.newPropertyFloat("logistic_k", config.get("logistic_k", 6.0))
    env.newPropertyFloat("drift_sigma", config.get("drift_sigma", 0.01))
    mobility = config.get("mobility", 0.1)
    env.newPropertyFloat("move_speed", mobility * radius)
    env.newPropertyFloat("space_size", space)
    env.newPropertyUInt("step_count", 0)

    # --- Spatial message (shared by influencers + population) ---
    msg = model.newMessageSpatial2D("social_signal")
    msg.newVariableID("id")
    msg.newVariableFloat("score")
    msg.newVariableInt("llm_index")  # -1 = population, 0..N = influencer
    msg.setRadius(radius)
    msg.setMin(0, 0)
    msg.setMax(space, space)

    # --- Influencer agent type ---
    influencer = model.newAgent("Influencer")
    influencer.newVariableFloat("x")
    influencer.newVariableFloat("y")
    influencer.newVariableFloat("score", 0.3)
    influencer.newVariableInt("llm_index", 0)

    fn_inf_out = influencer.newRTCFunction("influencer_output", INFLUENCER_OUTPUT_SRC)
    fn_inf_out.setMessageOutput("social_signal")

    # --- Population agent type ---
    population = model.newAgent("Population")
    population.newVariableFloat("x")
    population.newVariableFloat("y")
    population.newVariableFloat("score", 0.3)
    population.newVariableFloat("score_delta", 0.0)
    population.newVariableFloat("susceptibility", 0.4)
    population.newVariableFloat("resilience", 0.2)

    fn_pop_out = population.newRTCFunction("population_output", POPULATION_OUTPUT_SRC)
    fn_pop_out.setMessageOutput("social_signal")

    fn_pop_update = population.newRTCFunction("population_update", POPULATION_UPDATE_SRC)
    fn_pop_update.setMessageInput("social_signal")

    fn_pop_move = population.newRTCFunction("population_move", POPULATION_MOVE_SRC)
    fn_pop_move.setMessageInput("social_signal")

    # --- Execution order ---
    # Layer 1: All agents output their position + score
    layer1 = model.newLayer("broadcast")
    layer1.addAgentFunction(fn_inf_out)
    layer1.addAgentFunction(fn_pop_out)  # concurrent (different populations)

    # Layer 2: Population agents read messages and update scores
    layer2 = model.newLayer("update")
    layer2.addAgentFunction(fn_pop_update)

    # Layer 3: Population agents move in social space
    layer3 = model.newLayer("move")
    layer3.addAgentFunction(fn_pop_move)

    # --- Step function: increment step counter ---
    class StepCounter(pyflamegpu.HostFunction):
        def run(self, FLAMEGPU):
            step = FLAMEGPU.environment.getPropertyUInt("step_count")
            FLAMEGPU.environment.setPropertyUInt("step_count", step + 1)

    model.addStepFunction(StepCounter())

    # --- Logging ---
    log_cfg = pyflamegpu.LoggingConfig(model)
    pop_log = log_cfg.agent("Population")
    pop_log.logMean("score")
    pop_log.logStandardDev("score")
    pop_log.logMin("score")
    pop_log.logMax("score")
    pop_log.logCount()

    step_log = pyflamegpu.StepLoggingConfig(log_cfg)
    step_log.setFrequency(1)

    return model, log_cfg, step_log
