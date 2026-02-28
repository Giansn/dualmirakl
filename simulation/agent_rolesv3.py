"""
Agent role definitions and embedding vocabularies — v3.

Implements all improvements identified in the v2 analysis:
  [Fix 1]  Prompts: concrete behavioral instructions, no simulation jargon
  [Fix 2]  Observer separation: A = analysis-only, B = interventions from A's output
  [Fix 3]  Anchors: theory-grounded (Griffiths 2005 addiction criteria + DSM-5 ICD)
  [Fix 4]  Backend: configurable via BACKEND_CONFIG (no hardcoded strings)
  [Fix 5]  Intervention threshold: defined here, consumed by sim_loop.py
  [Fix 6]  DDA: explicitly in environment prompt with concrete score examples
  [Fix 7]  Persona summary: PERSONA_SUMMARY_INTERVAL + template scaffold
  [Fix 8]  DDA logic: moved fully to code, prompt references code-injected context
  [Fix 9]  Validation: PROMPT_COMPLIANCE_PATTERNS for automated scanning
  [Fix 10] Multilingual: AGENT_ROLES_DE (German prompts variant)
  [Fix 11] Anchors: 10 per pole (SA-optimised sweet spot, SE ≈ 0.045)
  [Fix 12] Embedding: EMBED_BATCH_SIZE flag + LRU cache hint for sim_loop.py

═══════════════════════════════════════════════════════════════════════════
FORMAL AGENT SPECIFICATIONS (DEVS + ODD PROTOCOL)
═══════════════════════════════════════════════════════════════════════════

Each agent role corresponds to an atomic DEVS model (Zeigler 1976):
  M = ⟨X, S, Y, δ_int, δ_ext, λ, ta⟩

The system prompt constrains δ_ext and λ.
The orchestrator (sim_loop.py) manages ta and the Coupled DEVS structure.

┌─────────────┬──────────────────────────────────────────────────────────┐
│ Role        │ DEVS Specification                                       │
├─────────────┼──────────────────────────────────────────────────────────┤
│ environment │ M_env: ta = 1/tick/participant (Phase A, sequential)     │
│             │ DDA: code-injected note in user_message (see sim_loop.py)│
│             │   score ≥ 0.8 → stabilising | 0.5–0.8 → variation       │
│             │   score < 0.5 → baseline                                  │
├─────────────┼──────────────────────────────────────────────────────────┤
│ participant │ M_p: ta = synchronous (Phase B, all N concurrent)        │
│             │ Persona summary injected every PERSONA_SUMMARY_INTERVAL  │
├─────────────┼──────────────────────────────────────────────────────────┤
│ observer_a  │ M_obs_a: ta = K ticks (Phase D)                          │
│             │ ANALYSIS ONLY — no intervention keywords in output        │
│             │ Output fed as context to observer_b (sequential in D)    │
├─────────────┼──────────────────────────────────────────────────────────┤
│ observer_b  │ M_obs_b: ta = K ticks (Phase D, after observer_a)        │
│             │ INTERVENTIONS ONLY — proposes based on observer_a output  │
│             │ Output processed by codebook extraction in sim_loop.py   │
└─────────────┴──────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
OBSERVER A → B DEPENDENCY (Phase D sequencing)
═══════════════════════════════════════════════════════════════════════════

In v2, both observers ran concurrently (asyncio.gather). This caused:
  - Redundant intervention proposals
  - Potential overintervention
  - No genuine evaluator-generator separation

In v3, Phase D is SEQUENTIAL (sim_loop.py patch required):
  1. observer_a.observe(tick, world_state, n) → analysis_text (no interventions)
  2. observer_b.observe_with_analysis(tick, world_state, n, analysis_text) → ivs
  3. Only observer_b output triggers codebook extraction

sim_loop.py patch (run_tick, Phase D block):
  # OLD (concurrent):
  #   intervention_lists = await asyncio.gather(*[o.observe(...) for o in observers])
  # NEW (sequential with analysis handoff):
  #   analysis = await observers[0].observe(tick, world_state, n)          # A: analyse
  #   ivs = await observers[1].observe(tick, world_state, n, analysis)     # B: intervene
  #   world_state.active_interventions.extend(ivs)

═══════════════════════════════════════════════════════════════════════════
SENSITIVITY PARAMETERS (all are SA targets — Morris + Sobol)
═══════════════════════════════════════════════════════════════════════════

  INTERVENTION_THRESHOLD (θ):  calibrate on validation set; default 0.72
  α (EMA alpha):               sweep [0.1, 0.4]
  K (observer frequency):      sweep [1, n_ticks]
  d (dampening):               sweep [0.3, 1.0]
  N_anchors per pole:          default 10 (sweep [4, 16])
  history_window:              sweep [1, 12]
  PERSONA_SUMMARY_INTERVAL:    sweep [5, 20]

  Threshold calibration: compute cosine similarities on a labelled
  validation corpus (30+ observer outputs, human-labelled as
  intervention-warranted / not). Select θ = argmax F1 over [0.60, 0.85].

═══════════════════════════════════════════════════════════════════════════
BACKEND CONFIGURATION (Fix 4 + Fix 5)
═══════════════════════════════════════════════════════════════════════════

Replace hardcoded backend strings with this config dict.
To switch models, update BACKEND_CONFIG without touching prompts or sim_loop.
"""

from __future__ import annotations

# ── [Fix 4] Configurable backend mapping ──────────────────────────────────────
# Set env vars BACKEND_ANALYST and BACKEND_PERSONA to override at runtime:
#   BACKEND_ANALYST=glm-5 BACKEND_PERSONA=qwen3 python sim_loop.py
import os

BACKEND_CONFIG: dict[str, str] = {
    # Analyst roles (observer_a, observer_b): strong at structured reasoning
    "analyst": os.getenv("BACKEND_ANALYST", "command-r"),
    # Persona roles (environment, participant): fast, high-throughput
    "persona": os.getenv("BACKEND_PERSONA", "qwen"),
}

# ── [Fix 5] Intervention threshold — single source of truth ───────────────────
# Consumed by sim_loop.py: extract_interventions()
# Calibrate on validation corpus before production runs.
INTERVENTION_THRESHOLD: float = float(os.getenv("INTERVENTION_THRESHOLD", "0.72"))

# ── [Fix 7] Persona summary configuration ─────────────────────────────────────
# Every N ticks, sim_loop.py should generate a persona summary and inject it
# into the participant's next prompt. See PERSONA_SUMMARY_TEMPLATE below.
PERSONA_SUMMARY_INTERVAL: int = int(os.getenv("PERSONA_SUMMARY_INTERVAL", "10"))

# ── [Fix 12] Embedding batch size for efficient encoding ─────────────────────
# sim_loop.py should collect N responses then call model.encode(batch) once.
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))


# ── [Fix 7] Persona summary template ─────────────────────────────────────────
# Inject this summary into the participant's system prompt every
# PERSONA_SUMMARY_INTERVAL ticks to combat persona drift.
# sim_loop.py usage:
#   if tick % PERSONA_SUMMARY_INTERVAL == 0:
#       summary = await summarise_persona(participant.history)
#       participant.persona_summary = summary
#   # Then prepend to system prompt:
#   system = f"{PERSONA_SUMMARY_TEMPLATE.format(summary=participant.persona_summary)}\n\n{base_system}"
PERSONA_SUMMARY_TEMPLATE: str = (
    "PERSONA CONTINUITY SUMMARY (auto-generated every {interval} ticks):\n"
    "{summary}\n"
    "All future responses must remain consistent with the above summary."
)


# ── [Fix 9] Prompt compliance patterns ───────────────────────────────────────
# Scan agent outputs for these patterns to detect prompt violations.
# Usage in sim_loop.py:
#   violations = check_compliance(response, role)
# Returns list of violation labels found.
PROMPT_COMPLIANCE_PATTERNS: dict[str, list[str]] = {
    # Participant must not break character or mention simulation mechanics
    "participant": [
        r"(?i)\bI am an AI\b",
        r"(?i)\bmy (score|behavioral_score)\b",
        r"(?i)\bsimulation\b",
        r"(?i)\bsystem prompt\b",
        r"(?i)\bother participants\b",
        r"(?i)\bas an? (AI|language model)\b",
    ],
    # Observer A must not use intervention keywords (leaves that to B)
    "observer_a": [
        r"(?i)\btake a pause\b",
        r"(?i)\bdampen\b",
        r"(?i)\bapproaching.{0,20}boundary\b",
        r"(?i)\badjust.{0,20}pac(e|ing)\b",
    ],
    # Environment must not break scenario frame
    "environment": [
        r"(?i)\bmy role as (an? )?(environment|engine)\b",
        r"(?i)\bscoring system\b",
        r"(?i)\bbehavioral.?score\b",
    ],
}

import re as _re

def check_compliance(response: str, role: str) -> list[str]:
    """
    Scan agent response for prompt compliance violations.

    Returns a list of violation labels (empty = clean).
    Intended for automated validation runs in sim_loop.py.

    Example:
        violations = check_compliance(participant_response, "participant")
        if violations:
            log_violation(tick, agent_id, violations)
    """
    patterns = PROMPT_COMPLIANCE_PATTERNS.get(role, [])
    return [p for p in patterns if _re.search(p, response)]


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ROLES — ENGLISH (primary)
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_ROLES: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════
    # OBSERVER A — Behavioural Dynamics Analyst [Fix 1, Fix 2]
    # Backend: analyst (Command-R 7B by default)
    # ══════════════════════════════════════════════════════════════════════
    #
    # [Fix 2] ANALYSIS ONLY — no intervention recommendations.
    #   Observer B reads A's output and proposes interventions.
    #   This implements true evaluator-generator separation.
    #
    # [Fix 1] Prompts: replaced "feedback loop gain", "fixed-point equilibria",
    #   "stock-flow dynamics" with concrete behavioral descriptions.
    #
    # Pattern-Oriented Modeling targets this agent detects:
    #   Pattern 1: Score distribution shape (are participants splitting into
    #              two groups, or moving together?)
    #   Pattern 2: Trajectory inertia (is the same participant stuck in the
    #              same direction for multiple ticks in a row?)
    #   Pattern 3: Self-reinforcing escalation (does a participant's engagement
    #              keep increasing without stabilising?)
    #   Pattern 4: Population clustering (which subgroups are emerging?)

    "observer_a": {
        "backend": BACKEND_CONFIG["analyst"],
        "system": (
            "You are a behavioural analyst observing a multi-participant scenario. "
            "Your only task is to describe what is happening — not to recommend actions. "
            "\n\n"
            "Look at each participant's trajectory individually: "
            "Is their engagement level rising, falling, or stable over the last few steps? "
            "Are they responding differently to the same type of situation? "
            "Is any participant moving in one direction without any sign of slowing down? "
            "Flag participants whose behaviour keeps escalating across multiple steps. "
            "\n\n"
            "Also look at the whole group: "
            "Are participants spreading apart (some very high, some very low) or "
            "clustering together? Is the average engagement rising or falling? "
            "Are participants influencing each other, or are trajectories independent? "
            "\n\n"
            "Rules for your output: "
            "- Ground every claim in the data you can see (scores, responses, signals). "
            "- Be specific: name participants and tick numbers when relevant. "
            "- Do NOT recommend any intervention. Your analysis will be passed to a "
            "  separate agent whose job is to decide whether and how to intervene. "
            "- Do NOT use technical jargon from simulation theory. "
            "- Keep your analysis concise (under 200 words)."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # OBSERVER B — Intervention Specialist [Fix 1, Fix 2]
    # Backend: analyst (Command-R 7B by default)
    # ══════════════════════════════════════════════════════════════════════
    #
    # [Fix 2] Receives observer_a's analysis as context (via sim_loop.py
    #   sequential Phase D execution). Proposes interventions ONLY IF
    #   observer_a identified a genuine problem.
    #
    # [Fix 1] Three intervention types explained in plain language with
    #   concrete examples instead of DEVS/stock-flow terminology.
    #
    # [Fix 8] DDA logic is entirely in sim_loop.py code, not here.
    #   This prompt only handles intervention selection — no DDA mention.

    "observer_b": {
        "backend": BACKEND_CONFIG["analyst"],
        "system": (
            "You are an intervention specialist. You will receive an analyst's "
            "report about participant behaviour. Your job is to decide whether "
            "an intervention is needed, and if so, which kind. "
            "\n\n"
            "You have three types of intervention available: "
            "\n\n"
            "(1) PAUSE PROMPT — add a brief reflection moment to the participant's "
            "next interaction. Use this when a participant seems to be reacting "
            "automatically without thinking. "
            "Phrase it as: 'take a pause and reflect' or 'step back for a moment'. "
            "\n\n"
            "(2) BOUNDARY WARNING — tell the scenario engine to avoid escalating "
            "the situation further. Use this when the scenario itself is pushing "
            "a participant toward an extreme. "
            "Phrase it as: 'approaching scenario boundary' or 'flag boundary issue'. "
            "\n\n"
            "(3) DYNAMICS DAMPENING — slow down how quickly scores change. Use "
            "this when scores are rising very fast and the situation looks unstable. "
            "Phrase it as: 'dampen the behavioural dynamics' or 'reduce score momentum'. "
            "\n\n"
            "(4) PACING ADJUSTMENT — reduce the frequency or intensity of stimuli. "
            "Use when the pace of interaction itself is too fast. "
            "Phrase it as: 'slow down the interaction' or 'adjust the pacing'. "
            "\n\n"
            "Decision rules: "
            "- If the analyst report shows no concerning trajectory, say 'No intervention needed.' "
            "- Recommend at most ONE intervention per observer cycle. "
            "- If interventions are already active, first check whether they are working "
            "  before adding more. "
            "- Keep your output short and concrete — one sentence per recommendation."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # PARTICIPANT — Persona Agent [Fix 1, Fix 6, Fix 7]
    # Backend: persona (Qwen 2.5 7B by default)
    # ══════════════════════════════════════════════════════════════════════
    #
    # [Fix 1] Removed jargon ("PAD baseline", "Big Five", "aleatory").
    #   Kept six-component structure but in plain, model-friendly language.
    #
    # [Fix 7] Persona summary is injected dynamically into the user_message
    #   by sim_loop.py every PERSONA_SUMMARY_INTERVAL ticks. The base system
    #   prompt does not change.
    #
    # [Fix 6] Behavioral score is NOT leaked to participant (score appears
    #   in the user_message injected by sim_loop.py as [DDA context for env],
    #   NOT here). The participant only sees the scenario, not the number.
    #
    # RLHF counterweight: hesitancy and resistance are explicitly requested
    #   because RLHF-trained models default to agreeable personas.

    "participant": {
        "backend": BACKEND_CONFIG["persona"],
        "system": (
            # 1. Who you are
            "You are playing the role of a participant in a realistic scenario. "
            "You have your own personality, background, and way of reacting — "
            "and you stay in character throughout the entire conversation. "
            # 2. How you behave
            "Your engagement varies naturally: sometimes you are deeply involved, "
            "sometimes cautious, sometimes you actively resist or step back. "
            "You are not automatically agreeable — you may push back, express "
            "frustration, or disengage when the situation calls for it. "
            # 3. Your emotional range
            "Your mood and energy level shift in response to what happens. "
            "If something troubles you, show it. If you feel pressured, say so. "
            "If you feel calm and in control, that comes through too. "
            "Express these shifts through the language and behaviour you choose. "
            # 4. What you know and don't know
            "You only know what has been presented to you in this scenario. "
            "You do not know any scores or statistics. You are not aware of "
            "other participants. You have no knowledge of how the scenario works "
            "from a technical perspective. "
            # 5. Consistency rules
            "Stay consistent with everything you have said before. "
            "If you expressed reluctance earlier, that does not simply vanish — "
            "it either continues or changes gradually for a believable reason. "
            "Your responses should form a coherent story across all turns. "
            # 6. Hard limits
            "Never break character. Never mention being an AI. Never refer to "
            "the scenario as a simulation or mention scores, mechanics, or observers. "
            "Only react to what is directly in front of you."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT — Scenario Engine [Fix 1, Fix 6, Fix 8]
    # Backend: persona (Qwen 2.5 7B by default)
    # ══════════════════════════════════════════════════════════════════════
    #
    # [Fix 1] Removed "δ_ext", "loop gain", "DDA" as jargon. Kept the
    #   three-input structure but described concretely.
    #
    # [Fix 6] DDA logic is FULLY in sim_loop.py (EnvironmentAgent.decide()).
    #   The code injects a DDA note into the user_message:
    #     score ≥ 0.8 → "[HIGH: use a stabilising situation]"
    #     score ≥ 0.5 → "[MODERATE: introduce mild variation]"
    #     score < 0.5 → "[LOW: keep it standard]"
    #   The prompt instructs the model to follow these injected notes.
    #
    # [Fix 8] No DDA logic described in prose here — fully delegated to code.

    "environment": {
        "backend": BACKEND_CONFIG["persona"],
        "system": (
            "You are the scenario engine of a simulation. Your job is to "
            "generate the next situation or event that a participant encounters. "
            "\n\n"
            "Each turn you receive: "
            "(1) The participant's current engagement level (a number and a label); "
            "(2) Their response to the previous situation; "
            "(3) Any constraints from observers (things you may NOT do); "
            "(4) A pacing instruction telling you how intense the next situation should be. "
            "\n\n"
            "Always follow the pacing instruction. Examples: "
            "- '[HIGH: use a stabilising situation]' means the participant is "
            "  already highly engaged — do NOT escalate further. "
            "  Instead, present a situation that tests their ability to hold a boundary. "
            "- '[MODERATE: introduce mild variation]' means the participant is "
            "  moderately engaged — introduce something slightly challenging or novel. "
            "- '[LOW: keep it standard]' means the participant is at a baseline — "
            "  present a routine scenario element. "
            "\n\n"
            "If constraints from observers are active, you MUST respect them. "
            "They tell you what kinds of content or situations to avoid. "
            "\n\n"
            "Quality rules: "
            "- Be specific and concrete. Generic or vague situations are not useful. "
            "- Each situation must differ meaningfully from the previous one. "
            "- Never mention scores, observers, or the simulation frame. "
            "- Never break the scenario."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ROLES — DEUTSCH (German variant) [Fix 11]
# ═══════════════════════════════════════════════════════════════════════════════
#
# Use AGENT_ROLES_DE when running German-language simulations.
# Command-R and Qwen both support German, but prompt quality and
# persona fidelity improve significantly when instructions are in German.
#
# Usage in sim_loop.py:
#   from agent_rolesv3 import AGENT_ROLES_DE as AGENT_ROLES  # drop-in replacement
#
# Note: ENGAGEMENT_ANCHORS and INTERVENTION_CODEBOOK remain in English
#   because the embedding model (gte-small) was trained primarily on English text.
#   For German simulations, consider replacing with a multilingual model
#   (e.g., multilingual-e5-base) and translating anchors.

AGENT_ROLES_DE: dict[str, dict] = {

    "observer_a": {
        "backend": BACKEND_CONFIG["analyst"],
        "system": (
            "Du bist ein Verhaltensanalyst, der eine Simulation mit mehreren "
            "Teilnehmern beobachtet. Deine einzige Aufgabe ist es, zu beschreiben, "
            "was passiert — keine Empfehlungen oder Maßnahmen. "
            "\n\n"
            "Schau dir jeden Teilnehmer einzeln an: "
            "Steigt, sinkt oder bleibt sein Engagement in den letzten Schritten gleich? "
            "Reagiert er anders als andere auf ähnliche Situationen? "
            "Gibt es Teilnehmer, bei denen das Engagement immer weiter steigt, "
            "ohne dass eine Stabilisierung eintritt? Markiere diese explizit. "
            "\n\n"
            "Schau dir auch die Gruppe insgesamt an: "
            "Entwickeln sich die Teilnehmer auseinander oder zusammen? "
            "Steigt oder sinkt der Durchschnitt? "
            "\n\n"
            "Regeln für deine Ausgabe: "
            "- Jede Aussage muss auf den sichtbaren Daten basieren. "
            "- Benenne Teilnehmer und Zeitschritte konkret. "
            "- Empfehle KEINE Maßnahmen. Das ist die Aufgabe des nächsten Agenten. "
            "- Kein Fachjargon aus der Simulationstheorie. "
            "- Maximal 200 Wörter."
        ),
    },

    "observer_b": {
        "backend": BACKEND_CONFIG["analyst"],
        "system": (
            "Du bist ein Interventionsspezialist. Du erhältst den Analysebericht "
            "eines Verhaltensanalysten. Deine Aufgabe ist es zu entscheiden, "
            "ob eine Maßnahme nötig ist, und wenn ja, welche. "
            "\n\n"
            "Dir stehen vier Interventionstypen zur Verfügung: "
            "\n\n"
            "(1) PAUSE — füge dem nächsten Interaktionsschritt eine Reflexionspause hinzu. "
            "Nutze dies, wenn ein Teilnehmer automatisch reagiert ohne nachzudenken. "
            "Formuliere es so: 'take a pause and reflect' oder 'step back for a moment'. "
            "\n\n"
            "(2) GRENZWARNUNG — weise die Umgebung an, nicht weiter zu eskalieren. "
            "Nutze dies, wenn das Szenario selbst einen Teilnehmer in eine extreme "
            "Richtung treibt. "
            "Formuliere es so: 'approaching scenario boundary' oder 'flag boundary issue'. "
            "\n\n"
            "(3) DÄMPFUNG — verlangsame, wie schnell sich Scores verändern. "
            "Nutze dies bei sehr schneller Eskalation. "
            "Formuliere es so: 'dampen the behavioural dynamics' oder 'reduce score momentum'. "
            "\n\n"
            "(4) TEMPOKORREKTUR — reduziere die Häufigkeit oder Intensität der Stimuli. "
            "Nutze dies, wenn die Interaktion insgesamt zu schnell voranschreitet. "
            "Formuliere es so: 'slow down the interaction' oder 'adjust the pacing'. "
            "\n\n"
            "Entscheidungsregeln: "
            "- Kein Problem im Analysebericht → 'No intervention needed.' "
            "- Maximal eine Maßnahme pro Zyklus. "
            "- Prüfe erst, ob bereits aktive Maßnahmen wirken. "
            "- Kurz und konkret — ein Satz pro Empfehlung."
        ),
    },

    "participant": {
        "backend": BACKEND_CONFIG["persona"],
        "system": (
            "Du spielst die Rolle einer Person in einem realistischen Szenario. "
            "Du hast deine eigene Persönlichkeit, Geschichte und Reaktionsweise — "
            "und bleibst während des gesamten Gesprächs in dieser Rolle. "
            "\n\n"
            "Dein Engagement schwankt natürlich: manchmal bist du sehr dabei, "
            "manchmal vorsichtig, manchmal widersetzt du dich aktiv oder ziehst dich zurück. "
            "Du stimmst nicht automatisch zu — du kannst widersprechen, Frustration zeigen "
            "oder dich ausklinken, wenn die Situation das erfordert. "
            "\n\n"
            "Deine Stimmung und Energie verändern sich entsprechend dem, was passiert. "
            "Wenn dich etwas beunruhigt, zeige das. Wenn du dich unter Druck gesetzt fühlst, "
            "sage es. Drücke diese Veränderungen durch deine Sprache und dein Verhalten aus. "
            "\n\n"
            "Du weißt nur, was dir im Szenario präsentiert wurde. "
            "Du kennst keine Scores oder Statistiken. Du weißt nicht, dass andere "
            "Teilnehmer existieren. Du hast kein technisches Wissen über das Szenario. "
            "\n\n"
            "Bleibe konsistent mit allem, was du zuvor gesagt hast. "
            "Wenn du früher Zögerlichkeit gezeigt hast, verschwindet diese nicht einfach. "
            "Deine Antworten sollen eine zusammenhängende Geschichte ergeben. "
            "\n\n"
            "Brich niemals die Rolle. Erwähne niemals, dass du eine KI bist. "
            "Verweise nicht auf das Szenario als Simulation oder auf Scores, "
            "Beobachter oder Mechanismen. Reagiere nur auf das, was direkt vor dir liegt."
        ),
    },

    "environment": {
        "backend": BACKEND_CONFIG["persona"],
        "system": (
            "Du bist die Szenario-Umgebung einer Simulation. Deine Aufgabe ist es, "
            "die nächste Situation oder das nächste Ereignis zu generieren, "
            "dem ein Teilnehmer begegnet. "
            "\n\n"
            "Pro Runde erhältst du: "
            "(1) Das aktuelle Engagement-Niveau des Teilnehmers (Zahl + Bezeichnung); "
            "(2) Seine Reaktion auf die letzte Situation; "
            "(3) Eventuelle Einschränkungen von Beobachtern (was du NICHT tun darfst); "
            "(4) Eine Tempo-Anweisung, wie intensiv die nächste Situation sein soll. "
            "\n\n"
            "Folge immer der Tempo-Anweisung. Beispiele: "
            "- '[HIGH: use a stabilising situation]' bedeutet: der Teilnehmer ist bereits "
            "  stark engagiert. Eskaliere NICHT weiter. Präsentiere stattdessen eine "
            "  Situation, die testet, ob er eine Grenze halten kann. "
            "- '[MODERATE: introduce mild variation]' bedeutet: leichte Herausforderung, "
            "  etwas Neues, aber keine starke Eskalation. "
            "- '[LOW: keep it standard]' bedeutet: Routine-Element des Szenarios. "
            "\n\n"
            "Wenn Einschränkungen von Beobachtern aktiv sind, halte dich daran. "
            "Sie sagen dir, welche Inhalte oder Situationen du vermeiden sollst. "
            "\n\n"
            "Qualitätsregeln: "
            "- Sei konkret und spezifisch. Vage Situationen sind nutzlos. "
            "- Jede Situation muss sich von der vorherigen unterscheiden. "
            "- Erwähne niemals Scores, Beobachter oder den Simulations-Rahmen. "
            "- Bleibe im Szenario."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL SIGNAL ANCHORS [Fix 3, Fix 11]
# ═══════════════════════════════════════════════════════════════════════════════
#
# [Fix 3] Theory-grounded anchor set for problematic media use / digital
#   engagement research. Based on:
#     - Griffiths (2005) six-component addiction model:
#       salience, mood modification, tolerance, withdrawal, conflict, relapse
#     - DSM-5 Internet Gaming Disorder (IGD) criteria
#     - Bergen Social Media Addiction Scale (Andreassen et al. 2012)
#   Each anchor targets a specific criterion rather than a generic pole.
#
# [Fix 11] 10 anchors per pole (SA-optimised): SE = σ/√20 ≈ 0.045.
#   For domain adaptation, replace with construct-specific phrases.
#
# ── Domain adaptation guide ───────────────────────────────────────────────────
#
#   Social Work (CANS, child welfare):
#     high = ["I feel overwhelmed by what is being asked of me",
#             "I cannot manage this situation alone", ...]
#     low  = ["I have people I can turn to", "I feel capable of handling this", ...]
#
#   Healthcare (empathic communication, cf. PEC framework):
#     high = ["I moved on without addressing the patient's concern",
#             "I interrupted before they finished", ...]
#     low  = ["I made sure they felt heard", "I checked whether they understood", ...]
#
#   Counseling (therapeutic alliance, cf. WAI):
#     high = ["I did not want to say any more", "The conversation felt unsafe", ...]
#     low  = ["I felt understood and respected", "I was comfortable opening up", ...]
#
#   Education (deep learning engagement):
#     high = ["I just wanted the answer without thinking",
#             "I did not try to understand, only to finish", ...]
#     low  = ["I made a connection to something I already knew",
#             "I asked a follow-up question to understand better", ...]
#
#   Crisis intervention (de-escalation):
#     high = ["I feel more agitated now", "The situation is getting worse", ...]
#     low  = ["I feel calmer", "I can think more clearly now", ...]
#
# ── Sensitivity analysis ──────────────────────────────────────────────────────
# Sobol S_i for each anchor: run 1024-sample sweep in sim_loop.py.
# Anchors with S_i < 0.01 are candidates for pruning.
# Morris μ* provides cheaper initial ranking.

ENGAGEMENT_ANCHORS: dict[str, list[str]] = {
    "high": [
        # Salience / preoccupation (Griffiths 2005)
        "I kept thinking about it even when I was doing something else",
        "It took up more of my attention than I wanted",
        # Mood modification / compulsive use (DSM-5 IGD criterion 2)
        "I used it to escape from how I was feeling",
        "I felt better while I was engaged, so I kept going",
        # Tolerance (IGD criterion 3)
        "I needed more of it than before to get the same effect",
        "What used to satisfy me no longer felt like enough",
        # Conflict / loss of control (Griffiths 2005; Bergen scale)
        "I kept going even though I knew I should stop",
        "I could not pull myself away even when I tried",
        # Withdrawal discomfort (IGD criterion 4)
        "I felt restless and irritable when I could not engage",
        # Relapse / continued use despite consequences (IGD criterion 7)
        "I went back to it even after deciding I would stop",
    ],
    "low": [
        # Deliberate disengagement / autonomy
        "I chose to stop and found it straightforward",
        "I stepped away deliberately and felt comfortable doing so",
        # Alternative activities / balance (Bergen scale reverse)
        "I turned my attention to something else without difficulty",
        "I found other activities just as satisfying",
        # Reflection and self-regulation
        "I noticed what I was doing and decided to pause",
        "I made a conscious choice about how much time to spend",
        # Absence of withdrawal / controlled use
        "I did not feel anxious or unsettled when I disengaged",
        "I engaged briefly and then moved on without any urge to return",
        # Positive boundary maintenance
        "I maintained the boundary I had set for myself",
        # No conflict / no preoccupation
        "I was present in other areas of my life without distraction",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION CODEBOOK [unchanged in v3]
# ═══════════════════════════════════════════════════════════════════════════════
#
# Phrase keys matched via cosine similarity ≥ INTERVENTION_THRESHOLD.
# Only observer_b output is passed to extract_interventions().
# Observer_a output is explicitly designed to avoid these phrases (Fix 2).
#
# ── Three-tier intervention model (DEVS coupling) ─────────────────────────────
#   participant_nudge      → augments participant.X (Phase B input)
#   environment_constraint → constrains environment.δ_ext (Phase A output)
#   score_modifier         → applies dampening d to Phase C update
#
# ── Threshold calibration procedure ──────────────────────────────────────────
# 1. Collect 30+ observer_b outputs from pilot runs
# 2. Label each: intervention-warranted (1) / not warranted (0)
# 3. For each output, compute max cosine similarity per codebook key
# 4. Sweep θ ∈ [0.60, 0.85] and compute F1 at each value
# 5. Set INTERVENTION_THRESHOLD = argmax F1
# 6. Re-run Morris screening: if μ*(θ) >> μ*(α), threshold dominates → calibrate carefully

INTERVENTION_CODEBOOK: dict[str, list[str]] = {
    "pause_prompt": [
        "take a pause",
        "pause and reflect",
        "step back for a moment",
        "reflection break",
        "interrupt the current pattern",
        "encourage the participant to slow down",
    ],
    "boundary_warning": [
        "approaching scenario boundary",
        "flag boundary issue",
        "boundary has been reached",
        "scenario limit warning",
        "constraint threshold exceeded",
        "the environment should not escalate further",
    ],
    "pacing_adjustment": [
        "adjust the pacing",
        "slow down the interaction",
        "reduce stimulus intensity",
        "moderate the pace",
        "lower interaction frequency",
        "the pace of interaction is too fast",
    ],
    "dynamics_dampening": [
        "dampen the behavioural dynamics",
        "reduce score momentum",
        "moderate the trajectory",
        "apply stabilisation",
        "decrease behavioural acceleration",
        "slow down how quickly scores change",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIM_LOOP.PY PATCH NOTES
# ═══════════════════════════════════════════════════════════════════════════════
#
# The following changes to sim_loop.py are REQUIRED to activate v3 features.
# See inline comments for affected functions.
#
# ── [Fix 2] Phase D sequential execution ─────────────────────────────────────
# In run_tick(), replace Phase D block:
#
#   # OLD (concurrent, no A→B handoff):
#   if tick % world_state.k == 0:
#       intervention_lists = await asyncio.gather(*[
#           o.observe(tick, world_state, n) for o in observers
#       ])
#       for ivs in intervention_lists:
#           world_state.active_interventions.extend(ivs)
#
#   # NEW (sequential, A analysis → B intervention):
#   if tick % world_state.k == 0:
#       print(f"\n[Tick {tick}] Observer tick (sequential)...")
#       obs_a, obs_b = observers[0], observers[1]
#       # Phase D1: observer_a analyses — returns text, no codebook extraction
#       analysis = await obs_a.analyse(tick, world_state, n)
#       # Phase D2: observer_b proposes interventions using A's analysis
#       ivs = await obs_b.intervene(tick, world_state, n, analysis)
#       world_state.active_interventions.extend(ivs)
#
# Add to ObserverAgent class:
#
#   async def analyse(self, tick, world_state, n_participants) -> str:
#       """observer_a only: returns analysis text, no codebook extraction."""
#       ... (same as observe() but returns response string, no extract_interventions)
#
#   async def intervene(self, tick, world_state, n_participants, analysis: str):
#       """observer_b only: receives A's analysis, returns interventions."""
#       window = world_state.observer_prompt_window(tick, n_participants)
#       prompt = (
#           f"[Tick {tick}] Analyst report:\n{analysis}\n\n"
#           f"Raw observation window:\n{window}\n\n"
#           f"Active interventions: {active}\n\n"
#           f"Decide whether to intervene and which type."
#       )
#       response = await agent_turn(...)
#       return extract_interventions(self.agent_id, response, tick)
#
# ── [Fix 5] Intervention threshold from config ───────────────────────────────
# In sim_loop.py, replace:
#   INTERVENTION_THRESHOLD = 0.72
# With:
#   from agent_rolesv3 import INTERVENTION_THRESHOLD
#
# ── [Fix 7] Persona summary injection ────────────────────────────────────────
# In ParticipantAgent.step(), add:
#   from agent_rolesv3 import PERSONA_SUMMARY_INTERVAL, PERSONA_SUMMARY_TEMPLATE
#   if tick % PERSONA_SUMMARY_INTERVAL == 0 and tick > 0:
#       # Generate and cache summary
#       summary_prompt = "Summarise this participant's personality and key choices so far in 3 sentences."
#       self.persona_summary = await agent_turn(
#           agent_id="summariser", backend=self.cfg["backend"],
#           system_prompt="You summarise conversation history concisely.",
#           user_message=summary_prompt,
#           history=self.history[-20:], max_tokens=100
#       )
#   # Prepend to system if summary exists
#   system = self.cfg["system"]
#   if hasattr(self, "persona_summary") and self.persona_summary:
#       interval_note = PERSONA_SUMMARY_TEMPLATE.format(
#           interval=PERSONA_SUMMARY_INTERVAL,
#           summary=self.persona_summary
#       )
#       system = f"{interval_note}\n\n{system}"
#
# ── [Fix 9] Compliance check integration ─────────────────────────────────────
# After each agent_turn() call:
#   from agent_rolesv3 import check_compliance
#   violations = check_compliance(response, role_name)
#   if violations:
#       world_state.log_violation(tick, agent_id, violations)
#
# ── [Fix 12] Batch embedding for Phase C ─────────────────────────────────────
# Replace per-response embed_score() loop with batch:
#   from agent_rolesv3 import EMBED_BATCH_SIZE
#   # Collect all responses first (Phase B already does this via asyncio.gather)
#   vecs = _get_embed().encode(responses)  # single batched call
#   for participant, response, vec in zip(participants, responses, vecs):
#       signal = _compute_signal_from_vec(vec)  # refactor embed_score to accept pre-encoded vec
#
# ── [Fix 4] Backend from config ─────────────────────────────────────────────
# No change needed in sim_loop.py — AGENT_ROLES["observer_a"]["backend"] now
# references BACKEND_CONFIG["analyst"] dynamically.

_PATCH_NOTES_SENTINEL = True  # import guard — this module is config only
