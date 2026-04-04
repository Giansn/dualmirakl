"""CLI entry point for the simulation."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

__all__ = [
    "_prompt",
]

def _prompt(label: str, default, cast, valid_range: str = "") -> any:
    hint = f"  ({valid_range})" if valid_range else ""
    raw = input(f"  {label:<22} [default: {default}]{hint}: ").strip()
    return cast(raw) if raw else default


if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse --scenario flag from argv before interactive prompts
    _scenario_path = None
    _argv = list(_sys.argv[1:])
    if "--scenario" in _argv:
        idx = _argv.index("--scenario")
        if idx + 1 < len(_argv):
            _scenario_path = _argv[idx + 1]
            _argv = _argv[:idx] + _argv[idx + 2:]

    print("\n-- multi-agent simulation (v3) --")
    print("Modes: [s]imulation (default) | [p]ipeline | [m]orris screening | [b]sobol analysis | [o]ptuna optimization | [c]alibrate threshold\n")

    mode = input("  Mode [s/m/b/o/c/p]: ").strip().lower() or "s"

    if mode in ("p", "pipeline"):
        from simulation.config.scenario import ScenarioConfig
        from simulation.analysis.output_pipeline import OutputPipeline

        if not _scenario_path:
            _scenario_path = input("  Scenario YAML [scenarios/social_dynamics.yaml]: ").strip()
            _scenario_path = _scenario_path or "scenarios/social_dynamics.yaml"

        _pcfg = ScenarioConfig.load(_scenario_path)
        _pcfg.validate_scenario(strict=True)
        print(f"  scenario: {_pcfg.meta.name} ({_scenario_path})")

        _p_n_runs = _prompt("Ensemble runs", 10, int, "3-100")
        _p_seed = _prompt("Base seed", 42, int, "any int")

        async def _run_pipeline():
            try:
                pipeline = OutputPipeline(
                    scenario_config=_pcfg,
                    on_stage=lambda name, status: print(f"  [{name}] {status}"),
                )
                res = await pipeline.run(n_runs=_p_n_runs, base_seed=_p_seed)

                print(f"\n  Pipeline completed in {res.total_duration_s:.1f}s")
                for stage in res.stages:
                    icon = {"completed": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(stage.status, "?")
                    line = f"    [{icon}] {stage.name} ({stage.duration_s:.1f}s)"
                    if stage.skip_reason:
                        line += f" -- {stage.skip_reason}"
                    if stage.error:
                        line += f" -- {stage.error}"
                    print(line)

                if res.validation:
                    v = res.validation
                    print(f"\n  Validation: {'PASSED' if v['passed'] else 'FAILED'}")
                    for name, rc in v.get("range_checks", {}).items():
                        status = "in range" if rc["in_range"] else "OUT OF RANGE"
                        print(f"    {name}: {rc['median_forecast']:.4f} "
                              f"[{rc['plausible_range'][0]}, {rc['plausible_range'][1]}] "
                              f"-- {status}")

                if res.possibility_report:
                    from simulation.analysis.possibility_report import render_cli
                    print(render_cli(res.possibility_report))

                print(f"\n  Artifacts: {pipeline.output_dir}")
            finally:
                from orchestrator import close_client
                await close_client()

        asyncio.run(_run_pipeline())

    elif mode in ("c", "calibrate"):
        from simulation.signal.sensitivity import calibrate_intervention_threshold
        data_dir = Path("data")
        if data_dir.exists():
            run_dirs = sorted([str(d) for d in data_dir.iterdir() if d.is_dir() and (d / "observations.json").exists()])
        else:
            run_dirs = []
        if not run_dirs:
            print("  No completed runs found in data/. Run pilot simulations first.")
        else:
            print(f"  Found {len(run_dirs)} completed run(s)")
            result = calibrate_intervention_threshold(run_dirs)
            if result["best_f1"] > 0:
                apply = input(f"  Apply θ={result['optimal_theta']:.2f}? [y/N]: ").strip().lower()
                if apply == "y":
                    os.environ["INTERVENTION_THRESHOLD"] = str(result["optimal_theta"])
                    print(f"  Set INTERVENTION_THRESHOLD={result['optimal_theta']:.2f} for this session")

    elif mode in ("m", "morris"):
        from simulation.signal.sensitivity import run_sensitivity_analysis
        run_sensitivity_analysis(mode="morris")

    elif mode in ("b", "sobol"):
        from simulation.signal.sensitivity import run_sensitivity_analysis
        run_sensitivity_analysis(mode="sobol")

    elif mode in ("o", "optuna"):
        from simulation.optimize.optuna import run_optimization
        _e = lambda k, d: os.environ.get(k, d)
        opt_mode = _prompt("Optuna mode", "fast", str, "fast|full")
        n_trials = _prompt("Number of trials", 100, int, "10-1000")
        include_flame = _prompt("Include FLAME params", _e("FLAME_ENABLED", "0"), str, "0|1") == "1"
        target_mean = _prompt("Target mean score", 0.5, float, "0.0-1.0")
        run_optimization(
            mode=opt_mode, n_trials=n_trials,
            include_flame=include_flame, target_mean=target_mean,
        )

    else:
        # Load scenario config if --scenario was provided
        _scenario_cfg = None
        if _scenario_path:
            from simulation.config.scenario import ScenarioConfig
            _scenario_cfg = ScenarioConfig.load(_scenario_path)
            _scenario_cfg.validate_scenario(strict=True)
            print(f"  scenario: {_scenario_cfg.meta.name} ({_scenario_path})")

        # Defaults from env vars (set in .env), overridable via interactive prompt
        _e = lambda k, d: os.environ.get(k, d)
        n_ticks        = _prompt("Ticks to simulate",      int(_e("SIM_N_TICKS", "12")),       int,   "1-168")
        n_participants = _prompt("Number of participants",  int(_e("SIM_N_PARTICIPANTS", "4")), int,   "1-50")
        k              = _prompt("Observer frequency K",    int(_e("SIM_OBSERVER_K", "4")),     int,   "1-n_ticks")
        alpha          = _prompt("EMA alpha",               float(_e("SIM_ALPHA", "0.15")),     float, "0.1-0.4")
        history_window = _prompt("History window (turns)",  int(_e("SIM_HISTORY_WINDOW", "4")), int,   "1-12")
        max_tokens     = _prompt("Context size (tokens)",   int(_e("SIM_MAX_TOKENS", "192")),   int,   "64-8192")
        seed           = _prompt("Random seed",             int(_e("SIM_SEED", "42")),          int,   "any int")
        score_mode     = _prompt("Score mode",              _e("SIM_SCORE_MODE", "ema"),        str,   "ema|logistic")
        logistic_k     = _prompt("Logistic steepness k",    float(_e("SIM_LOGISTIC_K", "6.0")), float, "3.0-10.0")

        # FLAME GPU 2 (optional 3rd GPU)
        flame_default = _e("FLAME_ENABLED", "0")
        flame_input = _prompt("FLAME GPU 2 (3rd GPU)",  flame_default,  str,   "0=off, 1=on")
        use_flame = flame_input == "1"

        async def main():
            try:
                await run_simulation(
                    n_ticks=n_ticks, n_participants=n_participants,
                    k=k, alpha=alpha, history_window=history_window,
                    max_tokens=max_tokens, seed=seed,
                    score_mode=score_mode, logistic_k=logistic_k,
                    flame_enabled=use_flame,
                    scenario_config=_scenario_cfg,
                )
            finally:
                await close_client()

        asyncio.run(main())
