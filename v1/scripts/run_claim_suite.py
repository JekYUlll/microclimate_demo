#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "v1" / "scripts" / "run_protocol_gate.py"

TASK_COLUMNS = [
    "snow_mass_flux_kg_m2_s",
    "snow_particle_mean_diameter_mm",
    "snow_particle_mean_velocity_ms",
]
TASK_SCALES = ["1e-4", "0.2", "5.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v1 multi-seed claim suite.")
    parser.add_argument(
        "--input-root",
        default="rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov",
        help="Directory containing per-seed inputs as <budget-tag>_seed<seed>/.",
    )
    parser.add_argument("--budget-tag", default="budget1p20")
    parser.add_argument("--out-root", default="v1/artifacts/claim_suite_semimarkov_n5")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=[
            "main",
            "main_safe",
            "safe_dagger3",
            "knn_safe",
            "knn_safe_dagger3",
            "cost_safe",
            "support1_safe",
            "support2_safe",
            "support3_safe",
            "support4_safe",
            "support5_safe",
            "support6_safe",
            "support8_safe",
            "support12_safe",
            "support_calib_safe",
            "mask_safe",
            "mask_anchor_safe",
            "hybrid_val_safe",
            "residual_safe",
            "oracle_context_safe",
            "value_residual_safe",
            "value_residual_no_dagger",
            "value_residual_oracle_objective",
            "learned_value_residual_safe",
            "learned_ensemble_value_safe",
            "learned_advantage_residual_safe",
            "learned_advantage_residual_calib_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_hybrid_residual_calib_safe",
            "learned_hybrid_residual_guarded_safe",
            "learned_event_threshold_guarded_safe",
            "learned_event_threshold_valguard_safe",
            "learned_event_threshold_strict_valguard_safe",
            "learned_event_threshold_riskcalib_safe",
            "learned_event_threshold_riskcenter_safe",
            "learned_hybrid_event_guarded_safe",
            "learned_hybrid_event_cycle_guarded_safe",
            "learned_hybrid_rate_guarded_safe",
            "learned_hybrid_rate_riskcenter_safe",
            "learned_hybrid_sequence_guarded_safe",
            "learned_hybrid_teacher_mix_guarded_safe",
            "learned_hybrid_contextual_duty_guarded_safe",
            "learned_hybrid_contextual_duty_guardcalib_safe",
            "learned_hybrid_contextual_duty_riskcenter_safe",
            "learned_hybrid_contextual_duty_riskband_safe",
            "learned_hybrid_sequence_mask_guarded_safe",
            "learned_hybrid_recurrent_value_guarded_safe",
            "learned_hybrid_recurrent_rank_guarded_safe",
            "learned_hybrid_recurrent_value_posguard_safe",
            "learned_hybrid_recurrent_rank_posguard_safe",
            "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
            "learned_hybrid_recurrent_advantage_posguard_safe",
            "learned_hybrid_option_planner_posguard_safe",
            "learned_option_planner_startguard_safe",
            "learned_option_runtime_risk_guard_safe",
            "learned_option_runtime_risk_denseval_safe",
            "learned_cost_knn_riskband_safe",
            "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_sequence_value_continuous_augmented_riskband_safe",
            "learned_teacher_improvement_gate_smoke",
            "learned_rollout_value_posguard_safe",
            "learned_rollout_value_self_posguard_safe",
            "learned_twin_rollout_posguard_safe",
            "learned_window_candidate_margin_safe",
            "learned_window_candidate_fullrollout_margin_safe",
            "learned_utility_planner_riskband_safe",
            "learned_proxy_mpc_riskband_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_hybrid_bc_guarded_safe",
            "learned_hybrid_planner_guarded_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
            "cost_support6_safe",
            "no_dagger",
            "oracle_objective",
            "no_anchor_guard",
        ],
        default=["main"],
    )
    parser.add_argument("--sensor-cfg", default="configs/sensors/windblown_sensors_physical_event_v4.yaml")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--bc-device", default="cpu")
    parser.add_argument("--selection", choices=["event_rich", "event_transport_rich", "uniform"], default="event_rich")
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.20)
    parser.add_argument("--startup-peak-budget", type=float, default=1.60)
    parser.add_argument("--energy-capacity", type=float, default=180.0)
    parser.add_argument("--initial-energy", type=float, default=180.0)
    parser.add_argument("--harvest-per-step", type=float, default=0.92)
    parser.add_argument("--reserve-energy", type=float, default=20.0)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--train-rollouts", type=int, default=4)
    parser.add_argument("--static-selection-steps", type=int, default=256)
    parser.add_argument("--static-selection-rollouts", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--task-error-weight", type=float, default=0.2)
    parser.add_argument("--gpu-ids", nargs="*", default=[])
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-rule-baselines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-validation-cyclic-policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-cyclic-top-k", type=int, default=4)
    parser.add_argument("--validation-cyclic-dwell-grid", nargs="*", type=int, default=[2, 4, 8, 16])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    if args.dry_run:
        for item in commands:
            print(format_command(item.command, env=item.env))
        return
    run_commands(commands, max_parallel=resolve_parallelism(args), continue_on_error=bool(args.continue_on_error))


class CommandItem:
    def __init__(self, command: list[str], out_dir: Path, log_path: Path, env: dict[str, str]) -> None:
        self.command = command
        self.out_dir = out_dir
        self.log_path = log_path
        self.env = env


def build_commands(args: argparse.Namespace) -> list[CommandItem]:
    commands: list[CommandItem] = []
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    gpu_ids = [str(item) for item in args.gpu_ids]
    job_index = 0
    for preset in args.presets:
        for seed in args.seeds:
            seed_dir = input_root / f"{args.budget_tag}_seed{int(seed)}"
            truth_csv = seed_dir / "truth_energy_split.csv"
            oracle_path = seed_dir / "v2_tcn_oracle.pt"
            if not truth_csv.exists() or not oracle_path.exists():
                raise FileNotFoundError(f"Missing seed inputs under {seed_dir}")
            out_dir = out_root / f"{preset}_seed{int(seed)}"
            if bool(args.skip_existing) and (out_dir / "gate_summary.json").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[job_index % len(gpu_ids)]
            command = base_command(
                seed=int(seed),
                truth_csv=truth_csv,
                oracle_path=oracle_path,
                out_dir=out_dir,
                preset=str(preset),
                sensor_cfg=str(args.sensor_cfg),
                oracle_device=str(args.oracle_device),
                bc_device=str(args.bc_device),
                selection=str(args.selection),
                max_active=int(args.max_active),
                budget=float(args.budget),
                startup_peak_budget=float(args.startup_peak_budget),
                energy_capacity=float(args.energy_capacity),
                initial_energy=float(args.initial_energy),
                harvest_per_step=float(args.harvest_per_step),
                reserve_energy=float(args.reserve_energy),
                train_steps=int(args.train_steps),
                train_rollouts=int(args.train_rollouts),
                static_selection_steps=int(args.static_selection_steps),
                static_selection_rollouts=int(args.static_selection_rollouts),
                eval_steps=int(args.eval_steps),
                eval_rollouts=int(args.eval_rollouts),
                task_error_weight=float(args.task_error_weight),
                include_rule_baselines=bool(args.include_rule_baselines),
                include_validation_cyclic_policy=bool(args.include_validation_cyclic_policy),
                validation_cyclic_top_k=int(args.validation_cyclic_top_k),
                validation_cyclic_dwell_grid=tuple(int(x) for x in args.validation_cyclic_dwell_grid),
            )
            commands.append(CommandItem(command=command, out_dir=out_dir, log_path=out_dir / "run.log", env=env))
            job_index += 1
    return commands


def base_command(
    *,
    seed: int,
    truth_csv: Path,
    oracle_path: Path,
    out_dir: Path,
    preset: str,
    sensor_cfg: str,
    oracle_device: str,
    bc_device: str,
    selection: str = "event_rich",
    max_active: int = 4,
    budget: float,
    startup_peak_budget: float,
    energy_capacity: float,
    initial_energy: float = 180.0,
    harvest_per_step: float,
    reserve_energy: float = 20.0,
    train_steps: int,
    train_rollouts: int,
    static_selection_steps: int,
    static_selection_rollouts: int,
    eval_steps: int,
    eval_rollouts: int,
    task_error_weight: float = 0.2,
    include_rule_baselines: bool,
    include_validation_cyclic_policy: bool = False,
    validation_cyclic_top_k: int = 4,
    validation_cyclic_dwell_grid: tuple[int, ...] = (2, 4, 8, 16),
) -> list[str]:
    common = [
        sys.executable,
        str(RUNNER),
        "--truth-csv",
        str(truth_csv),
        "--sensor-cfg",
        sensor_cfg,
        "--oracle-path",
        str(oracle_path),
        "--oracle-type",
        "tcn",
        "--oracle-device",
        oracle_device,
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--freq-s",
        "10800",
        "--split-ratios",
        "0.30",
        "0.45",
        "0.125",
        "0.125",
        "--selection",
        str(selection),
        "--selection-stride",
        "64",
        "--lookback",
        "20",
        "--horizon",
        "8",
        "--train-steps",
        str(int(train_steps)),
        "--train-rollouts",
        str(int(train_rollouts)),
        "--static-selection-steps",
        str(int(static_selection_steps)),
        "--static-selection-rollouts",
        str(int(static_selection_rollouts)),
        "--eval-steps",
        str(int(eval_steps)),
        "--eval-rollouts",
        str(int(eval_rollouts)),
        "--max-active",
        str(int(max_active)),
        "--budget",
        f"{float(budget):.6g}",
        "--startup-peak-budget",
        f"{float(startup_peak_budget):.6g}",
        "--energy-account",
        "--energy-capacity",
        f"{float(energy_capacity):.6g}",
        "--initial-energy",
        f"{float(initial_energy):.6g}",
        "--harvest-per-step",
        f"{float(harvest_per_step):.6g}",
        "--reserve-energy",
        f"{float(reserve_energy):.6g}",
        "--lambda-warmup-abort",
        "0.08",
        "--planning-horizon",
        "3",
        "--beam-width",
        "4",
        "--max-branch",
        "8",
        "--teacher-lambda-warmup-abort",
        "0.16",
        "--candidate-prior-weight",
        "0.5",
        "--candidate-prefilter-top-k",
        "24",
        "--teacher-anchor-source",
        "validation_best",
        "--anchor-improvement-margin",
        "0.002",
        "--bc-epochs",
        "100",
        "--bc-hidden-dim",
        "256",
        "--bc-device",
        bc_device,
        "--bc-batch-size",
        "128",
    ]
    if include_rule_baselines:
        common.append("--include-rule-baselines")
    else:
        common.append("--no-include-rule-baselines")
    if include_validation_cyclic_policy:
        common.extend(
            [
                "--include-validation-cyclic-policy",
                "--validation-cyclic-top-k",
                str(int(validation_cyclic_top_k)),
                "--validation-cyclic-dwell-grid",
                *[str(int(x)) for x in validation_cyclic_dwell_grid],
                "--validation-cyclic-preserve-warming",
            ]
        )
    else:
        common.append("--no-include-validation-cyclic-policy")

    if preset in {
        "main",
        "main_safe",
        "safe_dagger3",
        "knn_safe",
        "knn_safe_dagger3",
        "cost_safe",
        "support1_safe",
        "support2_safe",
        "support3_safe",
        "support4_safe",
        "support5_safe",
        "support6_safe",
        "support8_safe",
        "support12_safe",
        "support_calib_safe",
        "mask_safe",
        "mask_anchor_safe",
        "hybrid_val_safe",
        "residual_safe",
        "oracle_context_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_option_planner_startguard_safe",
        "learned_option_runtime_risk_guard_safe",
        "learned_option_runtime_risk_denseval_safe",
        "learned_cost_knn_riskband_safe",
        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_sequence_value_continuous_augmented_riskband_safe",
        "learned_teacher_improvement_gate_smoke",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
        "learned_twin_rollout_posguard_safe",
        "learned_window_candidate_margin_safe",
        "learned_window_candidate_fullrollout_margin_safe",
        "learned_utility_planner_riskband_safe",
        "learned_proxy_mpc_riskband_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
        "cost_support6_safe",
        "no_dagger",
        "no_anchor_guard",
    }:
        common.extend(
            [
                "--objective-mode",
                "task_composite",
                "--task-error-weight",
                f"{float(task_error_weight):.6g}",
                "--task-error-columns",
                *TASK_COLUMNS,
                "--task-error-scales",
                *TASK_SCALES,
            ]
        )
    elif preset in {"oracle_objective", "value_residual_oracle_objective"}:
        common.extend(["--objective-mode", "oracle", "--task-error-weight", "0.0"])
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if preset == "no_anchor_guard":
        common.append("--no-anchor-regret-guard")
    else:
        common.append("--anchor-regret-guard")

    if preset in {
        "main_safe",
        "safe_dagger3",
        "knn_safe",
        "knn_safe_dagger3",
        "cost_safe",
        "support1_safe",
        "support2_safe",
        "support3_safe",
        "support4_safe",
        "support5_safe",
        "support6_safe",
        "support8_safe",
        "support12_safe",
        "support_calib_safe",
        "mask_safe",
        "mask_anchor_safe",
        "hybrid_val_safe",
        "residual_safe",
        "oracle_context_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_option_planner_startguard_safe",
        "learned_option_runtime_risk_guard_safe",
        "learned_option_runtime_risk_denseval_safe",
        "learned_cost_knn_riskband_safe",
        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_teacher_improvement_gate_smoke",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
        "learned_rollout_value_oracle_regime_posguard_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_window_eligibility_posguard_safe",
        "learned_window_macro_eligibility_posguard_safe",
        "learned_utility_planner_riskband_safe",
        "learned_proxy_mpc_riskband_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
        "cost_support6_safe",
    }:
        common.append("--bc-preserve-warming")
    else:
        common.append("--no-bc-preserve-warming")

    support_top_k = {
        "support1_safe": 1,
        "support2_safe": 2,
        "support3_safe": 3,
        "support4_safe": 4,
        "support5_safe": 5,
        "support6_safe": 6,
        "support8_safe": 8,
        "support12_safe": 12,
        "cost_support6_safe": 6,
        "oracle_context_safe": 5,
        "value_residual_safe": 5,
        "value_residual_no_dagger": 5,
        "value_residual_oracle_objective": 5,
        "learned_value_residual_safe": 5,
        "learned_ensemble_value_safe": 8,
        "learned_advantage_residual_safe": 6,
        "learned_advantage_residual_calib_safe": 6,
        "learned_advantage_oracle_regime_posguard_safe": 12,
        "learned_hybrid_residual_calib_safe": 6,
        "learned_hybrid_residual_guarded_safe": 6,
        "learned_event_threshold_guarded_safe": 6,
        "learned_event_threshold_valguard_safe": 6,
        "learned_event_threshold_strict_valguard_safe": 6,
        "learned_event_threshold_riskcalib_safe": 6,
        "learned_event_threshold_riskcenter_safe": 6,
        "learned_hybrid_event_guarded_safe": 6,
        "learned_hybrid_event_cycle_guarded_safe": 6,
        "learned_hybrid_rate_guarded_safe": 6,
        "learned_hybrid_rate_riskcenter_safe": 6,
        "learned_hybrid_sequence_guarded_safe": 6,
        "learned_hybrid_teacher_mix_guarded_safe": 6,
        "learned_hybrid_contextual_duty_guarded_safe": 6,
        "learned_hybrid_contextual_duty_guardcalib_safe": 6,
        "learned_hybrid_contextual_duty_riskcenter_safe": 6,
        "learned_hybrid_contextual_duty_riskband_safe": 6,
        "learned_hybrid_sequence_mask_guarded_safe": 6,
        "learned_hybrid_recurrent_value_guarded_safe": 6,
        "learned_hybrid_recurrent_rank_guarded_safe": 6,
        "learned_hybrid_recurrent_value_posguard_safe": 6,
        "learned_hybrid_recurrent_rank_posguard_safe": 6,
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe": 6,
        "learned_hybrid_recurrent_advantage_posguard_safe": 6,
        "learned_hybrid_option_planner_posguard_safe": 6,
        "learned_option_planner_startguard_safe": 6,
        "learned_option_runtime_risk_guard_safe": 6,
        "learned_option_runtime_risk_denseval_safe": 6,
        "learned_cost_knn_riskband_safe": 6,
        "learned_macro_option_riskband_safe": 6,
        "learned_macro_option_dense_always_safe": 12,
        "learned_sequence_value_riskband_safe": 12,
        "learned_sequence_value_fullbank_riskband_safe": 12,
        "learned_sequence_value_oracle_context_fullbank_safe": 12,
        "learned_sequence_value_oracle_regime_fullbank_safe": 12,
        "learned_teacher_improvement_gate_smoke": 6,
        "learned_rollout_value_posguard_safe": 8,
        "learned_rollout_value_self_posguard_safe": 8,
        "learned_rollout_value_oracle_regime_posguard_safe": 12,
        "learned_window_eligibility_posguard_safe": 16,
        "learned_window_macro_eligibility_posguard_safe": 16,
        "learned_utility_planner_riskband_safe": 16,
        "learned_proxy_mpc_riskband_safe": 16,
        "learned_hybrid_bc_guarded_safe": 6,
        "learned_hybrid_planner_guarded_safe": 8,
    }.get(preset, 0)
    common.extend(["--bc-action-support-top-k", str(support_top_k)])
    if preset == "support_calib_safe":
        common.extend(["--bc-action-support-grid", "0", "4", "6", "8", "12"])
    if preset in {
        "mask_safe",
        "mask_anchor_safe",
        "residual_safe",
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_option_planner_startguard_safe",
        "learned_option_runtime_risk_guard_safe",
        "learned_option_runtime_risk_denseval_safe",
        "learned_cost_knn_riskband_safe",
        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_teacher_improvement_gate_smoke",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_window_eligibility_posguard_safe",
        "learned_window_macro_eligibility_posguard_safe",
        "learned_utility_planner_riskband_safe",
        "learned_proxy_mpc_riskband_safe",
        "learned_hybrid_planner_guarded_safe",
    }:
        common.append("--no-include-bc-policy")
    else:
        common.append("--include-bc-policy")

    if preset in {"knn_safe", "knn_safe_dagger3", "learned_hybrid_bc_guarded_safe"}:
        common.extend(["--include-knn-policy", "--knn-k", "7"])
    else:
        common.append("--no-include-knn-policy")
    if preset in {"mask_safe", "mask_anchor_safe", "hybrid_val_safe"}:
        common.append("--include-mask-bc-policy")
        if preset in {"mask_anchor_safe", "hybrid_val_safe"}:
            common.extend(["--mask-bc-anchor-bias", "0.25"])
    else:
        common.append("--no-include-mask-bc-policy")
    if preset in {
        "hybrid_val_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_option_planner_startguard_safe",
        "learned_option_runtime_risk_guard_safe",
        "learned_option_runtime_risk_denseval_safe",
        "learned_cost_knn_riskband_safe",
        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_teacher_improvement_gate_smoke",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
        "learned_window_eligibility_posguard_safe",
        "learned_window_macro_eligibility_posguard_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
    }:
        common.extend(["--deployable-selection", "validation"])
        if preset in {
            "learned_hybrid_residual_guarded_safe",
            "learned_event_threshold_guarded_safe",
            "learned_event_threshold_valguard_safe",
            "learned_event_threshold_strict_valguard_safe",
            "learned_event_threshold_riskcalib_safe",
            "learned_event_threshold_riskcenter_safe",
            "learned_hybrid_event_guarded_safe",
            "learned_hybrid_event_cycle_guarded_safe",
            "learned_hybrid_rate_guarded_safe",
            "learned_hybrid_rate_riskcenter_safe",
            "learned_hybrid_sequence_guarded_safe",
            "learned_hybrid_teacher_mix_guarded_safe",
            "learned_hybrid_contextual_duty_guarded_safe",
            "learned_hybrid_contextual_duty_guardcalib_safe",
            "learned_hybrid_contextual_duty_riskcenter_safe",
            "learned_hybrid_contextual_duty_riskband_safe",
            "learned_hybrid_sequence_mask_guarded_safe",
            "learned_hybrid_recurrent_value_guarded_safe",
            "learned_hybrid_recurrent_rank_guarded_safe",
            "learned_hybrid_recurrent_value_posguard_safe",
            "learned_hybrid_recurrent_rank_posguard_safe",
            "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
            "learned_hybrid_recurrent_advantage_posguard_safe",
            "learned_hybrid_option_planner_posguard_safe",
            "learned_option_planner_startguard_safe",
            "learned_option_runtime_risk_guard_safe",
            "learned_option_runtime_risk_denseval_safe",
            "learned_cost_knn_riskband_safe",
            "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_teacher_improvement_gate_smoke",
            "learned_rollout_value_posguard_safe",
            "learned_rollout_value_self_posguard_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
            "learned_hybrid_bc_guarded_safe",
            "learned_hybrid_planner_guarded_safe",
        }:
            min_mean_margin = (
                "0.001"
                if preset
                in {
                    "learned_hybrid_recurrent_value_posguard_safe",
                    "learned_hybrid_recurrent_rank_posguard_safe",
                    "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
                    "learned_hybrid_recurrent_advantage_posguard_safe",
                    "learned_hybrid_option_planner_posguard_safe",
                    "learned_option_planner_startguard_safe",
                    "learned_option_runtime_risk_guard_safe",
                    "learned_option_runtime_risk_denseval_safe",
                    "learned_cost_knn_riskband_safe",
                    "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
                    "learned_teacher_improvement_gate_smoke",
                    "learned_rollout_value_posguard_safe",
                    "learned_rollout_value_self_posguard_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
                    "learned_event_threshold_valguard_safe",
                    "learned_event_threshold_strict_valguard_safe",
                    "learned_event_threshold_riskcalib_safe",
                    "learned_event_threshold_riskcenter_safe",
                    "learned_hybrid_rate_riskcenter_safe",
                    "learned_hybrid_contextual_duty_riskcenter_safe",
                    "learned_hybrid_contextual_duty_riskband_safe",
                }
                else "0.0"
            )
            common.extend(
                [
                    "--deployable-selection-criterion",
                    "static_margin_risk"
                    if preset in {
                        "learned_event_threshold_riskcalib_safe",
                        "learned_event_threshold_riskcenter_safe",
                        "learned_hybrid_rate_riskcenter_safe",
                        "learned_hybrid_contextual_duty_riskcenter_safe",
                        "learned_hybrid_contextual_duty_riskband_safe",
                        "learned_option_runtime_risk_guard_safe",
                        "learned_option_runtime_risk_denseval_safe",
                        "learned_cost_knn_riskband_safe",
                        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
                        "learned_teacher_improvement_gate_smoke",
                        "learned_rollout_value_self_posguard_safe",
                        "learned_rollout_value_oracle_regime_posguard_safe",
                        "learned_advantage_oracle_regime_posguard_safe",
                        "learned_window_eligibility_posguard_safe",
                        "learned_window_macro_eligibility_posguard_safe",
                    }
                    else "static_margin_guard",
                    "--deployable-selection-min-mean-margin",
                    min_mean_margin,
                    "--deployable-selection-min-start-margin",
                    "-0.01",
                    "--deployable-selection-max-negative-starts",
                    "0" if preset == "learned_option_planner_startguard_safe" else "1",
                ]
            )
            if preset in {
                "learned_event_threshold_strict_valguard_safe",
                "learned_hybrid_option_planner_posguard_safe",
                "learned_option_planner_startguard_safe",
                "learned_rollout_value_posguard_safe",
                "learned_rollout_value_self_posguard_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
            }:
                common.append("--deployable-selection-require-guard-pass")
            if preset in {
                "learned_event_threshold_riskcenter_safe",
                "learned_hybrid_rate_riskcenter_safe",
                "learned_hybrid_contextual_duty_riskcenter_safe",
                "learned_hybrid_contextual_duty_riskband_safe",
                "learned_option_runtime_risk_guard_safe",
                "learned_option_runtime_risk_denseval_safe",
                "learned_cost_knn_riskband_safe",
                "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
                "learned_rollout_value_oracle_regime_posguard_safe",
                "learned_rollout_value_self_posguard_safe",
                "learned_advantage_oracle_regime_posguard_safe",
                "learned_window_eligibility_posguard_safe",
                "learned_window_macro_eligibility_posguard_safe",
                "learned_teacher_improvement_gate_smoke",
            }:
                common.append("--deployable-selection-require-positive-center")
            if preset in {
                "learned_hybrid_contextual_duty_riskband_safe",
                "learned_option_runtime_risk_denseval_safe",
                "learned_cost_knn_riskband_safe",
                "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_rollout_value_self_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
            }:
                q25_margin = "0.0" if preset in {
                    "learned_option_runtime_risk_denseval_safe",
                    "learned_cost_knn_riskband_safe",
                    "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_rollout_value_self_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
                } else "-0.005"
                max_negative_starts = "1" if preset in {
                    "learned_option_runtime_risk_denseval_safe",
                    "learned_cost_knn_riskband_safe",
                    "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
            "learned_sequence_value_oracle_context_fullbank_safe",
            "learned_sequence_value_oracle_regime_fullbank_safe",
            "learned_rollout_value_oracle_regime_posguard_safe",
            "learned_rollout_value_self_posguard_safe",
            "learned_advantage_oracle_regime_posguard_safe",
            "learned_window_eligibility_posguard_safe",
            "learned_window_macro_eligibility_posguard_safe",
                } else "4"
                common.extend(
                    [
                        "--deployable-selection-require-risk-band",
                        "--deployable-selection-risk-min-q25-margin",
                        q25_margin,
                        "--deployable-selection-risk-max-negative-starts",
                        max_negative_starts,
                    ]
                )
    if preset == "residual_safe":
        common.extend(
            [
                "--include-residual-bc-policy",
                "--residual-bc-support-top-k",
                "5",
                "--residual-deviation-threshold-grid",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.65",
                "0.8",
                "0.9",
                "0.98",
            ]
        )
    else:
        common.append("--no-include-residual-bc-policy")
    if preset in {
        "oracle_context_safe",
        "learned_sequence_value_oracle_context_fullbank_safe",
        "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_rollout_value_oracle_regime_posguard_safe",
        "learned_advantage_oracle_regime_posguard_safe",
    }:
        common.append("--forecast-truth-future")
    if preset in {
        "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_rollout_value_oracle_regime_posguard_safe",
        "learned_advantage_oracle_regime_posguard_safe",
    }:
        common.extend(
            [
                "--forecast-continuous-truth-future",
                "--forecast-continuous-columns",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "0.0001",
                "0.2",
                "5.0",
            ]
        )
    if preset in {
        "value_residual_safe",
        "value_residual_no_dagger",
        "value_residual_oracle_objective",
        "learned_value_residual_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
    }:
        common.extend(
            [
                "--include-value-residual-policy",
                "--value-residual-support-top-k",
                "5",
                "--value-residual-advantage-grid",
                "-1.0",
                "-0.5",
                "-0.2",
                "-0.1",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
                "1.0",
            ]
        )
    else:
        common.append("--no-include-value-residual-policy")
    if preset in {
        "learned_hybrid_planner_guarded_safe",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
        "learned_rollout_value_oracle_regime_posguard_safe",
    }:
        common.extend(
            [
                "--include-rollout-value-policy",
                "--rollout-value-support-top-k",
                "8",
                "--rollout-value-depth",
                "2",
                "--rollout-value-beam-width",
                "4",
                "--rollout-value-max-branch",
                "6",
                "--rollout-value-discount",
                "0.95",
                "--rollout-value-advantage-grid",
                "-1.0",
                "-0.5",
                "-0.2",
                "-0.1",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
                "1.0",
            ]
        )
        if preset == "learned_rollout_value_self_posguard_safe":
            common.extend(
                [
                    "--rollout-value-self-iters",
                    "1",
                    "--rollout-value-self-steps",
                    "128",
                    "--rollout-value-self-threshold",
                    "0.0",
                ]
            )
    else:
        common.append("--no-include-rollout-value-policy")
    if preset in {
        "learned_sequence_value_riskband_safe",
        "learned_sequence_value_fullbank_riskband_safe",
        "learned_sequence_value_oracle_context_fullbank_safe",
        "learned_sequence_value_oracle_regime_fullbank_safe",
    }:
        sequence_top_k = "128" if preset == "learned_sequence_value_riskband_safe" else "512"
        sequence_thresholds = (
            ["-0.05", "0.0", "0.01", "0.025", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5"]
            if preset in {
                "learned_sequence_value_fullbank_riskband_safe",
                "learned_sequence_value_oracle_context_fullbank_safe",
                "learned_sequence_value_oracle_regime_fullbank_safe",
            }
            else ["-0.05", "0.0", "0.01", "0.025", "0.05", "0.1"]
        )
        common.extend(
            [
                "--include-sequence-value-policy",
                "--sequence-value-segment-len",
                "8",
                "--sequence-value-snippet-stride",
                "4",
                "--sequence-value-negatives-per-state",
                "3",
                "--sequence-value-max-rows",
                "4096",
                "--sequence-value-top-k-sequences",
                sequence_top_k,
                "--sequence-value-advantage-grid",
                *sequence_thresholds,
            ]
        )
    else:
        common.append("--no-include-sequence-value-policy")
    if preset == "learned_ensemble_value_safe":
        common.extend(
            [
                "--include-ensemble-value-policy",
                "--ensemble-value-support-top-k",
                "8",
                "--ensemble-value-size",
                "5",
                "--ensemble-value-beta-grid",
                "0.0",
                "0.25",
                "0.5",
                "1.0",
                "--ensemble-value-advantage-grid",
                "-0.5",
                "-0.2",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
            ]
        )
    else:
        common.append("--no-include-ensemble-value-policy")
    if preset in {
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
    }:
        common.extend(
            [
                "--include-advantage-residual-policy",
                "--advantage-residual-support-top-k",
                "6",
                "--advantage-residual-grid",
                "-0.2",
                "-0.1",
                "0.0",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.8",
                "1.0",
            ]
        )
        if preset in {
            "learned_advantage_residual_calib_safe",
            "learned_hybrid_residual_calib_safe",
            "learned_hybrid_residual_guarded_safe",
            "learned_hybrid_event_guarded_safe",
            "learned_hybrid_event_cycle_guarded_safe",
            "learned_hybrid_rate_guarded_safe",
            "learned_hybrid_sequence_guarded_safe",
            "learned_hybrid_teacher_mix_guarded_safe",
        }:
            common.extend(["--advantage-residual-support-grid", "3", "5", "6", "8", "12"])
        if preset == "learned_advantage_oracle_regime_posguard_safe":
            common.extend(["--advantage-residual-support-grid", "6", "12"])
    else:
        common.append("--no-include-advantage-residual-policy")
    if preset in {
        "learned_value_residual_safe",
        "learned_ensemble_value_safe",
        "learned_advantage_residual_safe",
        "learned_advantage_residual_calib_safe",
        "learned_hybrid_residual_calib_safe",
        "learned_hybrid_residual_guarded_safe",
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_option_planner_startguard_safe",
        "learned_option_runtime_risk_guard_safe",
        "learned_option_runtime_risk_denseval_safe",
        "learned_cost_knn_riskband_safe",
        "learned_macro_option_riskband_safe",
            "learned_macro_option_dense_always_safe",
            "learned_sequence_value_riskband_safe",
            "learned_sequence_value_fullbank_riskband_safe",
        "learned_teacher_improvement_gate_smoke",
        "learned_rollout_value_posguard_safe",
        "learned_rollout_value_self_posguard_safe",
        "learned_window_eligibility_posguard_safe",
        "learned_window_macro_eligibility_posguard_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
    }:
        common.extend(
            [
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
            ]
        )
    if preset in {"cost_safe", "cost_support6_safe"}:
        common.extend(["--include-cost-policy", "--cost-epochs", "50", "--cost-hidden-dim", "256"])
    else:
        common.append("--no-include-cost-policy")
    if preset == "learned_cost_knn_riskband_safe":
        common.extend(
            [
                "--include-cost-knn-policy",
                "--cost-knn-support-top-k",
                "16",
                "--cost-knn-calibration-criterion",
                "static_margin_risk",
                "--cost-knn-k-grid",
                "4",
                "8",
                "16",
                "32",
                "--cost-knn-advantage-grid",
                "0.0",
                "0.01",
                "0.025",
                "0.05",
                "0.1",
                "--cost-knn-distance-weighting-grid",
                "inverse",
                "uniform",
            ]
        )
    else:
        common.append("--no-include-cost-knn-policy")
    if preset in {"learned_macro_option_riskband_safe", "learned_macro_option_dense_always_safe"}:
        if preset == "learned_macro_option_dense_always_safe":
            macro_segment_grid = ["8", "16", "32"]
            macro_k_grid = ["4", "8", "16"]
            macro_threshold_grid = ["0.0"]
            macro_refresh_grid = ["0", "8"]
            macro_max_lookahead = "8"
        else:
            macro_segment_grid = ["4", "8"]
            macro_k_grid = ["1", "4", "8"]
            macro_threshold_grid = ["0.4", "0.6", "0.8", "1.0"]
            macro_refresh_grid = ["0"]
            macro_max_lookahead = "4"
        common.extend(
            [
                "--include-macro-option-policy",
                "--macro-option-calibration-criterion",
                "static_margin_risk",
                "--macro-option-segment-grid",
                *macro_segment_grid,
                "--macro-option-k-grid",
                *macro_k_grid,
                "--macro-option-threshold-grid",
                *macro_threshold_grid,
                "--macro-option-aggregation-grid",
                "mean",
                "--macro-option-distance-weighting-grid",
                "inverse",
                "uniform",
                "--macro-option-refresh-grid",
                *macro_refresh_grid,
                "--macro-option-max-lookahead",
                macro_max_lookahead,
            ]
        )
    else:
        common.append("--no-include-macro-option-policy")
    if preset == "learned_teacher_improvement_gate_smoke":
        common.extend(
            [
                "--include-teacher-improvement-gate-policy",
                "--macro-option-calibration-criterion",
                "static_margin_risk",
                "--macro-option-segment-grid",
                "4",
                "8",
                "--macro-option-k-grid",
                "1",
                "4",
                "8",
                "--macro-option-aggregation-grid",
                "mean",
                "--macro-option-distance-weighting-grid",
                "inverse",
                "uniform",
                "--macro-option-refresh-grid",
                "0",
                "--macro-option-max-lookahead",
                "4",
                "--teacher-improvement-gate-hidden-dim",
                "128",
                "--teacher-improvement-gate-epochs",
                "40",
                "--teacher-improvement-gate-label-margin",
                "0.0",
                "--teacher-improvement-gate-threshold-grid",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
            ]
        )
    else:
        common.append("--no-include-teacher-improvement-gate-policy")
    if preset in {
        "learned_event_threshold_guarded_safe",
        "learned_event_threshold_valguard_safe",
        "learned_event_threshold_strict_valguard_safe",
        "learned_event_threshold_riskcalib_safe",
        "learned_event_threshold_riskcenter_safe",
        "learned_hybrid_event_guarded_safe",
        "learned_hybrid_event_cycle_guarded_safe",
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_sequence_guarded_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
        "learned_hybrid_sequence_mask_guarded_safe",
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        "learned_hybrid_recurrent_advantage_posguard_safe",
        "learned_hybrid_option_planner_posguard_safe",
        "learned_hybrid_bc_guarded_safe",
        "learned_hybrid_planner_guarded_safe",
    }:
        common.extend(
            [
                "--include-event-threshold-policy",
                "--event-threshold-support-top-k",
                "4",
                "--event-threshold-calibration-criterion",
                "static_margin_risk"
                if preset in {
                    "learned_event_threshold_riskcalib_safe",
                    "learned_event_threshold_riskcenter_safe",
                    "learned_hybrid_rate_riskcenter_safe",
                    "learned_hybrid_contextual_duty_riskcenter_safe",
                    "learned_hybrid_contextual_duty_riskband_safe",
                }
                else (
                    "static_margin_guard"
                    if preset
                    in {
                        "learned_event_threshold_valguard_safe",
                        "learned_event_threshold_strict_valguard_safe",
                    }
                    else "mean_objective"
                ),
                "--event-threshold-grid",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.65",
                "0.8",
                "--event-threshold-aggregation-grid",
                "max",
                "mean",
                "first",
            ]
        )
    else:
        common.append("--no-include-event-threshold-policy")
    if preset == "learned_hybrid_event_cycle_guarded_safe":
        common.extend(
            [
                "--include-event-support-cycle-policy",
                "--event-support-cycle-top-k",
                "6",
                "--event-support-cycle-grid",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.65",
                "0.8",
                "--event-support-cycle-aggregation-grid",
                "max",
                "mean",
                "first",
                "--event-support-cycle-period-grid",
                "1",
                "2",
                "4",
            ]
        )
    else:
        common.append("--no-include-event-support-cycle-policy")
    if preset in {"learned_hybrid_option_planner_posguard_safe", "learned_option_planner_startguard_safe"}:
        common.extend(
            [
                "--include-option-planner-policy",
                "--option-planner-support-top-k",
                "16",
                "--option-planner-calibration-criterion",
                "static_margin_guard",
                "--option-planner-threshold-grid",
                "0.35",
                "0.5",
                "0.65",
                "--option-planner-aggregation-grid",
                "max",
                "mean",
                "--option-planner-min-dwell-grid",
                "2",
                "4",
                "--option-planner-cooldown-grid",
                "0",
                "2",
                "--option-planner-target-rate-grid",
                "1.0",
                "--option-planner-rate-balance-grid",
                *(
                    ["0.0"]
                    if preset == "learned_option_planner_startguard_safe"
                    else ["0.0", "1.0", "3.0"]
                ),
                "--option-planner-freshness-grid",
                "0.25",
                "--option-planner-transport-grid",
                "0.0",
                "0.3",
                "--option-planner-power-grid",
                "0.05",
                "--option-planner-switch-grid",
                "0.05",
                "--option-planner-min-soc-grid",
                "0.0",
                "0.25",
            ]
        )
    elif preset in {"learned_option_runtime_risk_guard_safe", "learned_option_runtime_risk_denseval_safe"}:
        common.extend(
            [
                "--no-include-option-planner-policy",
                "--option-planner-support-top-k",
                "16",
                "--option-planner-calibration-criterion",
                "static_margin_risk",
                "--option-planner-threshold-grid",
                "0.35",
                "0.5",
                "0.65",
                "--option-planner-aggregation-grid",
                "max",
                "mean",
                "--option-planner-min-dwell-grid",
                "2",
                "4",
                "--option-planner-cooldown-grid",
                "0",
                "2",
                "--option-planner-target-rate-grid",
                "1.0",
                "--option-planner-rate-balance-grid",
                "0.0",
                "--option-planner-freshness-grid",
                "0.25",
                "--option-planner-transport-grid",
                "0.0",
                "0.3",
                "--option-planner-power-grid",
                "0.05",
                "--option-planner-switch-grid",
                "0.05",
                "--option-planner-min-soc-grid",
                "0.0",
                "0.25",
            ]
        )
    else:
        common.append("--no-include-option-planner-policy")
    if preset in {"learned_option_runtime_risk_guard_safe", "learned_option_runtime_risk_denseval_safe"}:
        if preset == "learned_option_runtime_risk_denseval_safe":
            runtime_thresholds = ["0.8", "1.0", "1.2"]
            runtime_aggregations = ["mean"]
            runtime_min_socs = ["0.0"]
        else:
            runtime_thresholds = ["0.4", "0.6", "0.8", "1.0", "1.2"]
            runtime_aggregations = ["max", "mean"]
            runtime_min_socs = ["0.0", "0.25"]
        common.extend(
            [
                "--include-runtime-risk-guard-policy",
                "--runtime-risk-guard-calibration-criterion",
                "static_margin_risk",
                "--runtime-risk-threshold-grid",
                *runtime_thresholds,
                "--runtime-risk-window-grid",
                "4",
                "8",
                "16",
                "--runtime-risk-aggregation-grid",
                *runtime_aggregations,
                "--runtime-risk-event-weight-grid",
                "1.0",
                "--runtime-risk-freshness-weight-grid",
                "0.0",
                "0.25",
                "--runtime-risk-transport-weight-grid",
                "0.0",
                "0.25",
                "--runtime-risk-soc-weight-grid",
                "0.0",
                "--runtime-risk-min-soc-grid",
                *runtime_min_socs,
            ]
        )
    else:
        common.append("--no-include-runtime-risk-guard-policy")
    if preset in {"learned_window_eligibility_posguard_safe", "learned_window_macro_eligibility_posguard_safe"}:
        common.extend(
            [
                "--include-window-eligibility-policy",
                "--window-eligibility-dynamic-grid",
                "macro" if preset == "learned_window_macro_eligibility_posguard_safe" else "option",
                "--window-eligibility-support-top-k",
                "16",
                "--window-eligibility-samples-per-start",
                "4",
                "--window-eligibility-max-train-windows",
                "96",
                "--window-eligibility-window-grid",
                "16",
                "32",
                "--window-eligibility-k-grid",
                "3",
                "--window-eligibility-margin-grid",
                "-0.005",
                "0.0",
                "0.005",
                "--window-eligibility-blend-grid",
                "0.5",
                "1.0",
                "--window-eligibility-min-dwell-grid",
                "2",
                "--window-eligibility-freshness-grid",
                "0.25",
                "--window-eligibility-transport-grid",
                "0.25",
                "--window-eligibility-power-grid",
                "0.05",
                "--window-eligibility-switch-grid",
                "0.05",
                "--window-eligibility-min-soc-grid",
                "0.0",
                "--window-eligibility-distance-weighting-grid",
                "inverse",
                "--window-eligibility-calibration-criterion",
                "static_margin_risk",
            ]
        )
        if preset == "learned_window_macro_eligibility_posguard_safe":
            common.extend(
                [
                    "--window-eligibility-macro-k-grid",
                    "4",
                    "8",
                    "--window-eligibility-macro-snippet-stride",
                    "1",
                    "--window-eligibility-macro-max-lookahead",
                    "16",
                ]
            )
    else:
        common.append("--no-include-window-eligibility-policy")
    if preset in {
        "learned_hybrid_rate_guarded_safe",
        "learned_hybrid_rate_riskcenter_safe",
        "learned_hybrid_teacher_mix_guarded_safe",
    }:
        common.extend(
            [
                "--include-teacher-rate-policy",
                "--teacher-rate-support-top-k",
                "16",
                "--teacher-rate-blend-grid",
                "0.5",
                "0.75",
                "1.0",
                "--teacher-rate-freshness-grid",
                "0.0",
                "0.1",
                "0.25",
                "0.5",
                "--teacher-rate-power-grid",
                "0.0",
                "0.03",
                "0.08",
            ]
        )
    else:
        common.append("--no-include-teacher-rate-policy")
    if preset in {
        "learned_hybrid_contextual_duty_guarded_safe",
        "learned_hybrid_contextual_duty_guardcalib_safe",
        "learned_hybrid_contextual_duty_riskcenter_safe",
        "learned_hybrid_contextual_duty_riskband_safe",
    }:
        common.extend(
            [
                "--include-contextual-duty-policy",
                "--contextual-duty-support-top-k",
                "16",
                "--contextual-duty-blend-grid",
                "0.5",
                "0.75",
                "1.0",
                "--contextual-duty-deficit-grid",
                "0.5",
                "1.0",
                "2.0",
                "--contextual-duty-freshness-grid",
                "0.0",
                "0.25",
                "--contextual-duty-power-grid",
                "0.0",
                "0.03",
                "0.08",
            ]
        )
        if preset == "learned_hybrid_contextual_duty_guardcalib_safe":
            common.extend(["--contextual-duty-calibration-criterion", "static_margin_guard"])
        elif preset in {
            "learned_hybrid_contextual_duty_riskcenter_safe",
            "learned_hybrid_contextual_duty_riskband_safe",
        }:
            common.extend(["--contextual-duty-calibration-criterion", "static_margin_risk"])
    else:
        common.append("--no-include-contextual-duty-policy")
    if preset == "learned_hybrid_sequence_mask_guarded_safe":
        common.extend(
            [
                "--include-sequence-mask-policy",
                "--sequence-mask-support-top-k",
                "16",
                "--sequence-mask-anchor-bias-grid",
                "0.0",
                "0.25",
                "0.5",
                "1.0",
                "--sequence-mask-power-grid",
                "0.0",
                "0.03",
                "0.08",
                "--sequence-mask-calibration-criterion",
                "static_margin_guard",
            ]
        )
    else:
        common.append("--no-include-sequence-mask-policy")
    if preset in {
        "learned_hybrid_recurrent_value_guarded_safe",
        "learned_hybrid_recurrent_rank_guarded_safe",
        "learned_hybrid_recurrent_value_posguard_safe",
        "learned_hybrid_recurrent_rank_posguard_safe",
        "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
    }:
        common.extend(
            [
                "--include-recurrent-value-policy",
                "--recurrent-value-support-top-k",
                "16",
                "--recurrent-value-advantage-grid",
                "-1.0",
                "-0.5",
                "-0.2",
                "-0.1",
                "0.0",
                "0.1",
                "0.2",
                "0.5",
                "1.0",
                "--recurrent-value-calibration-criterion",
                "static_margin_guard",
            ]
        )
        if preset in {
            "learned_hybrid_recurrent_rank_guarded_safe",
            "learned_hybrid_recurrent_rank_posguard_safe",
            "learned_hybrid_recurrent_rank_costdagger_posguard_safe",
        }:
            common.extend(["--recurrent-value-rank-weight", "0.5"])
        if preset == "learned_hybrid_recurrent_rank_costdagger_posguard_safe":
            common.extend(
                [
                    "--recurrent-value-cost-dagger-iters",
                    "1",
                    "--recurrent-value-cost-dagger-threshold",
                    "0.0",
                ]
            )
    else:
        common.append("--no-include-recurrent-value-policy")
    if preset == "learned_hybrid_recurrent_advantage_posguard_safe":
        common.extend(
            [
                "--include-recurrent-advantage-policy",
                "--recurrent-advantage-support-top-k",
                "16",
                "--recurrent-advantage-grid",
                "-0.2",
                "-0.1",
                "0.0",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "0.8",
                "1.0",
                "--recurrent-advantage-calibration-criterion",
                "static_margin_guard",
                "--recurrent-advantage-rank-weight",
                "0.5",
            ]
        )
    else:
        common.append("--no-include-recurrent-advantage-policy")
    if preset in {"learned_hybrid_sequence_guarded_safe", "learned_hybrid_teacher_mix_guarded_safe"}:
        common.extend(["--include-teacher-cycle-policy", "--teacher-cycle-max-lookahead", "64"])
    else:
        common.append("--no-include-teacher-cycle-policy")

    if preset in {
        "no_dagger",
        "value_residual_no_dagger",
        "learned_macro_option_dense_always_safe",
        "learned_sequence_value_riskband_safe",
        "learned_sequence_value_fullbank_riskband_safe",
        "learned_sequence_value_oracle_context_fullbank_safe",
        "learned_sequence_value_oracle_regime_fullbank_safe",
        "learned_rollout_value_self_posguard_safe",
        "learned_rollout_value_oracle_regime_posguard_safe",
        "learned_advantage_oracle_regime_posguard_safe",
        "learned_window_eligibility_posguard_safe",
        "learned_window_macro_eligibility_posguard_safe",
        "learned_utility_planner_riskband_safe",
        "learned_proxy_mpc_riskband_safe",
    }:
        common.extend(["--dagger-iters", "0"])
    elif preset in {"safe_dagger3", "knn_safe_dagger3"}:
        common.extend(["--dagger-iters", "3"])
    else:
        common.extend(["--dagger-iters", "1"])
    if preset == "learned_sequence_value_continuous_augmented_riskband_safe":
        common.extend(
            [
                "--dagger-iters",
                "0",
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
                "--learned-continuous-forecast",
                "--continuous-forecast-lookback",
                "8",
                "--continuous-forecast-hidden-dim",
                "128",
                "--continuous-forecast-epochs",
                "40",
                "--continuous-forecast-target-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "10.0",
                "20.0",
                "0.0001",
                "0.2",
                "5.0",
                "--no-include-bc-policy",
                "--no-include-knn-policy",
                "--no-include-mask-bc-policy",
                "--no-include-residual-bc-policy",
                "--no-include-value-residual-policy",
                "--no-include-ensemble-value-policy",
                "--no-include-advantage-residual-policy",
                "--no-include-event-threshold-policy",
                "--no-include-event-support-cycle-policy",
                "--no-include-option-planner-policy",
                "--no-include-macro-option-policy",
                "--no-include-teacher-improvement-gate-policy",
                "--no-include-teacher-rate-policy",
                "--no-include-contextual-duty-policy",
                "--no-include-sequence-mask-policy",
                "--no-include-teacher-cycle-policy",
                "--no-include-validation-cyclic-policy",
                "--no-include-runtime-risk-guard-policy",
                "--no-include-window-eligibility-policy",
                "--no-include-cost-policy",
                "--no-include-cost-knn-policy",
                "--no-include-rollout-value-policy",
                "--no-include-recurrent-value-policy",
                "--no-include-recurrent-advantage-policy",
                "--include-sequence-value-policy",
                "--sequence-value-augment-bank",
                "--sequence-value-static-top-k",
                "4",
                "--sequence-value-cycle-support-top-k",
                "8",
                "--sequence-value-cycle-dwell-grid",
                "1",
                "2",
                "4",
                "--sequence-value-cycle-max-sequences",
                "512",
                "--sequence-value-segment-len",
                "8",
                "--sequence-value-snippet-stride",
                "2",
                "--sequence-value-negatives-per-state",
                "8",
                "--sequence-value-max-rows",
                "8192",
                "--sequence-value-top-k-sequences",
                "1024",
                "--sequence-value-advantage-grid",
                "-0.05",
                "0.0",
                "0.01",
                "0.025",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "--deployable-selection",
                "validation",
                "--deployable-selection-criterion",
                "static_margin_risk",
                "--deployable-selection-min-mean-margin",
                "0.001",
                "--deployable-selection-min-start-margin",
                "-0.01",
                "--deployable-selection-max-negative-starts",
                "1",
                "--deployable-selection-require-positive-center",
                "--deployable-selection-require-risk-band",
                "--deployable-selection-risk-min-q25-margin",
                "0.0",
                "--deployable-selection-risk-max-negative-starts",
                "1",
            ]
        )
    if preset == "learned_twin_rollout_posguard_safe":
        common.extend(
            [
                "--dagger-iters",
                "0",
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
                "--learned-continuous-forecast",
                "--continuous-forecast-lookback",
                "8",
                "--continuous-forecast-hidden-dim",
                "128",
                "--continuous-forecast-epochs",
                "40",
                "--continuous-forecast-target-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "10.0",
                "20.0",
                "0.0001",
                "0.2",
                "5.0",
                "--bc-preserve-warming",
                "--no-include-bc-policy",
                "--no-include-knn-policy",
                "--no-include-mask-bc-policy",
                "--no-include-residual-bc-policy",
                "--no-include-value-residual-policy",
                "--no-include-ensemble-value-policy",
                "--no-include-advantage-residual-policy",
                "--no-include-event-threshold-policy",
                "--no-include-event-support-cycle-policy",
                "--no-include-option-planner-policy",
                "--no-include-macro-option-policy",
                "--no-include-teacher-improvement-gate-policy",
                "--no-include-teacher-rate-policy",
                "--no-include-contextual-duty-policy",
                "--no-include-sequence-mask-policy",
                "--no-include-teacher-cycle-policy",
                "--no-include-validation-cyclic-policy",
                "--no-include-runtime-risk-guard-policy",
                "--no-include-window-eligibility-policy",
                "--no-include-cost-policy",
                "--no-include-cost-knn-policy",
                "--no-include-sequence-value-policy",
                "--no-include-recurrent-value-policy",
                "--no-include-recurrent-advantage-policy",
                "--include-rollout-value-policy",
                "--rollout-value-cost-target",
                "executed_step",
                "--rollout-value-random-rollouts",
                "1",
                "--rollout-value-support-top-k",
                "12",
                "--rollout-value-depth",
                "3",
                "--rollout-value-beam-width",
                "4",
                "--rollout-value-max-branch",
                "8",
                "--rollout-value-discount",
                "0.95",
                "--rollout-value-advantage-grid",
                "-0.5",
                "-0.2",
                "-0.1",
                "0.0",
                "0.05",
                "0.1",
                "0.2",
                "0.35",
                "0.5",
                "--deployable-selection",
                "validation",
                "--deployable-selection-criterion",
                "static_margin_risk",
                "--deployable-selection-min-mean-margin",
                "0.001",
                "--deployable-selection-min-start-margin",
                "-0.01",
                "--deployable-selection-max-negative-starts",
                "1",
                "--deployable-selection-require-positive-center",
                "--deployable-selection-require-risk-band",
                "--deployable-selection-risk-min-q25-margin",
                "0.0",
                "--deployable-selection-risk-max-negative-starts",
                "1",
            ]
        )
    if preset in {"learned_window_candidate_margin_safe", "learned_window_candidate_fullrollout_margin_safe"}:
        common.extend(
            [
                "--dagger-iters",
                "0",
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
                "--learned-continuous-forecast",
                "--continuous-forecast-lookback",
                "8",
                "--continuous-forecast-hidden-dim",
                "128",
                "--continuous-forecast-epochs",
                "40",
                "--continuous-forecast-target-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "10.0",
                "20.0",
                "0.0001",
                "0.2",
                "5.0",
                "--bc-preserve-warming",
                "--no-include-bc-policy",
                "--no-include-knn-policy",
                "--no-include-mask-bc-policy",
                "--no-include-residual-bc-policy",
                "--no-include-value-residual-policy",
                "--no-include-ensemble-value-policy",
                "--no-include-advantage-residual-policy",
                "--no-include-event-threshold-policy",
                "--no-include-event-support-cycle-policy",
                "--no-include-option-planner-policy",
                "--no-include-macro-option-policy",
                "--no-include-teacher-improvement-gate-policy",
                "--no-include-teacher-rate-policy",
                "--no-include-contextual-duty-policy",
                "--no-include-sequence-mask-policy",
                "--no-include-teacher-cycle-policy",
                "--no-include-validation-cyclic-policy",
                "--no-include-runtime-risk-guard-policy",
                "--no-include-window-eligibility-policy",
                "--no-include-cost-policy",
                "--no-include-cost-knn-policy",
                "--no-include-rollout-value-policy",
                "--no-include-sequence-value-policy",
                "--no-include-recurrent-value-policy",
                "--no-include-recurrent-advantage-policy",
                "--include-window-candidate-policy",
                "--window-candidate-family-grid",
                "option",
                "macro",
                "rate",
                "--window-candidate-support-top-k",
                "16",
                "--window-candidate-samples-per-start",
                "4",
                "--window-candidate-max-train-windows",
                "96",
                "--window-candidate-window-grid",
                "16",
                "32",
                "--window-candidate-k-grid",
                "3",
                "5",
                "--window-candidate-margin-grid",
                "-0.005",
                "0.0",
                "0.005",
                "--window-candidate-quantile-grid",
                "0.25",
                "0.50",
                "--window-candidate-distance-weighting-grid",
                "inverse",
                "--window-candidate-max-candidates",
                "12",
                "--window-candidate-option-blend-grid",
                "0.5",
                "1.0",
                "--window-candidate-option-min-dwell-grid",
                "2",
                "--window-candidate-option-freshness-grid",
                "0.25",
                "--window-candidate-option-transport-grid",
                "0.25",
                "--window-candidate-option-power-grid",
                "0.05",
                "--window-candidate-option-switch-grid",
                "0.05",
                "--window-candidate-min-soc-grid",
                "0.0",
                "--window-candidate-macro-k-grid",
                "4",
                "8",
                "--window-candidate-macro-snippet-stride",
                "1",
                "--window-candidate-macro-max-lookahead",
                "16",
                "--window-candidate-rate-blend-grid",
                "0.5",
                "0.75",
                "1.0",
                "--window-candidate-rate-freshness-grid",
                "0.0",
                "0.25",
                "--window-candidate-rate-power-grid",
                "0.0",
                "0.03",
                "--window-candidate-calibration-criterion",
                "static_margin_risk",
                "--deployable-selection",
                "validation",
                "--deployable-selection-criterion",
                "static_margin_risk",
                "--deployable-selection-min-mean-margin",
                "0.001",
                "--deployable-selection-min-start-margin",
                "-0.01",
                "--deployable-selection-max-negative-starts",
                "1",
                "--deployable-selection-require-positive-center",
                "--deployable-selection-require-risk-band",
                "--deployable-selection-risk-min-q25-margin",
                "0.0",
                "--deployable-selection-risk-max-negative-starts",
                "1",
            ]
        )
        if preset == "learned_window_candidate_fullrollout_margin_safe":
            common.append("--window-candidate-full-rollout-calibration")
    if preset == "learned_utility_planner_riskband_safe":
        common.extend(
            [
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
                "--learned-continuous-forecast",
                "--continuous-forecast-lookback",
                "8",
                "--continuous-forecast-hidden-dim",
                "128",
                "--continuous-forecast-epochs",
                "40",
                "--continuous-forecast-target-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "10.0",
                "20.0",
                "0.0001",
                "0.2",
                "5.0",
                "--no-include-window-candidate-policy",
                "--include-utility-planner-policy",
                "--utility-planner-support-top-k",
                "16",
                "--utility-planner-event-weight-grid",
                "0.5",
                "1.5",
                "--utility-planner-magnitude-weight-grid",
                "1.0",
                "--utility-planner-variability-weight-grid",
                "0.5",
                "--utility-planner-freshness-grid",
                "0.25",
                "--utility-planner-target-rate-grid",
                "0.0",
                "0.5",
                "--utility-planner-anchor-bias-grid",
                "0.0",
                "0.1",
                "--utility-planner-power-grid",
                "0.03",
                "--utility-planner-switch-grid",
                "0.03",
                "--utility-planner-min-soc-grid",
                "0.0",
                "--utility-planner-dwell-grid",
                "2",
                "--utility-planner-aggregation-grid",
                "max",
                "mean",
                "--utility-planner-calibration-criterion",
                "static_margin_risk",
                "--deployable-selection",
                "validation",
                "--deployable-selection-criterion",
                "static_margin_risk",
                "--deployable-selection-min-mean-margin",
                "0.001",
                "--deployable-selection-min-start-margin",
                "-0.01",
                "--deployable-selection-max-negative-starts",
                "1",
                "--deployable-selection-require-positive-center",
                "--deployable-selection-require-risk-band",
                "--deployable-selection-risk-min-q25-margin",
                "0.0",
                "--deployable-selection-risk-max-negative-starts",
                "1",
            ]
        )
    else:
        common.append("--no-include-utility-planner-policy")
    if preset == "learned_proxy_mpc_riskband_safe":
        common.extend(
            [
                "--learned-event-forecast",
                "--event-forecast-lookback",
                "8",
                "--event-forecast-hidden-dim",
                "128",
                "--event-forecast-epochs",
                "40",
                "--learned-continuous-forecast",
                "--continuous-forecast-lookback",
                "8",
                "--continuous-forecast-hidden-dim",
                "128",
                "--continuous-forecast-epochs",
                "40",
                "--continuous-forecast-target-columns",
                "wind_speed_ms",
                "snow_surface_temperature_c",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-columns",
                "snow_mass_flux_kg_m2_s",
                "snow_particle_mean_diameter_mm",
                "snow_particle_mean_velocity_ms",
                "--forecast-continuous-scales",
                "0.0001",
                "0.2",
                "5.0",
                "--no-include-window-candidate-policy",
                "--include-proxy-mpc-policy",
                "--proxy-mpc-support-top-k",
                "16",
                "--proxy-mpc-event-weight-grid",
                "0.5",
                "1.5",
                "--proxy-mpc-magnitude-weight-grid",
                "1.0",
                "--proxy-mpc-variability-weight-grid",
                "0.5",
                "--proxy-mpc-freshness-grid",
                "0.25",
                "--proxy-mpc-target-rate-grid",
                "0.0",
                "0.5",
                "--proxy-mpc-anchor-bias-grid",
                "0.0",
                "--proxy-mpc-power-grid",
                "0.03",
                "--proxy-mpc-switch-grid",
                "0.03",
                "--proxy-mpc-min-soc-grid",
                "0.0",
                "--proxy-mpc-dwell-grid",
                "1",
                "2",
                "--proxy-mpc-aggregation-grid",
                "max",
                "--proxy-mpc-depth-grid",
                "2",
                "3",
                "--proxy-mpc-beam-width-grid",
                "4",
                "--proxy-mpc-max-branch-grid",
                "8",
                "--proxy-mpc-age-weight-grid",
                "0.25",
                "0.75",
                "--proxy-mpc-anchor-improvement-grid",
                "0.0",
                "--proxy-mpc-calibration-criterion",
                "static_margin_risk",
                "--deployable-selection",
                "validation",
                "--deployable-selection-criterion",
                "static_margin_risk",
                "--deployable-selection-min-mean-margin",
                "0.001",
                "--deployable-selection-min-start-margin",
                "-0.01",
                "--deployable-selection-max-negative-starts",
                "1",
                "--deployable-selection-require-positive-center",
                "--deployable-selection-require-risk-band",
                "--deployable-selection-risk-min-q25-margin",
                "0.0",
                "--deployable-selection-risk-max-negative-starts",
                "1",
            ]
        )
    else:
        common.append("--no-include-proxy-mpc-policy")
    return common


def run_commands(commands: list[CommandItem], *, max_parallel: int, continue_on_error: bool) -> None:
    if not commands:
        print("[run_claim_suite] no pending jobs")
        return
    pending = list(commands)
    running: list[tuple[subprocess.Popen[bytes], CommandItem, object]] = []
    failures: list[tuple[CommandItem, int]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            item = pending.pop(0)
            item.out_dir.mkdir(parents=True, exist_ok=True)
            log_file = item.log_path.open("wb")
            log_file.write((format_command(item.command, env=item.env) + "\n\n").encode("utf-8"))
            log_file.flush()
            print(f"[run_claim_suite] start {item.out_dir}")
            process = subprocess.Popen(
                item.command,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=item.env,
            )
            running.append((process, item, log_file))
        time.sleep(5.0)
        still_running: list[tuple[subprocess.Popen[bytes], CommandItem, object]] = []
        for process, item, log_file in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, item, log_file))
                continue
            log_file.close()
            print(f"[run_claim_suite] done {item.out_dir} rc={return_code}")
            if return_code != 0:
                failures.append((item, int(return_code)))
                if not continue_on_error:
                    for other, _, other_log in still_running:
                        other.terminate()
                        other_log.close()
                    raise SystemExit(f"Job failed: {item.out_dir} rc={return_code}")
        running = still_running
    if failures:
        for item, code in failures:
            print(f"[run_claim_suite] failed {item.out_dir} rc={code}")
        raise SystemExit(1)


def resolve_parallelism(args: argparse.Namespace) -> int:
    if args.max_parallel is not None:
        return max(1, int(args.max_parallel))
    if args.gpu_ids:
        return max(1, len(args.gpu_ids))
    return 1


def format_command(command: list[str], *, env: dict[str, str]) -> str:
    prefix = ""
    if "CUDA_VISIBLE_DEVICES" in env:
        prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
    return prefix + " ".join(shell_quote(part) for part in command)


def shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._/-:+=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
