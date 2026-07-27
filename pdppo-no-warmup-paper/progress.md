# Progress: No-Warmup PD-PPO Paper Track

## 2026-06-08

- Created isolated project directory:
  `pdppo-no-warmup-paper/`.
- Established non-conflict boundary with:
  - `v1/`;
  - `rl_sensor_scheduling_framework/.planning/`;
  - `rl_sensor_scheduling_framework/paper/`.
- Added fresh planning files, result-script skeletons, evidence ledger, and
  manuscript draft skeleton.
- Local branch check showed root repository currently reports `master`; no
  branch switch was performed.
- Added `scripts/collect_no_warmup_results.py` and verified it with
  `python -m py_compile`.
- Aggregated existing lightweight framework outputs into:
  - `results/tables/no_warmup_runs.csv`;
  - `results/tables/no_warmup_budget_summary.csv`;
  - `results/tables/claim_gate_summary.md`.
- Current aggregation found 18 runs:
  - 15 base no-warmup runs;
  - 3 hard-duty reduced runs.
- Key result:
  - base no-warmup: static wins `14/15`, best original fair wins `1/15`;
  - hard-duty reduced: static wins `2/3`, best original fair wins `0/3`,
    best constrained wins `1/3`, duty valid `3/3`.
- Added strict split DQN diagnostic tooling:
  - `scripts/run_no_warmup_dqn_split.py`;
  - `scripts/server_run_no_warmup_dqn_split_pilot.sh`;
  - `scripts/remote_start_no_warmup_dqn_split_pilot.sh`;
  - `scripts/collect_dqn_diagnostic.py`.
- Design:
  - reuse existing no-warmup truth, TCN oracle, split manifest, final-test
    starts, and validation-selected static candidates;
  - train DQN only from the recorded `rl_train` window;
  - evaluate DQN and baselines only on the recorded `final_test` starts.
- Verified locally by syntax checks only:
  - `python -m py_compile scripts/run_no_warmup_dqn_split.py scripts/collect_dqn_diagnostic.py`;
  - `bash -n` for the server and remote launch wrappers.
- No local experiment was run.
- First server launch failed immediately before training:
  - affected logs: `budget1p65_seed41.log`, `budget1p65_seed42.log`;
  - error: Python 3.12 dataclass import failed because the dynamically loaded
    DQN module was not inserted into `sys.modules` before `exec_module`;
  - fix: register the loaded module in `load_module()` before executing it.
- Second server launch entered training but used GPU4/5 because the local
  `GPU_IDS='1 5'` setting was not propagated into the remote tmux environment.
- Stopped that DQN pilot, cleared only
  `reports/v31_no_warmup_dqn_split_diagnostic`, and patched
  `remote_start_no_warmup_dqn_split_pilot.sh` to write a remote env file before
  starting tmux.
- Relaunched DQN split pilot:
  - tmux `pdppo_no_warmup_dqn_split_pilot_20260608c`;
  - GPU IDs `1 5`;
  - B=`1.65,1.70`, seeds `41,42,43`;
  - `60000` DQN steps per run;
  - initial pane/log check confirmed B=1.65 seed41/42 entered training.
- First two DQN results synced and aggregated:
  - `results/tables/dqn_split_diagnostic_runs.csv`;
  - `results/tables/dqn_split_diagnostic_summary.md`.
- Results:
  - B=1.65 seed41: DQN `0.11921`, source PD-PPO `0.11756`,
    AoI `0.11387`, round-robin `0.11477`, static `0.14494`;
  - B=1.65 seed42: DQN `0.14221`, source PD-PPO `0.13910`,
    static `0.13902`, AoI `0.14185`, round-robin `0.14210`.
- Interim DQN diagnostic:
  - wins source PD-PPO `0/2`;
  - wins validation/static `1/2`;
  - wins AoI `0/2`;
  - wins round-robin `0/2`;
  - duty is behaviorally better than base PD-PPO in these runs:
    mean `mid=7`, `always_on=0`, `always_off=0`.
- Updated after four DQN runs:
  - B=1.65 seed43: DQN `0.14344`, source PD-PPO `0.14408`,
    AoI `0.13977`, round-robin `0.14030`, static `0.17055`;
  - B=1.70 seed41: DQN `0.13814`, source PD-PPO `0.12376`,
    round-robin `0.11575`, AoI `0.11718`, static `0.13510`.
- 4-run interim gate:
  - wins source PD-PPO `1/4`;
  - wins validation/static `2/4`;
  - wins round-robin `0/4`;
  - wins AoI `0/4`;
  - wins best non-DQN `0/4`;
  - mean duty remains behaviorally valid:
    `mid=7.25`, `always_on=0`, `always_off=0`.
- Completed sync/aggregation after context resume:
  - server counts:
    `base=17`, `hard-duty=3`, `DQN=6`, `env-dwell12-trained=1`,
    `env-dwell12-replay=3`, `env-dwell6-replay=3`,
    `switch-limited-replay=2`;
  - commands:
    `python scripts/collect_no_warmup_results.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`;
    `python scripts/collect_dqn_diagnostic.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`.
- Added and verified compact evidence-table builder:
  - `scripts/build_minimal_policy_table.py`;
  - syntax check:
    `python -m py_compile scripts/collect_no_warmup_results.py scripts/build_minimal_policy_table.py`;
  - run command:
    `python scripts/build_minimal_policy_table.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`.
- Generated citation-ready tables:
  - `results/tables/no_warmup_minimal_policy_table.csv`:
    286 policy rows;
  - `results/tables/no_warmup_minimal_run_gate_table.csv`:
    35 run-gate rows;
  - `results/tables/no_warmup_minimal_gate_summary.md`.
- Full DQN diagnostic:
  - runs `6`;
  - wins source PD-PPO `1/6`;
  - wins validation/static `3/6`;
  - wins round-robin `0/6`;
  - wins AoI `1/6`;
  - wins best non-DQN `0/6`;
  - learned duty pass `6/6`.
- Updated no-warmup claim gate to include trained env-dwell12 reduced seed41:
  - `base_no_warmup`: static wins `16/17`, best non-learned wins `1/17`,
    learned duty pass `0/14`;
  - `hard_duty_reduced`: static wins `2/3`, best non-learned wins `0/3`,
    duty pass `3/3`;
  - `hard_duty_envdwell12_reduced`: static, dynamic, and constrained-baseline
    wins `1/1`, duty pass `1/1`.
- Remote status at `2026-06-08 02:57 CST`:
  - tmux `pdppo_no_warmup_20260607`: base B=1.70 seed48 running at update
    `43`; seeds `49` and `50` still pending;
  - tmux `pdppo_no_warmup_hguard_envdwell12_reduced_20260608`:
    seed42 running at update `10/20`, seed43 pending;
  - DQN pilot completed and no longer appears in tmux.
- Synced and aggregated two additional completed split results:
  - base no-warmup B=1.70 seed48;
  - hard-duty env-dwell12 reduced B=1.70 seed42.
- Updated table sizes:
  - `results/tables/no_warmup_runs.csv`: 23 rows;
  - `results/tables/no_warmup_minimal_policy_table.csv`: 302 policy rows;
  - `results/tables/no_warmup_minimal_run_gate_table.csv`: 37 run gates.
- New base seed48:
  - custom PPO `0.121979`;
  - validation/static `0.122815`;
  - round-robin `0.116968`;
  - AoI `0.117094`;
  - random `0.119079`;
  - duty invalid with `always_on=1`, `always_off=3`, `mid=4`.
  - Interpretation: reinforces the existing base pattern: static is weakened,
    but dynamic heuristics remain stronger.
- New env-dwell12 trained seed42:
  - custom PPO `0.149620`;
  - validation/static `0.138138`;
  - round-robin `0.154632`;
  - AoI `0.166598`;
  - best duty-constrained baseline `0.160709`;
  - duty valid with `always_on=0`, `always_off=0`, `mid=8`, switch rate
    `0.026860`.
  - Interpretation: env-dwell12 is now `2/2` against round-robin and AoI, but
    only `1/2` against static.
- Remote status at `2026-06-08 03:08 CST`:
  - env-dwell12 seed43 has created truth/dataset files but has not yet printed
    PPO updates;
  - base no-warmup seed49 has started and reached update `3`;
  - tmux sessions remain:
    `pdppo_no_warmup_20260607`,
    `pdppo_no_warmup_hguard_envdwell12_reduced_20260608`,
    and unrelated `v1_v7g_static_gate_20260608`.
- Monitored base no-warmup B=1.75 seed41 from tmux
  `pdppo_no_warmup_20260607`.
- Verified remote state at `2026-06-08 05:02 CST`:
  - `v2_custom_ppo_metrics.csv` existed remotely;
  - `v2_ppo_metadata.json` existed remotely;
  - `custom_ppo_training_history_live.json` had `49` records, last
    `timesteps=100000`;
  - B=1.75 seed42 had a run directory/log but no metrics yet.
- Synced only lightweight artifacts from
  `reports/v31_split_protocol_no_warmup`:
  CSV/JSON/log/done/evaluation CSV; excluded truth CSV, validation CSV,
  NPZ, checkpoints, zip, and pickle-like files.
- Verified local B=1.75 seed41 row:
  - custom PPO `0.126576`;
  - validation/static `0.144749`;
  - round-robin `0.115072`;
  - AoI `0.118588`;
  - random `0.121769`;
  - duty invalid with `always_on=1`, `always_off=3`, `mid=2`.
- Re-ran aggregation:
  - `python scripts/collect_no_warmup_results.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`;
  - `python scripts/build_minimal_policy_table.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`;
  - `python scripts/collect_dqn_diagnostic.py --framework-root ../rl_sensor_scheduling_framework --out-dir results/tables`.
- Updated aggregate:
  - `results/tables/no_warmup_runs.csv`: `27` rows;
  - base no-warmup: `21` runs, static wins `20/21`, best non-learned wins
    `2/21`, learned duty pass `0/18`;
  - hard-duty env-dwell12 reduced: `3` runs, static wins `2/3`,
    round-robin/AoI wins `3/3`, best constrained wins `3/3`, duty pass `3/3`.
- Wrote provenance note:
  `/home/horeb/agent/tmp/microclimate-codex-coordination/codex_notes/no_warmup_b175_monitor_20260608_045715.md`.
- Continued B=1.75 monitor at `2026-06-08 05:20 CST`.
- Remote state:
  - tmux `pdppo_no_warmup_20260607` active;
  - B=1.75 seed41 remains the only completed B=1.75 metric;
  - B=1.75 seed42 has run dir/log but no `v2_custom_ppo_metrics.csv`;
  - no B=1.75 seed43 run dir/log yet.
- Synced only light compact artifacts from
  `reports/v31_split_protocol_no_warmup`; rsync transferred:
  - `logs/budget1p75_seed42.log`;
  - `raw/budget1p75_seed42/custom_ppo_candidate_prior.csv`;
  - `raw/budget1p75_seed42/custom_ppo_training_history_live.json`.
- Verified local seed42 live status:
  - `records=22`;
  - `last_timesteps=45056`;
  - `v2_custom_ppo_metrics.csv` absent.
- Re-ran aggregation commands; outputs unchanged:
  - `no_warmup_runs.csv`: 27 rows;
  - `no_warmup_minimal_policy_table.csv`: 330 policy rows;
  - `no_warmup_minimal_run_gate_table.csv`: 41 run gates;
  - DQN rows: 6.
- Paper recommendation unchanged:
  - no manuscript update;
  - base no-warmup remains diagnostic only;
  - env-dwell12 trained reduced remains the first-paper positive line.
- Wrote provenance note:
  `/home/horeb/agent/tmp/microclimate-codex-coordination/codex_notes/no_warmup_b175_monitor_20260608_051906.md`.
- Cron monitor at `2026-06-08 05:25 CST`.
- Remote state:
  - tmux `pdppo_no_warmup_20260607` active;
  - B=1.75 run dirs: seed41 and seed42;
  - done marker only for seed41;
  - metrics only for seed41;
  - seed42 has no `v2_custom_ppo_metrics.csv` or `v2_ppo_metadata.json`;
  - no seed43 run dir observed.
- Remote seed42 live status:
  - records `28`;
  - last timesteps `57344`;
  - last loss `0.47026835948927326`.
- Synced only compact artifacts:
  - `logs/budget1p75_seed42.log`;
  - `raw/budget1p75_seed42/custom_ppo_training_history_live.json`.
- Re-ran aggregation; outputs unchanged:
  - `no_warmup_runs.csv`: 27 data rows plus header;
  - `no_warmup_minimal_policy_table.csv`: 330 data rows plus header;
  - `no_warmup_minimal_run_gate_table.csv`: 41 data rows plus header;
  - `dqn_split_diagnostic_runs.csv`: 6 data rows plus header.
- Recommendation unchanged:
  - no paper update;
  - B=1.75 seed42 is still non-reportable;
  - base no-warmup remains diagnostic only.
- Wrote provenance note:
  `/home/horeb/agent/tmp/microclimate-codex-coordination/codex_notes/no_warmup_b175_monitor_20260608_052258.md`.
- Auxiliary monitor at `2026-06-08 05:52 CST`.
- Remote state:
  - tmux `pdppo_no_warmup_20260607` active;
  - B=1.75 run dirs: seed41, seed42, seed43;
  - done markers: seed41 and seed42;
  - metrics: seed41 and seed42;
  - seed43 is running and reached update `8` in the synced log.
- Synced only compact artifacts for `reports/v31_split_protocol_no_warmup`;
  seed42 transferred:
  - `done/budget1p75_seed42.done`;
  - `logs/budget1p75_seed42.log`;
  - `raw/budget1p75_seed42/custom_ppo_training_history.json`;
  - `raw/budget1p75_seed42/custom_ppo_training_history_live.json`;
  - `raw/budget1p75_seed42/custom_ppo_training_log.csv`;
  - `raw/budget1p75_seed42/v2_custom_ppo_metrics.csv`;
  - `raw/budget1p75_seed42/v2_ppo_metadata.json`;
  - `raw/budget1p75_seed42/validation_static_candidates.csv`;
  - `raw/budget1p75_seed42/evaluation/v2_eval_overall.csv`.
- Re-ran aggregation sequentially after sync:
  - first parallel run raced with rsync and still showed 27 runs;
  - second sequential run showed `28` runs, `336` policy rows, and `42` gate
    rows.
- New B=1.75 seed42 result:
  - custom PPO `0.147619`;
  - validation/static `0.146270`;
  - round-robin `0.147178`;
  - AoI `0.151523`;
  - random `0.154893`;
  - best non-learned `feasible_static_projected=0.146270`;
  - duty invalid: `always_on=1`, `always_off=3`, `mid=4`;
  - switch rate `0.160549`.
- Updated aggregate:
  - base no-warmup: `22` runs;
  - wins validation/static `20/22`;
  - wins best non-learned `2/22`;
  - learned duty pass `0/19`.
- Recommendation unchanged:
  - no manuscript update;
  - B=1.75 remains diagnostic only;
  - env-dwell12 trained reduced remains the first-paper mainline.
- Wrote provenance note:
  `/home/horeb/agent/tmp/microclimate-codex-coordination/codex_notes/no_warmup_aux_monitor_20260608_055001.md`.
- Auxiliary monitor at `2026-06-08 06:19 CST`.
- Local state before remote sync:
  - B=1.75 seed41 and seed42 metrics already local;
  - B=1.75 seed43 had local live history/log but no metrics.
- Remote state:
  - tmux `pdppo_no_warmup_20260607` active;
  - B=1.75 run dirs: seed41, seed42, seed43;
  - done markers and metrics: seed41 and seed42 only;
  - seed43 had no `v2_custom_ppo_metrics.csv`, no metadata, no evaluation CSV;
  - seed43 live history: `records=44`, `last_timesteps=90112`,
    `last_loss=0.6279605135088786`.
- Synced only compact seed43 live artifacts:
  - `logs/budget1p75_seed43.log`;
  - `raw/budget1p75_seed43/custom_ppo_training_history_live.json`.
- Re-ran aggregation:
  - stdout: `wrote 28 runs`, `wrote 336 policy rows and 42 run gates`,
    `wrote 6 DQN diagnostic rows`;
  - no new result rows, because seed43 metrics are absent.
- Table line counts and hashes recorded in the provenance note.
- Recommendation unchanged:
  - no paper/mainline update;
  - base no-warmup remains diagnostic only;
  - wait for seed43 metric before any further base-grid interpretation.
- Wrote provenance note:
  `/home/horeb/agent/tmp/microclimate-codex-coordination/codex_notes/no_warmup_aux_monitor_20260608_061700.md`.
