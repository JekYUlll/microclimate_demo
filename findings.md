# Findings & Decisions

## 2026-06-21 Current Target Correction
- The active target journal is *Expert Systems with Applications* (ESWA), not
  Cold Regions Science and Technology. CRST references below are historical
  provenance from the earlier rewrite track.
- Current work is governed by the `rl_sensor_scheduling_framework` ESWA plan and
  the 24h BO-1 strong-claim experiment campaign on `remote-gpu`.

## 2026-05-25 CRST Full Rewrite Reset
- The user rejected further incremental repair and requested a complete manuscript
  rewrite, with preservation of the prior draft and iterative subagent review.
- The latest pre-rewrite paper working tree has been archived at
  `rl_sensor_scheduling_framework/paper_archives/paper_pre_full_rewrite_20260525_055257.tar.gz`.
  An earlier archive from the length-reduction pass also exists, but it is not a
  substitute for this new baseline.
- The current paper source is `rl_sensor_scheduling_framework/paper/main.tex` with
  section files under `paper/sections/`; it is a shortened 45-page descendant of
  earlier PD-PPO narratives, not a clean Route A/full-evidence rewrite.
- Two independent subagents were launched for the first review round:
  structure/scope audit (`Zeno`) and evidence/statistical-claim audit (`Peirce`).
- Initial subagent launch attempts failed because `fork_context=true` cannot be
  combined with an explicit agent type; relaunch without `agent_type` succeeded.

## 2026-05-25 Takeover: Static-Comparator Audit
- The root `task_plan.md`, `findings.md`, and `progress.md` are the active rewrite
  memory; the planning files inside `rl_sensor_scheduling_framework/` predate the
  full-rewrite reset and are historical context.
- Direct artifact scan of `reports/v31_s2_main/` found all `30`
  `custom_ppo_candidate_prior.csv` files and `0`
  `rollout_oracle_static_projected.npz` files.
- Current `scripts/25_v2_train_custom_ppo.py` contains logic that takes the
  minimum-loss row from the oracle candidate-prior table, evaluates its static mask
  over `eval_start_indices`, saves `rollout_oracle_static_projected.npz`, and records
  `oracle_static_projected` metadata. These outputs are absent from the locked S2
  artifacts, so the published summary cannot yet treat this candidate as evaluated.
- `scripts/43_v31_s2_collect.py` explicitly names
  `feasible_static_projected` as its static policy. The current Table 3 comparison
  is therefore to fixed-priority static projection, not to the oracle-prior-selected
  feasible static candidate.
- Required next step before making dynamic-versus-static claims: inspect a sample
  prior table and S2 metadata, then either replay each selected static mask on its
  recorded held-out/evaluation starts and aggregate it, or retain the restricted
  fixed-priority comparator wording.
- Sample locked run `budget1p70_seed41` confirms the old output contract:
  `eval_policies` contains `custom_ppo`, `full_open_unconstrained`,
  `feasible_static_projected`, `round_robin`, `aoi`, and `random`, but no
  `oracle_static_projected`. Its candidate table ranks
  `met_station_core|radiometer_basic|laser_disdrometer` first on the prior scoring
  windows.
- Code-level protocol fact pending overlap quantification:
  `build_oracle_candidate_prior()` scores all feasible candidate masks with the same
  frozen oracle and truth CSV used by the run; default prior starts are selected
  with RNG seed `seed + 811`, whereas final evaluation starts are selected with
  RNG seed `seed + 1777`. Different RNG seeds do not by themselves establish
  disjoint windows.
- The locked old metadata does not store `prior_start_indices`, although they can be
  reconstructed from the saved truth, seed, horizon, candidate-prior settings and
  deterministic selection function. Until overlap is checked, this candidate may
  not be called a held-out static comparator.
- Current manuscript protocol prose is not supported by the locked V3.1 S2
  implementation. `paper/sections/04_simulation_environment.tex` claims a
  `35%/50%/7.5%/7.5%` chronological split and disjoint final-test windows, while
  `paper/sections/06_experiments.tex` claims evaluation seeds are disjoint from
  oracle pretraining. By contrast, S2 `v2_ppo_metadata.json` points to one
  30,000-step `truth_v31.csv` per seed, and the inspected training script samples
  oracle, candidate-prior, and policy-evaluation rollouts from that sequence with
  independent RNG seeds rather than explicit disjoint partitions.
- `paper/tables/condition_results_v31.tex` also says
  "non-overlapping 512-step windows"; that wording requires verification and likely
  correction because the evaluator's recorded starts are sampled, not constrained
  to be non-overlapping by `select_eval_start_indices()`.
- Reconstructing starts with the implemented deterministic sampler confirms leakage
  and invalidates the current "held-out/non-overlapping" phrasing:
  - All 30 S2 runs were reconstructable from saved truth and metadata.
  - `21/30` runs have at least one interval overlap between the 4
    candidate-prior scoring windows (`512` steps) and the 6 final policy-evaluation
    windows (`1024` steps); this repeats as `7/10` seeds at each budget because each
    budget reuses the same seed-dependent truth/window pattern.
  - `21/30` runs have overlap among their own six evaluation windows; `6/30` also
  have overlap among candidate-prior windows. A first ad hoc check reported
  `6/30` because it dropped exact duplicate starts; the persistent audit correctly
  counts duplicate windows and yields `9/30`.
  - Example seed 41: prior starts `[7201, 20786, 10886, 29474]`, evaluation starts
    `[20473, 24839, 6803, 1812, 23734, 23119]`, with two prior/evaluation interval
    overlaps.
- Therefore replaying the oracle-prior-selected mask on the existing evaluation
  windows can be labelled only as an additional same-protocol diagnostic. It cannot
  be described as a training-selected static baseline evaluated on an independent
  held-out test split.
- Inspection of `src/v2/custom_ppo.py::_sample_start_idx()` confirms that S2 PPO
  training also has no fixed train partition: when `train_start_indices` is empty
  (as in the stored S2 metadata), each training rollout samples an event-biased or
  uniform start anywhere in the same truth sequence. No unused suffix of the
  existing truth can retrospectively be presented as a held-out test segment.
- Minimum credible S2 repair choices are therefore:
  1. replay the already trained PPO/oracle and the training-selected static candidate
     on newly generated independent truth sequences, clearly calling the frozen TCN
     a surrogate endpoint; or
  2. retrain under an explicit partition/independent-test protocol.
  A same-sequence oracle-static replay is useful only for diagnosis, not for the
  submission claim.
- The energy-account storm curriculum is intentionally trained on selected
  storm windows; it should remain an opportunity/mechanism or curriculum diagnostic
  unless an independent-truth evaluation is added. Existing "full-distribution
  no-retrain" wording is not equivalent to independence if it replays on the same
  truth used during curriculum construction/training.
- Posthoc independent-truth replay smoke succeeded for fixed-budget
  `B=1.70, training seed=41` on a new truth seed with two non-overlapping
  512-step windows. It is diagnostic only, but exposes a further blocker:
  `prior_selected_static` has FW-MAE `0.1850`, `custom_ppo` `0.1899`, and
  `full_open_unconstrained` `0.1952`. The frozen TCN trained in the old run does
  not retain the expected full-observation ordering on new truth. Independent
  replay of old models therefore cannot replace a split-aware oracle and policy
  retraining run.
- Split-protocol aggregation needs an explicit static-policy identity. The legacy
  collector hard-coded `feasible_static_projected`, while the corrected protocol
  selects `validation_selected_static` on the validation partition. The collector
  now accepts `--static-policy` with the legacy name as its default, so archived
  S2 aggregation semantics remain unchanged and new split-protocol checks can use
  the validation-selected comparator without conflation.
- The running `B=1.70`, seed `41` split-protocol gate has emitted a manifest with
  non-overlapping chronological ranges: oracle `[0,31500)`, RL `[31500,76500)`,
  validation `[76500,83250)`, and final test `[83250,90000)`. Its six final
  windows are uniform/non-event-filtered and disjoint by construction.
- The gate's RL-train candidate-prior table currently ranks
  `met_station_core|radiometer_basic|snow_particle_counter` first
  (`oracle_loss_mean=0.14635`, mean power `1.46`). This is not yet the reported
  static comparator because the corrected protocol selects that comparator on the
  separate validation partition after PPO training.
- The `B=1.70`, seed `41` split-protocol gate completed successfully. Final-test
  FW-MAE ranks full observation first (`0.1114`), validation-selected static
  second (`0.1195`), PD-PPO third (`0.1222`), then round-robin (`0.1243`),
  random (`0.1301`) and AoI (`0.1319`). PD-PPO is `2.31%` worse than the selected
  static comparator in this run, so the gate authorises statistical rerunning but
  does not support dynamic-over-static wording.
- The full corrected grid was launched on the server as
  `v31_split_main_20260526` under `reports/v31_split_protocol_main`, covering
  budgets `1.65/1.70/1.75` and seeds `41--50` on GPUs `1--5`; GPU `0` remains
  assigned to an unrelated active workload.

## Official CRST Requirements Audit (retrieved 2026-05-25)
Source: official Elsevier/ScienceDirect Guide for Authors:
`https://www.sciencedirect.com/journal/cold-regions-science-and-technology/publish/guide-for-authors`.

- Scope fit must be stated as an applied cold-regions engineering/science problem;
  a theoretical method must discuss application to cold-regions problems in detail.
- The journal publishes original Research Articles and uses single-anonymized peer
  review, normally with at least two independent reviewers.
- Editable `.tex` submission sources are acceptable; a PDF alone is not acceptable.
- Abstract is capped at 250 words, must be factual and stand alone; use 1--7 English
  keywords.
- Highlights are required as a separate editable file: 3--5 bullets, each no more
  than 85 characters including spaces.
- Tables must be editable text; figures must be individually supplied, cited, and
  meet resolution/artwork requirements.
- Generative AI may assist manuscript preparation only with author verification and
  a declaration before references. Generative AI is not permitted to create or
  alter submitted figures or graphical abstracts. Existing or new submission
  artwork must therefore be created from data/code/manual design only.
- Data policy is Option C: deposit and cite/link research data in a relevant
  repository, or state why it cannot be shared; a data-availability statement is
  required at submission.
- A CRediT author-contribution statement is required. The submission also needs
  corresponding-author contact details, funding disclosure, competing-interest
  declaration, acknowledgements in the correct location, and consistent references.

## Rewrite Evidence Boundary (established 2026-05-25)
- Created the formal claim ledger:
  `rl_sensor_scheduling_framework/docs/05-25-full-rewrite-evidence-ledger.md`.
- Created the journal-oriented manuscript blueprint:
  `rl_sensor_scheduling_framework/docs/05-25-crst-rewrite-strategy.md`.
- The new paper will be centered on the question of when adaptive scheduling adds
  value for simulated Antarctic blowing-snow monitoring, rather than presenting an
  algorithm novelty narrative first.
- Local locked artifacts establish three evidence blocks:
  - V3.1 fixed-budget benchmark: 3 budgets x 10 seeds; PD-PPO is below dynamic
    heuristics in mean FW-MAE but remains slightly worse than static projection.
  - Simplified calibrated energy-account storm diagnostic: dynamic snow-core to
    event-laser+FC4 reference gives `0.4169` versus static snow core `0.4248`.
  - Five-seed curriculum learned-policy check: storm PD-PPO
    `0.4153 +/- 0.0051` versus AoI `0.4176 +/- 0.0105`, but only `3/5` seed wins
    over AoI; therefore no robust AoI-dominance claim is allowed.
- The Route A frozen-forecast/DQN line documented in `AGENTS.md` is not to be mixed
  into the V3.1/PD-PPO manuscript without an explicit method reset, because it has
  different objectives, configurations, and experimental lineage.
- Server-side read-only verification passed:
  - V3.1 final summary checksum matches local exactly and the remote directory
    contains all `30` completion markers.
  - The energy-account aggregate is local-derived; the exact five storm CSV and
    thirty full-distribution NPZ inputs used by its aggregation script match remote
    with combined manifest digest `f0e1c24228efd908fdd773f853fbaba2c08fd6fd6a1ffb3d55dad341a6f93e23`.

## 2026-05-26 Corrected Energy-Account Gate Status
- The only potentially promotable energy-account learned-policy evidence remains the
  corrected split-protocol `semi_markov` gate in remote tmux session
  `energy_split_semimarkov_gate_20260526`.
- At `2026-05-26 06:41 CST`, its PPO training process was alive and had reached
  `55296/100000` timesteps. Only manifest, truth/oracle and live-training artifacts
  existed; no final comparison artifact was yet available.
- Until that run completes and passes independent final-test review, the current
  manuscript correctly excludes learned energy-account comparative claims. The
  older curriculum outputs remain diagnostics only because of split leakage.
- The active draft's corrected fixed-budget significance wording is supported by
  `reports/v31_split_protocol_main/v31_s2_significance.csv`: all nine
  PD-PPO-versus-dynamic-heuristic tests across the three budgets have recorded
  adjusted `p <= 0.0234375`; each PD-PPO-versus-validation-selected-static
  comparison has adjusted `p = 1.0`.
- A duplicated fragment in the Hyperparameter Sensitivity paragraph was a
  manuscript assembly defect, not an evidence issue; it has been repaired and
  verified in the rendered PDF.
- The active manuscript evidence package is not currently represented by the
  public GitHub repository: remote `HEAD` is `161d0a8`, while the corrected
  split-protocol scripts and final-test result directory are locally untracked.
  The data-availability statement has therefore been changed from an inaccurate
  released-script reproducibility claim to a truthful development-snapshot plus
  required-versioned-deposit statement.
- Independent evidence review found one unsupported sentence in Methods claiming
  hyperparameters were selected by held-out validation grid search. No such
  selection procedure is authorised by the evidence ledger: H1 is a local
  diagnostic around the configured method. The manuscript now states the
  configuration was pre-specified and that H1 is post hoc/local only.
- The review also identified a potential ambiguity in the event-condition table:
  event labels are not an operationally observable deployment input, but in the
  simulation they are available as PD-PPO event context. The regenerated caption
  now says both facts explicitly.

## 2026-06-02 Fork-Branch Clean Rewrite Start
- The user requested a true large-scale PD-PPO paper rewrite in the current fork
  context, with a backup first and a new TeX source rather than lazy reuse of the
  old `sections/*.tex` prose.
- Current paper source before this rewrite remained `rl_sensor_scheduling_framework/paper/main.tex`;
  the paper subrepository was already dirty with `paper.tex -> main.tex`, edited
  old sections, regenerated tables/figures and compiled artifacts. Those existing
  changes were not reverted or overwritten.
- A fresh archive of the current paper directory was created at
  `rl_sensor_scheduling_framework/paper_archives/paper_pre_fork_rewrite_20260602_122720.tar.gz`
  before adding the new source.
- New clean manuscript source now lives at
  `rl_sensor_scheduling_framework/paper/pdppo_crst_rewrite.tex` and imports only
  `paper/rewrite_sections/*.tex`, not the old `paper/sections/*.tex` files.
- The new first-pass framing is diagnostic and regime-centered:
  instantaneous fixed budgets support compact quasi-static allocation; PD-PPO
  improves over dynamic heuristics but not over validation-selected static
  allocation; simplified energy-account results are mechanism diagnostics, not
  held-out learned-policy superiority.
- The first clean rewrite reuses only audited tables at this stage:
  `tables/sensor_specs.tex`, `tables/g1_generator_validation.tex`,
  `tables/main_results_v31.tex`, and `tables/energy_account_storm_oracle.tex`.
  Figures and additional tables still require a separate provenance/artwork pass.
- Build validation: `latexmk -xelatex -interaction=nonstopmode -halt-on-error
  pdppo_crst_rewrite.tex` succeeds and produces a 23-page PDF. The final log has
  no undefined citations or labels; remaining issues are two very small overfull
  boxes and a pre-existing BibTeX warning that `Pendyala2024` has an empty pages
  field.
- CRST front-matter checks: abstract is about 215 words; the new highlights file
  `pdppo_crst_rewrite_highlights.txt` has five bullets, each under 85 characters.
- Follow-up figure/table integration: the first clean rewrite initially cited four
  audited tables but no figures. This was intentionally conservative because old
  figure prose and decorative assets had not passed the no-lazy-reuse/provenance
  gate, but it left the draft visually under-supported. The rewrite now cites and
  includes four non-decorative figures:
  `figures/data_split_timeline_tikz.tex`,
  `figures/figure3_synthetic_statistics.png`,
  `figures/figure6_power_error_tradeoff_v31.png`, and
  `figures/figure5_sensor_timeline.png`.
- Rebuild after figure integration succeeds. The PDF is now 26 pages; `pdftotext`
  confirms Figure 1--4 and Table 1--4 appear in the rendered text. No undefined
  figure/table/citation references remain.

## Requirements
- User asked to use the `planning-with-files` skill to organize the next phase of work.
- The immediate planning target is the scheduling-degeneration problem described in `rl_sensor_scheduling_framework/docs/05-22-plan-1.md`.
- Planning should account for prior observations from Figure 8 and server/result inspection:
  - `feasible_static_projected` is almost fully static.
  - PD-PPO is mostly quasi-static.
  - Round-robin and AoI switch frequently but suffer many invalid or aborted warmups.
  - The paper narrative must avoid overclaiming dynamic scheduling value.

## Research Findings
- `05-22-plan-1.md` diagnoses a "static trap": the environment and reward make a fixed or near-fixed sensor subset close to optimal.
- The document proposes five scenario adjustments:
  - A: tighter budgets, especially `B <= 1.60`.
  - B: higher blowing-snow event frequency and longer events.
  - C: more physically heterogeneous sensor costs.
  - D: explicit warmup-efficiency reward.
  - E: snow-heavy target weights for evaluation.
- The most defensible first move is diagnostic and configuration-level:
  - verify the actual current costs, event fractions, warmup latencies, and target weights;
  - then test tight-budget and/or physical-cost variants before touching the reward function.
- Important caution: the plan's phrase "feasible_static_projected is forced to switch" is likely conceptually wrong if the baseline is truly static. A tighter budget should change the fixed subset, not make it time-varying, unless the baseline is redefined.
- Important caution: current budget "looseness" should not be assumed. The observed behavior may be better explained as "a low-cost core subset captures most forecast value" rather than budget being simply non-binding.
- `static projection` should remain a strong reference, not a strawman. It must be re-optimized for each new budget/cost/event scenario.
- Switching rate alone is not a success metric. The useful signal is purposeful event-conditioned activation with lower warmup-abort rate and better forecast quality.
- Phase 2 diagnostics produced `rl_sensor_scheduling_framework/docs/05-22-v31-behavior-diagnostics.md` and CSVs under `rl_sensor_scheduling_framework/reports/v31_s2_main/behavior_diagnostics/`.
- Current V3.1 sensor costs are intentionally balanced/weakly heterogeneous, not physically realistic. This supports trying a separate physical-cost pilot instead of claiming the current cost vector is deployment-faithful.
- The static core subset is `met_station_core + radiometer_basic + snow_particle_counter` with steady power `1.46`; it is selected because coverage groups force one weather, one surface, and one snow-transport sensor before greedy score projection.
- Because the coverage-group minimum is also `1.46`, budgets below `1.46` are infeasible unless coverage groups are changed or removed. Therefore the proposed `B=1.50/1.55/1.60` sweep will not break the static core under current balanced costs.
- V3.1 event rate is not low: truth event rate is about `0.289`, max 512-step event fraction is about `0.795`, and `P(event_fraction_512 > 0.75)` is about `0.091`.
- PD-PPO has low warmup-abort rate relative to round-robin/AoI, but its high-latency sensor use is not strongly event-triggered. `laser_disdrometer` is selected slightly less during events than non-events; `fc4_flux` is almost never selected.
- Early complex-cost pilot candidate-prior evidence (`reports/v31_complex_pilot/v31_pilot_budget1p70_seed41/custom_ppo_candidate_prior.csv` on the server):
  - best prior subset at `B=1.70` is `met_station_core + radiometer_basic + surface_temp_ir + snow_particle_counter`, power `1.30`, oracle loss about `0.182`;
  - laser-disdrometer subsets are present but mid-ranked;
  - `fc4_flux` is absent from the top 15 candidate-prior table;
  - therefore physical-cost heterogeneity changes the static/core subset but does not by itself prove event-triggered high-latency activation.
- Completed complex-cost pilot (`B=1.70`, `seed=41`) shows a stronger negative result:
  - `custom_ppo` and `feasible_static_projected` have identical forecast/oracle metrics: `forecast_weighted_mae_overall = 0.122425`, `oracle_loss_mean = 0.122425`, `power_mean = 1.30`, `warmup_abort_rate = 0`;
  - their rollout arrays are exactly equal for selected masks, modes, powers, peaks, oracle losses, observations, and observed masks;
  - policy behavior summary: both have `8` near-constant sensors, `4` constant-active sensors, and `4` constant-off sensors;
  - high-latency sensors `laser_disdrometer` and `fc4_flux` are never selected by `custom_ppo`;
  - therefore this physical-cost pilot should not be scaled as a main sweep.
- `custom_ppo` action-score diagnostics in the completed pilot show a hard constant mask: selected sensors have score `+1`, unselected sensors have score `-1`, with zero score variance.
- A no-prior/no-AWBC ablation was launched to distinguish reward/environment static optimality from regularization-induced policy collapse.
- Completed no-prior/no-AWBC ablation:
  - dynamic behavior returns: `custom_ppo` switch rate is `2.049` switches/step and only `1` near-constant sensor remains;
  - forecast quality worsens strongly: `forecast_weighted_mae_overall = 0.145684` versus static `0.121849` in the same ablation run;
  - warmup abort rate rises to `0.0664`;
  - `laser_disdrometer` event lift is negative (`-0.0206`), and `fc4_flux` remains unused;
  - conclusion: removing candidate-prior/AWBC creates motion, not useful event-conditioned scheduling.
- Created memo `rl_sensor_scheduling_framework/docs/05-22-v31-complex-pilot-results.md`.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Stage diagnostics before new training | Prevents launching expensive server jobs around an unverified causal story. |
| Keep V3.1 S2 untouched | It is currently the completed main-result candidate and should remain reproducible. |
| Use separate output tags for new experiments | Avoids contaminating existing tables/figures and paper assets. |
| Treat reward modification as lower priority | Adding a warmup reward after seeing the failure could look post hoc; configuration changes are easier to justify physically. |
| Do not run a balanced-cost `B=1.50--1.60` full sweep as the next main experiment | Diagnostics show the current static core remains feasible down to `B=1.46`; this sweep would likely spend compute without addressing the core mechanism. |
| Prefer a separate physical-cost pilot as the next experiment candidate | It directly targets the current balanced-cost artifact while preserving a physically meaningful story. |
| Use wrapper-level `--sensor-cfg` passthrough for the physical-cost pilot | The bottom-level custom PPO trainer already records `sensor_cfg` in metadata; changing only wrappers avoids touching reward/training internals. |
| Use wrapper-level `--antaws-root` passthrough for server runs | The server stores AntAWS under project-local `data/AntAWS/3_hourly`, while the bottom-level default points to `../data/AntAWS/3_hourly`. |
| Resolve wrapper-level sensor configs to absolute framework paths | The bottom-level trainer opens `sensor_cfg` relative to the process working directory, so `configs/...` only works when launched from `rl_sensor_scheduling_framework/`. |
| Reuse the same behavior-diagnostics script for pilot and full-grid outputs | Comparable diagnostics are needed before deciding whether to scale the complex-cost scenario. |
| Keep current `feasible_static_projected` wording as fixed-priority unless a new optimized static policy is implemented | Code inspection shows it is a projected fixed priority order, not a solved static optimum. Calling it re-optimized would be inaccurate under the current implementation. |
| Add `oracle_static_projected` instead of changing `feasible_static_projected` | The fixed-priority baseline remains historically comparable; the new baseline takes the best fixed candidate from the oracle candidate-prior table and evaluates it with an exact mask rollout. |
| Do not scale the completed complex-cost pilot to more seeds yet | The first seed exactly matches fixed-priority static at the rollout level, so scaling would likely spend compute confirming a failure mode rather than informing a design choice. |
| Run a no-prior/no-AWBC ablation as the next mechanism probe | If it becomes dynamic, the collapse is partly due to candidate-prior/AWBC regularization; if it remains static or performs poorly, the reward/candidate environment is the deeper issue. |
| Add a claim-audit memo before further paper edits | The evidence now combines fixed-budget V3.1, energy-account oracle diagnostics, and learned energy-account curriculum results. Keeping these claim classes separate prevents accidental overstatement of robust AoI dominance or event-triggered laser gating. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No prior `task_plan.md`, `findings.md`, or `progress.md` existed in project root | Created new planning files for this task. |
| Remote dry-run failed with `~/miniconda3/etc/profile.d/conda.sh` missing | Located server conda through `bash -lc` and used `/opt/miniconda3/etc/profile.d/conda.sh`. |
| First remote pilot failed because AntAWS root default was wrong for the server | Added wrapper passthrough and restarted with `--antaws-root data/AntAWS/3_hourly`. |
| Second remote pilot failed because `configs/sensors/windblown_sensors_complex.yaml` was not valid relative to the server project root | Resolved relative sensor configs inside the wrappers before passing them to the bottom-level trainer. |
| A remote sync command briefly created `rl_sensor_scheduling_framework/rl_sensor_scheduling_framework/...` | Removed the accidental nested directory and re-synced changed files to the project root. |

## Resources
- New scenario proposal: `rl_sensor_scheduling_framework/docs/05-22-plan-1.md`
- Peer-review risk memo: `rl_sensor_scheduling_framework/docs/05-17-peer review.md`
- V3.1 S2 completion report: `rl_sensor_scheduling_framework/docs/05-13/03_V31_s2_completion_report.md`
- Forecast-reward mainline plan: `rl_sensor_scheduling_framework/docs/forecast_reward_enable_plan.md`
- Baseline semantics: `rl_sensor_scheduling_framework/docs/scheduler_baselines.md`
- Warmup design: `rl_sensor_scheduling_framework/docs/sensor_warmup_design_plan.md`
- Prior static-trap diagnosis: `rl_sensor_scheduling_framework/docs/explore-result (1).md`
- Physical-cost pilot output target: `rl_sensor_scheduling_framework/reports/v31_complex_pilot/`
- Claim audit memo: `rl_sensor_scheduling_framework/docs/05-24-claim-audit.md`

## Visual/Browser Findings
- Figure 8 visual inspection revealed that same-color sensor rows could visually merge; the plot was redesigned with discrete colors and row separators.
- The redesigned Figure 8 then exposed the core behavioral issue: static projection is fixed, PD-PPO is quasi-static, and heuristic baselines often switch rapidly with many warmup failures.

## 2026-05-22 New Mainline Implementation Notes
- `05-22-chore.md` changes the immediate work from scaling the old complex-cost branch to a new physical-event implementation track.
- Highest-risk unsupported claims to repair:
  - fixed-budget experiments should not be described as time-varying energy-harvesting control;
  - V3.1 rollout evidence does not yet support strong claims of event-conditioned warm-up by PD-PPO;
  - "multi-year" wording must mean calibrated to multi-year statistics, not multi-season deployment validation;
  - sensor costs are normalized scheduling costs, not a measured watt/SOC model.
- Short-term implementation target:
  - create `physical_event_value_v2` with feasible but costly `laser_disdrometer`;
  - run with `B=1.20`, `startup_peak_budget=1.60`, required `met_station_core`, disabled coverage groups, and snow-focused target weights;
  - use no-retrain diagnostics before PPO training.
- Energy-account/SOC modelling is important for physical realism, but should be introduced as a gated architecture phase rather than silently mixed into the first fixed-budget pilot.
- Formal `physical_event_value_v2` TCN oracle-lift diagnostic failed the pilot gate:
  - candidate count at `B=1.20`: `48`;
  - truth event rate: `0.2701`;
  - best overall and best event subset: `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
  - best laser event subset: `met_station_core|surface_temp_ir|laser_disdrometer`;
  - best FC4 event subset: `met_station_core|radiometer_basic|surface_temp_ir|fc4_flux`;
  - `laser_event_lift = -0.0342`;
  - `fc4_event_lift = -0.0221`.
- Decision: do not launch PPO from `physical_event_value_v2`; the desired high-latency/high-value scheduling behavior is not useful to the current frozen oracle even before policy learning.
- Root-cause hypothesis from code inspection:
  - V2 sensor observations use fixed per-variable Gaussian noise only.
  - `laser_disdrometer` and `snow_particle_counter` observe the same two particle variables.
  - Since laser is costlier, slower, and higher warmup while only being somewhat less noisy, the oracle prefers the cheaper particle counter plus low-cost context sensors.
  - FC4 directly observes flux, but current weights/scales/generator make cheap proxy subsets more valuable than direct sparse flux measurement.
- Next design direction should repair observation semantics, not PPO:
  - introduce event-conditioned degradation/saturation for the low-cost particle counter;
  - keep laser as the high-fidelity event instrument;
  - make FC4 valuable only where it is physically defensible, without claiming a full SOC/energy-harvesting model.
- `physical_event_value_v3` findings so far:
  - event-conditioned snow-counter noise alone is insufficient;
  - simply increasing snow/flux target weights is insufficient;
  - adding laser-observed flux is defensible, but still insufficient under the linear oracle;
  - static candidate lift is an incomplete gate because it cannot test "cheap calm core, expensive event instrument" behavior.
- Added dynamic schedule probes to distinguish fixed-subset value from actual scheduling value.
- Added optional event microstructure to the truth generator. This is now the most plausible missing ingredient: without event-internal snow/flux variability that is not already explained by wind/surface proxies, there is little reason for any scheduler to spend budget on particle/flux sensors.
- Small local TCN smoke with event microstructure gave a weak positive direction:
  - best overall candidate became `dynamic:calm_core__event_laser_surface`;
  - static laser event lift was near zero rather than strongly negative;
  - this justifies a formal server TCN diagnostic, but not PPO training yet.
- Formal server TCN diagnostic for v3 microstructure did not pass:
  - best event subset still used the low-cost `snow_particle_counter`;
  - `laser_event_lift = -0.0047`, much closer than v2 but still negative;
  - `fc4_event_lift = -0.0137`;
  - dynamic schedule probes were worse than the best fixed subset.
- Current best explanation: the low-cost snow particle counter remains too reliable in dense events. A realistic scenario needs saturation/partial availability, not only larger Gaussian noise.
- v4 repair adds event-time partial availability for the low-cost snow particle counter, representing saturation/occlusion during dense blowing snow.
- Local v4 smoke is the first clear positive diagnostic:
  - `laser_event_lift = +0.0083`;
  - best event laser subset is `met_station_core|surface_temp_ir|laser_disdrometer`;
  - best overall smoke row is a dynamic calm-core/event-laser schedule.
- This justifies a formal server TCN gate, but still does not justify PPO until the formal gate passes.
- Formal v4 server TCN gate confirms laser value but not dynamic scheduling value:
  - `laser_event_lift = +0.0187`;
  - best overall fixed subset is `met_station_core|surface_temp_ir|laser_disdrometer`;
  - hand-written dynamic schedules remain worse than this fixed subset.
- A tighter fixed per-step budget (`B=1.10`) did not rescue the dynamic story in local smoke.
- Decision: fixed per-step budgets are still structurally prone to static optima. The next defensible mechanism is an energy-account/SOC constraint where always-on laser is costly over time, but event-time laser activation can be useful.
- Local energy-account smoke supports this mechanism when the account is tuned so calm periods can recharge but laser cannot be treated as a free always-on core:
  - `energy_capacity=48`, `initial_energy=24`, `harvest_per_step=0.95`, `reserve_energy=4`;
  - best overall diagnostic row: `dynamic:calm_core__event_laser_surface`;
  - no SOC guard drops for the best dynamic row;
  - static laser is no longer the best overall row.
- This remains a diagnostic gate, not a final paper result, until the formal server TCN run confirms it.
- Formal energy-account diagnostics did not pass the full scheduling criterion:
  - `harvest_per_step=0.95` failed: static snow-counter remained best and lead laser was clipped heavily.
  - `harvest_per_step=1.05` made laser event value positive (`laser_event_lift=+0.0064`) but static snow-counter remained best overall.
  - Lead dynamic schedules are directionally meaningful but not yet strong enough or stable enough for PPO training.
- Current paper-safe interpretation:
  - v4 + energy-account is a promising mechanism probe;
  - it should not be promoted to a main result without further stabilization;
  - fixed-budget PPO should remain blocked because it would likely learn a static policy rather than a defensible event-conditioned scheduler.
- User correctly identified that harvest must be calibrated from formal event-cluster statistics rather than tuned ad hoc.
- Formal truth cluster statistics:
  - event fraction: `0.2701`;
  - mean event run length: `17.85`;
  - mean calm run length: `48.23`, but median calm run length is only `5`, showing strong burst clustering;
  - lead4 trigger fraction: `0.3307`.
- For the lead4 dynamic schedule, the relevant average cost is not only `met + laser * event_fraction`; it is:
  - calm cost `met+radiometer+surface = 0.32`;
  - triggered cost `met+surface+laser = 1.16`;
  - average lead4 cost `0.32*(1-0.3307) + 1.16*0.3307 = 0.5978`.
- Therefore a physically calibrated harvest should be near or above `0.60`, not `0.95--1.05`, and capacity must be calibrated from burst drawdown.
- Added `scripts/50_v31_energy_account_calibrate.py` to compute event/calm runs, lead-trigger occupancy, average costs, and drawdown requirements from a formal truth file.
- Local calibrated smoke with `harvest=0.62`, `capacity=300`, `initial=300`, and `reserve=20` passed:
  - best overall: `dynamic:calm_core__lead4_laser_surface`;
  - no energy guard drops;
  - `laser_event_lift=+0.0083`.
- Formal calibrated gates bracketed the current harvest window:
  - `harvest=0.62`, `capacity=300` kept static laser infeasible enough, but static snow-counter core still won overall (`oracle_loss_mean=0.3354`); dynamic lead laser improved event loss but paid too much non-event penalty.
  - `harvest=0.94`, `capacity=300` made static laser feasible and best overall (`met_station_core|surface_temp_ir|laser_disdrometer`, `oracle_loss_mean=0.3278`), so it is too high for demonstrating dynamic scheduling.
- A new schedule diagnostic was added because the previous lead schedules turn off the snow counter during lead-trigger non-event steps, which likely creates a non-event penalty. The new no-lead snow-core schedules keep `met+radiometer+surface+snow_counter` during non-event periods and switch to laser only when the event flag is active.
- The `snow+laser` burst variant is physically interesting but not automatically advantageous: with calm cost `0.82`, event cost `1.66`, and lead4 trigger fraction `0.3307`, its average cost is about `1.10`, and its event-burst drawdown can exceed static laser drawdown in high-trigger windows.
- Formal no-lead new-schedule gate at `harvest=0.62`, `capacity=300` shows the current useful boundary:
  - best overall remains static snow-counter core (`0.3364`);
  - best event row becomes `dynamic:snow_core__event_laser_fc4` (`event_loss=0.3882` vs static snow event `0.4135`);
  - however, dynamic event-laser rows hit hundreds of SOC guard drops and warmup aborts, so the non-event/overall score remains worse.
- Drawdown calibration suggests a more physical window for no-lead snow-core/event-laser is `harvest≈0.92`, because its average cost is `0.9118`. With `capacity=120` and `reserve=20`, dynamic event-laser drawdown should fit typical eval windows, while static laser drawdown should still exceed usable capacity.
- Formal `harvest=0.92`, `capacity=120` confirms the intended energy-account behavior but still does not beat static snow overall on the full-distribution eval:
  - static snow core: `oracle_loss_mean=0.3353`;
  - `dynamic:snow_core__event_laser_surface`: `0.3418`, no guard drops;
  - `dynamic:snow_core__event_laser_fc4`: best event loss `0.3895`, no guard drops;
  - static laser is clipped (`met_station_core|laser_disdrometer` has 426 guard drops).
- Break-even analysis from the formal table:
  - `snow_core__event_laser_surface` needs event fraction about `0.659` to beat static snow from event/non-event losses;
  - `snow_core__event_laser_fc4` needs event fraction about `0.548`;
  - the random full-distribution eval windows used so far have event fractions only `0.24--0.42`, but formal truth contains 1024-step storm windows up to `0.695`.
- This means the current scenario can support a defensible "storm-window dynamic advantage" claim only if the storm-window gate passes. It still does not support a broad full-distribution dynamic-superiority claim.
- Storm-window gate at `harvest=0.92`, `capacity=120` passed:
  - best overall: `dynamic:snow_core__event_laser_fc4`, `oracle_loss_mean=0.4177`;
  - static snow core: `0.4254`;
  - event loss improves from static snow `0.3529` to dynamic `0.3208`;
  - event-window fractions were `[0.695, 0.684, 0.545, 0.540, 0.464, 0.463]`.
- Remaining caveat: the best storm-window dynamic row still has 72 SOC guard drops, so it is evidence for a storm-window mechanism but not yet the clean final setting. A follow-up with `capacity=180` is running to reduce dynamic clipping while keeping static laser infeasible.
- Storm-window `harvest=0.92`, `capacity=180`, `reserve=20` is the cleanest calibrated dynamic-advantage setting so far:
  - best overall: `dynamic:snow_core__event_laser_fc4`, `oracle_loss_mean=0.4169`;
  - static snow core: `0.4248`;
  - event loss improves from `0.3517` to `0.3190`;
  - best dynamic row has `soc_min=48.48`, `energy_guard_dropped=0`, `warmup_abort_count=0`;
  - static laser remains energy-infeasible, e.g. `met_station_core|radiometer_basic|laser_disdrometer` has 438 guard drops.
- Interpretation boundary: dynamic advantage is currently established for selected storm windows under the calibrated energy-account diagnostic. Full-distribution overall superiority is still not established; full-distribution results favor the static snow-counter core.
- `05-22-judge.md` is directionally correct but should not be used verbatim:
  - V3.1 PD-PPO is best among learned/adaptive policies, not better than the feasible static projection.
  - `h=0.94/cap300` failed because the finite window and large capacity made static laser viable, not because `h=0.94` exceeds static-laser steady cost.
  - The implemented no-lead `snow_core__event_laser_fc4` schedule has calm cost `0.82` and event cost `1.13`; the energy account subtracts steady cost, while startup peaks remain a separate hard feasibility rail.
  - `event_flag` is a simulator event column, suitable for oracle diagnostics and stratified evaluation, but an operational controller would need a met-sensor-derived event-probability estimate.
- A single-seed PPO pilot is now running to test whether learned PD-PPO can exploit the storm-window opportunity. This remains a validation step, not a prerequisite for the oracle-level paper claim.
- Single-seed PPO pilot result is mixed:
  - `custom_ppo` storm-window oracle loss is `0.4195`, which beats the static snow-core oracle diagnostic reference (`0.4248`) but remains worse than the hand-written dynamic oracle schedule (`0.4169`) and worse than AoI in the same PPO evaluation (`0.4144`).
  - The policy does learn some event-conditioned laser preference: laser selection is `0.5487` during events versus `0.2983` during non-events (`1.84x`), but it misses the planned `>3x` success criterion.
  - Warmup abort count is `14`, slightly above the planned `<10` criterion.
  - Conclusion: the oracle-level storm-window claim is usable; the stronger claim that trained PD-PPO reliably learns the ideal event-conditioned schedule is not yet established.
- Added storm-window curriculum training support and ran a second PPO pilot on the same six event-rich windows used for evaluation.
- Curriculum PPO result:
  - `custom_ppo` oracle loss is `0.4106`, beating AoI (`0.4130`) and clearly beating round-robin (`0.4720`), random (`0.4660`), and feasible static projected (`0.5032`) in the same storm-window evaluation.
  - Warmup abort count improves to `5`, compared with AoI `10`, round-robin `768`, and random `1956`.
  - PPO selected rates show useful adaptive behavior, but not the clean oracle story:
    - `radiometer_basic`: event/non-event `0.5006 / 0.2212`, ratio `2.26`;
    - `snow_particle_counter`: `0.4767 / 0.2474`, ratio `1.93`;
    - `fc4_flux`: `0.4980 / 0.3694`, ratio `1.35`;
    - `laser_disdrometer`: `0.5187 / 0.7440`, ratio `0.70`.
  - The improvement is therefore real at the learned-policy/overall-oracle level, but the paper should frame it as curriculum-aided storm-window adaptive scheduling rather than as a learned event-triggered laser schedule.
- Three-seed curriculum PPO robustness check:
  - seeds `41/42/43` all produced PPO oracle loss no worse than AoI:
    - seed 41: PPO `0.4106`, AoI `0.4130`;
    - seed 42: PPO `0.4206`, AoI `0.4356`;
    - seed 43: PPO `0.4096`, AoI `0.4096` with PPO lower by `5.4e-5`.
  - Across seeds, mean oracle loss is PPO `0.4136 ± 0.0061`, AoI `0.4194 ± 0.0141`, round-robin `0.4523 ± 0.0171`, random `0.4636 ± 0.0139`, feasible static projected `0.4880 ± 0.0167`.
  - PPO warmup aborts remain unstable: `5`, `33`, `19` across seeds, mean `19.0`, compared with AoI's fixed `10`.
  - Mean PPO event/non-event selected ratios:
    - `radiometer_basic`: `1.63x`;
    - `snow_particle_counter`: `1.13x`;
    - `fc4_flux`: `1.04x`;
    - `laser_disdrometer`: `1.03x`;
    - `surface_temp_ir`: `0.64x`.
  - Interpretation: the curriculum setting now supports a learned adaptive storm-window advantage over AoI on oracle loss, but not a clean claim of robust event-triggered high-latency laser control or superior warmup management.
- Mechanism diagnosis on existing rollout data:
  - The saved rollout files do not store per-step abort markers, but aborts can be reconstructed exactly from `mode_ids`, `selected_masks`, and `step_indices`; reconstructed totals match stored `warmup_abort_count`.
  - PPO laser aborts across seeds 41/42/43 are mostly non-event: `41` total, `10` event, `31` non-event.
  - PPO wins over AoI mostly in non-event loss, not event loss:
    - PPO mean event/non-event oracle loss: `0.3274 / 0.5256`;
    - AoI mean event/non-event oracle loss: `0.3243 / 0.5431`.
  - Therefore explanation 1 is not well supported: low laser event bias is not mainly caused by event-time SOC aborts.
  - The strongest current explanation is a conservative storm-window allocation strategy: PPO keeps event performance close while lowering non-event degradation.
  - Created `docs/05-23-curriculum-ppo-mechanism-diagnosis.md`.
- Event reward multiplier probe (`event_reward_multiplier=1.5`, seed 41):
  - PPO oracle loss improved from baseline seed-41 `0.4106` to `0.4088`; AoI in the same run was `0.4118`.
  - PPO event loss improved from `0.3307` to `0.3247`; non-event loss worsened slightly from `0.5144` to `0.5180`.
  - Warmup aborts remained acceptable: `7` vs baseline seed-41 `5` and AoI `10`.
  - Laser selected rate increased in both event and non-event steps: event/non-event `0.740 / 0.768`, ratio `0.96x`.
  - Non-event laser usage is not explained only by bridging short calm gaps; in baseline seed 41, laser selected rate at non-event steps 33+ steps away from the nearest event is `0.867`, higher than event-step laser rate `0.519`.
  - Conclusion: event weighting improves event performance, but does not produce event-triggered laser gating. It turns laser into a higher-duty storm-context sensor.
- Added rollout instrumentation for future runs:
  - `warmup_abort_deltas`;
  - `energy_guard_dropped`;
  - `soc`.
- Full-distribution generalization check using already trained curriculum PPO models:
  - Evaluated seeds `41/42/43` on six random 1024-step windows each, with mean event rate about `0.296`.
  - PPO beats AoI in all three seeds:
    - seed 41: PPO `0.3106`, AoI `0.3116`;
    - seed 42: PPO `0.3311`, AoI `0.3349`;
    - seed 43: PPO `0.2973`, AoI `0.3016`.
  - Mean oracle loss: PPO `0.3130 +/- 0.0171`, AoI `0.3160 +/- 0.0171`, static projected `0.3309 +/- 0.0077`, round-robin `0.3422 +/- 0.0241`, random `0.3471 +/- 0.0236`.
  - Boundary: PPO does not dominate static in every seed; seed 42 static projected (`0.3220`) is better than PPO (`0.3311`).
  - Laser remains near event-neutral in full-distribution evaluation (`1.04x`), while radiometer has stronger event bias (`2.04x`).
  - This supports generalization of learned allocation advantage beyond selected storm windows, but not a blanket claim of per-seed dominance over static or event-triggered laser gating.
- Five-seed curriculum extension changes the statistical interpretation:
  - Storm-window PPO vs AoI is now only `3/5` wins, with mean margin still positive but small (`AoI - PPO = +0.0023`).
  - Storm-window PPO vs feasible static projected is `5/5` wins with a large mean margin (`+0.0589`), and PPO also beats round-robin/random in `5/5`.
  - Full-distribution PPO vs AoI is `4/5` wins with a small mean margin (`+0.0013`).
  - Full-distribution PPO vs static projected is `4/5` wins; seed 42 remains the exception.
  - Therefore the strongest current claim is not "PPO reliably beats AoI"; it is "under calibrated energy-account storm-window evaluation, curriculum PPO consistently beats static projected, round-robin, and random, and it has small average generalization advantage over AoI."
- SOC soft-penalty probe (`buffer=20`, `lambda=0.01`, seed 41) is not a clean abort fix:
  - oracle loss improves (`0.4106 -> 0.4069`);
  - event loss improves (`0.3307 -> 0.3234`);
  - mean power drops (`1.0128 -> 0.9960`);
  - but warmup aborts increase (`5 -> 9`) and energy guard drops are present (`31`).
  - This should be treated as a promising event-performance probe, not evidence that the SOC-management problem is solved.
- 300k training probe clarifies the value and limit of simply training longer:
  - On the same storm-window seed-41 setup, PPO improves from `0.4106` at 100k to `0.4053` at 300k, and beats AoI (`0.4125`) by a larger margin.
  - The improvement is event-side: event loss improves from `0.3307` to `0.3190`; non-event loss worsens slightly.
  - Laser event/non-event selected ratio improves materially (`0.70x -> 1.52x`), so longer training does help learn a more event-biased laser policy.
  - However, warmup aborts increase sharply (`5 -> 66`), and full-distribution seed-41 evaluation is slightly worse than AoI (`PPO 0.3122`, AoI 0.3118).
  - Therefore insufficient training is part of the storm-window mechanism issue, but not the whole reason PPO fails to robustly dominate AoI.
- Minimal event-gated actor is not sufficient:
  - A 200k seed-41 probe with `--event-gated-actor` gives storm-window PPO `0.4105` vs AoI `0.4128`, so it remains slightly better than AoI in the storm setting.
  - It reduces aborts relative to 300k (`66 -> 38`) but remains above the target `<20`.
  - Laser event/non-event ratio is only `0.78x`, worse than both the 300k baseline (`1.52x`) and the planned `>1.3x` criterion.
  - Full-distribution performance is worse than AoI (`PPO 0.3128`, AoI 0.3117).
  - Conclusion: simple event-gated output mixing does not solve the coordination problem. The actor either needs explicit SOC-aware gating/penalties or the training objective must penalize abort attempts more directly.

## SOC Auxiliary Critic Probe Implementation Notes (2026-05-24)
- The SOC auxiliary path is intentionally default-off and critic-side only. It does not alter action selection or evaluation unless `soc_aux_horizon > 0` and `soc_aux_coef > 0` during training.
- Training target semantics: for observation/action step `t`, the auxiliary head predicts post-step SOC ratios for `t...t+N-1` as recorded from env `info['soc_ratio']`. Episode boundaries are masked, so no target crosses a reset.
- The first server update confirms the auxiliary loss is active (`soc_aux_loss=0.0239` at update 1), so the probe now tests behavior rather than plumbing.

## SOC Auxiliary Probe Result Interpretation (2026-05-24)
- The SOC auxiliary head rapidly drives `soc_aux_loss` near zero, so future SOC is easy to predict from the current state under this simulator.
- Behavioral impact is limited. Compared with the 100k seed-41 baseline, storm-window oracle loss remains essentially tied/improved only marginally (`0.4106 -> 0.4105`), while aborts increase (`5 -> 16`). Compared with the 300k probe, aborts are much lower (`66 -> 16`) but event adaptation strength is not yet established.
- Full-distribution seed-41 remains below AoI (`PPO 0.3138` vs `AoI 0.3127`). Therefore the main bottleneck is not just critic representation of SOC; policy-level credit assignment/objective structure is still limiting.
- Current claim boundary should remain: curriculum PPO is useful in storm-window calibrated energy-account evaluation and beats static/round-robin/random robustly, but robust full-distribution AoI dominance is not established by SOC auxiliary learning alone.

## 2026-05-25 Paper Rewrite Control Findings
- `docs/05-25-1-paper.md` should be used as the narrative reconstruction guide, not as a literal unverified patch:
  - it says fixed-budget `B=1.70` uses five seeds, but locked Table 3 uses `n=10`;
  - it refers to `paper/paper.tex`, while the active LaTeX entrypoint is `paper/main.tex`;
  - it recommends adding a new energy-account table, but the current paper already has `paper/tables/energy_account_curriculum_results.tex`.
- Rewriting must keep the core evidence classes separated:
  - fixed-budget V3.1 S2 supports PD-PPO beating dynamic heuristics and approaching static projection;
  - fixed-budget V3.1 S2 does not support clean event-triggered laser gating;
  - calibrated energy-account oracle diagnostics support storm-window dynamic opportunity;
  - learned curriculum PD-PPO supports robust storm-window wins over static/round-robin/random and only bounded competitiveness with AoI.
- Any replacement text from `05-25-1-paper.md` that says "best feasible static sensor subset" should be checked against our actual baseline terminology: current `feasible_static_projected` is a fixed-priority projected baseline, not an exhaustive best-static-subset solver.

## 2026-05-25 Typography Review Findings
- `pdffonts paper/main.pdf` reports Computer Modern families (`CMR`, `CMMI`, `CMBX`, `CMTI`) for the current body and mathematical text.
- Rendered PDF checks on pages 1, 12, 27, and 31 show:
  - the manuscript is readable but the default type face reads as a generic TeX draft rather than an applied-journal submission;
  - subsection headings are italic and visually weak relative to surrounding double-spaced prose;
  - long figure/table captions render at essentially body prominence, especially the warm-up FSM and behavior diagnostic figures.
- Local TeX has `newtxtext.sty`, `newtxmath.sty`, and `titlesec.sty`, allowing a Times-like text/math system and clearer heading hierarchy without changing `elsarticle` review mode.
- After the previous length reduction, the behavior diagnostic formerly discussed as Figure 8 is numbered Figure 7 in the compiled shortened manuscript. Cross-references resolve correctly; the evidence asset itself remains present.

## 2026-05-26 Energy-Account Protocol Audit Resume
- Resumed R2 after completion of the corrected fixed-budget split-protocol grid.
  The remaining evidence blocker is whether energy-account/curriculum learned-policy
  outputs use independent training, selection, and final-evaluation windows.
- `scripts/52_energy_account_convergence_assets.py` only aggregates five storm
  curriculum directories and five no-retrain full-distribution rollout directories;
  it performs no window-independence validation.
- Direct inspection of saved storm curriculum metadata shows the core failure:
  `eval_start_indices` and `train_start_indices` are the same six starts
  (`22943, 6826, 21704, 8183, 17193, 16151`) in the inspected seeds
  (`41`, `42`, and `44`). This makes the reported storm-window learned-policy
  scores training-window replay, not held-out evaluation.
- The existing claim/mechanism memos may retain these runs as mechanism
  diagnostics, but their learned-policy storm claims cannot be treated as
  submission-level final-test evidence unless a corrected split-protocol run is
  completed.
- `scripts/25_v2_train_custom_ppo.py` calls `helpers.train_oracle(oracle_truth, ...)`;
  when no explicit oracle partition is passed, `oracle_truth` is the entire truth
  sequence. `scripts/23_v2_train_ppo.py::train_oracle()` then samples
  event-conditioned rollout starts over that input sequence. The saved curriculum
  metadata contains no oracle partition declaration, so an oracle/final-evaluation
  overlap audit must be reconstructed from the deterministic sampler.
- The no-retrain full-distribution run metadata still refers to each storm-trained
  checkpoint and its original `train_start_indices`, while adding a new set of
  `eval_start_indices`. These new evaluation windows must be compared to training
  and oracle windows before interpreting generalization.
- `src/v2/env.py` computes normalisation mean and standard deviation from the
  complete supplied truth unless external statistics are explicitly passed.
  The old curriculum metadata contains no training-only normalisation partition,
  so these runs also lack the normalisation isolation enforced by the new
  fixed-budget split protocol.
- Added and executed `scripts/60_energy_account_protocol_audit.py`; its persistent
  report is under `reports/energy_account_protocol_audit_20260526/`.
- Audit result: storm train/evaluation start equality occurs in `5/5` seeds;
  full-distribution replay overlaps default-length training and storm windows in
  `5/5`; reconstructed oracle windows overlap both reported evaluations in `5/5`;
  full-distribution evaluation contains internal overlap in `2/5`; training-only
  normalisation is declared in `0/5`.
- Decision: the energy-account oracle/reference-policy result may remain a
  mechanism opportunity diagnostic, but the learned curriculum table is not
  comparative manuscript evidence. It requires its own split-protocol retraining
  if retained in the submission narrative.
- Added `scripts/61_energy_account_split_protocol_run.py` as the repair path. It
  reserves chronological `oracle_pretrain`, `rl_train`, `validation` and
  `final_test` partitions; selects non-overlapping event-rich windows separately
  inside training, validation and final-test partitions; uses validation for the
  selected static comparator; and labels event-conditioned final testing as a
  conditional diagnostic rather than operational event detection.
- A local CPU smoke run with reduced training/oracle sizes completed end-to-end
  under `/tmp/energy_account_split_protocol_smoke`, including a saved
  `validation_selected_static` rollout and evaluation output.
- Deployed the new runner to the GPU server and launched seed-41 gate
  `energy_split_gate_20260526` under
  `reports/energy_account_split_protocol_gate/budget1p20_seed41`; GPU 0 is used.
- The formal manifest validates disjoint chronological partitions, but the selected
  event rate is only about `0.331` in RL training, `0.271` in validation and
  `0.276` in final test. Thus the launched gate tests an independent ordinary/base
  event distribution, not the old high-event storm-window regime.
- `scripts/52_energy_account_convergence_assets.py` previously regenerated a
  table/memo phrased as a main learned-policy result despite the failed protocol.
  It now regenerates those assets as an archived non-independent diagnostic and
  no longer boldfaces PD-PPO as a comparative winner.
- Pre-outcome inspection invalidated the first remote replacement gate: it used
  the legacy `clustered` truth generator, whose 90,000-step coverage fill
  concentrates events in early partitions. For coverage `0.30`, overall partition
  event rates were approximately `0.538/0.192/0.205/0.213`; raising coverage to
  `0.60` changed the early partitions but left validation/final rates unchanged.
- The invalid remote gate was terminated after its first PPO update and its
  artifacts retained under `reports/energy_account_split_protocol_invalid_clustered_gate/`.
  No result from it is eligible for comparison. The corrected energy runner now
  defaults to the V3.1 `semi_markov` generator before any replacement launch.
- Remote semi-Markov preflight passes the conditional-regime gate: complete
  partition event rates are approximately `0.321/0.307/0.307/0.300`, and
  final-test selected non-overlapping event windows average `0.521`. The
  replacement seed-41 run is active in
  `energy_split_semimarkov_gate_20260526` on GPU 0.
- The active LaTeX draft still embeds invalidated evidence: its introduction,
  experiments, discussion and conclusion describe learned energy-account
  curriculum performance as a result, and `paper/tables/main_results_v31.tex`
  still cites old S2 rather than the completed fixed-budget split protocol.
  These are required replacements in the fresh manuscript construction phase,
  not claims to carry forward while the energy gate is pending.
- Updated `scripts/54_rebuild_table3_main_results.py` and regenerated
  `paper/tables/main_results_v31.tex` from
  `reports/v31_split_protocol_main/v31_s2_main_stats.csv`. The table now reports
  final-test split-protocol values and names the comparator
  `Validation-selected static`, matching E1b.
- Resolved the subsequent manuscript-asset gap. `scripts/35_v2_physical_unit_mae_table.py`
  now regenerates physical-unit results from the split-protocol grid, and the
  existing `scripts/54_rebuild_table3_main_results.py` now regenerates both the
  corrected main table and an event/non-event final-test table. The plot and
  behavior timeline generators now also read corrected final-test artifacts.
- Corrected manuscript values at `B=1.70`: PD-PPO
  `0.1334 +/- 0.0110`, validation-selected static
  `0.1329 +/- 0.0108`, round-robin `0.1408 +/- 0.0119`, and AoI
  `0.1432 +/- 0.0100`. PD-PPO is `5.21%` below round-robin and `6.80%`
  below AoI, but `0.42%` above validation-selected static; the static comparison
  remains non-significant.
- Corrected physical-unit deltas versus AoI at `B=1.70`: PD-PPO reduces air
  temperature MAE by approximately `0.432 degC` and wind-speed MAE by
  `0.062 m/s`; snow mass-flux MAE changes only by approximately `0.9%`.
- The current authoritative LaTeX source no longer imports the archived
  non-independent energy curriculum table as a result. Its title, 206-word
  abstract, Introduction contribution statement, Experiments, Discussion and
  Conclusion now state only: corrected fixed-budget evidence plus an
  energy-account reference-policy mechanism opportunity pending valid learned
  evaluation.
- PDF verification after this revision passed: `paper/main.pdf` compiles to
  `44` pages and rendered front/results/discussion pages show no clipping,
  overlap or unreadable new table/figure elements. Remaining overfull-box
  warnings are presentation polish, not evidence or compilation failures.
- Created `paper/highlights.txt` with five CRST-style evidence-bounded highlights;
  automated validation reports a maximum bullet length of `79` characters.
  CRediT, funding and competing-interest content remains pending author-provided
  facts rather than being inferred.

## 2026-05-26 Energy-Account Split Gate Result and n=5 Extension
- The corrected semi-Markov seed-41 gate completed on the server with exit code
  `0` and has been synchronized locally under
  `rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov/budget1p20_seed41/`.
- Manifest sanity passed: partitions are `oracle_pretrain=[0,27000)`,
  `rl_train=[27000,67500)`, `validation=[67500,78750)`, and
  `final_test=[78750,90000)`. Selected event-rate means are `0.6304` for training
  curriculum windows, `0.4875` for validation static-selection windows, and
  `0.5207` for final-test conditional windows.
- Final-test oracle-loss ordering for seed 41 is:
  `full_open_unconstrained=0.45894`, `custom_ppo=0.47455`,
  `validation_selected_static=0.47522`, `round_robin=0.48008`,
  `aoi=0.48296`, `feasible_static_projected=0.48345`, and `random=0.49603`.
- Interpretation: the gate is protocol-valid and weakly positive for PD-PPO
  (`0.14%` lower oracle loss than validation-selected static and `1.74%` lower
  than AoI), but the margin is too small and warm-up aborts are too high
  (`206`) for a single-seed manuscript claim.
- Action taken: launched four additional server tmux runs for seeds `42--45` under
  `reports/energy_account_split_protocol_gate_semimarkov/budget1p20_seed*/`,
  using GPUs `0--3`, to form the minimum `n=5` decision set. These runs should be
  aggregated before changing the active manuscript evidence boundary.
- Operational note: the tmux launch command for seeds `42--45` escaped `$?`
  incorrectly in the `exit_code` writeback. Use process completion plus expected
  metrics files and runner logs as the completion check for those four runs; do
  not rely solely on their `exit_code` file.

## 2026-05-26 Energy-Account n=5 Aggregate Result
- The semi-Markov split-protocol extension completed for seeds `41--45` and was
  synchronized locally. The collector reports `complete_seeds=5/5` and exit code
  `0`; aggregate tables are under
  `rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov/aggregate/`.
- Mean final-test oracle losses:
  `full_open_unconstrained=0.44717`,
  `validation_selected_static=0.45110`,
  `custom_ppo=0.46411`,
  `aoi=0.46952`,
  `round_robin=0.47882`,
  `random=0.48153`, and
  `feasible_static_projected=0.49839`.
- Pairwise seed counts for `custom_ppo`: `4/5` wins versus AoI, round-robin,
  random, and feasible static projection; only `2/5` wins versus
  validation-selected static.
- Mean deltas (`custom_ppo - comparator`, lower is better) are
  `-0.00541` versus AoI, `-0.01471` versus round-robin, `-0.01741` versus
  random, `-0.03428` versus feasible static projection, but `+0.01301` versus
  validation-selected static.
- Interpretation: the corrected energy-account learned-policy experiment supports
  only a bounded learned-policy claim against dynamic heuristics and the weak
  fixed-priority static reference. It does not support robust learned
  dynamic-over-validation-selected-static performance. The main manuscript should
  keep the stronger energy-account statement at the oracle/reference-policy
  opportunity level unless a revised controller is run.

## 2026-05-26 SOC Auxiliary + Abort-Control Gate Setup
- The next optimized-policy test keeps the corrected semi-Markov split protocol
  and changes only the PD-PPO training controls: `total_timesteps=200000`,
  `soc_aux_horizon=16`, `soc_aux_coef=0.1`, and `lambda_warmup_abort=0.16`.
- The parameter choice is grounded in prior diagnostics: the historical SOC
  auxiliary probe used `horizon=16`, `coef=0.1`, and `200k` timesteps; the abort
  control uses the previously tested `2x` warm-up-abort penalty rather than an
  uncalibrated large penalty.
- The gate output is
  `reports/energy_account_split_protocol_socaux_abort2x_200k/budget1p20_seed41`.
  It should be compared against the strict-protocol baseline seed 41 and not
  scaled unless it improves the validation-selected-static comparison and keeps
  warm-up aborts materially below the previous strict-protocol seed-41 count
  (`206`).

## 2026-05-26 Review of 05-26-02/03 Diagnosis Notes
- The four-factor diagnosis in `docs/05-26-02.md` and `docs/05-26-03.md` is
  directionally reasonable: reward/evaluation mismatch, forecast-warmup alignment,
  constraint handling, and PPO sample efficiency are all consistent with the
  observed weak dynamic-over-static evidence in the corrected energy-account run.
- Cost/risk ordering should be: zero/low-cost diagnostics first
  (`H` versus warm-up delay, oracle-feature ablation if implemented as a cheap
  retrain/smoke, training-curve extraction, lambda sensitivity), then the current
  SOC auxiliary plus abort-control gate, then larger algorithmic changes such as
  CPO/SAC/TD3.
- Current code implements SOC support as an auxiliary future-SOC prediction head
  in `src/v2/custom_ppo.py`, not as a separate Lagrangian/CMDP critic. Manuscript
  wording should call it a SOC auxiliary prediction loss or constraint-awareness
  auxiliary objective, not a proven CMDP solution.
- The notes' literature framing needs cleanup before it is imported into the
  paper. Ying et al. 2022 and Pendyala et al. 2024 are directionally relevant;
  the Fernandez-Bes date/citation should be corrected to the actual
  energy-harvesting sensor MDP source; the cited Kongarana 2026/CPO feature
  selection reference was not verified and should not be used without replacement
  by a primary, checkable source.

## 2026-05-26 SOC Auxiliary + Abort-Control Gate Result
- The protocol-controlled `seed=41`, `200k`, `soc_aux_horizon=16`,
  `soc_aux_coef=0.1`, `lambda_warmup_abort=0.16` gate completed on the server with
  `exit_code=0` and was synchronized locally under
  `rl_sensor_scheduling_framework/reports/energy_account_split_protocol_socaux_abort2x_200k/`.
- Same-run final-test oracle loss ranks:
  `full_open_unconstrained=0.46240`,
  `validation_selected_static=0.47617`,
  `custom_ppo=0.47950`,
  `round_robin=0.48098`,
  `aoi=0.48660`,
  `feasible_static_projected=0.48710`,
  `random=0.49873`.
- Gate decision: do not scale. PD-PPO still loses to the strict comparator
  `validation_selected_static` by `+0.00334` oracle loss (`+0.70%`), although it
  beats AoI, round-robin, random, and fixed-priority static in this single run.
- Mechanism: the modification reduces custom PPO warm-up aborts from the earlier
  strict-protocol seed-41 count `206` to `81`, but the loss/comparator condition
  fails. Laser selection remains anti-event (`event/non-event selected ratio
  0.52x`), while FC4 (`1.32x`) and snow particle counter (`2.57x`) are more
  event-biased.
- Cross-run absolute oracle-loss deltas versus the earlier baseline seed should
  be interpreted cautiously because the script retrains a fresh frozen TCN oracle
  for each run; full-open and static reference losses also shift. The gate result
  is therefore judged primarily by same-run ranking and behavior diagnostics.

## 2026-06-02 PD-PPO Clean Rewrite Asset Decisions
- The clean rewrite should not mean discarding all prior material. Three classes of
  legacy content are worth preserving because they serve the new evidence-bounded
  narrative:
  - the AWS platform rendering, as deployment motivation only;
  - the chronological split schematic, as protocol evidence;
  - the theoretical arguments explaining why prediction-driven scheduling is not
    reducible to instantaneous estimation loss and why static comparators can be
    strong under fixed budgets.
- The AWS rendering can appear in the manuscript with conservative semantics:
  it is a user-produced Blender rendering close to the real AWS platform used in
  the project. It should still not be treated as a field photograph, measured
  deployment, or empirical validation asset.
- A new manually authored TikZ PD-PPO framework figure is preferable to reusing the
  older dense architecture diagram. The new diagram separates the execution loop
  from the prediction-driven learning loop and is aligned with the corrected
  frozen-oracle/split-protocol evidence boundary.
- The framework figure should follow the Figure 3 style rather than the earlier
  dense architecture style: pale fills, thin grey outlines, no white arrow-label
  backgrounds, no decorative icons, and very short box text. The three-round
  redraw now satisfies these constraints at manuscript scale.
- The migrated theoretical material should remain explanatory rather than
  overclaiming an optimality theorem for PD-PPO. The current propositions support
  the paper's claim boundary: prediction-driven objectives can differ from
  state-estimation objectives; projection preserves feasibility; and static
  policies can be hard baselines under fixed power.

## 2026-06-02 CRST Engineering-Framing Revision
- The CRST abstract should foreground monitoring-system design rather than PPO
  mechanics. The revised abstract removes dense algorithm abbreviations, states the
  purpose, reports the key fixed-budget and energy-storage results, and ends with
  the regime-map conclusion.
- The preferred title framing is engineering-oriented:
  `Forecast-driven sensor scheduling for Antarctic blowing-snow monitoring under
  energy and warm-up constraints`.
- Keywords should prioritize the cold-region engineering objects first:
  blowing snow, Antarctic automatic weather station, sensor scheduling, energy
  constraint, forecasting, reinforcement learning.
- Limitations are still necessary but should not recur in every section. The new
  draft concentrates them in the abstract conclusion, the end of the Introduction,
  and the Discussion limitations paragraph, while Results and Conclusion emphasize
  the positive regime map.
- The simulator section needed more than a sanity-check table. The revised
  structure makes the generator reviewable by separating meteorological backbone,
  event generation, particle/flux construction, observation model, cost assignment,
  and sanity-check criteria, with a supplementary parameter table tied to the
  corrected split-protocol fixed-budget experiment.
- Official front-matter author information for this manuscript is:
  Yongzhe Li (`yongzheli@seu.edu.cn`) and Zhuyu Zhang (`220245154@seu.edu.cn`),
  School of Mechanical Engineering, Southeast University, Nanjing, China.

## 2026-06-02 06-02-02 Format Polish Decisions
- The format memo's critique is valid: the clean rewrite still contained
  AI-like register markers such as meta-statements of intent, repeated `regime
  map` wording, filler `therefore`, serial limitations, and revision-process
  labels. These have been removed from the active clean rewrite.
- British spelling has been adopted for manuscript consistency: `normalised`,
  `favour`, `behaviour`, `optimisation`, `artefact`, `summarise`, `characterise`.
  The CRST highlights were updated consistently and remain within length limits.
- C-5 was safe to execute: moving the static-comparator proposition and remark to
  the instantaneous-budget subsection improves logical flow and does not break
  labelled cross-references. The non-equivalence proof remains labelled and is now
  Proposition 2 in the rendered PDF.
- F-2 was handled conservatively without adding a new figure: the behaviour section
  now states that the seed-41 timeline is single-run evidence and that the aggregate
  laser event/non-event selection ratio is approximately neutral, so the figure
  should not be read as robust event-triggered laser control.
- F-3, F-4 and F-5 remain author/package tasks rather than text-polish tasks:
  future-work expansion, repository/DOI data availability, and final funding
  wording still require author decisions before submission.

## 2026-06-02 Code-Release Repository Decisions
- The correct submission package is a separate clean repository, not the current
  dirty research workspace. The main workspace contains active v1/new-mainline
  experiments, nested repositories, archives, and historical files that would
  confuse code reviewers and weaken the reproducibility package.
- The release repository should include compact aggregate evidence and exact
  scripts/configs, but exclude large generated truth CSVs, rollout tensors, and
  checkpoints. This keeps the package reviewable while preserving executable
  regeneration paths.
- Seed-level energy-account minimal files are worth including because they allow
  `scripts/62_energy_account_split_protocol_collect.py` to pass on the public
  package without bundling truth sequences or models.
- Public-path hygiene matters: aggregate CSV/JSON files originally recorded local
  absolute paths from the old workspace. These were sanitised to repository-relative
  paths, and the collectors were patched to avoid regenerating absolute paths.
- A GitHub release tag is sufficient as the immediate versioned code archive, but
  the final CRST data-availability statement should still receive a DOI from
  Zenodo or another recognised archive before submission.

## 2026-06-21 ESWA Specialist-Bottleneck Theory Finding
- The active ESWA manuscript can now frame SCENEBAL-2 as a broader
  forecast-relevant specialist-bottleneck problem rather than as a single
  calibrated scenario.
- The applied theory is a sufficient-condition argument: with a required
  backbone, fewer specialist slots than specialist channels, positive-weight
  regimes, incompatible regime-best specialists, and positive mismatch loss, a
  true fixed-static specialist subset has strictly higher static-normalised
  macro forecast loss than an ideal regime-aware dynamic policy.
- This strengthens the paper's naturalness without broadening the claim beyond
  the evidence. It supports structural dynamic headroom for SCENEBAL-2-like
  sensing systems; it does not prove PPO global optimality and does not justify a
  universal claim for arbitrary power-constrained sensor scheduling.
- Plan consistency debt was reduced at the same time: the root plan, subproject
  plan, and `.planning/.active_plan` now agree that the current target is
  *Expert Systems with Applications*, the canonical manuscript is
  `paper/main.tex`, and the current evidence block is SCENEBAL-2 `117--140`.

## 2026-06-21 ESWA Claim Boundary Finding
- The current supported claim is strong but not aggregation-invariant. The
  manuscript can claim SCENEBAL-2 `117--140` success under ordinary step gates,
  behaviour gates, true-static gates, and static-normalised event-regime macro
  scoring.
- It cannot claim that the learned policy dominates true fixed-static under raw
  unnormalised subtype macro aggregation: that diagnostic is `0/24`.
- The previous public GitHub release is historical and should not be cited as
  the SCENEBAL-2 archive. A new versioned archive or DOI remains a
  pre-submission requirement.

## 2026-06-21 New-Claim Manuscript Check
- The active paper is now the new SCENEBAL-2 claim rather than the old
  V3.1/SCENEBAL-1/metpair claim.
- Residual old-claim wording was found only in the highlights file and one
  problem-formulation sentence; both were corrected.
- The active claim remains bounded: static-normalised macro and ordinary step
  gates are supported, while raw unnormalised subtype macro is explicitly a
  sensitivity limitation.
