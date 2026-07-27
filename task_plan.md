# Task Plan: ESWA Manuscript and Strong-Claim Experiment Alignment

## Goal
Bring the `rl_sensor_scheduling_framework` paper and experiment package into
alignment with the current target journal, *Expert Systems with Applications*
(ESWA). Every claim must be grounded in completed experiment artifacts, and the
final scheduler claim must remain bounded to protocols that verify forecast
quality, true fixed-static replay, and genuinely state-dependent behaviour
rather than fixed sensors or a simple rotation.

## Current Phase
ESWA manuscript completeness, metric-boundary consistency, and
submission-package preparation after the SCENEBAL-2 24-seed strong-claim
experiment.

## Supersession Notice
- Historical CRST rewrite notes in this root plan are retained as provenance only.
  They no longer define the target journal or current manuscript framing.
- The active target journal is *Expert Systems with Applications* (ESWA).
- The authoritative active subproject plan is
  `rl_sensor_scheduling_framework/.planning/2026-06-10-eswa-terminology-rewrite/`.
- The prior strong-claim exploration is complete for the stated objective:
  SCENEBAL-2 seeds `117--140` passed the operational, strict true fixed-static,
  explicit replay, macro, and behaviour gates in `24/24` seeds. Current work is
  manuscript consistency cleanup and submission packaging.
- The current paper claim is not aggregation-invariant. It is tied to ordinary
  step gates and static-normalised event-regime macro scoring; the learned-policy
  raw unnormalised subtype-macro diagnostic is `0/24` and must be treated as a
  limitation.

## Operating Invariant
- Remote GPU execution has exactly one valid local entry point: SSH alias
  `remote-gpu`. Do not use hardcoded host addresses, historical internal
  network paths, old tunnel/client scripts, or password-based sync helpers.
- Any future strong-claim search must keep PPO as the final learned scheduler.
  It may modify scene generation, teacher/oracle construction, PPO inputs,
  auxiliary heads, memory/policy structure, reward shaping, replay/evaluation
  protocol, and moderate simulated sensor/noise calibration, but it must not
  replace PPO with another final scheduling algorithm.
- For future experiment work, each modification direction has a maximum of `10`
  bounded work units without
  effective improvement. If a direction fails or becomes likely failed earlier,
  pivot to another layer of change instead of repeating conservative retries.
  A work unit means a seed wave, locked pilot, diagnostic branch, or equivalent
  bounded experiment with aggregate evidence and a written keep/pivot decision.
- Preserve the current microclimate sensor setup as the physical baseline.
  Moderate sensor-config variants are allowed only when they remain explainable
  as simulated variants of the existing setup, not as an arbitrary replacement
  sensing system.

## Historical Rewrite Track (started 2026-05-25, superseded by ESWA target)

### R1: Freeze Baseline and Establish Constraints
- [x] Set a persistent execution goal for full rewrite and iterative review.
- [x] Load the applicable planning, academic-writing, PDF, and experiment-server skills.
- [x] Archive the latest pre-rewrite paper state.
- [x] Retrieved the then-current CRST Guide for Authors requirements; this is now
  historical because the target journal has changed to ESWA.
- [ ] Receive independent structure and evidence audit memoranda from subagents.
- **Status:** in progress

### R2: Evidence Ledger and Manuscript Strategy
- [ ] Inventory final local and, where necessary, server-side result artifacts.
- [x] Close the static-comparator audit: V3.1 preserves oracle-prior candidate tables
  but no saved `oracle_static_projected` rollout; a reconstruction audit and
  independent-truth smoke replay establish that posthoc replay is diagnostic only.
- [x] Repair-decision gate for the protocol blocker: existing S2 prior/evaluation windows overlap
  (`21/30` runs) and the current manuscript falsely describes chronological,
  disjoint, non-overlapping testing. Independent-truth replay of an old checkpoint
  is inadequate; corrected retraining is required before using S2 as submission evidence.
- [x] Implement the retraining protocol with these non-negotiable controls:
  oracle pretraining, PPO optimisation, validation/static-candidate selection, and
  final testing operate on declared partitions; actor normalisation is fitted on
  training-authorised data only; final-test windows are non-overlapping and never
  consulted for policy or comparator selection.
- [x] Run a server-side split-protocol single-seed gate; scale only if the frozen
  oracle retains a credible full-observation ordering on final test and artifacts
  contain the declared split manifest.
- [x] Complete the launched split-protocol `3`-budget by `10`-seed grid and replace
  fixed-budget manuscript values only after final-test aggregation/review.
- [x] Audit energy-account curriculum/full-distribution independence: stored
  learned-policy results fail because training, oracle and evaluation windows are
  not separated; classify them as mechanism diagnostics only.
- [ ] If an energy-account learned-policy claim remains in scope, implement and run
  its own split-protocol gate before drafting comparative results.
- [x] Separate primary evidence, supporting diagnostics, and excluded/failed probes.
- [x] Define a single defensible research question and contribution set for CRST scope.
- [x] Produce a new section-by-section outline and compliance checklist.
- **Status:** pending

#### Energy-Account Split-Protocol Repair Status (2026-05-26)
- [x] Persist the audit establishing that existing curriculum outputs are
  same-protocol diagnostics only (`scripts/60_energy_account_protocol_audit.py`).
- [x] Implement and locally smoke-test a four-part energy-account runner
  (`scripts/61_energy_account_split_protocol_run.py`).
- [x] Abort and quarantine the initial server gate after pre-outcome manifest
  inspection identified a legacy `clustered` generator temporal-coverage bias:
  the final-test segment had mean selected event rate only about `0.276`.
- [x] Verify a corrected `semi_markov` split truth: partition-wide event rates are
  stable near `0.30`, and independent final conditional windows average `0.521`.
- [x] Complete the replacement single-seed server gate in `tmux` session
  `energy_split_semimarkov_gate_20260526`; do not interpret the aborted clustered run.
- [x] Complete and aggregate the `n=5` semi-Markov extension (`seeds 41--45`).
  Result: PD-PPO is better than AoI/round-robin/random in `4/5` seeds but is
  worse than validation-selected static on average and in `3/5` seeds, so no
  learned dynamic-over-static claim is authorised.
- [x] Run a protocol-controlled SOC-auxiliary plus warm-up-abort-control gate:
  `seed=41`, `200k` timesteps, `soc_aux_horizon=16`, `soc_aux_coef=0.1`,
  `lambda_warmup_abort=0.16`, same semi-Markov split protocol. Scale only if it
  improves the static comparison and controls aborts without breaking oracle
  sanity. Result: the gate controlled aborts (`206 -> 81`) but failed the static
  comparison (`custom_ppo=0.47950` vs `validation_selected_static=0.47617`) and
  did not repair laser event gating (`0.52x` event/non-event selected ratio), so
  it should not be scaled.

### R3: New Manuscript Construction
- [x] Create the new manuscript source from the approved outline rather than editing prose in place.
- [x] Create a first clean rewrite of title, abstract, highlights, introduction,
  formulation, framework/protocol, setup, results, discussion, conclusion and
  declarations in a new TeX entrypoint.
- [x] Replace the inherited fixed-budget result table, event-conditioned table,
  physical-unit table and fixed-budget diagnostic figures with artifacts generated
  from the corrected chronological final-test grid.
- [x] Remove the audited non-independent energy-account learned-policy comparison
  from current main-text claims; retain only the explicitly labelled mechanism
  diagnostic pending a valid replacement result.
- [x] Update the title, abstract, central contribution/result prose, discussion and
  conclusion to the corrected fixed-budget evidence boundary.
- [x] Add and validate CRST highlights (`5` bullets, maximum `79` characters).
- [x] Revise the CRST-facing title, abstract, keywords, and highlights toward an
  engineering-monitoring framing. Current abstract is `206` words; highlights are
  `68/67/75/67/67` characters.
- [x] Add a structured simulator-construction subsection and supplementary
  simulator/protocol parameter table for the corrected fixed-budget experiment.
- [x] Apply the `docs/06-02-02-ppo-format.md` writing/format patch set to the
  clean rewrite source: de-AI register edits, British spelling consistency,
  proposition relocation, clearer table captions, and a full compile check.
- [ ] Reuse only figures/tables that pass evidence and journal-artwork review.
  - Partial: migrated corrected-result tables, AWS rendering, chronological split
    schematic, generator-statistics figure, power-error figure, behavior-diagnostic
    figure, and a newly redrawn non-AI TikZ PD-PPO framework diagram into
    `paper/pdppo_crst_rewrite.tex`.
  - AWS rendering provenance: user confirmed it is a self-produced Blender render
    close to the real AWS used in the project. Record this provenance in the
    submission package rather than treating it as an AI-generated or stock figure.
  - Framework figure audit: completed three manual TikZ review rounds to match the
    Figure 3 low-saturation style, remove white label backgrounds, remove the
    crowded standalone prior block, and eliminate visible text/arrow overlaps.
- [ ] Finalize data availability, CRediT, funding/competing interests, and AI-use declarations.
- **Status:** in progress

### R4: Verification and Review Cycles
- [ ] Compile the new LaTeX manuscript and visually inspect rendered PDF pages.
  - Interim check: `pdppo_crst_rewrite.pdf` compiles to `31` pages with no
    undefined references/citations; migrated AWS and PD-PPO framework pages were
    visually inspected. Remaining issues are presentation polish only: two tiny
    overfull boxes, one underfull box, and the existing empty-pages BibTeX warning
    for `Pendyala2024`.
  - Current post-framework-redraw build compiles to `30` pages with the same
    non-blocking warnings and no undefined references/citations.
- [ ] Run claim-to-artifact, citations, word-count, highlight-length, and format checks.
  - Citation count interim check: clean rewrite now produces `28` bibliography
    entries after reusing archived vetted references. No undefined citations remain;
    four archived conference entries still lack page fields and produce non-blocking
    BibTeX `empty pages` warnings.
- [ ] Conduct at least two independent subagent review rounds on the new draft.
- [ ] Implement revisions and re-review material findings until resolved or explicitly documented.
- [x] Complete an interim post-evidence-repair build and mechanical audit; see
  `rl_sensor_scheduling_framework/docs/05-26-crst-draft-verification.md`.
  Repeat after any accepted energy-account learned-policy integration.
- **Status:** pending

### R5: Submission Package
- [x] Produce a provisional submission checklist with the active-source dependency
  boundary and explicit exclusion of superseded/diagnostic assets:
  `rl_sensor_scheduling_framework/docs/05-26-crst-submission-checklist.md`.
- [ ] Produce final manuscript PDF/source package after pending evidence,
  author declarations and versioned deposit are resolved.
- [ ] Record remaining author-only decisions (authorship approval, repositories, declarations).
- [ ] Mark persistent goal complete only after all required deliverables and reviews are complete.
- **Status:** pending

## Historical Experiment Track
The phases below record the completed physical-event/energy-account investigation
that supplies candidate evidence for the rewrite; they are not the active manuscript plan.

## Takeover Note (2026-05-25)
- The root planning files are authoritative for the full CRST rewrite.
- `rl_sensor_scheduling_framework/task_plan.md` records the completed earlier
  algorithm-first closure and should be treated as historical context except for its
  newly noted handoff audit.

## Phases

### Phase 1: Implementation Scaffold
- [x] Read `rl_sensor_scheduling_framework/docs/05-22-chore.md`.
- [x] Create a loader-compatible `physical_event_value_v2` sensor configuration.
- [x] Add wrapper passthroughs for target weights, required sensors, coverage disabling, and max-active controls.
- [x] Validate CLI dry-runs locally before touching the server.
- **Status:** complete

### Phase 2: No-Retrain Diagnostics
- [x] Build or reuse oracle/static-subset diagnostics for `physical_event_value_v2`.
- [x] Quantify feasible subsets at `B = 1.00/1.10/1.20` with required `met_station_core` and disabled coverage groups.
- [x] Estimate event-conditioned laser/FC4 lift before launching PPO training.
- [x] Gate pilot training on non-trivial event lift and feasible high-value subsets.
- **Status:** failed gate; do not launch PPO from `physical_event_value_v2`
- **Result:** formal TCN oracle-lift diagnostic at `B=1.20` found negative event lift for both high-value channels:
  - `laser_event_lift = -0.0342`
  - `fc4_event_lift = -0.0221`
  - best event subset was `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`, not a laser/FC4 subset.

### Phase 2b: Event-Sensitive Observation Repair
- [x] Document why `physical_event_value_v2` fails before any training: laser duplicates the snow particle counter while being costlier/slower, and FC4 is not rewarded enough to beat cheap proxy subsets.
- [x] Add a minimally invasive event-sensitive observation abstraction, preferably event-conditioned noise/saturation for the low-cost snow particle counter rather than arbitrary reward hacking.
- [x] Create `physical_event_value_v3` with defensible sensor roles:
  - low-cost snow counter as coarse/noisy event proxy;
  - laser disdrometer as high-fidelity particle measurement during blowing snow;
  - FC4 flux as direct but sparse/low-rate flux measurement.
- [x] Add an explicit event-microstructure truth-generator option so snow/flux channels can carry information not fully explained by smooth wind/surface proxies.
- [x] Rerun feasible-subset and oracle-lift diagnostics before any PPO training.
- **Status:** formal gate failed; do not launch PPO from this version
- **Current server run:** `physical_event_v3_micro_tcn_20260522`, output `reports/physical_event_v3_microstructure_tcn_b120_seed41`.
- **Result:** event microstructure narrowed the gap but did not invert the ranking:
  - best event subset: `met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`;
  - best laser event subset: `met_station_core|radiometer_basic|laser_disdrometer`;
  - `laser_event_lift = -0.0047`;
  - `fc4_event_lift = -0.0137`;
  - hand-written dynamic schedules were worse than the best fixed subset.

### Phase 2c: Event Saturation Repair
- [x] Add a defensible event-availability/saturation abstraction for low-cost particle sensing.
- [x] Create a separate config, not a silent overwrite, where `snow_particle_counter` can become partially unavailable during dense blowing-snow events.
- [x] Rerun local smoke and then formal server TCN gate.
- [x] Only if the formal gate shows positive event-conditioned value, launch a single-seed PPO pilot.
- **Status:** complete as diagnosis; fixed per-step PPO pilot remains blocked
- **Current server run:** `physical_event_v4_saturation_tcn_20260522`, output `reports/physical_event_v4_saturation_tcn_b120_seed41`.
- **Result:** formal `B=1.20` TCN gate gives positive laser event value, but the best overall policy is a fixed laser subset, not a dynamic schedule:
  - `best_overall = met_station_core|surface_temp_ir|laser_disdrometer`;
  - `laser_event_lift = +0.0187`;
  - dynamic schedules remain worse than the best fixed subset.
- **Decision:** do not launch fixed-budget PPO yet. The scenario now supports laser value, but fixed per-step constraints still encourage static activation.

### Phase 3: Single-Seed Pilot
- [ ] Check server state and ensure no old microclimate experiments are running.
- [ ] Launch one tmux pilot for `B=1.20`, seed 41, physical-event config, disabled coverage groups, and snow-focused weights.
- [ ] Sync outputs locally after completion.
- [ ] Run behavior diagnostics and inspect timelines before scaling.
- **Status:** blocked until Phase 2b produces positive oracle lift

### Phase 4: Paper Claim Repair
- [x] Downgrade unsupported event-conditioned warm-up claims unless pilot evidence supports them.
- [x] State that the current mainline uses a fixed per-step budget, not a time-varying energy-harvesting model.
- [x] Replace overbroad multi-year/multi-season wording with "calibrated to multi-year Antarctic AWS statistics".
- [x] Clarify that power values are normalized scheduling costs, not a full physical watt/SOC model.
- **Status:** complete

### Phase 5: Energy-Account Architecture Gate
- [x] Design a minimal energy-account/SOC extension with generated harvest, battery capacity, and SOC state input.
- [x] Implement a default-off energy-account diagnostic path.
- [x] Find a local smoke configuration where dynamic event-laser scheduling beats static fixed subsets without SOC violations.
- [x] Confirm the energy-account gate with a formal server TCN diagnostic.
- [ ] Decide whether energy-account results are required for the current paper or should remain future work.
- [ ] If promoted, implement separately from the fixed-budget physical-event pilot to avoid mixing claims.
- **Status:** storm-window oracle gate passed; PPO curriculum pilot produced first learned-policy win over AoI but still has behavioral caveats
- **Latest formal result:** `reports/physical_event_v4_energy_lead_h105_tcn_b120_seed41`
  - `laser_event_lift = +0.0064`, so event laser value is positive.
  - `best_overall = met_station_core|radiometer_basic|surface_temp_ir|snow_particle_counter`, so dynamic scheduling is not yet the best overall policy.
  - Lead dynamic schedules improve event loss but are still worse overall and can hit the energy guard in longer formal rollouts.
  - Decision: do not launch PPO yet.
- **Calibration correction:** harvest should be derived from formal truth event/lead-trigger statistics, and capacity should be derived from burst drawdown rather than average cost alone.
- **Formal calibrated results so far:**
  - `harvest=0.62`, `capacity=300`: static snow-counter core remained best overall, despite positive laser event lift.
  - `harvest=0.94`, `capacity=300`: static laser became feasible and best overall, so this is above the dynamic-advantage window.
- **Current calibration probe:** added no-lead `snow_core -> event laser` schedules to test whether keeping the static snow core during non-event periods avoids the non-event penalty of lead-trigger schedules. Formal server gate is running in `physical_event_v4_energy_cal_h062_cap300_newsched_tcn_20260522`.
- **Next formal probe:** `harvest=0.92`, `capacity=120`, `reserve=20` is running in `physical_event_v4_energy_cal_h092_cap120_newsched_tcn_20260522`; this is calibrated from no-lead snow-core/event-laser drawdown and is the first parameter pair intended to allow dynamic bursts while clipping static laser.
- **Storm-window probe:** because the full-distribution event fraction is too low for overall dynamic superiority, `physical_event_v4_energy_cal_h092_cap120_storm_tcn_20260522` evaluates the same calibrated energy account on the highest-event 1024-step windows in the formal truth.
- **Storm-window result:** `capacity=180` passes cleanly. Best overall is `dynamic:snow_core__event_laser_fc4` with no guard drops or warmup aborts, while static laser remains clipped. This supports a storm-window dynamic-advantage diagnostic, not a full-distribution superiority claim.
- **PPO pilot:** single-seed storm-window PPO completed. It partially passed: `custom_ppo` beats the static snow-core oracle reference but not AoI, and its event/non-event laser selection ratio is `1.84x` rather than the target `>3x`. Do not promote trained-PPO storm-window claims yet.
- **PPO curriculum pilot:** added `--train-start-indices` and trained on the same storm windows used for evaluation. This is the first learned-policy result that beats AoI:
  - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_seed41`;
  - `custom_ppo` oracle loss `0.4106` vs AoI `0.4130`, round-robin `0.4720`, random `0.4660`, feasible static projected `0.5032`;
  - warmup aborts: PPO `5`, AoI `10`, round-robin `768`, random `1956`;
  - caveat: PPO's `laser_disdrometer` selected rate is higher in non-event than event windows (`0.744` vs `0.519`), while event-conditioned increases are clearer for `radiometer_basic`, `snow_particle_counter`, and `fc4_flux`.
  - interpretation: curriculum improves learnability and supports a narrow "learned adaptive policy can beat heuristic baselines in storm windows" result, but not the cleaner claim that PPO has learned oracle-like event-triggered laser activation.
- **Three-seed curriculum check:** seeds `41/42/43` all beat AoI on storm-window oracle loss:
  - PPO: `0.4106 / 0.4206 / 0.4096`;
  - AoI: `0.4130 / 0.4356 / 0.4096`;
  - mean PPO `0.4136 ± 0.0061`, mean AoI `0.4194 ± 0.0141`;
  - PPO abort mean `19.0`, worse than AoI `10.0` but far better than round-robin `768` and random `2094`;
  - mean sensor bias shows `radiometer_basic` has the clearest event preference (`1.63x`), while `laser_disdrometer` is nearly event-neutral (`1.03x`).
  - decision: this is enough for a narrow curriculum-PPO storm-window result, but the manuscript must not claim robust learned event-laser gating.
- **Mechanism diagnosis:** completed on existing rollouts and documented in `docs/05-23-curriculum-ppo-mechanism-diagnosis.md`.
  - PPO laser aborts are mostly non-event (`10` event vs `31` non-event across three seeds), so event-time SOC abort is not the primary explanation for low laser event bias.
  - PPO beats AoI through better non-event loss (`0.5256` vs AoI `0.5431`) while AoI remains slightly better on event loss (`0.3243` vs PPO `0.3274`).
  - Next optimization should target event-step objective emphasis before SOC-only penalty tuning.
- **Current mechanism probe:** implemented default-off event-step reward multiplier and launched one seed with `event_reward_multiplier=1.5`.
  - output `reports/physical_event_v4_energy_ppo_h092_cap180_stormcurr_evt15_seed41`;
  - success criterion: event loss and/or laser event ratio improves without losing overall competitiveness against AoI.
- **Event multiplier result:** passed performance, failed laser-gating mechanism.
  - PPO oracle loss improved to `0.4088` vs AoI `0.4118`;
  - event loss improved to `0.3247` from baseline seed-41 `0.3307`;
  - aborts stayed acceptable at `7`;
  - laser event/non-event selected ratio stayed near neutral (`0.740 / 0.768 = 0.96x`).
  - decision: generic event-loss weighting is useful for event performance, but does not establish event-triggered laser gating.
- **Full-distribution generalization check:** completed without retraining.
  - Evaluated trained curriculum seeds `41/42/43` on random windows with mean event rate about `0.296`.
  - PPO beats AoI in all three seeds and on average (`0.3130` vs `0.3160`).
  - PPO beats static projected on average (`0.3130` vs `0.3309`), but not in every seed; seed 42 static projected is better.
  - mechanism remains high-duty/storm-context laser plus event-biased radiometer, not event-triggered laser gating.
- **Five-seed curriculum extension:** completed.
  - Storm-window n=5:
    - PPO `0.4153 +/- 0.0051`;
    - AoI `0.4176 +/- 0.0105`;
    - feasible static projected `0.4742 +/- 0.0236`;
    - round-robin `0.4451 +/- 0.0167`;
    - random `0.4565 +/- 0.0140`.
  - PPO wins:
    - vs AoI: `3/5`;
    - vs feasible static projected: `5/5`;
    - vs round-robin: `5/5`;
    - vs random: `5/5`.
  - Full-distribution n=5:
    - PPO `0.3155 +/- 0.0133`;
    - AoI `0.3168 +/- 0.0135`;
    - feasible static projected `0.3318 +/- 0.0062`;
    - round-robin `0.3375 +/- 0.0195`;
    - random `0.3431 +/- 0.0188`.
  - PPO wins in full distribution:
    - vs AoI: `4/5`;
    - vs feasible static projected: `4/5`;
    - vs round-robin/random: `5/5`.
  - Decision: do not claim robust AoI dominance. Claim consistent storm-window superiority over static projected and simple heuristic baselines, plus small average full-distribution generalization advantage.
- **SOC soft-penalty probe:** completed for seed 41.
  - PPO storm oracle loss improved from `0.4106` to `0.4069`;
  - event loss improved from `0.3307` to `0.3234`;
  - power dropped from `1.0128` to `0.9960`;
  - aborts increased from `5` to `9`, and energy guard dropped `31` selections.
  - Decision: do not merge SOC soft penalty as the default yet. It needs at least a 3-seed probe because its benefit is event performance, not clearly abort reduction.
- **300k training-length probe:** completed for seed 41.
  - Storm-window:
    - 100k PPO `0.4106`, AoI `0.4130`;
    - 300k PPO `0.4053`, AoI `0.4125`;
    - event loss improves to `0.3190`;
    - laser event/non-event ratio improves to `1.52x`;
    - aborts increase to `66`.
  - Full-distribution:
    - PPO `0.3122`, AoI `0.3118`;
    - PPO remains better than static projected, round-robin, and random;
    - aborts remain high (`75`).
  - Decision: longer training helps storm-window event adaptation but does not deliver robust full-distribution dominance over AoI. Do not scale 300k to five seeds until abort/SOC management or objective conditioning is improved.
- **Event-gated actor probe:** completed for seed 41 at 200k.
  - Storm-window:
    - PPO `0.4105`, AoI `0.4128`;
    - aborts `38`;
    - laser event/non-event ratio `0.78x`;
    - event/non-event loss `0.3249 / 0.5219`.
  - Full-distribution:
    - PPO `0.3128`, AoI `0.3117`;
    - static projected `0.3350`;
    - aborts `35`;
    - laser ratio `1.07x`.
  - Decision: failed the planned success criteria. Do not scale event-gated actor to n=5. Next candidate should combine stronger abort/SOC shaping with policy conditioning, or revise the claim away from full AoI dominance.
- **Next Phase: SOC auxiliary critic probe:** in progress.
  - Goal: test whether explicit future-SOC prediction improves credit assignment for long-horizon energy reserve behavior without redesigning the whole RL hierarchy.
  - Implementation plan:
    - add default-off `soc_aux_horizon` and `soc_aux_coef` to `CustomPPOConfig`;
    - collect per-step SOC ratio in rollouts;
    - add an auxiliary critic-side head that predicts future SOC ratios over `N` steps;
    - add masked MSE auxiliary loss to PPO update;
    - expose CLI flags in `25_v2_train_custom_ppo.py`;
    - run seed 41 probe before any multi-seed expansion.
  - Probe setting:
    - start with `soc_aux_horizon=16`, `soc_aux_coef=0.1`;
    - train `200k` steps on the same storm-window curriculum;
    - keep event-gated actor disabled to isolate SOC auxiliary effect.
  - Success criteria:
    - storm-window PPO remains competitive with AoI;
    - storm aborts lower than the 300k baseline and preferably below `20`;
    - full-distribution PPO beats AoI on seed 41;
    - no large degradation versus the existing n=5 main result.

### Phase 6: Results Convergence and Paper Alignment
- [x] Stop scaling new PPO architecture probes after SOC auxiliary produced only a partial pass.
- [x] Generate reproducible convergence assets from local result files.
- [x] Sync the latest SOC auxiliary server outputs into local reports.
- [x] Create the energy-account curriculum n=5 paper table.
- [x] Create a claim-boundary memo for the locked result.
- [x] Update `paper/sections/06_experiments.tex` to distinguish oracle gate, learned curriculum result, and AoI boundary.
- [x] Update `paper/sections/07_discussion.tex` with energy-account and latent-event limitations.
- [x] Compile the paper after edits.
- **Status:** complete as first pass
- **Generated assets:**
  - `rl_sensor_scheduling_framework/scripts/52_energy_account_convergence_assets.py`
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_main_long.csv`
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_main_summary.csv`
  - `rl_sensor_scheduling_framework/reports/energy_account_convergence_20260524/energy_account_probe_summary.csv`
  - `rl_sensor_scheduling_framework/docs/05-24-results-convergence.md`
  - `rl_sensor_scheduling_framework/paper/tables/energy_account_curriculum_results.tex`

### Phase 7: Claim Audit and Final Paper Alignment
- [x] Scan paper for PPO/AoI/static/dynamic scheduling claims after Phase 6 edits.
- [x] Correct feasible static projection wording so it is not described as solving an exhaustive static optimum.
- [x] Add regeneration-safe caption text to the energy-account curriculum table.
- [x] Create a claim-audit memo separating fixed-budget V3.1 evidence, energy-account oracle diagnostics, learned energy-account evidence, and unsupported claims.
- [x] Regenerate convergence assets after script edits.
- [x] Compile the paper and record verification status.
- **Status:** complete

### Phase 8: Full-Text Consistency Pass
- [x] Scan paper for remaining over-strong fixed-budget/dynamic/AoI mechanism wording.
- [x] Soften fixed-budget budget-sensitivity interpretation so it does not imply unproven event-sensor activation as the cause.
- [x] Soften V2-to-V3.1 diagnostic wording so it does not claim dynamic adaptation was fully corrected.
- [x] Soften EventAwareCritic motivation from guaranteed event-sensor payoff to possible event-regime value.
- [x] Soften conclusion wording from "best learned adaptive scheduler" to "best learned scheduler evaluated" in the final fixed-budget sweep.
- [x] Check main figure/table captions for static/AoI/dynamic claim consistency.
- [x] Soften the main-results table caption from "best learned adaptive policy" to "proposed learned policy".
- [x] Synchronize the legacy unused main-results table caption to avoid future asset drift.
- [x] Recompile paper and record verification.
- **Status:** complete

### Phase 9: Figure 4 Generator-Validation Redraw
- [x] Diagnose the old figure as an outdated V2-style diagnostic asset with an internal Figure title and weak validation semantics.
- [x] Add a reproducible V3.1 redraw script based on `reports/v3_supplement_assets`.
- [x] Replace `paper/figures/figure3_synthetic_statistics.png` and `.svg`.
- [x] Update the paper caption to describe the new six-panel figure.
- [x] Compile the paper and record verification.
- **Status:** complete

### Phase 10: Figure 2 State-Machine Redraw
- [x] Redraw the warm-up-aware sensor state machine to remove white label backgrounds and text blocking.
- [x] Simplify transition labels and move detailed equations into state boxes and lower rule cards.
- [x] Update the Figure 2 caption to refer to lower panels.
- [x] Compile the paper and visually inspect the rendered PDF page.
- **Status:** complete

### Phase 11: Figure 6 PD-PPO Architecture Redraw
- [x] Diagnose the old architecture figure as an outdated dense TikZ layout with white label backgrounds and overlapping micro-cards.
- [x] Redraw `pdppo_architecture_tikz.tex` with a cleaner runtime/core/training-signal structure.
- [x] Update the Figure 6 caption to match the revised framework diagram and current frozen-oracle wording.
- [x] Compile the paper and visually inspect the rendered PDF page.
- **Status:** complete

### Phase 12: Table 3 Main-Result Regeneration
- [x] Identify Table 3 as `paper/tables/main_results_v31.tex`.
- [x] Verify its values against `reports/v31_s2_main/v31_s2_main_stats.csv`.
- [x] Add a reproducible regeneration script for Table 3.
- [x] Regenerate the LaTeX table and compile the paper.
- [x] Render and visually inspect the Table 3 page.
- **Status:** complete

### Phase 13: 2026-05-25 Narrative Reconstruction
- [x] Read `rl_sensor_scheduling_framework/docs/05-25-1-paper.md`.
- [x] Compare the requested rewrite package against the current paper state.
- [x] Replace/align the abstract with the regime-dependent narrative while preserving locked n=10 fixed-budget provenance.
- [x] Rewrite the introduction contribution list around forecast-driven scheduling, fixed-budget diagnostics, energy-account regime separation, and bounded curriculum results.
- [x] Add fixed-budget simplification and event-flag deployment caveats in the problem/method sections; fix event-context notation drift (`e_t` vs `z_t`) where needed.
- [x] Rework experiments into fixed-budget benchmark, behavior diagnostic, ablation note, and energy-account/curriculum result with correct AoI positioning.
- [x] Rewrite discussion/limitations around regime-dependent value, SOC credit assignment, AoI boundary, and simulator/deployment assumptions.
- [x] Redraw Figure 8 as a fixed-budget behaviour diagnostic and update its caption.
- [x] Run forbidden-claim scan, bibliography key check, Figure 8 PDF-page visual check, and LaTeX compilation.
- **Status:** complete as first-pass narrative reconstruction
- **Important correction:** `05-25-1-paper.md` is directionally authoritative for narrative, but several details must be reconciled with locked assets:
  - fixed-budget Table 3 is `n=10`, not five seeds;
  - the paper entrypoint is `paper/main.tex`, not `paper.tex`;
  - energy-account table already exists as `paper/tables/energy_account_curriculum_results.tex`;
  - avoid promoting single-seed probes or clean laser-gating claims.

### Phase 14: 2026-05-25 Length Reduction and Review-Repair Loop
- [x] Archive the current paper before any shortening edits.
- [x] Read `rl_sensor_scheduling_framework/docs/05-25-2-paper.md`.
- [x] Attempt to create a new tool goal; record the tool limitation when the completed prior goal blocks a second goal.
- [x] Apply priority cuts in order: appendices, related work, problem formulation, simulation, methodology, experiments, discussion, introduction.
- [x] Preserve all load-bearing assets explicitly listed in `05-25-2-paper.md`: Table 3, Table 5, Table 6, Table 10, Figure 2, Figure 8, Proposition 1 sketch, Proposition 2, Fernandez-Bes anchor, laser-ratio diagnosis, and event-flag caveat.
- [x] Use subagents for independent review passes after major edits.
- [x] Recompile after each major edit batch.
- [x] Track page count and word count after each review-repair pass.
- [x] Continue review and repair until the remaining review verdict is close to minor revision or minor revision.
- **Status:** complete
- **Archive:**
  - `rl_sensor_scheduling_framework/paper_archives/paper_pre_0525_length_reduction_20260525_042051.tar.gz`
- **Objective:** reduce the 76-page manuscript toward the 40--50 page target while keeping the corrected regime-dependent narrative and locked experimental evidence intact.
- **Final current metrics:** 45 pages; texcount total 7,145 with 5,970 text words and 15 floats/tables/figures.
- **Review status:** first-round subagents judged the pre-edit version as major revision because of length and claim dilution. After the length/claim repairs, second-round subagent calls failed due account usage limit; local review against the same checklist finds no remaining major-blocker item and rates the current manuscript as close to minor revision, with residual risk concentrated in simulator external validity and eventual field validation.

### Phase 15: 2026-05-25 Typography and Format Polish
- [x] Inspect embedded PDF fonts and render representative manuscript pages.
- [x] Identify formatting issues in body type, heading hierarchy, and long captions.
- [ ] Introduce a publication-oriented but Elsevier-compatible font/caption/heading system while retaining review mode.
- [ ] Shorten visibly oversized captions only where their explanatory load duplicates main text.
- [ ] Recompile and visually inspect title, figure, and result-table pages.
- [ ] Record final typography changes, page count, and remaining non-blocking warnings.
- **Status:** in progress
- **Initial findings:** current PDF embeds Computer Modern fonts; subsection headings are low-contrast italic text; full-length Figure 2 and behavior-diagnostic captions render at near-body scale. Times-compatible `newtxtext/newtxmath` and `titlesec` are installed locally.

### Phase 16: 2026-06-02 CRST Code-Release Repository
- [x] Create a separate clean release repository instead of publishing the dirty
  main workspace or nested paper repository wholesale.
- [x] Include the current PD-PPO/V3.1 reproducibility surface: `src/v2`,
  `src/data_sources`, selected V3.1 scripts, sensor configs, compact aggregate
  result tables, manuscript source, highlights, figures, citation metadata, and
  data-availability notes.
- [x] Exclude large or noisy artefacts: generated truth CSVs, raw rollout NPZs,
  trained checkpoints, tmux logs, pycache, and LaTeX build outputs.
- [x] Add `README.md`, `REPRODUCIBILITY.md`, `DATA_AVAILABILITY.md`,
  `CITATION.cff`, `LICENSE`, `.gitignore`, and structured result-manifest docs.
- [x] Validate with focused pytest, py-compile, table regeneration, energy
  collector regeneration, Figure 6 regeneration, and LaTeX compilation.
- [x] Create and push the public GitHub repository and release tag.
- **Status:** complete
- **Repository:** `https://github.com/JekYUlll/forecast-driven-sensor-scheduling`
- **Release:** `https://github.com/JekYUlll/forecast-driven-sensor-scheduling/releases/tag/v0.1.0`
- **Remaining submission action:** connect the GitHub release to Zenodo or an
  equivalent archive service and insert the DOI before final journal submission.

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| Stop scaling the old complex-cost branch | It either collapses exactly to static or becomes dynamic but worse and non-event-conditioned. |
| Use `physical_event_value_v2` as the next runnable fixed-budget scenario | It keeps the high-value laser feasible under hard per-step masking while creating real competition with low-cost sensors. |
| Disable coverage groups for the physical-event pilot | The old coverage groups impose a static core and block the intended tight-budget mechanism. |
| Keep energy-account modelling gated | It is physically important, but mixing SOC changes into the first fixed-budget pilot would make causal interpretation muddy. |
| Prioritize claim repair even if the pilot succeeds | Current paper text overstates event-conditioned warm-up behavior relative to observed V3.1 rollouts. |
| Do not train PPO on `physical_event_value_v2` | No-retrain TCN diagnostics show laser/FC4 have negative event-conditioned lift, so PPO would be optimizing a scenario where the desired high-value scheduling behavior is not actually useful. |
| Repair observation semantics before policy learning | The failure is upstream of PPO: the oracle prefers cheap static proxy subsets even during events. |
| Do not train fixed-budget PPO immediately after v4 | v4 makes laser useful, but the fixed per-step budget prefers static laser activation at `B=1.20`; this would not demonstrate event-conditioned scheduling. |
| Move next to energy-account/SOC gate | A cumulative energy budget can make always-on laser infeasible while preserving event-triggered laser usefulness. |

## Acceptance Criteria
| Stage | Criteria |
|-------|----------|
| Scaffold | Sensor YAML loads; wrappers dry-run with desired CLI args; old defaults remain backward compatible. |
| Diagnostics | Laser and/or FC4 have measurable event-conditioned oracle lift; high-value subsets are feasible at target budgets. `physical_event_value_v2` failed this criterion. |
| Pilot | PD-PPO shows purposeful event-conditioned activation, lower warmup abort than round-robin/random, and competitive FW-MAE. |
| Paper | No claim says the fixed-budget simulator implements time-varying energy harvesting or proven event-conditioned warm-up unless supported by new results. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None in new implementation track | 1 | N/A |

## Notes
- Long server jobs must use the `microclimate-experiment-server` workflow and tmux.
- Do not overwrite V3.1 S2 main outputs.
- Energy-account/SOC is a planned architecture phase, not a silent modification to the fixed-budget experiment.
