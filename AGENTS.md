# Repository agent guide

## Current focus

- This repository contains several historical or secondary experiment pipelines. The current default task is refinement of the PD-PPO manuscript under `rl_sensor_scheduling_framework/`.
- Do not treat this file as a scientific project description. For scientific facts, inspect the active manuscript, executable implementation/configuration, and frozen evidence artifacts.
- Do not revive old Route A notes, historical run tags, backup paths, commit snapshots, or legacy script inventories unless the user explicitly requests historical work.

## Environment and safety

- Local tools: `uv`, `pip`, and conda are available. The main conda environment is `darts`.
- The local machine has no dedicated GPU. Use it for source editing, CPU checks, plotting, and LaTeX builds only.
- The only valid remote GPU entry is SSH alias `remote-gpu`. Never use historical IP addresses, old tunnels, or hard-coded hosts. Load the relevant remote-server skill before remote work; use `tmux` for long jobs.
- Never write credentials into the repository or print them in notes. Replace any credential encountered in manuscript-facing material with `[REDACTED]`.
- Ask before major root-level or destructive `sudo` operations.

## Worktree discipline

- The root repository, `rl_sensor_scheduling_framework/`, and `rl_sensor_scheduling_framework/paper/` are separate Git worktrees. Check `git status` in all three before editing or reporting a clean state.
- Preserve unrelated modified and untracked files. Never reset, clean, overwrite, or reformat work outside the explicit task scope.
- Put temporary plans, audits, extracted pages, and intermediate artifacts under `~/agent/tmp/`, not in the repository.
- Do not commit, push, rewrite history, or publish a release unless the user explicitly asks.

## Active manuscript surface

- Canonical English manuscript: `rl_sensor_scheduling_framework/paper/main.tex`.
- In this file, shorter `paper/...` paths are relative to `rl_sensor_scheduling_framework/`.
- `paper/anonymous_manuscript.tex` is a thin anonymous wrapper around `main.tex`; do not create a divergent scientific version there.
- Supplement: `paper/supplementary_material.tex`. Treat it as independently readable and apply first-use definitions there.
- Title page: `paper/title_page.tex`.
- `paper/raw.tex` is a Chinese translation mirror. Do not edit or synchronize it unless the user explicitly requests Chinese-version work.
- Resolve current highlights, cover letter, declarations, submission trees, and other attachments from the live repository before editing; do not assume a historical path still exists.

## Mandatory pre-edit protocol for every refinement round

1. Read `rl_sensor_scheduling_framework/docs/PD-PPO-TERMINOLOGY.md` in full before modifying any manuscript, supplement, figure text, table, highlight, cover letter, declaration, or submission metadata.
2. Read `paper/main.tex` and recursively inspect the relevant included sections, tables, captions, bibliography entries, and authoritative figure sources.
3. Recheck both Git worktrees and confirm that no other agent is actively writing the same paper tree. Do not resume or feed an existing Codex session during manuscript work.
4. Establish the edit allowlist and the evidence boundary. For a substantive round, create a source/PDF backup immediately before the first paper edit and record its checksum.
5. Treat `docs/08-02-01-word.md` and other review memos as editorial input, not scientific authority. Rebase every suggestion onto the current active source.

Evidence priority:

1. current executable implementation and configuration;
2. frozen evidence artifacts and verified manifests;
3. active manuscript equations and protocol definitions;
4. `docs/PD-PPO-TERMINOLOGY.md` as the canonical writing contract;
5. editorial memos and historical prose.

## Manuscript ownership and edit style

- Hermes must perform all paper modifications directly. Do not delegate manuscript prose, equations, tables, figure sources/exports, supplements, highlights, cover letters, declarations, packaging, LaTeX builds, or visual QA to Codex.
- Non-Codex subagents may perform read-only audits. Hermes owns every accepted edit, integration step, build, and final verification.
- Use surgical edits. Preserve scientific meaning, numerical values, statistical interpretation, citation meaning, equation semantics, and baseline definitions unless the user explicitly authorizes a scientific correction.
- Use objective academic narration and avoid first person unless the venue or user requires it. Do not synonym-cycle technical terms or introduce self-coined labels merely for stylistic variety.
- Ordinary refinement must not rerun training, policy rollout, simulation, model selection, or frozen statistics. New scientific claims require explicit authorization and new evidence.

## Non-negotiable PD-PPO boundaries

The terminology contract contains the full definitions and exact claim templates. At minimum, preserve these distinctions:

- The **PD-PPO scheduler** is the complete online method; the **PD-PPO policy** is the trainable categorical neural policy.
- Main empirical effects belong to the **complete PD-PPO training configuration**. Do not attribute them to forecast-loss reward, PPO, warnings, behavior cloning, AWBC, or the auxiliary classifier without an isolating experiment.
- Keep candidate, feasible, proposed, and executed actions distinct. Preserve the frozen proposed-action likelihood versus executed-transition convention.
- Keep `forecaster-training`, `policy-training`, `calibration/validation`, and `test` partitions distinct, including the difference between chronological placement and computational order.
- Keep the forecast-target set and observed-variable AoI set separate.
- The two co-primary endpoints are mean forecast loss and macro-averaged normalized forecast loss.
- Seeds 119--140 are the 22 post-selection evaluation seeds. Seeds 117--118 are pilot/model-selection seeds; the combined 24-seed aggregate is descriptive only.
- If a paired confidence interval includes zero, do not claim significance, equivalence, no difference, or a population-level directional advantage.
- Reward-objective rows are matched PPO training configurations with different scalar objectives. Double DQN is a training-configuration comparison, not a component ablation.
- Labels or next-step test targets unavailable to a deployed scheduler must be disclosed as privileged information in the comparator's first-use label.
- Do not strengthen the frozen core claim or change its attribution unit without verified new evidence.

## Figures, tables, and generated assets

- Figure 1 authority: `paper/figures/figure_pdppo_framework.drawio`. Edit the Draw.io source, then export PDF/PNG; do not patch the exports directly. Preserve the 82-cell topology unless the user explicitly authorizes a structural change.
- Generated scientific assets are owned by `rl_sensor_scheduling_framework/scripts/95_v31_build_clean_paper_assets.py`. Change reader-facing generated content through the authoritative generator and frozen inputs.
- A source, generator, or figure change invalidates downstream PDFs, submission trees, ZIP files, checksums, and archives. Rebuild and reverify the dependency chain after the final upstream edit.
- Judge figure readability both standalone and on the final embedded manuscript page.

## Verification before completion

- Search the complete active reader-facing surface for deprecated terms and near-synonyms listed in the terminology contract, including captions, tables, supplementary material, figure source, highlights, and cover-letter text.
- Build every affected independently readable document. For the main manuscript use, from `rl_sensor_scheduling_framework/paper/`:

  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

- If `main.tex` or included scientific content changed, also build `anonymous_manuscript.tex`. Build the supplement and title page when affected. Locate any other live attachment entry point before building it.
- Inspect the final target log for fatal errors and undefined references/citations. Visually inspect every affected PDF page; a successful LaTeX exit alone is insufficient.
- From `rl_sensor_scheduling_framework/`, run `conda run -n darts pytest -q tests` when repository code, tests, or generated-asset code is affected. Pure prose-only edits do not require retraining or unrelated tests.
- If a submission package changes, verify it from a fresh extraction: safe paths, no duplicate members, archive integrity, member/hash equality, independent build/tests, anonymity, and checksum sidecars.
- Run `git diff --check` in the affected Git worktree(s), review the final diff, and report any pre-existing unrelated dirty state separately.

## Non-paper code and experiments

- For code changes, inspect definitions, usages, neighboring imports, manifests, and tests before editing. Fix root causes and avoid unrelated refactors.
- Multi-file or non-trivial code implementation may be delegated to Codex under the general coding policy; the manuscript exception above always takes priority.
- If the user explicitly requests experiments, use the current implementation/configuration as authority, load the remote-GPU skill, launch long jobs in `tmux`, and monitor them to completion. Do not infer experiment status from this file.

## Keep this file lean

- Record only stable operating rules and current authoritative entry points here.
- Put scientific descriptions in the manuscript/docs, runnable commands in README or scripts, task progress in `~/agent/tmp/`, and historical outcomes in Git/CHANGELOG/evidence records.
- Do not add dated run status, exhaustive file catalogs, commit hashes, backup inventories, or prose copies of the full terminology contract to `AGENTS.md`.