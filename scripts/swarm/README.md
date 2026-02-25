# Swarm Local Context Pack

This folder contains offline-friendly scripts for building SWARM inputs from canonical local artifacts (WT FASTA, receptor PDB, ligand SDF/PDB, docking pose, pocket residues) and running a single recursive mutation loop.

## Scripts

- `14a_prepare_inputs.py`
  - Canonical input stage for protein/ligand ingestion from local files or APIs.
  - Produces canonical files consumed by SWARM/fpocket:
    - `OUTDIR/reference_protein.pdb`
    - `OUTDIR/enzyme_wt.fasta`
    - `OUTDIR/ligand.sdf`
    - `OUTDIR/ligand.pdb`
  - Writes provenance: `OUTDIR/swarm/input_manifest.json`.
- `15a_build_context_pack.py`
  - Builds `OUTDIR/swarm/context_pack.json`.
  - Merges API context from `OUTDIR/swarm_api/context_api.json` when present.
- `15b_build_site_cards.py`
  - Builds `OUTDIR/swarm/site_cards.jsonl`.
  - Merges residue constraints from `OUTDIR/swarm_api/residue_constraints.jsonl` when present.
  - Emits both `do_not_mutate` (annotation provenance) and `do_not_mutate_hard` (structural hard-block).
  - Writes cached geometry arrays in `OUTDIR/swarm/cache/`.
- `15f_enrich_site_cards_prolif.py`
  - Adds ProLIF residue interaction fingerprints to `site_cards.jsonl`.
  - Writes `OUTDIR/swarm/prolif_features.json`.
- `18b_generate_stat_neighborhood_candidates.py`
  - Unified generator for all recursive rounds (`round>=0`).
  - Mutation-level multi-objective deep-ensemble surrogate:
    - uses previous-round labels when available (`proposals_vespag_r{k-1}.jsonl`),
    - cold-starts from bootstrap priors when no previous labels exist.
  - Optimizes constrained EHVI using objectives:
    - function preservation,
    - binding relevance,
    - stability proxy,
    - sequence plausibility.
  - Writes:
    - `OUTDIR/swarm/proposals_rk.jsonl`
    - `OUTDIR/swarm/stat_model_diagnostics_rk.json`
    - `OUTDIR/swarm/manifest_rk.json`
  - Gate policy:
    - hard blocks only structural lock constraints (`DISULFIDE`/`CROSSLINK` class),
    - functional/catalytic annotations remain soft-risk modeled (protein-ligand agnostic default).
- `16a_make_vespag_mutation_file.py`
  - Converts round proposals into VespaG mutation input CSV.
- `16b_run_vespag_round.py`
  - Runs VespaG scoring.
  - Writes canonical per-round scores to `OUTDIR/swarm/vespag_scores_rk.csv`.
- `16c_join_vespag_scores.py`
  - Joins VespaG scores (`vespag_scores_rk.csv`) and applies Bayesian tri-band gating.
- `16d_update_vespag_policy.py`
  - Updates policy memory from accepted/rejected mutations.
- `16e_fast_binding_delta.py`
  - Fast binding alteration check without AlphaFold:
    - local mutation-aware structural rebuild via `pdbfixer`,
    - `gnina` `--score_only` CNN rescoring for WT and mutant,
    - writes WT->mutant `delta`-based `p_bind_fastdl` into `proposals_vespag.jsonl`.
  - Runtime requirements: `pdbfixer`, `openmm`, and a runnable `gnina` binary (`--gnina-bin`, optional `--binding-ld-library-path`).
- `17c_select_candidates.py`
  - Cluster-free mutation-level panel selector.
  - Two-lane exploitation/exploration with per-position caps, utility weighting, and MMR diversity.
  - Writes:
    - `OUTDIR/swarm/swarm_panel_rk.tsv`
    - `OUTDIR/swarm/swarm_panel_summary_rk.json`
- `20_run_recursive_adaptive_flow.py`
  - Primary orchestration entrypoint for the single recursive loop:
    - all rounds: `18b -> 16a/16b/16c/16e -> 17c -> 16d`
  - Tracks per-round metrics and adaptive threshold/proposal-budget knobs.
  - Uses VOI stop rule from `stat_model_diagnostics.json` (`expected_hvi_max`).
  - Enforces a preflight WT consistency QC between `site_cards.jsonl` and `enzyme_wt.fasta` (override: `--allow-site-card-wt-mismatch`).
  - Reuses existing proposal files only when manifest mode/script/input fingerprint and generation configuration all match.
  - Supports a global panel budget (`--global-panel-budget`) to keep cumulative final candidates concentrated.
  - Writes:
    - `OUTDIR/swarm/recursive_adaptive_summary.json`
    - `OUTDIR/swarm/recursive_iteration_metrics_rk.json`
- `19b_make_janus_input.py`
  - Converts panel TSV to JanusDDG input CSV.
- `19c_run_janusddg.py`
  - Runs JanusDDG and writes `janus_scores.csv`.
- `19d_join_janus_panel.py`
  - Joins JanusDDG output onto the panel and writes ranked output.
- `19e_run_final_janus.py`
  - Final scorer across rounds; merges panels, dedupes by `variant_id`, runs Janus once.

## Usage (Single Recursive System)

Recommended end-to-end entrypoint:

```bash
python scripts/swarm/run_swarm_bootstrap.py --outdir /content/run_LB --iterations 10 --panel-total 200 --global-panel-budget 96
```

Manual execution:

```bash
python scripts/swarm/14a_prepare_inputs.py --outdir /content/run_LB
python scripts/swarm/15a_build_context_pack.py --outdir /content/run_LB
python scripts/swarm/15b_build_site_cards.py --outdir /content/run_LB
python scripts/swarm/15f_enrich_site_cards_prolif.py --outdir /content/run_LB
python scripts/swarm/20_run_recursive_adaptive_flow.py --outdir /content/run_LB --start-round 0 --iterations 10 --panel-total 200 --global-panel-budget 96
```

Fast binding-focused mode (no AlphaFold in-loop):

```bash
python scripts/swarm/20_run_recursive_adaptive_flow.py \
  --outdir /content/run_LB \
  --start-round 0 \
  --iterations 10 \
  --panel-total 200 \
  --global-panel-budget 96 \
  --fast-binding-check \
  --binding-workers 4 \
  --binding-cnn-model fast
```

Multi-point mode (pairwise variants in same recursive loop):

```bash
python scripts/swarm/20_run_recursive_adaptive_flow.py \
  --outdir /content/run_LB \
  --start-round 0 \
  --iterations 10 \
  --panel-total 200 \
  --max-mutations-per-variant 2 \
  --multi-point-fraction 0.35 \
  --multi-seed-size 120 \
  --multi-max-candidates 1200
```

`20_run_recursive_adaptive_flow.py` is the primary execution system.
Use `--force-regenerate-proposals` to force proposal refresh even when round files exist.

Optional JanusDDG integration:

```bash
export JANUS_REPO=/path/to/JanusDDG
python scripts/swarm/20_run_recursive_adaptive_flow.py --outdir /content/run_LB --start-round 0 --iterations 10 --with-janus-final
```

Bootstrap notes:
- `python scripts/swarm/run_swarm_bootstrap.py --outdir /content/run_LB`
- Runs `14a_prepare_inputs.py` first by default (`--skip-input-prep` to disable).
- Runs the recursive loop by default (`--no-run-recursive` to disable).
- Accepts pass-through input flags such as `--protein-source/--protein-id/--ligand-source/--ligand-id/--input-spec`.
