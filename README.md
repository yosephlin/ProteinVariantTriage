# ProteinVariantTriage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yosephlin/ProteinVariantTriage/blob/main/VariantTriage_swarm_pre_esm_colab.ipynb)

Protein variant prioritization pipeline for structure-aware ligand-focused mutation triage.

The current repository is organized around a SWARM-style recursive proposal loop, VespaG sequence-function scoring, JanusDDG stability scoring, and an optional post-filter ColabFold/AF2 stage. The example configuration in this repo is `ecDHFR + methotrexate`, but the SWARM system itself is intended to be reusable for other protein-ligand systems once canonical inputs are prepared.

## What This Repo Does

At a high level, the pipeline:

1. fetches or prepares canonical WT inputs
2. builds a local context pack and residue/site features
3. generates mutation proposals with the SWARM recursive loop
4. scores proposals with VespaG and optional fast binding checks
5. selects a compact candidate panel
6. scores final candidates with JanusDDG
7. optionally runs ColabFold on the short-listed variants

The repo currently contains three notebook entrypoints:

- `VariantTriage.ipynb`
  - original notebook pipeline
- `VariantTriage_swarm_pre_esm_offline.ipynb`
  - local/offline-oriented SWARM notebook
- `VariantTriage_swarm_pre_esm_colab.ipynb`
  - Colab-first notebook with runtime bootstrap and dependency materialization

## Recommended Entry Point

Use `VariantTriage_swarm_pre_esm_colab.ipynb` if you want to run this from the Colab website.

That notebook is designed to:

- clone the repo into the Colab runtime
- materialize pinned third-party dependencies
- install the minimal runtime stack needed for the SWARM path
- auto-tune some GPU-sensitive defaults for Colab

## Colab Quick Start

### 1. Open the notebook

Use the badge above, or open directly:

```text
https://colab.research.google.com/github/yosephlin/ProteinVariantTriage/blob/main/VariantTriage_swarm_pre_esm_colab.ipynb
```

### 2. Select a GPU runtime

In Colab:

- `Runtime` -> `Change runtime type`
- set `Hardware accelerator` to `GPU`

### 3. Run the bootstrap cells in order

Run these cells first:

1. `00 - Clone Repo + Enter Workdir`
2. `01 - Install Runtime Dependencies (Colab)`
3. `02 - Configuration`
4. `03 - GPU Preflight + Auto-Tune`

After that, continue with the notebook cells in order.

### 4. What cell `00` now does

The Colab notebook does not require `VespaG/` and `JanusDDG/` to be vendored into the parent repo checkout.

Instead it uses:

- `colab_dependency_manifest.json`
- `scripts/colab/materialize_dependencies.py`
- tracked overlay patches in `third_party_overlays/`

to recreate the dependency state inside the Colab runtime.

### 5. Optional environment variables

The Colab notebook supports these overrides:

```bash
PVT_REPO_URL
PVT_GIT_REF
PVT_REPO_DIR
PVT_OUTDIR
PVT_UNIPROT_ID
PVT_VESPAG_REPO_URL
PVT_VESPAG_REF
PVT_JANUSDDG_REPO_URL
PVT_JANUSDDG_REF
```

Examples:

```bash
export PVT_UNIPROT_ID=P0ABQ4
export PVT_OUTDIR=/content/run_P0ABQ4
```

## Local Usage

If you are running locally and already have the toolchain installed, use:

- `VariantTriage_swarm_pre_esm_offline.ipynb`
- or the script entrypoint `scripts/swarm/run_swarm_bootstrap.py`

Recommended local SWARM bootstrap:

```bash
python scripts/swarm/run_swarm_bootstrap.py \
  --outdir /path/to/run \
  --iterations 10 \
  --panel-total 200 \
  --global-panel-budget 96
```

Manual local flow:

```bash
python scripts/swarm/14a_prepare_inputs.py --outdir /path/to/run
python scripts/swarm/15a_build_context_pack.py --outdir /path/to/run
python scripts/swarm/15b_build_site_cards.py --outdir /path/to/run
python scripts/swarm/15f_enrich_site_cards_prolif.py --outdir /path/to/run
python scripts/swarm/20_run_recursive_adaptive_flow.py \
  --outdir /path/to/run \
  --start-round 0 \
  --iterations 10 \
  --panel-total 200 \
  --global-panel-budget 96
```

## Dependency Model

This repo intentionally does not track the local `VespaG/` and `JanusDDG/` working trees in the parent repository.

Instead, the tracked parent repo contains:

- `colab_dependency_manifest.json`
  - pinned upstream source and commit for each dependency
- `scripts/colab/materialize_dependencies.py`
  - clones or updates those repos into the runtime
- `third_party_overlays/`
  - tracked local patches applied on top of pinned upstream commits

Current pinned dependencies:

- `VespaG`
  - source: `https://github.com/JSchlensok/VespaG.git`
  - ref: `05a021674c1a1626d4deeedf3741e99e79de6464`
- `JanusDDG`
  - source: `https://github.com/compbiomed-unito/JanusDDG.git`
  - ref: `7ef53fde72c9b9b715b0a9f2f80910ebf927bd59`

Current tracked overlays:

- `third_party_overlays/VespaG/0001-lazy-import-cli.patch`
  - makes the VespaG CLI import path lighter by deferring heavy imports in `vespag/__main__.py`
- `third_party_overlays/VespaG/0002-transformers-tokenizer-compat.patch`
  - updates VespaG embedding generation for current `transformers` tokenizer APIs used in Colab
- `third_party_overlays/JanusDDG/0001-resolve-model-path.patch`
  - makes JanusDDG resolve model checkpoints from `JanusDDG/models/` instead of assuming a repo-root checkpoint

The Colab recursive SWARM cell also resolves GNINA explicitly:

- first from config or `PATH`
- then by downloading a pinned GNINA release asset when missing
- with a CPU fallback when the CUDA build is unavailable in the runtime

## Repository Layout

```text
ProteinVariantTriage/
├── VariantTriage.ipynb
├── VariantTriage_swarm_pre_esm_offline.ipynb
├── VariantTriage_swarm_pre_esm_colab.ipynb
├── requirements.txt
├── requirements-colab.txt
├── colab_dependency_manifest.json
├── scripts/
│   ├── colab/
│   │   └── materialize_dependencies.py
│   └── swarm/
│       ├── 14a_prepare_inputs.py
│       ├── 15a_build_context_pack.py
│       ├── 15b_build_site_cards.py
│       ├── 15f_enrich_site_cards_prolif.py
│       ├── 16a_make_vespag_mutation_file.py
│       ├── 16b_run_vespag_round.py
│       ├── 16c_join_vespag_scores.py
│       ├── 16d_update_vespag_policy.py
│       ├── 16e_fast_binding_delta.py
│       ├── 17c_select_candidates.py
│       ├── 19b_make_janus_input.py
│       ├── 19c_run_janusddg.py
│       ├── 19d_join_janus_panel.py
│       ├── 19e_run_final_janus.py
│       ├── 20_run_recursive_adaptive_flow.py
│       └── run_swarm_bootstrap.py
└── third_party_overlays/
    └── VespaG/
        └── 0001-lazy-import-cli.patch
```

## Main SWARM Stages

The most important SWARM scripts are:

- `scripts/swarm/14a_prepare_inputs.py`
  - canonical WT protein and ligand input preparation
- `scripts/swarm/15a_build_context_pack.py`
  - builds the local context pack
- `scripts/swarm/15b_build_site_cards.py`
  - residue/site-level mutation context
- `scripts/swarm/15f_enrich_site_cards_prolif.py`
  - adds ProLIF interaction features
- `scripts/swarm/18b_generate_stat_neighborhood_candidates.py`
  - proposal generator for recursive rounds
- `scripts/swarm/16a_make_vespag_mutation_file.py`
  - converts proposals to VespaG input
- `scripts/swarm/16b_run_vespag_round.py`
  - VespaG scoring
- `scripts/swarm/16c_join_vespag_scores.py`
  - joins VespaG results and applies gating
- `scripts/swarm/16e_fast_binding_delta.py`
  - optional fast binding-delta screen without in-loop AF2
- `scripts/swarm/17c_select_candidates.py`
  - panel selection
- `scripts/swarm/19b_make_janus_input.py`
  - prepares JanusDDG input
- `scripts/swarm/19c_run_janusddg.py`
  - runs JanusDDG
- `scripts/swarm/19d_join_janus_panel.py`
  - joins Janus results to the panel
- `scripts/swarm/20_run_recursive_adaptive_flow.py`
  - primary recursive orchestration entrypoint

Additional script-level notes are in:

- `scripts/swarm/README.md`

## Output Conventions

Most generated pipeline artifacts are written under:

```text
data/<UNIPROT_ID>/
```

For SWARM runs, the important outputs typically live under:

```text
data/<UNIPROT_ID>/swarm/
```

Common outputs include:

- `context_pack.json`
- `site_cards.jsonl`
- `proposals_rk.jsonl`
- `vespag_scores_rk.csv`
- `swarm_panel_rk.tsv`
- `janus_input.csv`
- `swarm_final_with_janus.tsv`
- `recursive_iteration_metrics_rk.json`
- `recursive_adaptive_summary.json`

These outputs are intentionally ignored from Git.

## Colab Runtime Notes

### Dependency installation

The Colab notebook installs:

- Python dependencies from `requirements-colab.txt`
- system/runtime dependencies via `mamba`
- ColabFold
- local editable `VespaG` after dependency materialization

### GPU use

The Colab notebook attempts to use GPU for:

- ESM embedding stages when available
- ColabFold/AF2 stages

It also auto-tunes some AF2 chunking settings based on observed GPU memory.

### Practical Colab constraints

Hosted Colab is still constrained by:

- variable GPU type
- runtime duration limits
- ephemeral filesystem
- slow first-time dependency installation
- ColabFold/AF2 memory sensitivity

For heavy AF2 use, a local runtime or a better-provisioned environment is still the more reliable option.

## Protein/Ligand Scope

The SWARM design in this repo is intended to be more general than a single protein-ligand pair, but the included example notebook configuration is currently centered on:

- protein: `ecDHFR`
- UniProt: `P0ABQ4`
- ligand example: `methotrexate`

That means:

- the orchestration and candidate-generation system is reusable
- the included defaults and sample outputs are still DHFR/MTX-specific until you replace the inputs/configuration

## Git Policy

Tracked:

- notebooks
- scripts
- manifests
- overlay patches
- requirements files

Ignored:

- `data/`
- notebook checkpoints
- Python caches
- local dependency working trees
- downloaded tools and scratch artifacts

This keeps the parent repo reproducible without forcing large generated outputs or nested dependency repos into version control.

## Minimal Reproducible Colab Flow

If you only want the shortest path to a run:

1. open `VariantTriage_swarm_pre_esm_colab.ipynb`
2. switch Colab to GPU
3. run cells `00`, `01`, `02`, `03`
4. continue through the notebook

If cell `00` succeeds, the dependency materialization path is working.
If cell `01` succeeds, the Colab runtime has the packages and pinned dependency assets needed by the notebook.

## Maintainer Notes

If you change local dependency behavior in `VespaG/` or `JanusDDG/`, do not silently rely on untracked nested repo state.

Instead:

1. pin the dependency commit in `colab_dependency_manifest.json`
2. capture the parent-required delta as a patch in `third_party_overlays/`
3. keep the Colab notebook bootstrap consuming the manifest rather than local-only assumptions

That is the mechanism that keeps Colab and local behavior aligned.
