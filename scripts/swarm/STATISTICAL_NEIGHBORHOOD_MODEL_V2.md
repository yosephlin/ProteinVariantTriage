# Statistical Mutation-Level Surrogate (Current)

This document describes the current focused-round generator (`18b_generate_stat_neighborhood_candidates.py`).

## Goal

Replace cluster-driven focused generation with a mutation-level recursive model that:
- learns online from prior-round scored proposals,
- predicts multiple objectives jointly,
- selects candidates by constrained expected hypervolume improvement (EHVI).

## Inputs

For round `k >= 1`, the generator uses:
- `swarm/site_cards.jsonl`
- `swarm/context_pack.json`
- `swarm/round_{k-1}/proposals_vespag.jsonl` (training labels)

## Surrogate

- Deep ensemble of MLP regressors (per objective).
- Objectives:
  - function preservation,
  - binding relevance,
  - stability proxy,
  - sequence plausibility.
- Prediction output per candidate:
  - posterior mean and uncertainty for each objective.

## Acquisition

For each candidate mutation:
- estimate EHVI via Monte Carlo,
- estimate feasibility probability against minimum objective thresholds,
- acquisition score = `EHVI * feasibility_probability`.

Candidates are ranked by acquisition, then filtered by:
- hard constraints,
- dedupe against previous rounds,
- per-position caps.

## Outputs

Per round `k`:
- `swarm/round_k/proposals.jsonl`
- `swarm/round_k/stat_model_diagnostics.json`
- `swarm/round_k/manifest.json`

## Orchestration

`20_run_recursive_adaptive_flow.py` runs:
- round 0: seed generator (`15c + 15e`) + VespaG + selector,
- rounds `k>0`: mutation-level surrogate generator (`18b`) + VespaG + selector.

No clustering/focus-plan stages are part of the active pipeline.
