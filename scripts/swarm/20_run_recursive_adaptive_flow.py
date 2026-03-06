#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from artifact_paths import (
        final_swarm_panel_path,
        final_swarm_panel_production_path,
        final_swarm_panel_summary_path,
        final_with_janus_production_path,
        final_with_janus_production_summary_path,
        final_with_janus_summary_path,
        final_with_janus_path,
        panel_path,
        panel_summary_path,
        proposals_path,
        proposals_vespag_path,
        recursive_iteration_metrics_path,
        round_diagnostics_path,
        round_manifest_path,
        swarm_root,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        final_swarm_panel_path,
        final_swarm_panel_production_path,
        final_swarm_panel_summary_path,
        final_with_janus_production_path,
        final_with_janus_production_summary_path,
        final_with_janus_summary_path,
        final_with_janus_path,
        panel_path,
        panel_summary_path,
        proposals_path,
        proposals_vespag_path,
        recursive_iteration_metrics_path,
        round_diagnostics_path,
        round_manifest_path,
        swarm_root,
    )

try:
    from mutation_utils import row_mutations
except ImportError:
    from scripts.swarm.mutation_utils import row_mutations


EXPECTED_PROPOSAL_GENERATOR = "mutation_level_deep_ensemble_v4_multi_point"
AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def run(cmd: List[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def std_or_zero(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean_or_zero(values)
    var = sum((float(x) - mu) ** 2 for x in values) / float(len(values))
    return float(math.sqrt(max(0.0, var)))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open() as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def load_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def parse_fasta_sequence(path: Path) -> str:
    if not path.exists():
        return ""
    seq_parts: List[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    break
                continue
            seq_parts.append(line)
    return "".join(seq_parts).upper()


def site_card_sequence_qc(
    outdir: Path,
    mismatch_max_frac: float,
    mismatch_min_checked: int,
) -> Dict[str, Any]:
    site_cards_path = outdir / "swarm" / "site_cards.jsonl"
    fasta_path = outdir / "enzyme_wt.fasta"
    cards = load_jsonl(site_cards_path)
    seq = parse_fasta_sequence(fasta_path)
    checked = 0
    mismatch = 0
    out_of_range = 0
    missing_wt = 0

    if not cards or not seq:
        return {
            "site_cards_path": str(site_cards_path),
            "fasta_path": str(fasta_path),
            "checked": int(checked),
            "mismatch": int(mismatch),
            "out_of_range": int(out_of_range),
            "missing_wt": int(missing_wt),
            "mismatch_fraction": 0.0,
            "passes": True,
            "skipped": True,
        }

    for c in cards:
        if not isinstance(c, dict):
            continue
        wt = str(c.get("wt") or "").upper()
        if wt not in AA_STANDARD:
            missing_wt += 1
            continue
        try:
            pos = int(c.get("pos"))
        except Exception:
            continue
        if pos <= 0 or pos > len(seq):
            out_of_range += 1
            continue
        checked += 1
        if seq[pos - 1] != wt:
            mismatch += 1

    mismatch_fraction = float(mismatch / float(max(1, checked)))
    enough = checked >= int(max(1, mismatch_min_checked))
    passes = (not enough) or (mismatch_fraction <= float(max(0.0, mismatch_max_frac)))
    return {
        "site_cards_path": str(site_cards_path),
        "fasta_path": str(fasta_path),
        "checked": int(checked),
        "mismatch": int(mismatch),
        "out_of_range": int(out_of_range),
        "missing_wt": int(missing_wt),
        "mismatch_fraction": round(float(mismatch_fraction), 6),
        "mismatch_max_frac": round(float(max(0.0, mismatch_max_frac)), 6),
        "mismatch_min_checked": int(max(1, mismatch_min_checked)),
        "passes": bool(passes),
        "skipped": False,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_generation_input_fingerprint(outdir: Path, focus_round: int) -> Dict[str, Any]:
    if int(focus_round) >= 0:
        train_labels_path = proposals_vespag_path(outdir=outdir, round_id=int(focus_round))
    else:
        train_labels_path = outdir / "swarm" / "bootstrap_prior_labels.jsonl"
    targets = {
        "train_labels_path": train_labels_path,
        "site_cards_path": outdir / "swarm" / "site_cards.jsonl",
        "context_pack_path": outdir / "swarm" / "context_pack.json",
    }
    fp: Dict[str, Any] = {"focus_round": int(focus_round)}
    for key, path in targets.items():
        p = Path(path)
        fp[key] = str(p)
        fp[key.replace("_path", "_sha256")] = file_sha256(p) if p.exists() else None
    return fp


def build_generation_config(
    round_id: int,
    knobs: "AdaptiveKnobs",
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "target_proposals": int(knobs.proposal_total),
        "max_per_position": int(knobs.proposal_max_per_position),
        "max_mutations_per_variant": int(max(1, int(args.max_mutations_per_variant))),
        "multi_point_fraction": round(float(clamp(float(args.multi_point_fraction), 0.0, 1.0)), 6),
        "multi_seed_size": int(max(8, int(args.multi_seed_size))),
        "multi_max_candidates": int(max(0, int(args.multi_max_candidates))),
        "multi_min_position_separation": int(max(0, int(args.multi_min_position_separation))),
        "multi_max_position_separation": int(max(0, int(args.multi_max_position_separation))),
        "critical_blosum_min": int(args.critical_blosum_min),
        "strict_evo_conservation_threshold": round(float(args.strict_evo_conservation_threshold), 6),
        "strict_evo_blosum_min": int(args.strict_evo_blosum_min),
        "functional_exploratory_enable": bool(args.functional_exploratory_enable),
        "functional_exploratory_blosum_min": int(args.functional_exploratory_blosum_min),
        "functional_exploratory_max_extra": int(max(0, int(args.functional_exploratory_max_extra))),
        "functional_exploratory_ligand_shell": round(float(max(0.0, float(args.functional_exploratory_ligand_shell))), 6),
        "chemistry_coverage_enable": bool(args.chemistry_coverage_enable),
        "chemistry_coverage_max_extra": int(max(0, int(args.chemistry_coverage_max_extra))),
        "chemistry_coverage_blosum_min": int(args.chemistry_coverage_blosum_min),
        "chemistry_coverage_distal_enable": bool(args.chemistry_coverage_distal_enable),
        "chemistry_coverage_distal_ligand_shell": round(float(max(0.0, float(args.chemistry_coverage_distal_ligand_shell))), 6),
        "dedupe_scope": str(args.dedupe_scope),
        "dedupe_lookback_rounds": int(max(0, int(args.dedupe_lookback_rounds))),
        "ensemble_models": int(args.ensemble_models),
        "ensemble_max_iter": int(args.ensemble_max_iter),
        "min_train_samples": int(max(1, int(args.min_train_samples))),
        "ehvi_mc": int(max(1, int(args.ehvi_mc))),
        "min_function": round(float(knobs.min_function), 6),
        "min_binding": round(float(knobs.min_binding), 6),
        "min_stability": round(float(knobs.min_stability), 6),
        "min_plausibility": round(float(knobs.min_plausibility), 6),
        "seed": int(args.proposal_seed + (round_id * 7)),
        "functional_site_hard_filter": bool(args.functional_site_hard_filter),
        "near_functional_hard_filter": bool(args.near_functional_hard_filter),
    }


def should_regenerate_proposals(
    outdir: Path,
    round_id: int,
    focus_round: int,
    expected_generation_config: Dict[str, Any],
) -> bool:
    props = proposals_path(outdir=outdir, round_id=int(round_id))
    if not props.exists():
        return True
    manifest_path = round_manifest_path(outdir=outdir, round_id=int(round_id))
    if not manifest_path.exists():
        return True
    manifest = load_json(manifest_path, {})
    if not isinstance(manifest, dict):
        return True

    source = str(manifest.get("generation_mode") or "").strip().lower()
    if source != EXPECTED_PROPOSAL_GENERATOR:
        return True

    generator_script = Path(__file__).resolve().parent / "18b_generate_stat_neighborhood_candidates.py"
    expected_generator_sha = file_sha256(generator_script)
    current_generator_sha = str(manifest.get("generator_script_sha256") or "").strip().lower()
    if current_generator_sha != expected_generator_sha:
        return True

    expected_fp = build_generation_input_fingerprint(outdir=outdir, focus_round=focus_round)
    current_fp = manifest.get("input_fingerprint") if isinstance(manifest.get("input_fingerprint"), dict) else None
    if current_fp != expected_fp:
        return True

    current_cfg = manifest.get("generation_config") if isinstance(manifest.get("generation_config"), dict) else None
    if current_cfg != expected_generation_config:
        return True
    return False


@dataclass
class AdaptiveKnobs:
    proposal_total: int
    proposal_max_per_position: int
    panel_max_per_position: int
    tau_func_green: float
    tau_func_amber: float
    min_function: float
    min_binding: float
    min_stability: float
    min_plausibility: float


@dataclass
class AdaptiveSelectorKnobs:
    exploit_frac: float
    exploit_min_func: float
    binding_challenger_frac: float
    binding_challenger_min: int
    min_binding_challenger_selected: int
    chemistry_challenger_frac: float
    chemistry_challenger_min: int
    min_chemistry_challenger_selected: int
    w_bind: float
    w_func: float
    w_stability: float
    w_plausibility: float
    w_novel: float
    mmr_exploit: float
    mmr_explore: float
    mmr_repeat_penalty: float
    explore_new_position_bonus: float


def compute_objective(metrics: Dict[str, Any]) -> float:
    top_decile = safe_float(metrics.get("vespag_top_decile_mean"), 0.0)
    pass_rate = safe_float(metrics.get("vespag_gate_pass_rate"), 0.0)
    bind = safe_float(metrics.get("panel_bind_mean"), safe_float(metrics.get("candidate_bind_mean"), 0.0))
    func = safe_float(metrics.get("panel_func_mean"), 0.0)
    stability = safe_float(metrics.get("panel_stability_mean"), safe_float(metrics.get("candidate_stability_mean"), 0.0))
    plaus = safe_float(metrics.get("panel_plausibility_mean"), safe_float(metrics.get("candidate_plausibility_mean"), 0.0))
    panel_fill = safe_float(metrics.get("panel_fill"), 0.0)
    green_fraction = safe_float(metrics.get("panel_green_fraction"), 0.0)
    red_fraction = safe_float(metrics.get("selector_red_fraction"), 0.0)
    diversity = safe_float(metrics.get("diversity_ratio"), 0.0)
    prolif_fraction = safe_float(metrics.get("panel_prolif_contact_fraction"), 0.0)

    score = (
        (0.22 * top_decile)
        + (0.14 * pass_rate)
        + (0.14 * bind)
        + (0.14 * func)
        + (0.14 * stability)
        + (0.08 * plaus)
        + (0.05 * green_fraction)
        + (0.04 * panel_fill)
        + (0.03 * diversity)
        + (0.02 * prolif_fraction)
        - (0.08 * red_fraction)
    )
    return float(clamp(score, 0.0, 1.0))


def summarize_round(outdir: Path, round_id: int, panel_total: int) -> Dict[str, Any]:
    props = load_jsonl(proposals_path(outdir=outdir, round_id=int(round_id)))
    scored = load_jsonl(proposals_vespag_path(outdir=outdir, round_id=int(round_id)))
    panel_summary = load_json(panel_summary_path(outdir=outdir, round_id=int(round_id)), {})
    panel_rows = load_tsv(panel_path(outdir=outdir, round_id=int(round_id)))
    stat_diag = load_json(round_diagnostics_path(outdir=outdir, round_id=int(round_id)), {})

    posterior_vals: List[float] = []
    pass_n = 0
    candidate_bind: List[float] = []
    candidate_stability: List[float] = []
    candidate_plaus: List[float] = []

    for r in scored:
        p = r.get("vespag_shrunk_posterior", r.get("vespag_posterior"))
        x = safe_float(p, float("nan"))
        if math.isfinite(x):
            posterior_vals.append(float(clamp(x, 0.0, 1.0)))
        if bool(r.get("vespag_gate_pass")):
            pass_n += 1

        b = safe_float(r.get("p_bind"), float("nan"))
        if not math.isfinite(b):
            stat_model = r.get("stat_model") if isinstance(r.get("stat_model"), dict) else {}
            b = safe_float(stat_model.get("bind_relevance"), float("nan"))
        if math.isfinite(b):
            candidate_bind.append(float(clamp(b, 0.0, 1.0)))

        risk = safe_float(r.get("mechanistic_risk"), float("nan"))
        if not math.isfinite(risk):
            stat_model = r.get("stat_model") if isinstance(r.get("stat_model"), dict) else {}
            risk = safe_float(stat_model.get("mechanistic_risk"), float("nan"))
        if math.isfinite(risk):
            candidate_stability.append(float(clamp(1.0 - risk, 0.0, 1.0)))

        seq = safe_float(r.get("seq_prior_ensemble_plausibility"), float("nan"))
        if math.isfinite(seq):
            candidate_plaus.append(float(clamp(seq, 0.0, 1.0)))

    posterior_vals.sort(reverse=True)
    n_post = len(posterior_vals)
    k_top = max(1, int(round(0.1 * n_post))) if n_post > 0 else 0
    top_decile_mean = (sum(posterior_vals[:k_top]) / float(k_top)) if k_top > 0 else 0.0

    unique_positions = set()
    for r in props:
        muts = row_mutations(r)
        if muts:
            for m in muts:
                try:
                    unique_positions.add((str(m.get("chain") or "A"), int(m.get("pos") or -1)))
                except Exception:
                    continue
            continue
        if r.get("pos") is not None:
            try:
                unique_positions.add((str(r.get("chain") or "A"), int(r.get("pos") or -1)))
            except Exception:
                pass
    unique_positions = {x for x in unique_positions if x[1] > 0}
    proposal_total = int(len(props))
    diversity_ratio = (len(unique_positions) / float(proposal_total)) if proposal_total > 0 else 0.0

    panel_bind = [safe_float(r.get("p_bind"), float("nan")) for r in panel_rows]
    panel_bind = [float(clamp(x, 0.0, 1.0)) for x in panel_bind if math.isfinite(x)]

    panel_func = [safe_float(r.get("p_func"), float("nan")) for r in panel_rows]
    panel_func = [float(clamp(x, 0.0, 1.0)) for x in panel_func if math.isfinite(x)]

    panel_stability = [safe_float(r.get("p_stability"), float("nan")) for r in panel_rows]
    panel_stability = [float(clamp(x, 0.0, 1.0)) for x in panel_stability if math.isfinite(x)]

    panel_plaus = [safe_float(r.get("p_plausibility"), float("nan")) for r in panel_rows]
    panel_plaus = [float(clamp(x, 0.0, 1.0)) for x in panel_plaus if math.isfinite(x)]

    panel_prolif = [safe_float(r.get("prolif_persist"), float("nan")) for r in panel_rows]
    panel_prolif = [float(clamp(x, 0.0, 1.0)) for x in panel_prolif if math.isfinite(x)]

    selected_total = int(safe_float(panel_summary.get("selected_total"), float(len(panel_rows))))
    panel_fill = selected_total / float(max(1, int(panel_total)))

    selected_green = int(safe_float(panel_summary.get("selected_green"), 0.0))
    selected_amber = int(safe_float(panel_summary.get("selected_amber"), 0.0))
    if selected_total <= 0:
        green_fraction = 0.0
        amber_fraction = 0.0
    else:
        green_fraction = float(selected_green / float(selected_total))
        amber_fraction = float(selected_amber / float(selected_total))

    selector_effective_min_binding = clamp(
        safe_float(panel_summary.get("effective_min_binding"), safe_float(panel_summary.get("min_binding"), 0.0)),
        0.0,
        1.0,
    )
    selector_effective_min_stability = clamp(
        safe_float(panel_summary.get("effective_min_stability"), safe_float(panel_summary.get("min_stability"), 0.0)),
        0.0,
        1.0,
    )
    selector_effective_min_plausibility = clamp(
        safe_float(panel_summary.get("effective_min_plausibility"), safe_float(panel_summary.get("min_plausibility"), 0.0)),
        0.0,
        1.0,
    )
    selector_reject_counts = panel_summary.get("reject_counts") if isinstance(panel_summary.get("reject_counts"), dict) else {}
    selector_role_counts = panel_summary.get("role_counts") if isinstance(panel_summary.get("role_counts"), dict) else {}
    selector_lane_counts = panel_summary.get("lane_counts") if isinstance(panel_summary.get("lane_counts"), dict) else {}
    role_total = float(sum(max(0.0, safe_float(v, 0.0)) for v in selector_role_counts.values()))
    if role_total > 0.0:
        role_name, role_count = max(
            ((str(k), float(max(0.0, safe_float(v, 0.0)))) for k, v in selector_role_counts.items()),
            key=lambda kv: kv[1],
        )
        selector_dominant_role = role_name
        selector_dominant_role_fraction = float(role_count / role_total)
    else:
        selector_dominant_role = ""
        selector_dominant_role_fraction = 0.0
    selector_qualified_total = int(safe_float(panel_summary.get("qualified_total"), 0.0))
    selector_qualified_ratio = float(selector_qualified_total / float(max(1, panel_total)))

    metrics = {
        "proposal_total": proposal_total,
        "unique_positions": int(len(unique_positions)),
        "diversity_ratio": round(float(clamp(diversity_ratio, 0.0, 1.0)), 6),
        "vespag_scored_total": int(len(scored)),
        "vespag_gate_pass_rate": round(float(pass_n / float(max(1, len(scored)))), 6),
        "vespag_top_decile_mean": round(float(clamp(top_decile_mean, 0.0, 1.0)), 6),
        "candidate_bind_mean": round(float(mean_or_zero(candidate_bind)), 6),
        "candidate_stability_mean": round(float(mean_or_zero(candidate_stability)), 6),
        "candidate_plausibility_mean": round(float(mean_or_zero(candidate_plaus)), 6),
        "candidate_bind_std": round(float(std_or_zero(candidate_bind)), 6),
        "candidate_stability_std": round(float(std_or_zero(candidate_stability)), 6),
        "candidate_plausibility_std": round(float(std_or_zero(candidate_plaus)), 6),
        "panel_selected_total": selected_total,
        "panel_fill": round(float(clamp(panel_fill, 0.0, 1.0)), 6),
        "panel_green_fraction": round(float(clamp(green_fraction, 0.0, 1.0)), 6),
        "panel_amber_fraction": round(float(clamp(amber_fraction, 0.0, 1.0)), 6),
        "panel_bind_mean": round(float(mean_or_zero(panel_bind)), 6),
        "panel_func_mean": round(float(mean_or_zero(panel_func)), 6),
        "panel_stability_mean": round(float(mean_or_zero(panel_stability)), 6),
        "panel_plausibility_mean": round(float(mean_or_zero(panel_plaus)), 6),
        "panel_bind_std": round(float(std_or_zero(panel_bind)), 6),
        "panel_func_std": round(float(std_or_zero(panel_func)), 6),
        "panel_stability_std": round(float(std_or_zero(panel_stability)), 6),
        "panel_plausibility_std": round(float(std_or_zero(panel_plaus)), 6),
        "panel_prolif_contact_fraction": round(
            float(sum(1 for x in panel_prolif if x >= 0.18) / float(max(1, len(panel_prolif)))),
            6,
        ),
        "selector_qualified_total": int(selector_qualified_total),
        "selector_qualified_ratio": round(float(max(0.0, selector_qualified_ratio)), 6),
        "selector_qualified_before_minima_total": int(safe_float(panel_summary.get("qualified_before_minima_total"), 0.0)),
        "selector_red_rescued_total": int(safe_float(panel_summary.get("red_rescued_total"), 0.0)),
        "selector_binding_challenger_pool_total": int(safe_float(panel_summary.get("binding_challenger_pool_total"), 0.0)),
        "selector_binding_challenger_added": int(safe_float(panel_summary.get("binding_challenger_added"), 0.0)),
        "selector_selected_binding_challengers": int(safe_float(panel_summary.get("selected_binding_challengers"), 0.0)),
        "selector_selected_chemistry_challengers": int(
            safe_float(panel_summary.get("selected_chemistry_challengers"), 0.0)
        ),
        "selector_red_fraction": round(float(clamp(1.0 - green_fraction - amber_fraction, 0.0, 1.0)), 6),
        "selector_minima_relaxed": bool(panel_summary.get("minima_relaxed")),
        "selector_fallback_mode": str(panel_summary.get("fallback_mode") or ""),
        "selector_effective_min_binding": round(float(selector_effective_min_binding), 6),
        "selector_effective_min_stability": round(float(selector_effective_min_stability), 6),
        "selector_effective_min_plausibility": round(float(selector_effective_min_plausibility), 6),
        "selector_reject_counts": selector_reject_counts,
        "selector_role_counts": selector_role_counts,
        "selector_lane_counts": selector_lane_counts,
        "selector_dominant_role": selector_dominant_role,
        "selector_dominant_role_fraction": round(float(clamp(selector_dominant_role_fraction, 0.0, 1.0)), 6),
        "expected_hvi_max": round(float(clamp(safe_float((stat_diag or {}).get("expected_hvi_max"), 0.0), 0.0, 1.0)), 10),
        "expected_hvi_mean_top10": round(float(clamp(safe_float((stat_diag or {}).get("expected_hvi_mean_top10"), 0.0), 0.0, 1.0)), 10),
        "expected_hvi_median": round(float(clamp(safe_float((stat_diag or {}).get("expected_hvi_median"), 0.0), 0.0, 1.0)), 10),
    }
    metrics["objective_score"] = round(float(compute_objective(metrics)), 6)
    return metrics


def adapt_knobs(
    knobs: AdaptiveKnobs,
    metrics: Dict[str, Any],
    max_proposal_total: int,
    diversity_floor: float,
) -> Tuple[AdaptiveKnobs, List[str]]:
    next_knobs = deepcopy(knobs)
    notes: List[str] = []

    diversity = safe_float(metrics.get("diversity_ratio"), 0.0)
    pass_rate = safe_float(metrics.get("vespag_gate_pass_rate"), 0.0)
    panel_fill = safe_float(metrics.get("panel_fill"), 0.0)
    panel_bind = safe_float(metrics.get("panel_bind_mean"), safe_float(metrics.get("candidate_bind_mean"), 0.0))
    panel_stability = safe_float(metrics.get("panel_stability_mean"), safe_float(metrics.get("candidate_stability_mean"), 0.0))
    panel_plaus = safe_float(metrics.get("panel_plausibility_mean"), safe_float(metrics.get("candidate_plausibility_mean"), 0.0))
    panel_plaus_std = safe_float(metrics.get("panel_plausibility_std"), 0.0)
    candidate_plaus_std = safe_float(metrics.get("candidate_plausibility_std"), 0.0)
    selector_minima_relaxed = bool(metrics.get("selector_minima_relaxed"))
    selector_effective_min_plaus = safe_float(metrics.get("selector_effective_min_plausibility"), panel_plaus)
    selector_reject_counts = metrics.get("selector_reject_counts") if isinstance(metrics.get("selector_reject_counts"), dict) else {}
    reject_plaus = safe_float(selector_reject_counts.get("reject_plausibility_below_min"), 0.0)
    reject_total = float(sum(max(0.0, safe_float(v, 0.0)) for v in selector_reject_counts.values()))
    reject_plaus_frac = (reject_plaus / reject_total) if reject_total > 0 else 0.0
    hvi_max = safe_float(metrics.get("expected_hvi_max"), 0.0)
    plausibility_signal_collapsed = bool(
        selector_minima_relaxed
        or panel_plaus_std < 0.015
        or candidate_plaus_std < 0.015
    )

    if diversity < diversity_floor:
        next_knobs.proposal_total = min(max_proposal_total, int(round(next_knobs.proposal_total * 1.15)))
        next_knobs.proposal_max_per_position = max(2, int(next_knobs.proposal_max_per_position) - 1)
        notes.append("low_diversity_expand_acquisition")

    if pass_rate < 0.35:
        next_knobs.min_function = clamp(next_knobs.min_function + 0.03, 0.20, 0.90)
        next_knobs.tau_func_green = clamp(next_knobs.tau_func_green + 0.02, 0.55, 0.95)
        next_knobs.tau_func_amber = clamp(next_knobs.tau_func_amber + 0.02, 0.25, 0.88)
        notes.append("low_function_pass_tighten_function_thresholds")

    if panel_bind < 0.40:
        next_knobs.proposal_total = min(max_proposal_total, int(round(next_knobs.proposal_total * 1.08)))
        next_knobs.min_binding = clamp(next_knobs.min_binding + 0.02, 0.20, 0.85)
        notes.append("low_binding_raise_binding_constraint_and_expand")

    if panel_stability < 0.45:
        next_knobs.min_stability = clamp(next_knobs.min_stability + 0.03, 0.20, 0.90)
        notes.append("low_stability_raise_stability_constraint")

    if panel_plaus < 0.45:
        plausibility_regime_low = bool(panel_plaus < 0.30 or reject_plaus_frac >= 0.35)
        if plausibility_signal_collapsed or plausibility_regime_low:
            # When sequence plausibility is collapsed/flat, tightening this gate only causes
            # false negatives and selector fallback. Keep the gate aligned to effective support.
            target_min = min(next_knobs.min_plausibility, selector_effective_min_plaus + 0.01)
            next_knobs.min_plausibility = clamp(target_min, 0.05, 0.90)
            notes.append("plausibility_signal_low_hold_or_relax_sequence_constraint")
        else:
            next_knobs.min_plausibility = clamp(next_knobs.min_plausibility + 0.02, 0.20, 0.90)
            notes.append("low_plausibility_raise_sequence_constraint")

    if panel_fill < 0.75:
        next_knobs.proposal_total = min(max_proposal_total, int(round(next_knobs.proposal_total * 1.12)))
        next_knobs.tau_func_amber = clamp(next_knobs.tau_func_amber - 0.02, 0.20, 0.88)
        next_knobs.min_binding = clamp(next_knobs.min_binding - 0.01, 0.20, 0.85)
        if reject_plaus_frac >= 0.25:
            next_knobs.min_plausibility = clamp(min(next_knobs.min_plausibility, selector_effective_min_plaus) - 0.02, 0.05, 0.90)
            notes.append("panel_underfilled_expand_loosen_plausibility_gate")
        else:
            notes.append("panel_underfilled_expand_and_loosen_explore_gate")

    if hvi_max > 0.02 and panel_fill > 0.90:
        next_knobs.proposal_total = min(max_proposal_total, int(round(next_knobs.proposal_total * 1.05)))
        notes.append("high_value_of_information_expand_budget")

    if not notes:
        next_knobs.min_function = clamp(next_knobs.min_function + 0.005, 0.20, 0.90)
        next_knobs.min_binding = clamp(next_knobs.min_binding + 0.005, 0.20, 0.85)
        next_knobs.min_stability = clamp(next_knobs.min_stability + 0.005, 0.20, 0.90)
        if plausibility_signal_collapsed:
            target_min = min(next_knobs.min_plausibility, selector_effective_min_plaus + 0.01)
            next_knobs.min_plausibility = clamp(target_min, 0.05, 0.90)
            notes.append("stable_iteration_mild_anneal_plausibility_hold")
        else:
            next_knobs.min_plausibility = clamp(next_knobs.min_plausibility + 0.005, 0.20, 0.90)
            notes.append("stable_iteration_mild_anneal")

    if next_knobs.tau_func_amber >= (next_knobs.tau_func_green - 0.05):
        next_knobs.tau_func_amber = clamp(next_knobs.tau_func_green - 0.05, 0.20, 0.88)

    next_knobs.proposal_total = max(80, int(next_knobs.proposal_total))
    next_knobs.proposal_max_per_position = max(2, int(next_knobs.proposal_max_per_position))
    next_knobs.panel_max_per_position = max(1, int(next_knobs.panel_max_per_position))
    return next_knobs, notes


def adapt_selector_knobs(
    selector_knobs: AdaptiveSelectorKnobs,
    metrics: Dict[str, Any],
    panel_total: int,
) -> Tuple[AdaptiveSelectorKnobs, List[str]]:
    next_knobs = deepcopy(selector_knobs)
    notes: List[str] = []

    panel_fill = safe_float(metrics.get("panel_fill"), 0.0)
    panel_func = safe_float(metrics.get("panel_func_mean"), 0.0)
    panel_bind = safe_float(metrics.get("panel_bind_mean"), 0.0)
    qualified_ratio = safe_float(metrics.get("selector_qualified_ratio"), 0.0)
    dominant_role_frac = safe_float(metrics.get("selector_dominant_role_fraction"), 0.0)
    red_fraction = safe_float(metrics.get("selector_red_fraction"), 0.0)
    minima_relaxed = bool(metrics.get("selector_minima_relaxed"))
    selected_binding_ch = int(max(0.0, safe_float(metrics.get("selector_selected_binding_challengers"), 0.0)))
    selected_chem_ch = int(max(0.0, safe_float(metrics.get("selector_selected_chemistry_challengers"), 0.0)))

    # Primary guardrail: recover function first.
    if panel_func < 0.55:
        next_knobs.exploit_frac = clamp(next_knobs.exploit_frac + 0.08, 0.45, 0.85)
        next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func + 0.05, 0.50, 0.80)
        next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac - 0.03, 0.02, 0.20)
        notes.append("selector_func_guardrail_raise_exploit_reduce_chemistry")
    elif panel_func > 0.68 and panel_fill > 0.95:
        next_knobs.exploit_frac = clamp(next_knobs.exploit_frac - 0.04, 0.45, 0.85)
        next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func - 0.02, 0.50, 0.80)
        next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac + 0.01, 0.02, 0.20)
        notes.append("selector_high_func_release_more_explore")

    # If we are underfilled or close to qualification bottleneck, loosen exploit floor modestly.
    if panel_fill < 0.95 or qualified_ratio < 1.35:
        next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func - 0.04, 0.50, 0.80)
        next_knobs.binding_challenger_frac = clamp(next_knobs.binding_challenger_frac + 0.02, 0.05, 0.25)
        if panel_func >= 0.55:
            next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac + 0.01, 0.02, 0.20)
        notes.append("selector_underfill_relax_exploit_floor")

    # Reduce source-role collapse when function is acceptable.
    if dominant_role_frac > 0.65 and panel_func >= 0.55:
        next_knobs.exploit_frac = clamp(next_knobs.exploit_frac - 0.03, 0.45, 0.85)
        next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac + 0.03, 0.02, 0.20)
        notes.append("selector_role_skew_expand_chemistry_diversity")

    # If selector fell back/relaxed minima, bias toward stronger exploit quality next round.
    if minima_relaxed:
        next_knobs.exploit_frac = clamp(next_knobs.exploit_frac + 0.03, 0.45, 0.85)
        next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func + 0.02, 0.50, 0.80)
        notes.append("selector_minima_relaxed_reinforce_exploit_quality")

    # If too many selected variants are red, auto-bias toward higher-quality exploitation.
    if red_fraction > 0.50:
        next_knobs.exploit_frac = clamp(next_knobs.exploit_frac + 0.05, 0.45, 0.85)
        next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func + 0.03, 0.50, 0.80)
        next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac - 0.02, 0.02, 0.20)
        notes.append("selector_high_red_fraction_raise_quality_pressure")

    # Utility weight auto-balancing.
    w_bind = float(next_knobs.w_bind)
    w_func = float(next_knobs.w_func)
    w_stability = float(next_knobs.w_stability)
    w_plaus = float(next_knobs.w_plausibility)
    w_novel = float(next_knobs.w_novel)
    if panel_func < 0.55:
        w_func += 0.04
        w_bind -= 0.02
        w_novel -= 0.02
        notes.append("selector_weights_shift_to_function")
    elif panel_bind < 0.25 and panel_func >= 0.60:
        w_bind += 0.03
        w_novel -= 0.02
        w_plaus -= 0.01
        notes.append("selector_weights_shift_to_binding")
    w_bind = max(0.05, w_bind)
    w_func = max(0.15, w_func)
    w_stability = max(0.10, w_stability)
    w_plaus = max(0.05, w_plaus)
    w_novel = max(0.02, w_novel)
    w_sum = w_bind + w_func + w_stability + w_plaus + w_novel
    next_knobs.w_bind = float(w_bind / w_sum)
    next_knobs.w_func = float(w_func / w_sum)
    next_knobs.w_stability = float(w_stability / w_sum)
    next_knobs.w_plausibility = float(w_plaus / w_sum)
    next_knobs.w_novel = float(w_novel / w_sum)

    # Keep challenger pressure in proportion to panel size and realized coverage.
    target_binding_selected = max(1, int(round(panel_total * 0.02)))
    target_chem_selected = max(1, int(round(panel_total * 0.02)))
    if selected_binding_ch < target_binding_selected:
        next_knobs.binding_challenger_frac = clamp(next_knobs.binding_challenger_frac + 0.01, 0.05, 0.25)
        notes.append("selector_binding_challenger_coverage_up")
    if selected_chem_ch > int(round(panel_total * 0.22)) and panel_func < 0.55:
        next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac - 0.02, 0.02, 0.20)
        notes.append("selector_chemistry_challenger_pressure_down")

    next_knobs.binding_challenger_min = max(1, int(round(next_knobs.binding_challenger_frac * panel_total * 0.20)))
    next_knobs.min_binding_challenger_selected = max(
        1, int(round(next_knobs.binding_challenger_frac * panel_total * 0.12))
    )
    next_knobs.min_binding_challenger_selected = min(
        next_knobs.min_binding_challenger_selected,
        next_knobs.binding_challenger_min,
    )
    next_knobs.chemistry_challenger_min = max(1, int(round(next_knobs.chemistry_challenger_frac * panel_total * 0.18)))
    next_knobs.min_chemistry_challenger_selected = max(
        1, int(round(next_knobs.chemistry_challenger_frac * panel_total * 0.10))
    )
    next_knobs.min_chemistry_challenger_selected = min(
        next_knobs.min_chemistry_challenger_selected,
        next_knobs.chemistry_challenger_min,
    )

    # Keep range-limited controls stable.
    next_knobs.exploit_frac = clamp(next_knobs.exploit_frac, 0.45, 0.85)
    next_knobs.exploit_min_func = clamp(next_knobs.exploit_min_func, 0.50, 0.80)
    next_knobs.binding_challenger_frac = clamp(next_knobs.binding_challenger_frac, 0.05, 0.25)
    next_knobs.chemistry_challenger_frac = clamp(next_knobs.chemistry_challenger_frac, 0.02, 0.20)
    next_knobs.mmr_exploit = clamp(next_knobs.mmr_exploit, 0.50, 0.95)
    next_knobs.mmr_explore = clamp(next_knobs.mmr_explore, 0.40, 0.90)
    next_knobs.mmr_repeat_penalty = max(0.0, float(next_knobs.mmr_repeat_penalty))
    next_knobs.explore_new_position_bonus = max(0.0, float(next_knobs.explore_new_position_bonus))
    return next_knobs, notes


def run_round_pipeline(
    outdir: Path,
    py: str,
    round_id: int,
    focus_round: int,
    knobs: AdaptiveKnobs,
    selector_knobs: AdaptiveSelectorKnobs,
    args: argparse.Namespace,
    panel_total_current: int,
) -> bool:
    swarm_root(outdir).mkdir(parents=True, exist_ok=True)

    train_labels_path = proposals_vespag_path(outdir=outdir, round_id=int(focus_round))
    available_train_n = 0
    if focus_round >= 0 and train_labels_path.exists():
        available_train_n = len(load_jsonl(train_labels_path))

    effective_min_train_samples = int(args.min_train_samples)
    if available_train_n > 0:
        effective_min_train_samples = min(int(args.min_train_samples), int(available_train_n))
    effective_min_train_samples = max(1, int(effective_min_train_samples))

    expected_generation_config = build_generation_config(
        round_id=int(round_id),
        knobs=knobs,
        args=args,
    )
    expected_generation_config["min_train_samples"] = int(effective_min_train_samples)

    need_regen = bool(args.force_regenerate_proposals) or should_regenerate_proposals(
        outdir=outdir,
        round_id=int(round_id),
        focus_round=int(focus_round),
        expected_generation_config=expected_generation_config,
    )
    if need_regen:
        cmd_18b = [
            py,
            "scripts/swarm/18b_generate_stat_neighborhood_candidates.py",
            "--outdir", str(outdir),
            "--round", str(round_id),
            "--focus-round", str(focus_round),
            "--target-proposals", str(int(knobs.proposal_total)),
            "--max-per-position", str(int(knobs.proposal_max_per_position)),
            "--max-mutations-per-variant", str(max(1, int(args.max_mutations_per_variant))),
            "--multi-point-fraction", str(float(clamp(float(args.multi_point_fraction), 0.0, 1.0))),
            "--multi-seed-size", str(max(8, int(args.multi_seed_size))),
            "--multi-max-candidates", str(max(0, int(args.multi_max_candidates))),
            "--multi-min-position-separation", str(max(0, int(args.multi_min_position_separation))),
            "--multi-max-position-separation", str(max(0, int(args.multi_max_position_separation))),
            "--critical-blosum-min", str(int(args.critical_blosum_min)),
            "--strict-evo-conservation-threshold", str(float(args.strict_evo_conservation_threshold)),
            "--strict-evo-blosum-min", str(int(args.strict_evo_blosum_min)),
            "--functional-exploratory-blosum-min", str(int(args.functional_exploratory_blosum_min)),
            "--functional-exploratory-max-extra", str(max(0, int(args.functional_exploratory_max_extra))),
            "--functional-exploratory-ligand-shell", str(max(0.0, float(args.functional_exploratory_ligand_shell))),
            "--chemistry-coverage-max-extra", str(max(0, int(args.chemistry_coverage_max_extra))),
            "--chemistry-coverage-blosum-min", str(int(args.chemistry_coverage_blosum_min)),
            "--chemistry-coverage-distal-ligand-shell", str(max(0.0, float(args.chemistry_coverage_distal_ligand_shell))),
            "--dedupe-scope", str(args.dedupe_scope),
            "--dedupe-lookback-rounds", str(max(0, int(args.dedupe_lookback_rounds))),
            "--ensemble-models", str(int(args.ensemble_models)),
            "--ensemble-max-iter", str(int(args.ensemble_max_iter)),
            "--min-train-samples", str(int(effective_min_train_samples)),
            "--ehvi-mc", str(int(args.ehvi_mc)),
            "--min-function", str(float(knobs.min_function)),
            "--min-binding", str(float(knobs.min_binding)),
            "--min-stability", str(float(knobs.min_stability)),
            "--min-plausibility", str(float(knobs.min_plausibility)),
            "--seed", str(int(args.proposal_seed + (round_id * 7))),
        ]
        if bool(args.functional_site_hard_filter):
            cmd_18b.append("--functional-site-hard-filter")
        if bool(args.near_functional_hard_filter):
            cmd_18b.append("--near-functional-hard-filter")
        cmd_18b.append(
            "--functional-exploratory-enable"
            if bool(args.functional_exploratory_enable)
            else "--no-functional-exploratory-enable"
        )
        cmd_18b.append(
            "--chemistry-coverage-enable"
            if bool(args.chemistry_coverage_enable)
            else "--no-chemistry-coverage-enable"
        )
        cmd_18b.append(
            "--chemistry-coverage-distal-enable"
            if bool(args.chemistry_coverage_distal_enable)
            else "--no-chemistry-coverage-distal-enable"
        )
        run(cmd_18b)

    round_props = proposals_path(outdir=outdir, round_id=int(round_id))
    if not round_props.exists():
        raise SystemExit(f"Round-{round_id} proposals missing after generation step: {round_props}")

    run([py, "scripts/swarm/16a_make_vespag_mutation_file.py", "--outdir", str(outdir), "--round", str(round_id)])

    cmd_16b = [py, "scripts/swarm/16b_run_vespag_round.py", "--outdir", str(outdir), "--round", str(round_id)]
    if args.hf_home:
        cmd_16b.extend(["--hf-home", args.hf_home])
    if args.model_weights_dir:
        cmd_16b.extend(["--model-weights-dir", args.model_weights_dir])
    if bool(args.cpu_embeddings):
        cmd_16b.append("--cpu-embeddings")
    else:
        cmd_16b.append("--gpu-embeddings")
    run(cmd_16b)

    run([py, "scripts/swarm/16c_join_vespag_scores.py", "--outdir", str(outdir), "--round", str(round_id)])

    if bool(args.fast_binding_check):
        cmd_16e = [
            py,
            "scripts/swarm/16e_fast_binding_delta.py",
            "--outdir", str(outdir),
            "--round", str(round_id),
            "--workers", str(max(1, int(args.binding_workers))),
            "--gnina-cpu", str(max(1, int(args.binding_cpu_per_job))),
            "--cnn-model", str(args.binding_cnn_model),
            "--autobox-add", str(float(args.binding_autobox_add)),
            "--progress-every", str(max(1, int(args.binding_progress_every))),
            "--relax-max-iterations", str(max(1, int(args.binding_relax_max_iterations))),
            "--relax-heavy-restraint-k", str(float(args.binding_relax_heavy_restraint_k)),
            "--binding-context", str(args.binding_context),
            "--ternary-keep-resnames", str(args.binding_ternary_keep_resnames),
            "--context-blend-weight-binary", str(float(args.binding_context_blend_weight_binary)),
            "--context-blend-weight-ternary", str(float(args.binding_context_blend_weight_ternary)),
            "--context-coupling-penalty", str(float(args.binding_context_coupling_penalty)),
        ]
        cmd_16e.append("--relax-mutants" if bool(args.binding_relax_mutants) else "--no-relax-mutants")
        if args.binding_wt_pdb_binary:
            cmd_16e.extend(["--wt-pdb-binary", str(args.binding_wt_pdb_binary)])
        if args.binding_wt_pdb_ternary:
            cmd_16e.extend(["--wt-pdb-ternary", str(args.binding_wt_pdb_ternary)])
        if args.gnina_bin:
            cmd_16e.extend(["--gnina-bin", str(args.gnina_bin)])
        if args.binding_ld_library_path:
            cmd_16e.extend(["--ld-library-path", str(args.binding_ld_library_path)])
        if int(args.binding_max_variants) > 0:
            cmd_16e.extend(["--max-variants", str(int(args.binding_max_variants))])
        if bool(args.binding_score_all):
            cmd_16e.append("--score-all")
        else:
            cmd_16e.append("--no-score-all")
        run(cmd_16e)

    selector_cmd = [
        py,
        "scripts/swarm/17c_select_candidates.py",
        "--outdir", str(outdir),
        "--round", str(round_id),
        "--total", str(int(panel_total_current)),
        "--tau-func-green", str(float(knobs.tau_func_green)),
        "--tau-func-amber", str(float(knobs.tau_func_amber)),
        "--min-binding", str(float(knobs.min_binding)),
        "--min-stability", str(float(knobs.min_stability)),
        "--min-plausibility", str(float(knobs.min_plausibility)),
        "--max-per-position", str(int(knobs.panel_max_per_position)),
        "--exploit-frac", str(float(clamp(float(selector_knobs.exploit_frac), 0.0, 1.0))),
        "--exploit-min-func", str(float(clamp(float(selector_knobs.exploit_min_func), 0.0, 1.0))),
        "--w-bind", str(float(selector_knobs.w_bind)),
        "--w-func", str(float(selector_knobs.w_func)),
        "--w-stability", str(float(selector_knobs.w_stability)),
        "--w-plausibility", str(float(selector_knobs.w_plausibility)),
        "--w-novel", str(float(selector_knobs.w_novel)),
        "--mmr-exploit", str(float(clamp(float(selector_knobs.mmr_exploit), 0.0, 1.0))),
        "--mmr-explore", str(float(clamp(float(selector_knobs.mmr_explore), 0.0, 1.0))),
        "--mmr-repeat-penalty", str(float(max(0.0, float(selector_knobs.mmr_repeat_penalty)))),
        "--explore-new-position-bonus", str(float(max(0.0, float(selector_knobs.explore_new_position_bonus)))),
        "--binding-mode", str(args.binding_mode),
        "--functional-site-binding-floor", str(float(clamp(float(args.functional_site_binding_floor), 0.0, 1.0))),
        "--binding-challenger-frac", str(float(clamp(float(selector_knobs.binding_challenger_frac), 0.0, 1.0))),
        "--binding-challenger-min", str(max(0, int(selector_knobs.binding_challenger_min))),
        "--binding-challenger-max", str(max(0, int(args.binding_challenger_max))),
        "--binding-challenger-min-binding", str(float(clamp(float(args.binding_challenger_min_binding), 0.0, 1.0))),
        "--binding-challenger-uncertainty-min", str(max(0.0, float(args.binding_challenger_uncertainty_min))),
        "--binding-challenger-max-signal", str(max(0.0, float(args.binding_challenger_max_signal))),
        "--binding-challenger-min-func", str(float(clamp(float(args.binding_challenger_min_func), 0.0, 1.0))),
        "--min-binding-challenger-selected", str(max(0, int(selector_knobs.min_binding_challenger_selected))),
        "--chemistry-challenger-frac", str(float(clamp(float(selector_knobs.chemistry_challenger_frac), 0.0, 1.0))),
        "--chemistry-challenger-min", str(max(0, int(selector_knobs.chemistry_challenger_min))),
        "--chemistry-challenger-max", str(max(0, int(args.chemistry_challenger_max))),
        "--chemistry-challenger-min-binding", str(float(clamp(float(args.chemistry_challenger_min_binding), 0.0, 1.0))),
        "--chemistry-challenger-uncertainty-min", str(max(0.0, float(args.chemistry_challenger_uncertainty_min))),
        "--chemistry-challenger-max-signal", str(max(0.0, float(args.chemistry_challenger_max_signal))),
        "--min-chemistry-challenger-selected", str(max(0, int(selector_knobs.min_chemistry_challenger_selected))),
    ]
    selector_cmd.append(
        "--enable-functional-binding-challenger"
        if bool(args.enable_functional_binding_challenger)
        else "--no-enable-functional-binding-challenger"
    )
    run(selector_cmd)

    # Update policy only after panel selection so the policy can use realized selection outcomes.
    run([py, "scripts/swarm/16d_update_vespag_policy.py", "--outdir", str(outdir), "--round", str(round_id)])
    return True


def run_final_janus(outdir: Path, py: str, rounds: List[int], args: argparse.Namespace) -> None:
    janus_cmd = args.janus_cmd or os.environ.get("JANUS_CMD")
    janus_repo = args.janus_repo or os.environ.get("JANUS_REPO")
    if not janus_cmd and not janus_repo:
        raise SystemExit(
            "--with-janus-final set but Janus is not configured. "
            "Provide --janus-cmd or --janus-repo/JANUS_REPO."
        )
    rounds_spec = ",".join(str(int(r)) for r in sorted(set(int(r) for r in rounds)))
    cmd_19e = [
        py,
        "scripts/swarm/19e_run_final_janus.py",
        "--outdir", str(outdir),
        "--rounds", rounds_spec,
        "--with-janus",
    ]
    final_cap = int(args.final_max_candidates) if int(args.final_max_candidates) > 0 else int(args.global_panel_budget)
    if final_cap > 0:
        cmd_19e.extend(["--max-candidates", str(final_cap)])
    cmd_19e.extend(["--janus-panel-mode", str(args.janus_panel_mode)])
    if janus_cmd:
        cmd_19e.extend(["--janus-cmd", janus_cmd])
    if janus_repo:
        cmd_19e.extend(["--janus-repo", janus_repo])
    run(cmd_19e)


def prune_swarm_artifacts(
    outdir: Path,
    chosen_final_round: int,
    with_janus_final: bool,
    janus_panel_mode: str,
) -> Dict[str, Any]:
    swarm_dir = swarm_root(outdir=outdir)
    swarm_dir.mkdir(parents=True, exist_ok=True)

    keep_paths = {
        panel_path(outdir=outdir, round_id=int(chosen_final_round)).resolve(),
        panel_summary_path(outdir=outdir, round_id=int(chosen_final_round)).resolve(),
        (swarm_dir / "recursive_adaptive_summary.json").resolve(),
    }

    if with_janus_final:
        mode = str(janus_panel_mode).strip().lower()
        if mode == "production":
            keep_paths.add(final_swarm_panel_production_path(outdir=outdir).resolve())
            keep_paths.add(final_with_janus_production_path(outdir=outdir).resolve())
            keep_paths.add(final_with_janus_production_summary_path(outdir=outdir).resolve())
            # Keep legacy mirrors for compatibility with downstream readers.
            keep_paths.add(final_with_janus_path(outdir=outdir).resolve())
            keep_paths.add(final_with_janus_summary_path(outdir=outdir).resolve())
        else:
            keep_paths.add(final_swarm_panel_path(outdir=outdir).resolve())
            keep_paths.add(final_with_janus_path(outdir=outdir).resolve())
            keep_paths.add(final_with_janus_summary_path(outdir=outdir).resolve())
    else:
        keep_paths.add(final_swarm_panel_path(outdir=outdir).resolve())
        keep_paths.add(final_swarm_panel_summary_path(outdir=outdir).resolve())

    cleanup_globs = [
        "proposals_r*.jsonl",
        "proposals_vespag_r*.jsonl",
        "proposals_*_readable*.tsv",
        "vespag_round*_mutations.csv",
        "vespag_scores_r*.csv",
        "swarm_panel_r*.tsv",
        "swarm_panel_summary_r*.json",
        "swarm_panel_with_janus_r*.tsv",
        "swarm_panel_with_janus_summary_r*.json",
        "recursive_iteration_metrics_r*.json",
        "binding_fastdl_cache_r*.json",
        "binding_fastdl_summary_r*.json",
        "manifest_r*.json",
        "stat_model_diagnostics_r*.json",
        "janus_input*.csv",
        "janus_scores*.csv",
        "swarm_final_panel*.tsv",
        "swarm_final_panel_summary.json",
        "swarm_final_with_janus*.tsv",
        "swarm_final_with_janus*.json",
        "vespag_policy_state.json",
    ]

    removed_files = 0
    removed_dirs = 0
    protected_matches = 0

    for pat in cleanup_globs:
        for p in swarm_dir.glob(pat):
            rp = p.resolve()
            if rp in keep_paths:
                protected_matches += 1
                continue
            if p.is_file():
                p.unlink(missing_ok=True)
                removed_files += 1

    for p in swarm_dir.glob("binding_fastdl_mutants_r*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed_dirs += 1

    # Remove empty round-local mutant directories left by partial runs.
    for p in swarm_dir.glob("binding_fastdl_mutants_r*"):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass

    return {
        "mode": "minimal",
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "kept_files": sorted(str(p) for p in keep_paths if p.exists()),
        "protected_matches": int(protected_matches),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Single recursive SWARM mutation loop (unified mutation-level generator across all rounds)."
    )
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--focus-round", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--force-regenerate-proposals", action="store_true", default=False)

    ap.add_argument("--panel-total", type=int, default=200)
    ap.add_argument("--global-panel-budget", type=int, default=0,
                    help="Global cap across all round panels for concentration (0 disables global cap).")
    ap.add_argument("--final-max-candidates", type=int, default=0,
                    help="Optional final merged cap for 19e final panel (0 uses global-panel-budget or no cap).")
    ap.add_argument("--proposal-total", type=int, default=0)
    ap.add_argument("--proposal-max-per-position", type=int, default=4)
    ap.add_argument(
        "--max-mutations-per-variant",
        type=int,
        default=2,
        help="Maximum mutations per generated proposal (1 disables multi-point generation).",
    )
    ap.add_argument(
        "--multi-point-fraction",
        type=float,
        default=0.35,
        help="Target share of final per-round panel for multi-point variants.",
    )
    ap.add_argument("--multi-seed-size", type=int, default=120)
    ap.add_argument("--multi-max-candidates", type=int, default=1200)
    ap.add_argument("--multi-min-position-separation", type=int, default=1)
    ap.add_argument("--multi-max-position-separation", type=int, default=0)
    ap.add_argument("--critical-blosum-min", type=int, default=-1,
                    help="Conservativeness gate at critical/API-functional sites (BLOSUM threshold).")
    ap.add_argument("--strict-evo-conservation-threshold", type=float, default=0.95,
                    help="Conservation cutoff for strict evolution gate.")
    ap.add_argument("--strict-evo-blosum-min", type=int, default=-1,
                    help="Conservativeness gate for strict evolution out-of-family proposals (BLOSUM threshold).")
    ap.add_argument(
        "--functional-exploratory-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow bounded nonconservative exploration at functional/ligand-proximal sites.",
    )
    ap.add_argument(
        "--functional-exploratory-blosum-min",
        type=int,
        default=-2,
        help="BLOSUM floor for functional exploratory substitutions.",
    )
    ap.add_argument(
        "--functional-exploratory-max-extra",
        type=int,
        default=3,
        help="Max exploratory substitutions admitted per site.",
    )
    ap.add_argument(
        "--functional-exploratory-ligand-shell",
        type=float,
        default=8.0,
        help="Ligand distance threshold (A) to classify functional exploration eligibility.",
    )
    ap.add_argument(
        "--chemistry-coverage-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable protein-agnostic chemistry coverage substitutions during proposal generation.",
    )
    ap.add_argument(
        "--chemistry-coverage-max-extra",
        type=int,
        default=2,
        help="Maximum chemistry-coverage substitutions admitted per site.",
    )
    ap.add_argument(
        "--chemistry-coverage-blosum-min",
        type=int,
        default=-3,
        help="BLOSUM floor for chemistry-coverage substitutions.",
    )
    ap.add_argument(
        "--chemistry-coverage-distal-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow chemistry-coverage substitutions for distal but ligand-reachable sites.",
    )
    ap.add_argument(
        "--chemistry-coverage-distal-ligand-shell",
        type=float,
        default=14.0,
        help="Ligand distance threshold (A) for distal chemistry-coverage eligibility.",
    )
    ap.add_argument("--functional-site-hard-filter", action="store_true", default=False,
                    help="If set, prefilter out functional annotated sites before mutation generation.")
    ap.add_argument("--near-functional-hard-filter", action="store_true", default=False,
                    help="If set, prefilter out residues within functional-distance-min cutoff.")
    ap.add_argument(
        "--site-card-wt-mismatch-max-frac",
        type=float,
        default=0.10,
        help="Max tolerated WT mismatch fraction between site_cards and enzyme_wt.fasta.",
    )
    ap.add_argument(
        "--site-card-wt-mismatch-min-checked",
        type=int,
        default=20,
        help="Minimum checked residues before mismatch QC is enforced.",
    )
    ap.add_argument(
        "--allow-site-card-wt-mismatch",
        action="store_true",
        default=False,
        help="Continue even when site-card/fasta WT mismatch QC fails.",
    )
    ap.add_argument(
        "--dedupe-scope",
        choices=["none", "panel", "all"],
        default="panel",
        help="Mutation dedupe scope across rounds: panel-only (default), all proposals, or none.",
    )
    ap.add_argument(
        "--dedupe-lookback-rounds",
        type=int,
        default=2,
        help="Lookback window for dedupe scope (0=all previous rounds).",
    )
    ap.add_argument("--proposal-seed", type=int, default=13)
    ap.add_argument("--ensemble-models", type=int, default=5)
    ap.add_argument("--ensemble-max-iter", type=int, default=700)
    ap.add_argument("--min-train-samples", type=int, default=80)
    ap.add_argument("--ehvi-mc", type=int, default=32)
    ap.add_argument("--min-function", type=float, default=0.40)
    ap.add_argument("--min-binding", type=float, default=0.35)
    ap.add_argument("--min-stability", type=float, default=0.40)
    ap.add_argument("--min-plausibility", type=float, default=0.40)
    ap.add_argument(
        "--functional-site-binding-floor",
        type=float,
        default=0.35,
        help="Lower early binding floor for functional/ligand-proximal mutations.",
    )
    ap.add_argument(
        "--enable-functional-binding-challenger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow uncertain functional mutations to bypass strict early binding floor.",
    )
    ap.add_argument("--binding-challenger-frac", type=float, default=0.12)
    ap.add_argument("--binding-challenger-min", type=int, default=2)
    ap.add_argument("--binding-challenger-max", type=int, default=24)
    ap.add_argument("--binding-challenger-min-binding", type=float, default=0.02)
    ap.add_argument("--binding-challenger-uncertainty-min", type=float, default=0.08)
    ap.add_argument("--binding-challenger-max-signal", type=float, default=0.35)
    ap.add_argument("--binding-challenger-min-func", type=float, default=0.35)
    ap.add_argument("--min-binding-challenger-selected", type=int, default=2)
    ap.add_argument("--chemistry-challenger-frac", type=float, default=0.08)
    ap.add_argument("--chemistry-challenger-min", type=int, default=1)
    ap.add_argument("--chemistry-challenger-max", type=int, default=16)
    ap.add_argument("--chemistry-challenger-min-binding", type=float, default=0.01)
    ap.add_argument("--chemistry-challenger-uncertainty-min", type=float, default=0.05)
    ap.add_argument("--chemistry-challenger-max-signal", type=float, default=0.55)
    ap.add_argument("--min-chemistry-challenger-selected", type=int, default=1)

    ap.add_argument("--tau-func-green", type=float, default=0.70)
    ap.add_argument("--tau-func-amber", type=float, default=0.45)
    ap.add_argument("--max-per-position", type=int, default=3)
    ap.add_argument("--exploit-frac", type=float, default=0.60)
    ap.add_argument("--exploit-min-func", type=float, default=0.60)
    ap.add_argument("--w-bind", type=float, default=0.30)
    ap.add_argument("--w-func", type=float, default=0.35)
    ap.add_argument("--w-stability", type=float, default=0.20)
    ap.add_argument("--w-plausibility", type=float, default=0.10)
    ap.add_argument("--w-novel", type=float, default=0.05)
    ap.add_argument("--mmr-exploit", type=float, default=0.82)
    ap.add_argument("--mmr-explore", type=float, default=0.58)
    ap.add_argument("--mmr-repeat-penalty", type=float, default=0.04)
    ap.add_argument("--explore-new-position-bonus", type=float, default=0.05)

    ap.add_argument("--model-weights-dir", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--cpu-embeddings", action="store_true", default=True)
    ap.add_argument("--gpu-embeddings", dest="cpu_embeddings", action="store_false")

    ap.add_argument("--adaptive", action="store_true", default=True)
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    ap.add_argument("--max-proposal-total", type=int, default=2400)
    ap.add_argument("--diversity-floor", type=float, default=0.20)
    ap.add_argument("--voi-cost-threshold", type=float, default=0.0025)
    ap.add_argument("--voi-patience", type=int, default=2)
    ap.add_argument("--min-iterations", type=int, default=3)
    ap.add_argument(
        "--objective-improvement-eps",
        type=float,
        default=0.005,
        help="Minimum objective improvement to reset objective plateau counter.",
    )
    ap.add_argument(
        "--objective-patience",
        type=int,
        default=0,
        help="Optional objective-only early-stop patience in rounds (0 disables objective-only stop).",
    )
    ap.add_argument(
        "--min-budget-fraction-before-voi-stop",
        type=float,
        default=0.75,
        help="Require this consumed budget fraction before VOI early-stop can trigger when budgeted.",
    )
    ap.add_argument(
        "--quality-guardrail-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable hard guardrail stop when panel quality collapses (low function + high red).",
    )
    ap.add_argument(
        "--quality-guardrail-min-func",
        type=float,
        default=0.45,
        help="Guardrail lower bound for panel function mean.",
    )
    ap.add_argument(
        "--quality-guardrail-max-red",
        type=float,
        default=0.50,
        help="Guardrail upper bound for selected red fraction.",
    )
    ap.add_argument(
        "--quality-guardrail-patience",
        type=int,
        default=1,
        help="Number of consecutive guardrail breaches before early stop.",
    )

    ap.add_argument("--fast-binding-check", action="store_true", default=True)
    ap.add_argument("--no-fast-binding-check", dest="fast_binding_check", action="store_false")
    ap.add_argument("--gnina-bin", default=None)
    ap.add_argument("--binding-ld-library-path", default=None)
    ap.add_argument("--binding-cnn-model", default="fast")
    ap.add_argument(
        "--binding-mode",
        choices=["robust", "direct_ligand", "cofactor_coupled"],
        default="robust",
        help="Binding interpretation mode for panel selection.",
    )
    ap.add_argument("--binding-context", choices=["auto", "single", "dual"], default="dual")
    ap.add_argument("--binding-wt-pdb-binary", default=None)
    ap.add_argument("--binding-wt-pdb-ternary", default=None)
    ap.add_argument(
        "--binding-ternary-keep-resnames",
        default="",
        help="Comma-separated HETATM residue names to retain from ternary template during fast binding rescoring.",
    )
    ap.add_argument("--binding-context-blend-weight-binary", type=float, default=0.50)
    ap.add_argument("--binding-context-blend-weight-ternary", type=float, default=0.50)
    ap.add_argument("--binding-context-coupling-penalty", type=float, default=0.10)
    ap.add_argument("--binding-workers", type=int, default=4)
    ap.add_argument("--binding-cpu-per-job", type=int, default=1)
    ap.add_argument("--binding-autobox-add", type=float, default=6.0)
    ap.add_argument("--binding-progress-every", type=int, default=10,
                    help="Progress print interval for fast binding scoring stage.")
    ap.add_argument("--binding-relax-mutants", action="store_true", default=False,
                    help="Run local mutant relaxation before GNINA in fast-binding stage.")
    ap.add_argument("--no-binding-relax-mutants", dest="binding_relax_mutants", action="store_false")
    ap.add_argument("--binding-relax-max-iterations", type=int, default=120,
                    help="Max minimization iterations per mutant in fast-binding stage.")
    ap.add_argument("--binding-relax-heavy-restraint-k", type=float, default=25.0,
                    help="Heavy-atom restraint strength during mutant relaxation.")
    ap.add_argument("--binding-max-variants", type=int, default=0,
                    help="Optional cap for fast binding rescoring stage per round (0=all eligible).")
    ap.add_argument("--binding-score-all", dest="binding_score_all", action="store_true", default=True,
                    help="Score all proposals in fast binding stage, not only green/amber gate.")
    ap.add_argument("--no-binding-score-all", dest="binding_score_all", action="store_false")

    ap.add_argument("--with-janus-final", action="store_true", default=False)
    ap.add_argument(
        "--artifact-mode",
        choices=["full", "minimal"],
        default="minimal",
        help="full: keep all round/intermediate artifacts; minimal: keep only core final outputs.",
    )
    ap.add_argument(
        "--final-round-policy",
        choices=["last", "best_objective"],
        default="best_objective",
        help="Choose round used for final reporting/final Janus merge policy.",
    )
    ap.add_argument("--janus-panel-mode", choices=["all", "production"], default="production")
    ap.add_argument("--janus-cmd", default=None)
    ap.add_argument("--janus-repo", default=None)
    args = ap.parse_args()

    if int(args.start_round) < 0:
        raise SystemExit("--start-round must be >= 0.")
    if int(args.iterations) <= 0:
        raise SystemExit("--iterations must be >= 1.")

    outdir = Path(args.outdir).resolve()
    py = sys.executable
    site_card_qc = site_card_sequence_qc(
        outdir=outdir,
        mismatch_max_frac=float(clamp(float(args.site_card_wt_mismatch_max_frac), 0.0, 1.0)),
        mismatch_min_checked=max(1, int(args.site_card_wt_mismatch_min_checked)),
    )
    if not bool(site_card_qc.get("passes", True)) and not bool(args.allow_site_card_wt_mismatch):
        raise SystemExit(
            "site_cards WT mismatch QC failed. "
            f"checked={site_card_qc.get('checked')} mismatch={site_card_qc.get('mismatch')} "
            f"mismatch_fraction={site_card_qc.get('mismatch_fraction')} "
            f"max_allowed={site_card_qc.get('mismatch_max_frac')}. "
            "Regenerate site_cards for this target, or rerun with --allow-site-card-wt-mismatch to override."
        )
    if bool(site_card_qc.get("skipped")):
        print("[recursive] site-card QC skipped (missing site_cards.jsonl or enzyme_wt.fasta)")
    else:
        print(
            "[recursive] site-card QC: "
            f"checked={site_card_qc.get('checked')} "
            f"mismatch={site_card_qc.get('mismatch')} "
            f"fraction={site_card_qc.get('mismatch_fraction')} "
            f"threshold={site_card_qc.get('mismatch_max_frac')}"
        )

    proposal_total = int(args.proposal_total) if int(args.proposal_total) > 0 else max(int(args.panel_total) * 3, int(args.panel_total) + 60)
    knobs = AdaptiveKnobs(
        proposal_total=int(proposal_total),
        proposal_max_per_position=max(1, int(args.proposal_max_per_position)),
        panel_max_per_position=max(1, int(args.max_per_position)),
        tau_func_green=float(clamp(float(args.tau_func_green), 0.0, 1.0)),
        tau_func_amber=float(clamp(float(args.tau_func_amber), 0.0, 1.0)),
        min_function=float(clamp(float(args.min_function), 0.0, 1.0)),
        min_binding=float(clamp(float(args.min_binding), 0.0, 1.0)),
        min_stability=float(clamp(float(args.min_stability), 0.0, 1.0)),
        min_plausibility=float(clamp(float(args.min_plausibility), 0.0, 1.0)),
    )
    selector_knobs = AdaptiveSelectorKnobs(
        exploit_frac=float(clamp(float(args.exploit_frac), 0.0, 1.0)),
        exploit_min_func=float(clamp(float(args.exploit_min_func), 0.0, 1.0)),
        binding_challenger_frac=float(clamp(float(args.binding_challenger_frac), 0.0, 1.0)),
        binding_challenger_min=max(0, int(args.binding_challenger_min)),
        min_binding_challenger_selected=max(0, int(args.min_binding_challenger_selected)),
        chemistry_challenger_frac=float(clamp(float(args.chemistry_challenger_frac), 0.0, 1.0)),
        chemistry_challenger_min=max(0, int(args.chemistry_challenger_min)),
        min_chemistry_challenger_selected=max(0, int(args.min_chemistry_challenger_selected)),
        w_bind=float(args.w_bind),
        w_func=float(args.w_func),
        w_stability=float(args.w_stability),
        w_plausibility=float(args.w_plausibility),
        w_novel=float(args.w_novel),
        mmr_exploit=float(clamp(float(args.mmr_exploit), 0.0, 1.0)),
        mmr_explore=float(clamp(float(args.mmr_explore), 0.0, 1.0)),
        mmr_repeat_penalty=float(max(0.0, float(args.mmr_repeat_penalty))),
        explore_new_position_bonus=float(max(0.0, float(args.explore_new_position_bonus))),
    )

    history: List[Dict[str, Any]] = []
    objective_best = -1.0
    voi_plateau_count = 0
    objective_plateau_count = 0
    quality_guardrail_count = 0
    final_round = None
    global_budget = max(0, int(args.global_panel_budget))
    selected_cumulative = 0

    for i in range(int(args.iterations)):
        round_id = int(args.start_round) + i
        focus_round = int(args.focus_round) if (i == 0 and args.focus_round is not None) else (round_id - 1)
        swarm_root(outdir).mkdir(parents=True, exist_ok=True)

        if global_budget > 0:
            remaining_budget = global_budget - selected_cumulative
            if remaining_budget <= 0:
                print(f"[recursive] early-stop: global panel budget reached ({global_budget})")
                break
            rounds_left = max(1, int(args.iterations) - i)
            even_split_target = int(math.ceil(float(remaining_budget) / float(rounds_left)))
            panel_total_current = min(int(args.panel_total), int(remaining_budget), max(1, even_split_target))
        else:
            remaining_budget = None
            panel_total_current = int(args.panel_total)

        knobs_before = asdict(knobs)
        selector_knobs_before = asdict(selector_knobs)
        ran_round = run_round_pipeline(
            outdir=outdir,
            py=py,
            round_id=round_id,
            focus_round=focus_round,
            knobs=knobs,
            selector_knobs=selector_knobs,
            args=args,
            panel_total_current=panel_total_current,
        )
        if not ran_round:
            break

        metrics = summarize_round(outdir=outdir, round_id=int(round_id), panel_total=int(panel_total_current))
        metrics.update(
            {
                "round": int(round_id),
                "iteration_index": int(i),
                "focus_round": int(focus_round),
                "panel_total_requested": int(panel_total_current),
                "global_panel_budget": int(global_budget),
                "global_budget_remaining_before_round": int(remaining_budget) if remaining_budget is not None else None,
                "knobs_before": knobs_before,
                "selector_knobs_before": selector_knobs_before,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        current_obj = safe_float(metrics.get("objective_score"), 0.0)
        delta_best = current_obj - objective_best
        metrics["objective_improvement_vs_best"] = round(float(delta_best), 6)
        if current_obj > objective_best:
            objective_best = current_obj
        improve_eps = float(max(0.0, args.objective_improvement_eps))
        if delta_best > improve_eps:
            objective_plateau_count = 0
        else:
            objective_plateau_count += 1
        metrics["objective_improvement_eps"] = round(float(improve_eps), 6)
        metrics["objective_plateau_count"] = int(objective_plateau_count)

        expected_hvi_max = safe_float(metrics.get("expected_hvi_max"), 0.0)
        voi_threshold = float(max(0.0, args.voi_cost_threshold))
        metrics["voi_cost_threshold"] = round(float(voi_threshold), 10)
        metrics["voi_margin"] = round(float(expected_hvi_max - voi_threshold), 10)
        if expected_hvi_max <= voi_threshold:
            voi_plateau_count += 1
        else:
            voi_plateau_count = 0
        metrics["voi_plateau_count"] = int(voi_plateau_count)
        metrics["plateau_count"] = int(voi_plateau_count)

        guardrail_min_func = float(clamp(float(args.quality_guardrail_min_func), 0.0, 1.0))
        guardrail_max_red = float(clamp(float(args.quality_guardrail_max_red), 0.0, 1.0))
        guardrail_enabled = bool(args.quality_guardrail_enable)
        guardrail_patience = max(1, int(args.quality_guardrail_patience))
        panel_func_mean = safe_float(metrics.get("panel_func_mean"), 0.0)
        red_fraction = safe_float(metrics.get("selector_red_fraction"), 0.0)
        quality_guardrail_breached = bool(
            guardrail_enabled
            and panel_func_mean < guardrail_min_func
            and red_fraction > guardrail_max_red
        )
        if quality_guardrail_breached:
            quality_guardrail_count += 1
        else:
            quality_guardrail_count = 0
        metrics["quality_guardrail_enabled"] = bool(guardrail_enabled)
        metrics["quality_guardrail_min_func"] = round(float(guardrail_min_func), 6)
        metrics["quality_guardrail_max_red"] = round(float(guardrail_max_red), 6)
        metrics["quality_guardrail_patience"] = int(guardrail_patience)
        metrics["quality_guardrail_breached"] = bool(quality_guardrail_breached)
        metrics["quality_guardrail_count"] = int(quality_guardrail_count)

        adaptation_notes: List[str] = []
        if bool(args.adaptive) and i < (int(args.iterations) - 1):
            knobs, adaptation_notes = adapt_knobs(
                knobs=knobs,
                metrics=metrics,
                max_proposal_total=max(120, int(args.max_proposal_total)),
                diversity_floor=float(clamp(float(args.diversity_floor), 0.0, 1.0)),
            )
            selector_knobs, selector_notes = adapt_selector_knobs(
                selector_knobs=selector_knobs,
                metrics=metrics,
                panel_total=int(panel_total_current),
            )
            adaptation_notes.extend(selector_notes)
        metrics["adaptation_notes"] = adaptation_notes
        metrics["knobs_after"] = asdict(knobs)
        metrics["selector_knobs_after"] = asdict(selector_knobs)

        selected_cumulative += int(safe_float(metrics.get("panel_selected_total"), 0.0))
        metrics["global_selected_cumulative"] = int(selected_cumulative)
        metrics["global_budget_remaining_after_round"] = (
            int(max(0, global_budget - selected_cumulative)) if global_budget > 0 else None
        )
        recursive_iteration_metrics_path(outdir=outdir, round_id=int(round_id)).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2)
        )

        history.append(metrics)
        final_round = int(round_id)

        budget_fraction_used = 1.0
        min_budget_fraction_before_voi_stop = float(clamp(float(args.min_budget_fraction_before_voi_stop), 0.0, 1.0))
        if global_budget > 0:
            budget_fraction_used = float(clamp(float(selected_cumulative) / float(max(1, global_budget)), 0.0, 1.0))
        metrics["budget_fraction_used"] = round(float(budget_fraction_used), 6)
        metrics["min_budget_fraction_before_voi_stop"] = round(float(min_budget_fraction_before_voi_stop), 6)

        objective_patience = max(0, int(args.objective_patience))
        if (
            objective_patience > 0
            and i + 1 >= int(args.min_iterations)
            and objective_plateau_count >= objective_patience
            and i < (int(args.iterations) - 1)
        ):
            print(
                "[recursive] early-stop: "
                f"objective_plateau_count={objective_plateau_count} "
                f"objective_patience={objective_patience}"
            )
            break

        if (
            bool(args.quality_guardrail_enable)
            and quality_guardrail_count >= guardrail_patience
            and i < (int(args.iterations) - 1)
        ):
            print(
                "[recursive] early-stop: "
                f"quality_guardrail_breach_count={quality_guardrail_count} "
                f"panel_func_mean={panel_func_mean:.6f} "
                f"red_fraction={red_fraction:.6f} "
                f"thresholds(func<{guardrail_min_func:.3f}, red>{guardrail_max_red:.3f})"
            )
            break

        if (
            i + 1 >= int(args.min_iterations)
            and voi_plateau_count >= int(args.voi_patience)
            and objective_plateau_count >= int(args.voi_patience)
            and budget_fraction_used >= min_budget_fraction_before_voi_stop
            and i < (int(args.iterations) - 1)
        ):
            print(
                "[recursive] early-stop: "
                f"voi_plateau_count={voi_plateau_count} "
                f"objective_plateau_count={objective_plateau_count} "
                f"expected_hvi_max={expected_hvi_max:.6f} "
                f"threshold={voi_threshold:.6f}"
            )
            break

        if global_budget > 0 and selected_cumulative >= global_budget and i < (int(args.iterations) - 1):
            print(f"[recursive] early-stop: global panel budget exhausted ({selected_cumulative}/{global_budget})")
            break

    if final_round is None:
        raise SystemExit("No rounds executed.")

    chosen_final_round = int(final_round)
    if str(args.final_round_policy).strip().lower() == "best_objective" and history:
        best_entry = max(
            history,
            key=lambda h: (
                safe_float(h.get("objective_score"), -1.0),
                int(h.get("round", -1)),
            ),
        )
        chosen_final_round = int(best_entry.get("round", final_round))

    summary = {
        "run_mode": "single_recursive_swarm_v5_unified_single_loop_fast_binding",
        "started_at_utc": history[0].get("generated_at_utc") if history else None,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "start_round": int(args.start_round),
        "iterations_requested": int(args.iterations),
        "iterations_completed": int(len(history)),
        "final_round": int(chosen_final_round),
        "final_round_last_executed": int(final_round),
        "final_round_selected": int(chosen_final_round),
        "final_round_policy": str(args.final_round_policy),
        "rounds_executed": [int(h.get("round")) for h in history],
        "adaptive_enabled": bool(args.adaptive),
        "global_panel_budget": int(global_budget),
        "global_selected_cumulative": int(selected_cumulative),
        "objective_best": round(float(objective_best), 6),
        "voi_cost_threshold": float(max(0.0, args.voi_cost_threshold)),
        "voi_patience": int(max(1, args.voi_patience)),
        "quality_guardrail_enabled": bool(args.quality_guardrail_enable),
        "quality_guardrail_min_func": round(float(clamp(float(args.quality_guardrail_min_func), 0.0, 1.0)), 6),
        "quality_guardrail_max_red": round(float(clamp(float(args.quality_guardrail_max_red), 0.0, 1.0)), 6),
        "quality_guardrail_patience": int(max(1, args.quality_guardrail_patience)),
        "selector_knobs_final": asdict(selector_knobs),
        "site_card_qc": site_card_qc,
        "history": history,
    }
    summary_path = outdir / "swarm" / "recursive_adaptive_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if bool(args.with_janus_final):
        janus_rounds = [int(h.get("round")) for h in history if h.get("round") is not None]
        if str(args.final_round_policy).strip().lower() == "best_objective":
            janus_rounds = [int(chosen_final_round)]
        run_final_janus(
            outdir=outdir,
            py=py,
            rounds=janus_rounds,
            args=args,
        )

    if str(args.artifact_mode).strip().lower() == "minimal":
        prune_stats = prune_swarm_artifacts(
            outdir=outdir,
            chosen_final_round=int(chosen_final_round),
            with_janus_final=bool(args.with_janus_final),
            janus_panel_mode=str(args.janus_panel_mode),
        )
        print(
            "[recursive] artifact-prune: "
            f"removed_files={prune_stats.get('removed_files', 0)} "
            f"removed_dirs={prune_stats.get('removed_dirs', 0)}"
        )

    print("[recursive] flow complete")
    print(f"[recursive] summary: {summary_path}")
    print(f"[recursive] final panel: {panel_path(outdir=outdir, round_id=int(chosen_final_round))}")
    if bool(args.with_janus_final):
        print(f"[recursive] final with Janus: {final_with_janus_path(outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
