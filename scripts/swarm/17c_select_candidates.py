import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    from artifact_paths import panel_path, panel_summary_path, proposals_vespag_path
except ImportError:
    from scripts.swarm.artifact_paths import panel_path, panel_summary_path, proposals_vespag_path

try:
    from mutation_utils import row_mutations, row_variant_id
except ImportError:
    from scripts.swarm.mutation_utils import row_mutations, row_variant_id


AA_CLASSES = {
    "hydrophobic": set("AVLIMFWY"),
    "aromatic": set("FYW"),
    "polar": set("STNQC"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "special": set("GP"),
}

BINDING_MODE_ALIASES = {
    "robust": "robust",
    "direct_ligand": "direct_ligand",
    "direct_mtx": "direct_ligand",  # backward-compatible alias
    "cofactor_coupled": "cofactor_coupled",
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def aa_class(aa: str) -> str:
    aa = (aa or "").upper()
    for k, vals in AA_CLASSES.items():
        if aa in vals:
            return k
    return "other"


def canonical_row_mutations(row: Dict) -> List[Dict]:
    return [
        {
            "chain": str(m.get("chain") or "A"),
            "pos": int(m.get("pos")),
            "wt": str(m.get("wt") or "").upper(),
            "mut": str(m.get("mut") or "").upper(),
        }
        for m in row_mutations(row)
    ]


def bind_proxy_probability(p: Dict) -> float:
    fast = safe_float(p.get("p_bind_fastdl", p.get("p_bind")), float("nan"))
    if math.isfinite(fast):
        return clamp(fast, 0.0, 1.0)

    lig_contact = 1.0 if bool(p.get("ligand_contact")) else 0.0
    dist_lig = safe_float(p.get("dist_ligand"), float("nan"))
    prolif_freq = safe_float(p.get("prolif_contact_freq"), 0.0)
    if math.isfinite(dist_lig):
        near = math.exp(-max(0.0, dist_lig) / 5.5)
    else:
        near = 0.35
    return clamp(0.10 + 0.55 * near + 0.22 * prolif_freq + 0.13 * lig_contact, 0.0, 1.0)


def normalize_binding_mode(mode: str) -> str:
    key = str(mode or "robust").strip().lower()
    return BINDING_MODE_ALIASES.get(key, "robust")


def bind_surrogate_probability(p: Dict, binding_mode: str = "robust") -> float:
    binary = safe_float(p.get("p_bind_binary_fastdl", p.get("p_bind_binary")), float("nan"))
    ternary = safe_float(p.get("p_bind_ternary_fastdl", p.get("p_bind_ternary")), float("nan"))
    blended = safe_float(p.get("p_bind_fastdl", p.get("p_bind")), float("nan"))
    proxy = bind_proxy_probability(p)

    mode = normalize_binding_mode(binding_mode)
    if mode == "direct_ligand":
        if math.isfinite(binary):
            return clamp(binary, 0.0, 1.0)
        if math.isfinite(blended):
            return clamp(blended, 0.0, 1.0)
        if math.isfinite(ternary):
            return clamp(ternary, 0.0, 1.0)
        return proxy
    if mode == "cofactor_coupled":
        if math.isfinite(ternary):
            return clamp(ternary, 0.0, 1.0)
        if math.isfinite(blended):
            return clamp(blended, 0.0, 1.0)
        if math.isfinite(binary):
            return clamp(binary, 0.0, 1.0)
        return proxy

    # robust mode: require agreement between contexts where available.
    if math.isfinite(binary) and math.isfinite(ternary):
        coupling = safe_float(p.get("p_bind_coupling_abs"), abs(binary - ternary))
        robust = (0.5 * binary) + (0.5 * ternary) - (0.10 * coupling)
        return clamp(robust, 0.0, 1.0)
    if math.isfinite(blended):
        return clamp(blended, 0.0, 1.0)
    if math.isfinite(binary):
        return clamp(binary, 0.0, 1.0)
    if math.isfinite(ternary):
        return clamp(ternary, 0.0, 1.0)
    return proxy


def mechanistic_risk(p: Dict) -> float:
    risk = 0.0
    if bool(p.get("functional_site")) or bool(p.get("critical")):
        # Soft functional annotations should discourage but not eliminate candidates.
        risk += 0.10
    hard = {str(x or "").upper() for x in (p.get("hard_constraints") or [])}
    if any(x in hard for x in ("DISULFIDE", "DISULFID", "CROSSLINK", "CROSSLNK")):
        risk += 0.40
    if any(x in hard for x in ("ACTIVE_SITE", "ACT_SITE", "METAL")):
        risk += 0.22
    if any(x in hard for x in ("CATALYTIC", "MCSA_CATALYTIC", "BINDING_SITE", "COFACTOR")):
        risk += 0.10
    if bool(p.get("buried_core")):
        risk += 0.15
    evo_cons = clamp(safe_float(p.get("evolution_conservation"), 0.5), 0.0, 1.0)
    allowed = set(p.get("evolution_allowed_aas") or [])
    mut = str(p.get("mut") or "").upper()
    if allowed and mut and mut not in allowed:
        risk += 0.12 + 0.18 * evo_cons
    return clamp(risk, 0.0, 1.0)


def prolif_contact_persistence(p: Dict) -> float:
    v = safe_float(p.get("prolif_contact_freq"), float("nan"))
    if math.isfinite(v):
        return clamp(v, 0.0, 1.0)
    if bool(p.get("ligand_contact")):
        return 0.45
    return 0.0


def prolif_retention_probability(p: Dict) -> float:
    persist = prolif_contact_persistence(p)
    p_func = clamp(
        safe_float(
            p.get("vespag_posterior", p.get("vespag_score_norm", 0.0)),
            0.0,
        ),
        0.0,
        1.0,
    )
    return clamp(0.35 + 0.45 * persist + 0.20 * p_func, 0.0, 1.0)


def acquisition_uncertainty(p: Dict) -> float:
    stat = p.get("stat_model") if isinstance(p.get("stat_model"), dict) else {}
    obj_std = stat.get("objective_std") if isinstance(stat.get("objective_std"), dict) else {}
    vals = [
        max(0.0, safe_float(obj_std.get("function"), 0.0)),
        max(0.0, safe_float(obj_std.get("binding"), 0.0)),
        max(0.0, safe_float(obj_std.get("stability"), 0.0)),
        max(0.0, safe_float(obj_std.get("plausibility"), 0.0)),
    ]
    mean_std = float(sum(vals) / float(max(1, len(vals))))
    ehvi_std = max(0.0, safe_float(stat.get("expected_hvi_std"), 0.0))
    return clamp(mean_std + (0.20 * ehvi_std), 0.0, 1.0)


def binding_signal_strength(p: Dict, binding_mode: str = "robust") -> float:
    fast = p.get("binding_fastdl") if isinstance(p.get("binding_fastdl"), dict) else {}
    mode = normalize_binding_mode(binding_mode)
    q_pairs: List[Tuple[float, float]] = []
    q_aff_binary = safe_float(fast.get("delta_affinity_quantile_binary"), float("nan"))
    q_score_binary = safe_float(fast.get("delta_score_quantile_binary"), float("nan"))
    q_aff_ternary = safe_float(fast.get("delta_affinity_quantile_ternary"), float("nan"))
    q_score_ternary = safe_float(fast.get("delta_score_quantile_ternary"), float("nan"))
    q_aff_legacy = safe_float(fast.get("delta_affinity_quantile"), float("nan"))
    q_score_legacy = safe_float(fast.get("delta_score_quantile"), float("nan"))

    if mode == "direct_ligand":
        q_pairs = [(q_aff_binary, q_score_binary), (q_aff_legacy, q_score_legacy), (q_aff_ternary, q_score_ternary)]
    elif mode == "cofactor_coupled":
        q_pairs = [(q_aff_ternary, q_score_ternary), (q_aff_legacy, q_score_legacy), (q_aff_binary, q_score_binary)]
    else:
        q_pairs = [(q_aff_binary, q_score_binary), (q_aff_ternary, q_score_ternary), (q_aff_legacy, q_score_legacy)]

    vals: List[float] = []
    for q_aff, q_score in q_pairs:
        if math.isfinite(q_aff):
            vals.append(abs(q_aff - 0.5) * 2.0)
        if math.isfinite(q_score):
            vals.append(abs(q_score - 0.5) * 2.0)
        if vals:
            break
    if vals:
        return clamp(float(max(vals)), 0.0, 1.0)

    p_bind = bind_surrogate_probability(p, binding_mode=mode)
    return clamp(abs(float(p_bind) - 0.5) * 2.0, 0.0, 1.0)


def is_noncontact_functional_probe(x: Dict) -> bool:
    dist_lig = safe_float(x.get("dist_ligand"), float("nan"))
    return bool(
        x.get("functional_site")
        and not x.get("ligand_contact")
        and math.isfinite(dist_lig)
        and dist_lig <= 7.5
    )


def challenger_priority(x: Dict) -> float:
    unc = safe_float(x.get("acq_uncertainty"), 0.0)
    signal = safe_float(x.get("binding_signal"), 0.5)
    p_func = safe_float(x.get("p_func"), 0.0)
    p_bind = safe_float(x.get("p_bind"), 0.0)
    score = float((0.45 * unc) + (0.24 * (1.0 - signal)) + (0.15 * p_func) + (0.10 * (1.0 - p_bind)))
    if is_noncontact_functional_probe(x):
        score += 0.15
    return score


def chemistry_challenger_priority(x: Dict) -> float:
    unc = safe_float(x.get("acq_uncertainty"), 0.0)
    signal = safe_float(x.get("binding_signal"), 0.5)
    p_func = safe_float(x.get("p_func"), 0.0)
    p_bind = safe_float(x.get("p_bind"), 0.0)
    score = float((0.42 * unc) + (0.24 * (1.0 - signal)) + (0.18 * p_bind) + (0.10 * (1.0 - p_func)))
    if bool(x.get("functional_focus")):
        score += 0.08
    return score


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def apply_minima(
    records: List[Dict],
    min_binding: float,
    min_stability: float,
    min_plausibility: float,
    functional_binding_floor: float,
) -> List[Dict]:
    out: List[Dict] = []
    for x in records:
        bind_floor = float(min_binding)
        if bool(x.get("functional_focus")):
            bind_floor = min(bind_floor, float(functional_binding_floor))
        if safe_float(x.get("p_bind"), 0.0) < bind_floor:
            continue
        if safe_float(x.get("p_stability"), 0.0) < float(min_stability):
            continue
        if safe_float(x.get("p_plausibility"), 0.0) < float(min_plausibility):
            continue
        out.append(x)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Select final SWARM panel using mutation-level utility (cluster-free mode).")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--proposals", default=None, help="Default: OUTDIR/swarm/proposals_vespag_rK.jsonl")
    ap.add_argument("--out", default=None, help="Default: OUTDIR/swarm/swarm_panel_rK.tsv")
    ap.add_argument("--summary", default=None, help="Default: OUTDIR/swarm/swarm_panel_summary_rK.json")

    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--tau-func-green", type=float, default=0.70)
    ap.add_argument("--tau-func-amber", type=float, default=0.45)
    ap.add_argument("--max-per-position", type=int, default=4)
    ap.add_argument("--exploit-frac", type=float, default=0.70)
    ap.add_argument(
        "--binding-mode",
        choices=["robust", "direct_ligand", "direct_mtx", "cofactor_coupled"],
        default="robust",
        help=(
            "Interpret p_bind as robust blend, direct ligand-focused binary context, "
            "or cofactor-coupled ternary context. "
            "'direct_mtx' is accepted as a backward-compatible alias."
        ),
    )
    ap.add_argument("--min-binding", type=float, default=0.0)
    ap.add_argument("--min-stability", type=float, default=0.0)
    ap.add_argument("--min-plausibility", type=float, default=0.0)
    ap.add_argument(
        "--functional-site-binding-floor",
        type=float,
        default=0.35,
        help="Binding minimum used for functional/ligand-proximal mutations during early filtering.",
    )
    ap.add_argument(
        "--enable-functional-binding-challenger",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow uncertain functional hypotheses to bypass strict early binding floors.",
    )
    ap.add_argument(
        "--binding-challenger-frac",
        type=float,
        default=0.0,
        help="Max fraction of target panel admitted as functional binding challengers.",
    )
    ap.add_argument(
        "--binding-challenger-min",
        type=int,
        default=0,
        help="Minimum number of functional binding challengers to retain when available.",
    )
    ap.add_argument(
        "--binding-challenger-max",
        type=int,
        default=24,
        help="Maximum number of functional binding challengers admitted pre-selection.",
    )
    ap.add_argument(
        "--binding-challenger-min-binding",
        type=float,
        default=0.02,
        help="Absolute lower bound for challenger p_bind.",
    )
    ap.add_argument(
        "--binding-challenger-uncertainty-min",
        type=float,
        default=0.08,
        help="Minimum acquisition uncertainty to trigger challenger admission.",
    )
    ap.add_argument(
        "--binding-challenger-max-signal",
        type=float,
        default=0.35,
        help="Maximum binding signal strength interpreted as ambiguous/low-confidence.",
    )
    ap.add_argument(
        "--binding-challenger-min-func",
        type=float,
        default=0.20,
        help="Minimum functional score for challenger admission.",
    )

    ap.add_argument("--w-bind", type=float, default=0.45)
    ap.add_argument("--w-func", type=float, default=0.25)
    ap.add_argument("--w-stability", type=float, default=0.18)
    ap.add_argument("--w-plausibility", type=float, default=0.05)
    ap.add_argument("--w-novel", type=float, default=0.07)
    ap.add_argument("--enable-red-rescue", action=argparse.BooleanOptionalAction, default=True,
                    help="Allow rescue of red-band variants when binding evidence is strong.")
    ap.add_argument("--red-rescue-min-binding", type=float, default=0.70)
    ap.add_argument("--red-rescue-min-func", type=float, default=0.35)

    ap.add_argument("--mmr-exploit", type=float, default=0.88)
    ap.add_argument("--mmr-explore", type=float, default=0.62)
    ap.add_argument("--prolif-threshold", type=float, default=0.18)
    ap.add_argument("--min-prolif-selected", type=int, default=8)
    ap.add_argument("--binding-focus-threshold", type=float, default=0.55)
    ap.add_argument("--min-binding-focused-selected", type=int, default=10)
    ap.add_argument("--min-functional-selected", type=int, default=4)
    ap.add_argument("--min-binding-challenger-selected", type=int, default=0)
    ap.add_argument("--chemistry-challenger-frac", type=float, default=0.0)
    ap.add_argument("--chemistry-challenger-min", type=int, default=0)
    ap.add_argument("--chemistry-challenger-max", type=int, default=24)
    ap.add_argument("--chemistry-challenger-min-binding", type=float, default=0.01)
    ap.add_argument("--chemistry-challenger-uncertainty-min", type=float, default=0.05)
    ap.add_argument("--chemistry-challenger-max-signal", type=float, default=0.55)
    ap.add_argument("--min-chemistry-challenger-selected", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    binding_mode = normalize_binding_mode(str(args.binding_mode))
    round_id = int(args.round)
    proposals_path = Path(args.proposals) if args.proposals else proposals_vespag_path(outdir=outdir, round_id=round_id)
    out_path = Path(args.out) if args.out else panel_path(outdir=outdir, round_id=round_id)
    summary_path = Path(args.summary) if args.summary else panel_summary_path(outdir=outdir, round_id=round_id)

    proposals = load_jsonl(proposals_path)
    if not proposals:
        raise SystemExit(f"No proposals found at {proposals_path}")

    reject_stats: Counter = Counter()
    prepared: List[Dict] = []
    pre_min_qualified: List[Dict] = []
    binding_challengers: List[Dict] = []
    chemistry_challengers: List[Dict] = []
    red_rescued_total = 0

    for p in proposals:
        muts = canonical_row_mutations(p)
        if not muts:
            reject_stats["reject_missing_fields"] += 1
            continue
        anchor = muts[0]
        wt = str(anchor.get("wt") or "").upper()
        mut = str(anchor.get("mut") or "").upper()
        chain = str(anchor.get("chain") or "A")
        pos_int = int(anchor.get("pos"))

        p_func = clamp(
            safe_float(
                p.get("vespag_posterior", p.get("vespag_score_norm", 0.0)),
                0.0,
            ),
            0.0,
            1.0,
        )
        p_bind = bind_surrogate_probability(p, binding_mode=binding_mode)
        bind_signal = binding_signal_strength(p, binding_mode=binding_mode)
        p_bind_binary = safe_float(
            p.get("p_bind_binary_fastdl", p.get("p_bind_binary", p.get("p_bind"))),
            float("nan"),
        )
        p_bind_ternary = safe_float(p.get("p_bind_ternary_fastdl", p.get("p_bind_ternary")), float("nan"))
        if not math.isfinite(p_bind_binary):
            p_bind_binary = p_bind
        p_bind_coupling_abs = safe_float(
            p.get("p_bind_coupling_abs"),
            abs(p_bind_binary - p_bind_ternary) if math.isfinite(p_bind_binary) and math.isfinite(p_bind_ternary) else 0.0,
        )

        band = str(p.get("vespag_gate_band") or "").strip().lower()
        if band not in ("green", "amber", "red"):
            band = "red"
            reject_stats["relabel_unknown_band_to_red"] += 1
        red_rescued = False
        if (
            bool(args.enable_red_rescue)
            and band == "red"
            and p_bind >= float(args.red_rescue_min_binding)
            and p_func >= float(args.red_rescue_min_func)
        ):
            band = "amber"
            red_rescued = True
            red_rescued_total += 1

        if band == "green" and p_func < float(args.tau_func_green):
            band = "amber"
            reject_stats["downgrade_green_to_amber_below_tau"] += 1
        amber_floor = float(args.red_rescue_min_func) if red_rescued else float(args.tau_func_amber)
        if band == "amber" and p_func < amber_floor:
            band = "red"
            reject_stats["downgrade_amber_to_red_below_tau"] += 1

        risk = mechanistic_risk(p)
        p_stability = clamp(1.0 - risk, 0.0, 1.0)
        p_plaus = clamp(safe_float(p.get("seq_prior_ensemble_plausibility"), 0.5), 0.0, 1.0)
        p_persist = prolif_contact_persistence(p)
        p_retain = prolif_retention_probability(p)
        dist_lig = safe_float(p.get("dist_ligand"), float("nan"))
        functional_focus = bool(p.get("functional_site")) or bool(p.get("ligand_contact")) or (
            math.isfinite(dist_lig) and dist_lig <= 6.0
        )
        unc = acquisition_uncertainty(p)
        move_primary = str(p.get("move_primary") or "").strip().lower()
        move_tags_raw = p.get("move_tags") if isinstance(p.get("move_tags"), list) else []
        move_tags = {str(x or "").strip().lower() for x in move_tags_raw if str(x or "").strip()}
        nonconservative_hint = bool(move_tags) and ("conservative" not in move_tags)
        chemistry_challenger = bool(p.get("chemistry_challenger"))
        if not chemistry_challenger:
            chemistry_challenger = bool(
                functional_focus
                and nonconservative_hint
                and move_primary in {"charge_shift", "polarity_shift", "aromatic_change", "size_shift", "multi_point"}
            )

        rec = {
            **p,
            "variant_id": row_variant_id({**p, "mutations": muts}),
            "chain": chain,
            "pos": pos_int,
            "wt": wt,
            "mut": mut,
            "mutations": muts,
            "mutation_count": int(len(muts)),
            "gate_band": band,
            "p_bind": round(p_bind, 6),
            "p_bind_binary": (round(float(p_bind_binary), 6) if math.isfinite(p_bind_binary) else None),
            "p_bind_ternary": (round(float(p_bind_ternary), 6) if math.isfinite(p_bind_ternary) else None),
            "p_bind_coupling_abs": round(float(p_bind_coupling_abs), 6),
            "p_func": round(p_func, 6),
            "p_stability": round(p_stability, 6),
            "p_plausibility": round(p_plaus, 6),
            "prolif_persist": round(p_persist, 6),
            "prolif_retention": round(p_retain, 6),
            "red_rescued": bool(red_rescued),
            "functional_focus": bool(functional_focus),
            "acq_uncertainty": round(float(unc), 6),
            "binding_signal": round(float(bind_signal), 6),
            "binding_mode": str(binding_mode),
            "binding_challenger": False,
            "chemistry_challenger": bool(chemistry_challenger),
            "_chain_positions": [(str(m["chain"]), int(m["pos"])) for m in muts],
            "_aa_class": aa_class(mut),
        }
        pre_min_qualified.append(rec)

        bind_floor = float(args.min_binding)
        if functional_focus:
            bind_floor = max(bind_floor, float(args.functional_site_binding_floor))
        if p_bind < bind_floor:
            challenger_ok = (
                bool(args.enable_functional_binding_challenger)
                and bool(functional_focus)
                and p_stability >= float(args.min_stability)
                and p_plaus >= float(args.min_plausibility)
                and p_func >= float(args.binding_challenger_min_func)
                and p_bind >= float(args.binding_challenger_min_binding)
                and (
                    unc >= float(args.binding_challenger_uncertainty_min)
                    or bind_signal <= float(args.binding_challenger_max_signal)
                )
            )
            if challenger_ok:
                rec["binding_challenger"] = True
                binding_challengers.append(rec)
                reject_stats["binding_challenger_admitted_pre"] += 1
                continue
            chemistry_challenger_ok = (
                bool(rec.get("chemistry_challenger"))
                and p_stability >= float(args.min_stability)
                and p_plaus >= float(args.min_plausibility)
                and p_bind >= float(args.chemistry_challenger_min_binding)
                and (
                    unc >= float(args.chemistry_challenger_uncertainty_min)
                    or bind_signal <= float(args.chemistry_challenger_max_signal)
                )
            )
            if chemistry_challenger_ok:
                chemistry_challengers.append(rec)
                reject_stats["chemistry_challenger_admitted_pre"] += 1
                continue
            reject_stats["reject_binding_below_min"] += 1
            continue
        if p_stability < float(args.min_stability):
            reject_stats["reject_stability_below_min"] += 1
            continue
        if p_plaus < float(args.min_plausibility):
            reject_stats["reject_plausibility_below_min"] += 1
            continue

        prepared.append(rec)

    strict_min_binding = float(args.min_binding)
    strict_min_stability = float(args.min_stability)
    strict_min_plausibility = float(args.min_plausibility)
    effective_min_binding = strict_min_binding
    effective_min_stability = strict_min_stability
    effective_min_plausibility = strict_min_plausibility
    minima_relaxed = False
    fallback_mode = "strict"
    requested_total = max(1, int(args.total))
    support_target = min(requested_total, len(pre_min_qualified))
    need_relaxation = bool(pre_min_qualified) and (
        (not prepared) or (len(prepared) < support_target)
    )

    if need_relaxation:
        arr_bind = np.asarray([safe_float(x.get("p_bind"), 0.0) for x in pre_min_qualified], dtype=float)
        arr_stab = np.asarray([safe_float(x.get("p_stability"), 0.0) for x in pre_min_qualified], dtype=float)
        arr_plaus = np.asarray([safe_float(x.get("p_plausibility"), 0.0) for x in pre_min_qualified], dtype=float)

        # Data-driven fail-soft for both empty and underfilled panels.
        base_q = 0.15 if not prepared else clamp((float(support_target) / float(max(1, len(pre_min_qualified)))) * 1.10, 0.05, 0.30)
        q_schedule = sorted({float(base_q), min(float(base_q), 0.10), min(float(base_q), 0.05), 0.02, 0.0}, reverse=True)

        best: List[Dict] = list(prepared)
        best_q = float(base_q)
        best_min_binding = float(effective_min_binding)
        best_min_stability = float(effective_min_stability)
        best_min_plausibility = float(effective_min_plausibility)

        for q in q_schedule:
            cand_min_binding = min(strict_min_binding, float(np.quantile(arr_bind, q)))
            cand_min_stability = min(strict_min_stability, float(np.quantile(arr_stab, q)))
            cand_min_plausibility = min(strict_min_plausibility, float(np.quantile(arr_plaus, q)))
            cand = apply_minima(
                pre_min_qualified,
                min_binding=cand_min_binding,
                min_stability=cand_min_stability,
                min_plausibility=cand_min_plausibility,
                functional_binding_floor=float(args.functional_site_binding_floor),
            )
            if len(cand) > len(best):
                best = cand
                best_q = float(q)
                best_min_binding = float(cand_min_binding)
                best_min_stability = float(cand_min_stability)
                best_min_plausibility = float(cand_min_plausibility)
            if len(cand) >= support_target:
                break

        prepared = best
        effective_min_binding = best_min_binding
        effective_min_stability = best_min_stability
        effective_min_plausibility = best_min_plausibility
        minima_relaxed = True
        fallback_mode = f"quantile_relax_q{int(round(best_q * 100)):02d}"
        if not prepared:
            print(
                "[selector] strict minima yielded empty set; quantile relaxation failed to recover candidates."
            )
        else:
            reason = "empty_set" if len(prepared) == 0 else ("underfilled_panel" if len(prepared) < support_target else "strict_recovery")
            print(
                "[selector] minima relaxation "
                f"({reason}): q={best_q:.3f}, "
                f"(binding={effective_min_binding:.4f}, stability={effective_min_stability:.4f}, plausibility={effective_min_plausibility:.4f}), "
                f"qualified={len(prepared)}/{support_target}"
            )

        if not prepared:
            effective_min_binding = 0.0
            effective_min_stability = 0.0
            effective_min_plausibility = 0.0
            prepared = list(pre_min_qualified)
            fallback_mode = "disable_minima_keep_band_func"
            print("[selector] quantile-relaxed minima still empty; disabling minima and keeping band+function-qualified set.")

    binding_challenger_added = 0
    if bool(args.enable_functional_binding_challenger) and binding_challengers:
        requested_total = max(1, int(args.total))
        challenger_target = int(round(float(requested_total) * clamp(float(args.binding_challenger_frac), 0.0, 1.0)))
        challenger_target = max(int(max(0, args.binding_challenger_min)), challenger_target)
        challenger_target = min(int(max(0, args.binding_challenger_max)), challenger_target)
        if challenger_target > 0:
            seen_vids = {str(x.get("variant_id") or "") for x in prepared}
            ranked_challengers = sorted(
                binding_challengers,
                key=lambda x: (
                    challenger_priority(x),
                    safe_float(x.get("p_func"), 0.0),
                    safe_float(x.get("acq_uncertainty"), 0.0),
                ),
                reverse=True,
            )
            for cand in ranked_challengers:
                if binding_challenger_added >= challenger_target:
                    break
                vid = str(cand.get("variant_id") or "")
                if not vid or vid in seen_vids:
                    continue
                prepared.append(cand)
                seen_vids.add(vid)
                binding_challenger_added += 1

    chemistry_challenger_added = 0
    if chemistry_challengers:
        requested_total = max(1, int(args.total))
        chemistry_target = int(round(float(requested_total) * clamp(float(args.chemistry_challenger_frac), 0.0, 1.0)))
        chemistry_target = max(int(max(0, args.chemistry_challenger_min)), chemistry_target)
        chemistry_target = min(int(max(0, args.chemistry_challenger_max)), chemistry_target)
        if chemistry_target > 0:
            seen_vids = {str(x.get("variant_id") or "") for x in prepared}
            ranked_chem = sorted(
                chemistry_challengers,
                key=lambda x: (
                    chemistry_challenger_priority(x),
                    safe_float(x.get("acq_uncertainty"), 0.0),
                    safe_float(x.get("p_bind"), 0.0),
                ),
                reverse=True,
            )
            for cand in ranked_chem:
                if chemistry_challenger_added >= chemistry_target:
                    break
                vid = str(cand.get("variant_id") or "")
                if not vid or vid in seen_vids:
                    continue
                prepared.append(cand)
                seen_vids.add(vid)
                chemistry_challenger_added += 1

    if not prepared:
        raise SystemExit("No candidates remain after mutation-level panel filters.")

    class_counts = Counter(x["_aa_class"] for x in prepared)
    pos_counts_all: Counter = Counter()
    for x in prepared:
        for cp in x.get("_chain_positions", []):
            pos_counts_all[tuple(cp)] += 1
    max_class = max(class_counts.values()) if class_counts else 1
    max_pos = max(pos_counts_all.values()) if pos_counts_all else 1

    for x in prepared:
        n_class = 1.0 - (class_counts.get(x["_aa_class"], 0) / max_class)
        pos_freq = 0.0
        chain_positions = list(x.get("_chain_positions", []))
        if chain_positions:
            pos_freq = float(np.mean([pos_counts_all.get(tuple(cp), 0) for cp in chain_positions]))
        n_pos = 1.0 - (pos_freq / max_pos)
        novelty = clamp(0.65 * n_class + 0.35 * n_pos, 0.0, 1.0)
        x["novelty"] = round(novelty, 6)

    comp_names = ["p_bind", "p_func", "p_stability", "p_plausibility", "novelty"]
    base_w = np.array([
        max(1e-6, float(args.w_bind)),
        max(1e-6, float(args.w_func)),
        max(1e-6, float(args.w_stability)),
        max(1e-6, float(args.w_plausibility)),
        max(1e-6, float(args.w_novel)),
    ], dtype=float)
    base_w = base_w / np.sum(base_w)

    mat = np.array([[safe_float(x.get(k), 0.0) for k in comp_names] for x in prepared], dtype=float)
    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0) + 1e-6

    for x in prepared:
        vals = np.array([safe_float(x.get(k), 0.0) for k in comp_names], dtype=float)
        z = (vals - means) / stds
        utility = float(np.dot(base_w, z))
        soft = {str(s or "").upper() for s in (x.get("soft_constraints") or [])}
        hard = {str(s or "").upper() for s in (x.get("hard_constraints") or [])}
        if "SIGNAL" in soft:
            utility -= 0.45
        dist_lig = safe_float(x.get("dist_ligand"), float("nan"))
        if math.isfinite(dist_lig) and dist_lig > 14.0 and not bool(x.get("ligand_contact")):
            utility -= 0.12
        if "MCSA_CATALYTIC" in hard and bool(x.get("red_rescued")):
            utility += 0.08
        if x.get("gate_band") == "red" and bool(x.get("functional_focus")):
            utility += 0.10
        if x.get("gate_band") == "red" and safe_float(x.get("p_bind"), 0.0) >= float(args.functional_site_binding_floor):
            utility += 0.06
        if bool(x.get("binding_challenger")):
            utility += 0.06
            utility += 0.03 * (1.0 - safe_float(x.get("binding_signal"), 0.5))
            if is_noncontact_functional_probe(x):
                utility += 0.05
        if bool(x.get("chemistry_challenger")):
            utility += 0.05
            utility += 0.04 * safe_float(x.get("acq_uncertainty"), 0.0)
            utility += 0.02 * (1.0 - safe_float(x.get("p_func"), 0.0))
        utility += 0.05 * safe_float(x.get("acq_uncertainty"), 0.0)
        x["utility"] = round(float(utility), 6)

    total = min(max(1, int(args.total)), len(prepared))
    exploit_frac = clamp(float(args.exploit_frac), 0.10, 0.95)
    exploit_target = min(total, max(0, int(round(total * exploit_frac))))
    explore_target = total - exploit_target

    # Keep an explicit exploration share for red-band functional hypotheses.
    red_functional_pool = [x for x in prepared if x["gate_band"] == "red" and bool(x.get("functional_focus"))]
    if red_functional_pool:
        reserve = min(len(red_functional_pool), max(1, int(round(0.25 * total))))
        if explore_target < reserve:
            explore_target = reserve
            exploit_target = max(0, total - explore_target)

    max_per_pos = max(1, int(args.max_per_position))

    green_pool = [x for x in prepared if x["gate_band"] == "green"]
    green_median = float(np.median([x["p_func"] for x in green_pool])) if green_pool else 0.5
    bind_focus_thr = clamp(float(args.binding_focus_threshold), 0.0, 1.0)
    def is_binding_focused(it: Dict) -> bool:
        return safe_float(it.get("p_bind"), 0.0) >= bind_focus_thr

    exploit_pool = [x for x in prepared if x["gate_band"] in ("green", "amber")]
    if not exploit_pool:
        exploit_pool = list(prepared)
    exploit_pool = sorted(exploit_pool, key=lambda x: x["utility"], reverse=True)
    explore_pool = [
        x
        for x in prepared
        if x["gate_band"] == "red"
        or bool(x.get("binding_challenger"))
        or bool(x.get("chemistry_challenger"))
        or (x["gate_band"] == "amber" and safe_float(x.get("p_bind"), 0.0) >= bind_focus_thr)
        or (x["gate_band"] == "green" and x["p_func"] <= green_median)
    ]
    if not explore_pool:
        explore_pool = [x for x in prepared if x["variant_id"] not in {y["variant_id"] for y in exploit_pool[:exploit_target]}]
    explore_pool = sorted(explore_pool, key=lambda x: x["utility"], reverse=True)

    vectors: Dict[str, np.ndarray] = {}
    for x in prepared:
        wt = (x.get("wt") or "").upper()
        mut = (x.get("mut") or "").upper()
        core = np.array([
            safe_float(x["p_bind"]),
            safe_float(x["p_func"]),
            safe_float(x["p_stability"]),
            safe_float(x["p_plausibility"]),
            safe_float(x["novelty"]),
        ], dtype=float)
        aa_flags = np.array([
            1.0 if wt in AA_CLASSES["hydrophobic"] else 0.0,
            1.0 if mut in AA_CLASSES["hydrophobic"] else 0.0,
            1.0 if wt in AA_CLASSES["positive"] else 0.0,
            1.0 if mut in AA_CLASSES["positive"] else 0.0,
            1.0 if wt in AA_CLASSES["negative"] else 0.0,
            1.0 if mut in AA_CLASSES["negative"] else 0.0,
            1.0 if wt in AA_CLASSES["aromatic"] else 0.0,
            1.0 if mut in AA_CLASSES["aromatic"] else 0.0,
        ], dtype=float)
        vectors[x["variant_id"]] = np.concatenate([core, aa_flags])

    pos_sel: Dict[Tuple[str, int], int] = Counter()
    selected: List[Dict] = []
    selected_ids: Set[str] = set()

    def fits_caps(item: Dict) -> bool:
        cps = [tuple(cp) for cp in item.get("_chain_positions", [])]
        if not cps:
            return False
        return all(pos_sel.get(cp, 0) < max_per_pos for cp in cps)

    def mmr_pick(pool: List[Dict], n_take: int, lane: str, lam: float) -> int:
        taken = 0
        while taken < n_take:
            best = None
            best_score = -1e18
            for item in pool:
                vid = item["variant_id"]
                if vid in selected_ids:
                    continue
                if not fits_caps(item):
                    continue
                vec = vectors[vid]
                if selected:
                    sim = max(cosine_similarity(vec, vectors[s["variant_id"]]) for s in selected)
                else:
                    sim = 0.0
                mmr = lam * item["utility"] + (1.0 - lam) * (1.0 - sim)
                if lane == "explore" and item["gate_band"] == "amber":
                    mmr += 0.04
                if lane == "explore" and item["gate_band"] == "red":
                    mmr += 0.06
                if lane == "explore" and bool(item.get("red_rescued")):
                    mmr += 0.06
                if lane == "explore" and bool(item.get("binding_challenger")):
                    mmr += 0.08
                if lane == "explore" and bool(item.get("chemistry_challenger")):
                    mmr += 0.09
                if lane == "explore" and bool(item.get("functional_focus")):
                    mmr += 0.05
                if lane == "explore" and safe_float(item.get("prolif_persist"), 0.0) >= float(args.prolif_threshold):
                    mmr += 0.04
                if lane == "explore":
                    mmr += 0.03 * safe_float(item.get("acq_uncertainty"), 0.0)
                if mmr > best_score:
                    best_score = mmr
                    best = item
            if best is None:
                break
            out = dict(best)
            out["selection_lane"] = lane
            selected.append(out)
            selected_ids.add(out["variant_id"])
            for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                pos_sel[cp] += 1
            taken += 1
        return taken

    exploit_taken = mmr_pick(exploit_pool, exploit_target, "exploit", clamp(float(args.mmr_exploit), 0.10, 0.98))
    explore_target = min(total - exploit_taken, explore_target + (exploit_target - exploit_taken))
    mmr_pick(explore_pool, explore_target, "explore", clamp(float(args.mmr_explore), 0.10, 0.98))

    remaining = total - len(selected)
    if remaining > 0:
        rem_pool = sorted([x for x in prepared if x["variant_id"] not in selected_ids], key=lambda x: x["utility"], reverse=True)
        mmr_pick(rem_pool, remaining, "rebalance", 0.74)

    prolif_thr = clamp(float(args.prolif_threshold), 0.0, 1.0)
    target_prolif = max(0, int(args.min_prolif_selected))
    if target_prolif > 0 and selected:
        def is_contact_informed(it: Dict) -> bool:
            return safe_float(it.get("prolif_persist"), 0.0) >= prolif_thr

        selected_prolif = sum(1 for it in selected if is_contact_informed(it))
        if selected_prolif < target_prolif:
            needed = target_prolif - selected_prolif
            candidates = [x for x in prepared if x["variant_id"] not in selected_ids and is_contact_informed(x)]
            candidates.sort(key=lambda x: x["utility"], reverse=True)
            for cand in candidates:
                if needed <= 0:
                    break
                victims = [s for s in selected if not is_contact_informed(s)]
                victims.sort(key=lambda x: x["utility"])
                replaced = False
                for victim in victims:
                    cps_v = [tuple(c) for c in victim.get("_chain_positions", [])]
                    for cp_v in cps_v:
                        pos_sel[cp_v] -= 1
                    if fits_caps(cand):
                        selected.remove(victim)
                        selected_ids.remove(victim["variant_id"])
                        out = dict(cand)
                        out["selection_lane"] = "prolif_topup"
                        selected.append(out)
                        selected_ids.add(out["variant_id"])
                        for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                            pos_sel[cp] += 1
                        needed -= 1
                        replaced = True
                        break
                    for cp_v in cps_v:
                        pos_sel[cp_v] += 1
                if not replaced:
                    continue

    target_bind_focus = max(0, int(args.min_binding_focused_selected))
    if target_bind_focus > 0 and selected:
        selected_bind_focus = sum(1 for it in selected if is_binding_focused(it))
        if selected_bind_focus < target_bind_focus:
            needed = target_bind_focus - selected_bind_focus
            candidates = [x for x in prepared if x["variant_id"] not in selected_ids and is_binding_focused(x)]
            candidates.sort(
                key=lambda x: (
                    safe_float(x.get("p_bind"), 0.0),
                    safe_float(x.get("utility"), 0.0),
                    safe_float(x.get("p_func"), 0.0),
                ),
                reverse=True,
            )
            for cand in candidates:
                if needed <= 0:
                    break
                victims = [s for s in selected if not is_binding_focused(s)]
                victims.sort(
                    key=lambda x: (
                        safe_float(x.get("p_bind"), 0.0),
                        safe_float(x.get("utility"), 0.0),
                    )
                )
                replaced = False
                for victim in victims:
                    cps_v = [tuple(c) for c in victim.get("_chain_positions", [])]
                    for cp_v in cps_v:
                        pos_sel[cp_v] -= 1
                    if fits_caps(cand):
                        selected.remove(victim)
                        selected_ids.remove(victim["variant_id"])
                        out = dict(cand)
                        out["selection_lane"] = "binding_topup"
                        selected.append(out)
                        selected_ids.add(out["variant_id"])
                        for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                            pos_sel[cp] += 1
                        needed -= 1
                        replaced = True
                        break
                    for cp_v in cps_v:
                        pos_sel[cp_v] += 1
                if not replaced:
                    continue

    target_functional = max(0, int(args.min_functional_selected))
    if target_functional > 0 and selected:
        def is_functional_focus(it: Dict) -> bool:
            return bool(it.get("functional_focus"))

        selected_functional = sum(1 for it in selected if is_functional_focus(it))
        if selected_functional < target_functional:
            needed = target_functional - selected_functional
            candidates = [x for x in prepared if x["variant_id"] not in selected_ids and is_functional_focus(x)]
            candidates.sort(
                key=lambda x: (
                    safe_float(x.get("utility"), 0.0),
                    safe_float(x.get("p_bind"), 0.0),
                    safe_float(x.get("acq_uncertainty"), 0.0),
                ),
                reverse=True,
            )
            for cand in candidates:
                if needed <= 0:
                    break
                victims = [s for s in selected if not is_functional_focus(s)]
                victims.sort(
                    key=lambda x: (
                        safe_float(x.get("utility"), 0.0),
                        safe_float(x.get("p_bind"), 0.0),
                    )
                )
                replaced = False
                for victim in victims:
                    cps_v = [tuple(c) for c in victim.get("_chain_positions", [])]
                    for cp_v in cps_v:
                        pos_sel[cp_v] -= 1
                    if fits_caps(cand):
                        selected.remove(victim)
                        selected_ids.remove(victim["variant_id"])
                        out = dict(cand)
                        out["selection_lane"] = "functional_topup"
                        selected.append(out)
                        selected_ids.add(out["variant_id"])
                        for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                            pos_sel[cp] += 1
                        needed -= 1
                        replaced = True
                        break
                    for cp_v in cps_v:
                        pos_sel[cp_v] += 1
                if not replaced:
                    continue

    target_binding_challenger = max(0, int(args.min_binding_challenger_selected))
    if target_binding_challenger > 0 and selected:
        def is_binding_challenger(it: Dict) -> bool:
            return bool(it.get("binding_challenger"))

        selected_challenger = sum(1 for it in selected if is_binding_challenger(it))
        if selected_challenger < target_binding_challenger:
            needed = target_binding_challenger - selected_challenger
            candidates = [x for x in prepared if x["variant_id"] not in selected_ids and is_binding_challenger(x)]
            candidates.sort(
                key=lambda x: (
                    safe_float(x.get("utility"), 0.0),
                    safe_float(x.get("acq_uncertainty"), 0.0),
                    safe_float(x.get("p_func"), 0.0),
                ),
                reverse=True,
            )
            for cand in candidates:
                if needed <= 0:
                    break
                bind_focus_selected = sum(1 for s in selected if is_binding_focused(s))
                victims = [s for s in selected if not is_binding_challenger(s)]
                victims.sort(
                    key=lambda x: (
                        0 if not is_binding_focused(x) else 1,
                        safe_float(x.get("utility"), 0.0),
                    )
                )
                replaced = False
                for victim in victims:
                    victim_bind = is_binding_focused(victim)
                    cand_bind = is_binding_focused(cand)
                    projected_bind = bind_focus_selected - (1 if victim_bind else 0) + (1 if cand_bind else 0)
                    if projected_bind < target_bind_focus:
                        continue
                    cps_v = [tuple(c) for c in victim.get("_chain_positions", [])]
                    for cp_v in cps_v:
                        pos_sel[cp_v] -= 1
                    if fits_caps(cand):
                        selected.remove(victim)
                        selected_ids.remove(victim["variant_id"])
                        out = dict(cand)
                        out["selection_lane"] = "challenger_topup"
                        selected.append(out)
                        selected_ids.add(out["variant_id"])
                        for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                            pos_sel[cp] += 1
                        needed -= 1
                        replaced = True
                        break
                    for cp_v in cps_v:
                        pos_sel[cp_v] += 1
                if not replaced:
                    continue

    target_chemistry_challenger = max(0, int(args.min_chemistry_challenger_selected))
    if target_chemistry_challenger > 0 and selected:
        def is_chemistry_challenger(it: Dict) -> bool:
            return bool(it.get("chemistry_challenger"))

        selected_chem = sum(1 for it in selected if is_chemistry_challenger(it))
        if selected_chem < target_chemistry_challenger:
            needed = target_chemistry_challenger - selected_chem
            candidates = [x for x in prepared if x["variant_id"] not in selected_ids and is_chemistry_challenger(x)]
            candidates.sort(
                key=lambda x: (
                    safe_float(x.get("utility"), 0.0),
                    safe_float(x.get("acq_uncertainty"), 0.0),
                    safe_float(x.get("p_bind"), 0.0),
                ),
                reverse=True,
            )
            for cand in candidates:
                if needed <= 0:
                    break
                bind_focus_selected = sum(1 for s in selected if is_binding_focused(s))
                victims = [s for s in selected if not is_chemistry_challenger(s)]
                victims.sort(
                    key=lambda x: (
                        0 if not is_binding_focused(x) else 1,
                        safe_float(x.get("utility"), 0.0),
                    )
                )
                replaced = False
                for victim in victims:
                    victim_bind = is_binding_focused(victim)
                    cand_bind = is_binding_focused(cand)
                    projected_bind = bind_focus_selected - (1 if victim_bind else 0) + (1 if cand_bind else 0)
                    if projected_bind < target_bind_focus:
                        continue
                    cps_v = [tuple(c) for c in victim.get("_chain_positions", [])]
                    for cp_v in cps_v:
                        pos_sel[cp_v] -= 1
                    if fits_caps(cand):
                        selected.remove(victim)
                        selected_ids.remove(victim["variant_id"])
                        out = dict(cand)
                        out["selection_lane"] = "chemistry_topup"
                        selected.append(out)
                        selected_ids.add(out["variant_id"])
                        for cp in [tuple(c) for c in out.get("_chain_positions", [])]:
                            pos_sel[cp] += 1
                        needed -= 1
                        replaced = True
                        break
                    for cp_v in cps_v:
                        pos_sel[cp_v] += 1
                if not replaced:
                    continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "variant_id",
                "mutation_count",
                "mutations_json",
                "chain",
                "pos",
                "wt",
                "mut",
                "tier",
                "source_role",
                "vespag_score_norm",
                "vespag_posterior",
                "vespag_shrunk_posterior",
                "vespag_gate_band",
                "vespag_gate_pass",
                "vespag_strict_pass",
                "selection_lane",
                "binding_mode",
                "p_bind",
                "p_bind_binary",
                "p_bind_ternary",
                "p_bind_coupling_abs",
                "binding_signal",
                "p_func",
                "p_stability",
                "p_plausibility",
                "prolif_persist",
                "prolif_retention",
                "red_rescued",
                "binding_challenger",
                "chemistry_challenger",
                "novelty",
                "utility",
            ]
        )
        for it in selected:
            writer.writerow(
                [
                    it["variant_id"],
                    int(it.get("mutation_count", len(it.get("mutations", [])) or 1)),
                    json.dumps(it.get("mutations") or [], ensure_ascii=False),
                    it.get("chain"),
                    it.get("pos"),
                    it.get("wt"),
                    it.get("mut"),
                    it.get("tier"),
                    it.get("source_role"),
                    it.get("vespag_score_norm"),
                    it.get("vespag_posterior"),
                    it.get("vespag_shrunk_posterior"),
                    it.get("gate_band"),
                    it.get("vespag_gate_pass"),
                    it.get("vespag_strict_pass"),
                    it.get("selection_lane"),
                    it.get("binding_mode"),
                    it.get("p_bind"),
                    it.get("p_bind_binary"),
                    it.get("p_bind_ternary"),
                    it.get("p_bind_coupling_abs"),
                    it.get("binding_signal"),
                    it.get("p_func"),
                    it.get("p_stability"),
                    it.get("p_plausibility"),
                    it.get("prolif_persist"),
                    it.get("prolif_retention"),
                    it.get("red_rescued"),
                    it.get("binding_challenger"),
                    it.get("chemistry_challenger"),
                    it.get("novelty"),
                    it.get("utility"),
                ]
            )

    summary = {
        "round": int(args.round),
        "mode": "mutation_level_panel_selector_v3_context_aware",
        "binding_mode": str(binding_mode),
        "input_total": len(proposals),
        "qualified_total": len(prepared),
        "qualified_green": len([x for x in prepared if x["gate_band"] == "green"]),
        "qualified_amber": len([x for x in prepared if x["gate_band"] == "amber"]),
        "qualified_before_minima_total": len(pre_min_qualified),
        "requested_total": int(args.total),
        "target_after_qualification": int(total),
        "selected_total": len(selected),
        "selected_green": sum(1 for x in selected if x.get("gate_band") == "green"),
        "selected_amber": sum(1 for x in selected if x.get("gate_band") == "amber"),
        "selected_bind_mean": round(float(np.mean([safe_float(x.get("p_bind"), 0.0) for x in selected])) if selected else 0.0, 6),
        "selected_func_mean": round(float(np.mean([safe_float(x.get("p_func"), 0.0) for x in selected])) if selected else 0.0, 6),
        "selected_stability_mean": round(float(np.mean([safe_float(x.get("p_stability"), 0.0) for x in selected])) if selected else 0.0, 6),
        "selected_plausibility_mean": round(float(np.mean([safe_float(x.get("p_plausibility"), 0.0) for x in selected])) if selected else 0.0, 6),
        "qualified_prolif_contacts": sum(1 for x in prepared if safe_float(x.get("prolif_persist"), 0.0) >= prolif_thr),
        "selected_prolif_contacts": sum(1 for x in selected if safe_float(x.get("prolif_persist"), 0.0) >= prolif_thr),
        "prolif_threshold": prolif_thr,
        "qualified_binding_focused": sum(1 for x in prepared if safe_float(x.get("p_bind"), 0.0) >= bind_focus_thr),
        "selected_binding_focused": sum(1 for x in selected if safe_float(x.get("p_bind"), 0.0) >= bind_focus_thr),
        "binding_focus_threshold": bind_focus_thr,
        "binding_challenger_pool_total": int(len(binding_challengers)),
        "binding_challenger_added": int(binding_challenger_added),
        "qualified_binding_challengers": sum(1 for x in prepared if bool(x.get("binding_challenger"))),
        "selected_binding_challengers": sum(1 for x in selected if bool(x.get("binding_challenger"))),
        "chemistry_challenger_pool_total": int(len(chemistry_challengers)),
        "chemistry_challenger_added": int(chemistry_challenger_added),
        "qualified_chemistry_challengers": sum(1 for x in prepared if bool(x.get("chemistry_challenger"))),
        "selected_chemistry_challengers": sum(1 for x in selected if bool(x.get("chemistry_challenger"))),
        "lane_counts": dict(Counter(x.get("selection_lane", "unknown") for x in selected)),
        "role_counts": dict(Counter(x.get("source_role") or "unknown" for x in selected)),
        "tier_counts": dict(Counter(str(x.get("tier")) for x in selected)),
        "tau_func_green": float(args.tau_func_green),
        "tau_func_amber": float(args.tau_func_amber),
        "min_binding": float(args.min_binding),
        "min_stability": float(args.min_stability),
        "min_plausibility": float(args.min_plausibility),
        "effective_min_binding": float(effective_min_binding),
        "effective_min_stability": float(effective_min_stability),
        "effective_min_plausibility": float(effective_min_plausibility),
        "minima_relaxed": bool(minima_relaxed),
        "fallback_mode": str(fallback_mode),
        "enable_red_rescue": bool(args.enable_red_rescue),
        "red_rescue_min_binding": float(args.red_rescue_min_binding),
        "red_rescue_min_func": float(args.red_rescue_min_func),
        "enable_functional_binding_challenger": bool(args.enable_functional_binding_challenger),
        "binding_challenger_frac": float(args.binding_challenger_frac),
        "binding_challenger_min": int(args.binding_challenger_min),
        "binding_challenger_max": int(args.binding_challenger_max),
        "binding_challenger_min_binding": float(args.binding_challenger_min_binding),
        "binding_challenger_uncertainty_min": float(args.binding_challenger_uncertainty_min),
        "binding_challenger_max_signal": float(args.binding_challenger_max_signal),
        "binding_challenger_min_func": float(args.binding_challenger_min_func),
        "min_binding_challenger_selected": int(args.min_binding_challenger_selected),
        "chemistry_challenger_frac": float(args.chemistry_challenger_frac),
        "chemistry_challenger_min": int(args.chemistry_challenger_min),
        "chemistry_challenger_max": int(args.chemistry_challenger_max),
        "chemistry_challenger_min_binding": float(args.chemistry_challenger_min_binding),
        "chemistry_challenger_uncertainty_min": float(args.chemistry_challenger_uncertainty_min),
        "chemistry_challenger_max_signal": float(args.chemistry_challenger_max_signal),
        "min_chemistry_challenger_selected": int(args.min_chemistry_challenger_selected),
        "red_rescued_total": int(red_rescued_total),
        "max_per_position": int(max_per_pos),
        "reject_counts": dict(reject_stats),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
