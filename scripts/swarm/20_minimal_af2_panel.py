import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from artifact_paths import (
        minimal_af2_candidates_path,
        minimal_af2_panel_path,
        minimal_af2_summary_path,
        swarm_root,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        minimal_af2_candidates_path,
        minimal_af2_panel_path,
        minimal_af2_summary_path,
        swarm_root,
    )

try:
    from mutation_utils import mutations_to_id
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id

try:
    from proposal_utils import load_site_cards
except ImportError:
    from scripts.swarm.proposal_utils import load_site_cards


RISKY_AAS = {"C", "G", "P"}
HYDROPHOBIC = {"A", "V", "I", "L", "M"}
AROMATIC = {"F", "Y", "W", "H"}
NEGATIVE = {"D", "E"}
POSITIVE = {"K", "R"}
POLAR = {"S", "T", "N", "Q"}
SMALL_POLAR = {"S", "T", "N", "Q", "H"}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def read_single_fasta(path: Path) -> str:
    seq_lines: List[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).strip().upper()
    if not seq:
        raise ValueError(f"WT FASTA has no sequence: {path}")
    return seq


def wt_mismatch_fraction(cards: Sequence[Dict[str, Any]], wt_seq: str) -> Tuple[int, int, float]:
    checked = 0
    mismatches = 0
    for card in cards:
        pos = int(card.get("pos") or 0)
        wt = str(card.get("wt") or "").strip().upper()
        if pos <= 0 or pos > len(wt_seq) or len(wt) != 1:
            continue
        checked += 1
        if wt_seq[pos - 1] != wt:
            mismatches += 1
    frac = (float(mismatches) / float(checked)) if checked else 0.0
    return checked, mismatches, frac


def prolif_tags(card: Dict[str, Any]) -> List[str]:
    prolif = card.get("prolif") or {}
    tags = []
    for rec in prolif.get("top_interactions") or []:
        if isinstance(rec, (list, tuple)) and rec:
            tags.append(str(rec[0]).strip().lower())
        elif isinstance(rec, str):
            tags.append(rec.strip().lower())
    return [t for t in tags if t]


def pose_contact_freq(card: Dict[str, Any]) -> float:
    prolif = card.get("prolif") or {}
    raw = card.get("ligand_pose_support")
    if raw in (None, ""):
        raw = prolif.get("contact_freq")
    return clamp(safe_float(raw, 0.0), 0.0, 1.0)


def pose_contact_uncertainty(card: Dict[str, Any]) -> float:
    raw = card.get("ligand_pose_uncertainty")
    if raw not in (None, ""):
        return clamp(safe_float(raw, 1.0), 0.0, 1.0)
    return 1.0 - pose_contact_freq(card)


def classify_site(card: Dict[str, Any], no_touch_conservation: float, direct_shell: float, first_shell: float, distal_shell: float) -> Optional[str]:
    hard = {str(x) for x in (card.get("hard_constraints") or [])}
    dist = safe_float(card.get("dist_ligand"), 999.0)
    conservation = safe_float(card.get("evolution_conservation"), float("nan"))
    if bool(card.get("do_not_mutate_hard")) or bool(card.get("functional_site")) or hard:
        return "no_touch"
    if math.isfinite(conservation) and conservation >= float(no_touch_conservation):
        return "no_touch"
    if bool(card.get("ligand_contact")) or dist <= float(direct_shell):
        return "direct_ligand"
    if dist <= float(first_shell):
        return "first_shell"
    if dist <= float(distal_shell) or bool(card.get("buried_core")) or bool(card.get("structural_lock")):
        return "distal_support"
    return None


def site_class_weight(site_class: str) -> float:
    return {
        "direct_ligand": 1.0,
        "first_shell": 0.82,
        "distal_support": 0.64,
        "no_touch": 0.0,
    }.get(str(site_class), 0.0)


def interaction_context(card: Dict[str, Any]) -> str:
    tags = prolif_tags(card)
    if any("pi" in t or "stack" in t or "cation" in t for t in tags):
        return "aromatic_contact"
    if any("hydrogen" in t or "hbond" in t or "donor" in t or "acceptor" in t for t in tags):
        return "hbond_contact"
    if bool(card.get("ligand_contact")):
        return "direct_contact"
    if safe_float(card.get("dist_ligand"), 999.0) <= 8.0:
        return "proximal_shell"
    return "structural_support"


def generation_prefilter_allows(card: Dict[str, Any], site_class: str) -> bool:
    if bool(card.get("site_prefilter_keep")):
        return True
    pose_support = pose_contact_freq(card)
    dist = safe_float(card.get("dist_ligand"), 999.0)
    if site_class == "direct_ligand" and (bool(card.get("ligand_contact")) or pose_support >= 0.15):
        return True
    if pose_support >= 0.35:
        return True
    if bool(card.get("fpocket_member")) and dist <= 6.0:
        return True
    return False


def allowed_mutations(card: Dict[str, Any], site_class: str) -> List[Tuple[str, str]]:
    wt = str(card.get("wt") or "").strip().upper()
    if wt in RISKY_AAS:
        return []

    allowed: List[Tuple[str, str]] = []
    tags = prolif_tags(card)
    evo_allowed = {str(x).strip().upper() for x in (card.get("evolution_allowed_aas") or []) if str(x).strip()}
    context = interaction_context(card)

    def add(targets: Iterable[str], chemotype: str) -> None:
        for aa in targets:
            aa = str(aa).strip().upper()
            if not aa or aa == wt or aa in RISKY_AAS:
                continue
            if evo_allowed and aa not in evo_allowed:
                continue
            tup = (aa, chemotype)
            if tup not in allowed:
                allowed.append(tup)

    if site_class == "direct_ligand":
        if wt in HYDROPHOBIC:
            add(HYDROPHOBIC, "hydrophobe_preserve")
        elif wt in AROMATIC:
            add(AROMATIC, "aromatic_preserve")
        elif wt in NEGATIVE:
            add(NEGATIVE, "charge_preserve")
        elif wt in POSITIVE:
            add(POSITIVE | {"H"}, "charge_preserve")
        elif wt in POLAR:
            add(POLAR | {"H"}, "polar_preserve")
        if context == "hbond_contact":
            add(SMALL_POLAR, "hbond_tune")
    elif site_class == "first_shell":
        if wt in HYDROPHOBIC:
            add(HYDROPHOBIC | {"F", "Y"}, "packing_tune")
        elif wt in AROMATIC:
            add(AROMATIC | {"L", "M", "Q"}, "aromatic_tune")
        elif wt in NEGATIVE:
            add(NEGATIVE | {"N", "Q"}, "hbond_tune")
        elif wt in POSITIVE:
            add(POSITIVE | {"H", "Q", "N"}, "hbond_tune")
        elif wt in POLAR:
            add(POLAR | NEGATIVE | {"H"}, "hbond_tune")
        if any("salt" in t for t in tags):
            add(NEGATIVE | POSITIVE | {"H"}, "electrostatic_tune")
    elif site_class == "distal_support":
        if wt in HYDROPHOBIC:
            add(HYDROPHOBIC, "stability_conservative")
        elif wt in AROMATIC:
            add({"F", "Y", "W"}, "stability_conservative")
        elif wt in NEGATIVE:
            add(NEGATIVE, "stability_conservative")
        elif wt in POSITIVE:
            add(POSITIVE | {"H"}, "stability_conservative")
        elif wt in POLAR:
            add(POLAR, "stability_conservative")
        else:
            add(HYDROPHOBIC | POLAR | NEGATIVE | POSITIVE, "stability_conservative")
    return allowed


def candidate_scores(card: Dict[str, Any], site_class: str, mut: str, chemotype: str, ranking_mode: str = "protein_agnostic") -> Dict[str, float]:
    wt = str(card.get("wt") or "").strip().upper()
    dist = safe_float(card.get("dist_ligand"), 999.0)
    plddt = safe_float(card.get("plddt"), 70.0)
    exposure = safe_float(card.get("exposure"), 0.5)
    conservation = safe_float(card.get("evolution_conservation"), 0.5)
    buried_core = bool(card.get("buried_core"))
    structural_lock = bool(card.get("structural_lock"))
    fpocket_conf = clamp(safe_float(card.get("fpocket_occupancy_confidence"), 0.0), 0.0, 1.0)
    prefilter_keep = bool(card.get("site_prefilter_keep"))
    proxy_functional = bool(card.get("proxy_functional_site"))
    pose_support = pose_contact_freq(card)
    pose_uncertainty = pose_contact_uncertainty(card)
    dist_functional = safe_float(card.get("dist_functional"), 999.0)

    site_weight = site_class_weight(site_class)
    protein_mode = str(ranking_mode or "protein_agnostic").strip().lower() != "ligand_context"
    plddt_term = clamp((plddt - 55.0) / 40.0, 0.0, 1.0)
    dist_term = clamp((8.0 - min(dist, 8.0)) / 8.0, 0.0, 1.0)
    exposure_mid = 1.0 - abs(exposure - 0.45) / 0.45 if math.isfinite(exposure) else 0.5
    exposure_mid = clamp(exposure_mid, 0.0, 1.0)
    conservation_penalty = clamp((conservation - 0.70) / 0.25, 0.0, 1.0) if math.isfinite(conservation) else 0.0
    func_proximity_penalty = clamp((2.5 - dist_functional) / 2.5, 0.0, 1.0) if math.isfinite(dist_functional) and dist_functional < 2.5 else 0.0

    plaus = 0.52 + (0.15 * plddt_term) + (0.08 * fpocket_conf) + (0.05 * exposure_mid)
    if protein_mode:
        if site_class == "direct_ligand":
            plaus -= 0.05
        elif site_class == "first_shell":
            plaus -= 0.02
    else:
        plaus += (0.10 * site_weight) + (0.07 * pose_support)
    if chemotype in {"hydrophobe_preserve", "aromatic_preserve", "charge_preserve", "polar_preserve", "stability_conservative"}:
        plaus += 0.08
    if chemotype in {"packing_tune", "hbond_tune", "electrostatic_tune", "aromatic_tune"}:
        plaus += 0.04
    plaus -= 0.12 * conservation_penalty
    plaus -= 0.05 * func_proximity_penalty
    plaus -= 0.05 if not prefilter_keep else 0.0
    plaus -= 0.05 if proxy_functional else 0.0
    if not protein_mode and site_class == "direct_ligand":
        plaus -= 0.04 * pose_uncertainty
    if wt in RISKY_AAS or mut in RISKY_AAS:
        plaus -= 0.20

    p_func = 0.58 + (0.12 * plddt_term) + (0.04 * exposure_mid) - (0.20 * conservation_penalty)
    if not protein_mode:
        p_func += 0.05 * pose_support
    if site_class == "direct_ligand":
        p_func -= 0.05 if protein_mode else 0.03
    if chemotype in {"hbond_tune", "electrostatic_tune", "aromatic_tune"} and site_class == "direct_ligand":
        p_func -= 0.05
    if chemotype in {"stability_conservative", "hydrophobe_preserve", "charge_preserve", "polar_preserve"}:
        p_func += 0.04
    p_func -= 0.08 if proxy_functional else 0.0
    p_func -= 0.07 * func_proximity_penalty
    p_func -= 0.05 if not prefilter_keep else 0.0

    p_stability = 0.56 + (0.15 * plddt_term) + (0.05 * exposure_mid)
    if site_class == "distal_support":
        p_stability += 0.12 if not protein_mode else 0.14
    elif protein_mode and site_class == "direct_ligand":
        p_stability -= 0.05
    elif protein_mode and site_class == "first_shell":
        p_stability -= 0.02
    if chemotype in {"stability_conservative", "packing_tune", "hydrophobe_preserve"}:
        p_stability += 0.08
    if buried_core or structural_lock:
        p_stability += 0.05
    if exposure > 0.65 and chemotype == "stability_conservative":
        p_stability -= 0.04
    p_stability -= 0.03 if proxy_functional else 0.0

    if site_class == "direct_ligand":
        ligand_align = 0.64
    elif site_class == "first_shell":
        ligand_align = 0.52
    else:
        ligand_align = 0.34
    ligand_align += 0.10 * dist_term
    ligand_align += 0.12 * pose_support
    ligand_align += 0.08 * fpocket_conf
    if chemotype in {"hbond_tune", "electrostatic_tune", "aromatic_tune"}:
        ligand_align += 0.08
    if chemotype in {"hydrophobe_preserve", "aromatic_preserve"} and dist <= 6.5:
        ligand_align += 0.06
    if chemotype == "stability_conservative" and site_class == "distal_support":
        ligand_align -= 0.04
    ligand_align -= 0.04 if not prefilter_keep else 0.0
    ligand_align -= 0.06 if proxy_functional else 0.0

    protein_agnostic = (
        0.38 * clamp(p_func, 0.0, 1.0)
        + 0.30 * clamp(plaus, 0.0, 1.0)
        + 0.32 * clamp(p_stability, 0.0, 1.0)
    )
    if protein_mode:
        triage = protein_agnostic
    else:
        triage = 0.84 * protein_agnostic + 0.16 * clamp(ligand_align, 0.0, 1.0)
    return {
        "p_func": round(clamp(p_func, 0.0, 1.0), 4),
        "p_plausibility": round(clamp(plaus, 0.0, 1.0), 4),
        "p_stability": round(clamp(p_stability, 0.0, 1.0), 4),
        "ligand_context_alignment": round(clamp(ligand_align, 0.0, 1.0), 4),
        "protein_agnostic_score": round(clamp(protein_agnostic, 0.0, 1.0), 4),
        "triage_score": round(clamp(triage, 0.0, 1.0), 4),
    }


def mutation_row(card: Dict[str, Any], site_class: str, mut: str, chemotype: str, ranking_mode: str = "protein_agnostic") -> Dict[str, Any]:
    m = {
        "chain": str(card.get("chain") or "A"),
        "pos": int(card.get("pos")),
        "wt": str(card.get("wt") or "").strip().upper(),
        "mut": mut,
    }
    scores = candidate_scores(card, site_class, mut, chemotype, ranking_mode=ranking_mode)
    tags = [str(x) for x in (card.get("tags") or [])]
    row = {
        "variant_id": mutations_to_id([m]),
        "chain": m["chain"],
        "pos": m["pos"],
        "wt": m["wt"],
        "mut": m["mut"],
        "mutation_count": 1,
        "mutations_json": json.dumps([m], ensure_ascii=False),
        "selection_lane": "single",
        "source_role": site_class,
        "site_class": site_class,
        "chemotype": chemotype,
        "generation_rule": f"{site_class}:{chemotype}",
        "api_context_tags": "|".join(sorted({*tags, *[str(x) for x in (card.get('hard_constraints') or [])]})),
        "ligand_context_tags": "|".join(sorted(set(prolif_tags(card) + [interaction_context(card)]))),
        "dist_ligand": round(safe_float(card.get("dist_ligand"), 999.0), 4),
        "ligand_contact": bool(card.get("ligand_contact")),
        "functional_site": bool(card.get("functional_site")),
        "proxy_functional_site": bool(card.get("proxy_functional_site")),
        "do_not_mutate": bool(card.get("do_not_mutate")),
        "site_prefilter_keep": bool(card.get("site_prefilter_keep")),
        "evolution_conservation": round(safe_float(card.get("evolution_conservation"), float("nan")), 6)
        if math.isfinite(safe_float(card.get("evolution_conservation"), float("nan")))
        else "",
        "site_plddt": round(safe_float(card.get("plddt"), 70.0), 4),
        "site_exposure": round(safe_float(card.get("exposure"), 0.5), 4),
        "ligand_pose_support": round(pose_contact_freq(card), 4),
        "ligand_pose_uncertainty": round(pose_contact_uncertainty(card), 4),
        "score_model": f"heuristic_context_v2:{ranking_mode}",
    }
    row.update(scores)
    return row


def pair_template_allowed(a: Dict[str, Any], b: Dict[str, Any], min_sep: int, max_sep: int) -> bool:
    class_a = str(a.get("site_class"))
    class_b = str(b.get("site_class"))
    allowed_pairs = {
        ("direct_ligand", "first_shell"),
        ("first_shell", "distal_support"),
        ("direct_ligand", "distal_support"),
    }
    pair_key = tuple(sorted((class_a, class_b)))
    if pair_key not in {tuple(sorted(x)) for x in allowed_pairs}:
        return False
    pos_a = int(a.get("pos"))
    pos_b = int(b.get("pos"))
    if pos_a == pos_b:
        return False
    sep = abs(pos_a - pos_b)
    if sep < int(min_sep):
        return False
    if int(max_sep) > 0 and sep > int(max_sep):
        return False
    if class_a == "direct_ligand" and class_b == "distal_support":
        allowed_chemo = {"hydrophobe_preserve", "aromatic_preserve", "charge_preserve", "polar_preserve", "stability_conservative"}
        if str(a.get("chemotype")) not in allowed_chemo or str(b.get("chemotype")) not in allowed_chemo:
            return False
    if str(a.get("chemotype")) in {"electrostatic_tune", "aromatic_tune"} and str(b.get("chemotype")) in {"electrostatic_tune", "aromatic_tune"}:
        return False
    return True


def combine_pair(a: Dict[str, Any], b: Dict[str, Any], ranking_mode: str = "protein_agnostic") -> Dict[str, Any]:
    muts = json.loads(a["mutations_json"]) + json.loads(b["mutations_json"])
    muts = sorted(muts, key=lambda x: (str(x["chain"]), int(x["pos"])))
    vid = mutations_to_id(muts)
    protein_agnostic = 0.52 * max(float(a["protein_agnostic_score"]), float(b["protein_agnostic_score"])) + 0.48 * (
        0.5 * float(a["protein_agnostic_score"]) + 0.5 * float(b["protein_agnostic_score"])
    )
    ligand_align = 0.5 * float(a["ligand_context_alignment"]) + 0.5 * float(b["ligand_context_alignment"])
    p_func = 0.5 * float(a["p_func"]) + 0.5 * float(b["p_func"]) - 0.03
    p_plausibility = math.sqrt(max(1e-8, float(a["p_plausibility"])) * max(1e-8, float(b["p_plausibility"])))
    p_stability = 0.5 * float(a["p_stability"]) + 0.5 * float(b["p_stability"]) - 0.02
    protein_mode = str(ranking_mode or "protein_agnostic").strip().lower() != "ligand_context"
    triage = protein_agnostic if protein_mode else (0.78 * protein_agnostic + 0.22 * ligand_align)
    return {
        "variant_id": vid,
        "chain": str(muts[0]["chain"]),
        "pos": int(muts[0]["pos"]),
        "wt": str(muts[0]["wt"]),
        "mut": str(muts[0]["mut"]),
        "mutation_count": 2,
        "mutations_json": json.dumps(muts, ensure_ascii=False),
        "selection_lane": "double",
        "source_role": f"{a['site_class']}+{b['site_class']}",
        "site_class": f"{a['site_class']}+{b['site_class']}",
        "chemotype": f"{a['chemotype']}+{b['chemotype']}",
        "generation_rule": f"pair:{a['site_class']}+{b['site_class']}",
        "api_context_tags": "|".join(sorted(set(filter(None, str(a.get("api_context_tags", "")).split("|") + str(b.get("api_context_tags", "")).split("|"))))),
        "ligand_context_tags": "|".join(sorted(set(filter(None, str(a.get("ligand_context_tags", "")).split("|") + str(b.get("ligand_context_tags", "")).split("|"))))),
        "dist_ligand": round(min(safe_float(a.get("dist_ligand"), 999.0), safe_float(b.get("dist_ligand"), 999.0)), 4),
        "ligand_contact": bool(a.get("ligand_contact")) or bool(b.get("ligand_contact")),
        "functional_site": False,
        "proxy_functional_site": bool(a.get("proxy_functional_site")) or bool(b.get("proxy_functional_site")),
        "do_not_mutate": False,
        "site_prefilter_keep": bool(a.get("site_prefilter_keep")) and bool(b.get("site_prefilter_keep")),
        "evolution_conservation": "",
        "site_plddt": round(0.5 * (safe_float(a.get("site_plddt"), 70.0) + safe_float(b.get("site_plddt"), 70.0)), 4),
        "site_exposure": round(0.5 * (safe_float(a.get("site_exposure"), 0.5) + safe_float(b.get("site_exposure"), 0.5)), 4),
        "ligand_pose_support": round(0.5 * (pose_contact_freq(a) + pose_contact_freq(b)), 4),
        "ligand_pose_uncertainty": round(0.5 * (pose_contact_uncertainty(a) + pose_contact_uncertainty(b)), 4),
        "score_model": f"heuristic_context_v2:{ranking_mode}",
        "p_func": round(clamp(p_func, 0.0, 1.0), 4),
        "p_plausibility": round(clamp(p_plausibility, 0.0, 1.0), 4),
        "p_stability": round(clamp(p_stability, 0.0, 1.0), 4),
        "ligand_context_alignment": round(clamp(ligand_align, 0.0, 1.0), 4),
        "protein_agnostic_score": round(clamp(protein_agnostic, 0.0, 1.0), 4),
        "triage_score": round(clamp(triage, 0.0, 1.0), 4),
    }


def select_top_per_position(rows: Sequence[Dict[str, Any]], max_per_position: int, limit: int) -> List[Dict[str, Any]]:
    pos_counts: Dict[Tuple[str, int], int] = {}
    selected: List[Dict[str, Any]] = []
    for row in rows:
        muts = json.loads(row["mutations_json"])
        keys = [(str(m["chain"]), int(m["pos"])) for m in muts]
        if any(pos_counts.get(k, 0) >= int(max_per_position) for k in keys):
            continue
        selected.append(row)
        for key in keys:
            pos_counts[key] = pos_counts.get(key, 0) + 1
        if int(limit) > 0 and len(selected) >= int(limit):
            break
    return selected


def row_bucket(row: Dict[str, Any]) -> str:
    site_class = str(row.get("site_class") or "")
    if "direct_ligand" in site_class:
        return "direct_ligand"
    if "first_shell" in site_class:
        return "first_shell"
    return "distal_support"


def select_balanced_rows(rows: Sequence[Dict[str, Any]], max_per_position: int, limit: int, ranking_mode: str = "protein_agnostic") -> List[Dict[str, Any]]:
    if int(limit) <= 0:
        return []
    if str(ranking_mode or "protein_agnostic").strip().lower() == "protein_agnostic":
        return select_top_per_position(rows, max_per_position=max_per_position, limit=limit)
    buckets = {"direct_ligand": [], "first_shell": [], "distal_support": []}
    for row in rows:
        buckets.setdefault(row_bucket(row), []).append(row)

    targets = {
        "direct_ligand": max(1, int(round(limit * 0.30))),
        "first_shell": max(1, int(round(limit * 0.40))),
    }
    targets["distal_support"] = max(0, int(limit) - targets["direct_ligand"] - targets["first_shell"])

    selected: List[Dict[str, Any]] = []
    pos_counts: Dict[Tuple[str, int], int] = {}
    selected_ids = set()

    def can_take(row: Dict[str, Any]) -> bool:
        vid = str(row.get("variant_id") or "")
        if not vid or vid in selected_ids:
            return False
        muts = json.loads(row["mutations_json"])
        keys = [(str(m["chain"]), int(m["pos"])) for m in muts]
        return not any(pos_counts.get(k, 0) >= int(max_per_position) for k in keys)

    def take(row: Dict[str, Any]) -> bool:
        if not can_take(row):
            return False
        selected.append(row)
        selected_ids.add(str(row["variant_id"]))
        for m in json.loads(row["mutations_json"]):
            key = (str(m["chain"]), int(m["pos"]))
            pos_counts[key] = pos_counts.get(key, 0) + 1
        return True

    for bucket in ("direct_ligand", "first_shell", "distal_support"):
        want = int(targets.get(bucket, 0))
        if want <= 0:
            continue
        for row in buckets.get(bucket, []):
            if len([x for x in selected if row_bucket(x) == bucket]) >= want:
                break
            take(row)

    for row in rows:
        if len(selected) >= int(limit):
            break
        take(row)
    return selected[: int(limit)]


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a minimal AF2-first SWARM candidate panel from site cards.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--site-cards", default=None)
    ap.add_argument("--wt-fasta", default=None)
    ap.add_argument("--max-singles", type=int, default=36)
    ap.add_argument("--max-doubles", type=int, default=24)
    ap.add_argument("--panel-size", type=int, default=48)
    ap.add_argument("--max-per-position", type=int, default=2)
    ap.add_argument("--min-p-func", type=float, default=0.45)
    ap.add_argument("--min-p-plausibility", type=float, default=0.45)
    ap.add_argument("--min-p-stability", type=float, default=0.45)
    ap.add_argument("--direct-shell", type=float, default=5.0)
    ap.add_argument("--first-shell", type=float, default=8.0)
    ap.add_argument("--distal-shell", type=float, default=12.0)
    ap.add_argument("--no-touch-conservation", type=float, default=0.98)
    ap.add_argument("--multi-min-position-separation", type=int, default=4)
    ap.add_argument("--multi-max-position-separation", type=int, default=24)
    ap.add_argument("--site-card-wt-mismatch-max-frac", type=float, default=0.10)
    ap.add_argument("--site-card-wt-mismatch-min-checked", type=int, default=20)
    ap.add_argument("--ranking-mode", choices=["protein_agnostic", "ligand_context"], default="protein_agnostic")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    site_cards_path = Path(args.site_cards) if args.site_cards else outdir / "swarm" / "site_cards.jsonl"
    wt_fasta = Path(args.wt_fasta) if args.wt_fasta else outdir / "enzyme_wt.fasta"
    panel_path = minimal_af2_panel_path(outdir)
    summary_path = minimal_af2_summary_path(outdir)
    candidates_path = minimal_af2_candidates_path(outdir)

    cards = load_site_cards(site_cards_path)
    if not cards:
        raise SystemExit(f"No site cards found at {site_cards_path}")
    wt_seq = read_single_fasta(wt_fasta)
    checked, mismatches, mismatch_frac = wt_mismatch_fraction(cards, wt_seq)
    if checked >= int(args.site_card_wt_mismatch_min_checked) and mismatch_frac > float(args.site_card_wt_mismatch_max_frac):
        raise SystemExit(
            f"Site-card WT mismatch fraction too high ({mismatch_frac:.3f}; checked={checked}, mismatches={mismatches}). "
            "Likely stale FASTA/PDB pairing."
        )

    classified_sites: List[Dict[str, Any]] = []
    site_class_counts: Dict[str, int] = {}
    for raw in cards:
        card = dict(raw)
        site_class = classify_site(
            card=card,
            no_touch_conservation=float(args.no_touch_conservation),
            direct_shell=float(args.direct_shell),
            first_shell=float(args.first_shell),
            distal_shell=float(args.distal_shell),
        )
        if site_class is None:
            continue
        card["site_class"] = site_class
        classified_sites.append(card)
        site_class_counts[site_class] = site_class_counts.get(site_class, 0) + 1

    eligible_sites = [c for c in classified_sites if c.get("site_class") != "no_touch"]
    if not eligible_sites:
        raise SystemExit("No eligible mutable sites after site classification.")

    single_rows: List[Dict[str, Any]] = []
    for card in eligible_sites:
        site_class = str(card.get("site_class"))
        if not generation_prefilter_allows(card, site_class):
            continue
        muts = allowed_mutations(card, site_class)
        for mut, chemotype in muts:
            row = mutation_row(card, site_class, mut, chemotype, ranking_mode=args.ranking_mode)
            if (
                float(row["p_func"]) < float(args.min_p_func)
                or float(row["p_plausibility"]) < float(args.min_p_plausibility)
                or float(row["p_stability"]) < float(args.min_p_stability)
            ):
                continue
            single_rows.append(row)
    single_rows.sort(
        key=lambda r: (
            float(r["triage_score"]),
            float(r["ligand_context_alignment"]),
            float(r["p_func"]),
            float(r.get("ligand_pose_support") or 0.0),
            -float(r["dist_ligand"]) if r["dist_ligand"] != "" else -999.0,
        ),
        reverse=True,
    )
    single_rows = select_balanced_rows(
        single_rows,
        max_per_position=int(args.max_per_position),
        limit=int(args.max_singles),
        ranking_mode=args.ranking_mode,
    )

    double_seed_rows = single_rows[: max(8, min(len(single_rows), int(max(12, args.max_singles // 2))))]
    double_rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for a, b in combinations(double_seed_rows, 2):
        if not pair_template_allowed(a, b, min_sep=int(args.multi_min_position_separation), max_sep=int(args.multi_max_position_separation)):
            continue
        row = combine_pair(a, b, ranking_mode=args.ranking_mode)
        vid = row["variant_id"]
        if vid in seen_pairs:
            continue
        if (
            float(row["p_func"]) < float(args.min_p_func)
            or float(row["p_plausibility"]) < float(args.min_p_plausibility)
            or float(row["p_stability"]) < float(args.min_p_stability)
        ):
            continue
        seen_pairs.add(vid)
        double_rows.append(row)
    double_rows.sort(
        key=lambda r: (
            float(r["triage_score"]),
            float(r["ligand_context_alignment"]),
            float(r["p_func"]),
            float(r.get("ligand_pose_support") or 0.0),
        ),
        reverse=True,
    )
    double_rows = select_balanced_rows(
        double_rows,
        max_per_position=int(args.max_per_position),
        limit=int(args.max_doubles),
        ranking_mode=args.ranking_mode,
    )

    all_rows = single_rows + double_rows
    all_rows.sort(
        key=lambda r: (
            float(r["triage_score"]),
            float(r["protein_agnostic_score"]),
            float(r["ligand_context_alignment"]),
            float(r["p_func"]),
            float(r.get("ligand_pose_support") or 0.0),
        ),
        reverse=True,
    )
    panel_rows = select_balanced_rows(
        all_rows,
        max_per_position=int(args.max_per_position),
        limit=int(args.panel_size),
        ranking_mode=args.ranking_mode,
    )
    if not panel_rows:
        raise SystemExit("No candidates survived minimal AF2 panel construction.")

    write_jsonl(candidates_path, all_rows)
    write_tsv(panel_path, panel_rows)

    summary = {
        "site_cards_path": str(site_cards_path),
        "wt_fasta": str(wt_fasta),
        "site_card_wt_checked": int(checked),
        "site_card_wt_mismatches": int(mismatches),
        "site_card_wt_mismatch_frac": round(float(mismatch_frac), 6),
        "site_class_counts": site_class_counts,
        "eligible_sites": len(eligible_sites),
        "single_candidates_total": len(single_rows),
        "double_candidates_total": len(double_rows),
        "all_candidates_total": len(all_rows),
        "panel_total": len(panel_rows),
        "max_singles": int(args.max_singles),
        "max_doubles": int(args.max_doubles),
        "panel_size": int(args.panel_size),
        "ranking_mode": str(args.ranking_mode),
        "min_p_func": float(args.min_p_func),
        "min_p_plausibility": float(args.min_p_plausibility),
        "min_p_stability": float(args.min_p_stability),
        "outputs": {
            "panel_tsv": str(panel_path),
            "candidates_jsonl": str(candidates_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {panel_path}")
    print(f"Wrote: {candidates_path}")
    print(f"Wrote: {summary_path}")
    print(
        f"[minimal-af2] eligible_sites={len(eligible_sites)} singles={len(single_rows)} "
        f"doubles={len(double_rows)} panel={len(panel_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
