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


def classify_site(card: Dict[str, Any], no_touch_conservation: float, direct_shell: float, first_shell: float, distal_shell: float) -> Optional[str]:
    hard = {str(x) for x in (card.get("hard_constraints") or [])}
    dist = safe_float(card.get("dist_ligand"), 999.0)
    conservation = safe_float(card.get("evolution_conservation"), float("nan"))
    if bool(card.get("do_not_mutate")) or bool(card.get("functional_site")) or hard:
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


def candidate_scores(card: Dict[str, Any], site_class: str, mut: str, chemotype: str) -> Dict[str, float]:
    wt = str(card.get("wt") or "").strip().upper()
    dist = safe_float(card.get("dist_ligand"), 999.0)
    plddt = safe_float(card.get("plddt"), 70.0)
    exposure = safe_float(card.get("exposure"), 0.5)
    conservation = safe_float(card.get("evolution_conservation"), 0.5)
    buried_core = bool(card.get("buried_core"))
    structural_lock = bool(card.get("structural_lock"))

    site_weight = site_class_weight(site_class)
    plddt_term = clamp((plddt - 55.0) / 40.0, 0.0, 1.0)
    conservation_penalty = clamp((conservation - 0.70) / 0.25, 0.0, 1.0) if math.isfinite(conservation) else 0.0

    plaus = 0.55 + (0.18 * plddt_term) + (0.15 * site_weight)
    if chemotype in {"hydrophobe_preserve", "aromatic_preserve", "charge_preserve", "polar_preserve", "stability_conservative"}:
        plaus += 0.08
    if chemotype in {"packing_tune", "hbond_tune", "electrostatic_tune", "aromatic_tune"}:
        plaus += 0.03
    plaus -= 0.14 * conservation_penalty
    if wt in RISKY_AAS or mut in RISKY_AAS:
        plaus -= 0.20

    p_func = 0.62 + (0.10 * plddt_term) - (0.22 * conservation_penalty)
    if site_class == "direct_ligand":
        p_func -= 0.04
    if chemotype in {"hbond_tune", "electrostatic_tune", "aromatic_tune"} and site_class == "direct_ligand":
        p_func -= 0.05
    if chemotype in {"stability_conservative", "hydrophobe_preserve", "charge_preserve", "polar_preserve"}:
        p_func += 0.04

    p_stability = 0.58 + (0.14 * plddt_term)
    if site_class == "distal_support":
        p_stability += 0.12
    if chemotype in {"stability_conservative", "packing_tune", "hydrophobe_preserve"}:
        p_stability += 0.08
    if buried_core or structural_lock:
        p_stability += 0.05
    if exposure > 0.65 and chemotype == "stability_conservative":
        p_stability -= 0.04

    if site_class == "direct_ligand":
        ligand_align = 0.78
    elif site_class == "first_shell":
        ligand_align = 0.66
    else:
        ligand_align = 0.48
    if chemotype in {"hbond_tune", "electrostatic_tune", "aromatic_tune"}:
        ligand_align += 0.08
    if chemotype in {"hydrophobe_preserve", "aromatic_preserve"} and dist <= 6.5:
        ligand_align += 0.06
    if chemotype == "stability_conservative" and site_class == "distal_support":
        ligand_align -= 0.04

    protein_agnostic = (
        0.40 * clamp(p_func, 0.0, 1.0)
        + 0.30 * clamp(plaus, 0.0, 1.0)
        + 0.30 * clamp(p_stability, 0.0, 1.0)
    )
    triage = 0.80 * protein_agnostic + 0.20 * clamp(ligand_align, 0.0, 1.0)
    return {
        "p_func": round(clamp(p_func, 0.0, 1.0), 6),
        "p_plausibility": round(clamp(plaus, 0.0, 1.0), 6),
        "p_stability": round(clamp(p_stability, 0.0, 1.0), 6),
        "ligand_context_alignment": round(clamp(ligand_align, 0.0, 1.0), 6),
        "protein_agnostic_score": round(clamp(protein_agnostic, 0.0, 1.0), 6),
        "triage_score": round(clamp(triage, 0.0, 1.0), 6),
    }


def mutation_row(card: Dict[str, Any], site_class: str, mut: str, chemotype: str) -> Dict[str, Any]:
    m = {
        "chain": str(card.get("chain") or "A"),
        "pos": int(card.get("pos")),
        "wt": str(card.get("wt") or "").strip().upper(),
        "mut": mut,
    }
    scores = candidate_scores(card, site_class, mut, chemotype)
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
        "do_not_mutate": bool(card.get("do_not_mutate")),
        "evolution_conservation": round(safe_float(card.get("evolution_conservation"), float("nan")), 6)
        if math.isfinite(safe_float(card.get("evolution_conservation"), float("nan")))
        else "",
        "site_plddt": round(safe_float(card.get("plddt"), 70.0), 4),
        "site_exposure": round(safe_float(card.get("exposure"), 0.5), 4),
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


def combine_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
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
    triage = 0.78 * protein_agnostic + 0.22 * ligand_align
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
        "do_not_mutate": False,
        "evolution_conservation": "",
        "site_plddt": round(0.5 * (safe_float(a.get("site_plddt"), 70.0) + safe_float(b.get("site_plddt"), 70.0)), 4),
        "site_exposure": round(0.5 * (safe_float(a.get("site_exposure"), 0.5) + safe_float(b.get("site_exposure"), 0.5)), 4),
        "p_func": round(clamp(p_func, 0.0, 1.0), 6),
        "p_plausibility": round(clamp(p_plausibility, 0.0, 1.0), 6),
        "p_stability": round(clamp(p_stability, 0.0, 1.0), 6),
        "ligand_context_alignment": round(clamp(ligand_align, 0.0, 1.0), 6),
        "protein_agnostic_score": round(clamp(protein_agnostic, 0.0, 1.0), 6),
        "triage_score": round(clamp(triage, 0.0, 1.0), 6),
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


def select_balanced_rows(rows: Sequence[Dict[str, Any]], max_per_position: int, limit: int) -> List[Dict[str, Any]]:
    if int(limit) <= 0:
        return []
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
        muts = allowed_mutations(card, site_class)
        for mut, chemotype in muts:
            row = mutation_row(card, site_class, mut, chemotype)
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
            -float(r["dist_ligand"]) if r["dist_ligand"] != "" else -999.0,
        ),
        reverse=True,
    )
    single_rows = select_balanced_rows(single_rows, max_per_position=int(args.max_per_position), limit=int(args.max_singles))

    double_seed_rows = single_rows[: max(8, min(len(single_rows), int(max(12, args.max_singles // 2))))]
    double_rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for a, b in combinations(double_seed_rows, 2):
        if not pair_template_allowed(a, b, min_sep=int(args.multi_min_position_separation), max_sep=int(args.multi_max_position_separation)):
            continue
        row = combine_pair(a, b)
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
        ),
        reverse=True,
    )
    double_rows = select_balanced_rows(double_rows, max_per_position=int(args.max_per_position), limit=int(args.max_doubles))

    all_rows = single_rows + double_rows
    all_rows.sort(
        key=lambda r: (
            float(r["triage_score"]),
            float(r["protein_agnostic_score"]),
            float(r["ligand_context_alignment"]),
            float(r["p_func"]),
        ),
        reverse=True,
    )
    panel_rows = select_balanced_rows(all_rows, max_per_position=int(args.max_per_position), limit=int(args.panel_size))
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
