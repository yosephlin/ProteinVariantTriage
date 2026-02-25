import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from proposal_utils import AA_LIST, compact_site_card, load_site_cards
try:
    from artifact_paths import (
        panel_path,
        proposals_path,
        proposals_vespag_path,
        round_diagnostics_path,
        round_manifest_path,
        swarm_root,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        panel_path,
        proposals_path,
        proposals_vespag_path,
        round_diagnostics_path,
        round_manifest_path,
        swarm_root,
    )

try:
    from mutation_utils import mutations_to_id, row_mutations, row_variant_id
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id, row_mutations, row_variant_id


GENERATOR_TAG = "mutation_level_deep_ensemble_v4_multi_point"
OBJECTIVE_NAMES = ["function", "binding", "stability", "plausibility"]

HYDROPHOBIC_AA = set("AVLIMFWY")
AROMATIC_AA = set("FYW")
POLAR_AA = set("STNQCH")
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
SPECIAL_AA = set("GP")
METAL_COMPAT_AA = set("HDEC")

CRITICAL_CONSTRAINTS = {"ACTIVE_SITE", "ACT_SITE", "BINDING_SITE", "METAL", "DISULFID", "DISULFIDE", "CROSSLINK"}
FUNCTIONAL_CONSTRAINT_TYPES = {
    "ACTIVE_SITE",
    "ACT_SITE",
    "CATALYTIC",
    "CATALYTIC_SITE",
    "BINDING_SITE",
    "METAL",
    "COFACTOR",
    "MOTIF",
}

HARD_BLOCK_CONSTRAINTS = {
    "DISULFIDE_BOND",
    "DISULFIDE",
    "DISULFID",
    "CROSSLNK",
    "CROSSLINK",
}

# Keep prior/risk scoring aligned with the same conservativeness gates used for mutation admission.
GATE_BLOSUM_MIN = 0
GATE_CRITICAL_BLOSUM_MIN = -1
GATE_STRICT_EVO_CONSERVATION_THRESHOLD = 0.95
GATE_STRICT_EVO_BLOSUM_MIN = -1
GATE_FUNCTIONAL_EXPLORATORY_ENABLE = True
GATE_FUNCTIONAL_EXPLORATORY_BLOSUM_MIN = -2
GATE_FUNCTIONAL_EXPLORATORY_MAX_EXTRA = 3
GATE_FUNCTIONAL_EXPLORATORY_LIGAND_SHELL = 8.0
GATE_CHEMISTRY_COVERAGE_ENABLE = True
GATE_CHEMISTRY_COVERAGE_MAX_EXTRA = 2
GATE_CHEMISTRY_COVERAGE_BLOSUM_MIN = -3
GATE_CHEMISTRY_COVERAGE_DISTAL_ENABLE = True
GATE_CHEMISTRY_COVERAGE_DISTAL_LIGAND_SHELL = 14.0


BLOSUM62 = {
    "A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1, "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 0, "W": -3, "Y": -2, "V": 0},
    "R": {"A": -1, "R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0, "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3, "P": -2, "S": -1, "T": -1, "W": -3, "Y": -2, "V": -3},
    "N": {"A": -2, "R": 0, "N": 6, "D": 1, "C": -3, "Q": 0, "E": 0, "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3, "P": -2, "S": 1, "T": 0, "W": -4, "Y": -2, "V": -3},
    "D": {"A": -2, "R": -2, "N": 1, "D": 6, "C": -3, "Q": 0, "E": 2, "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3, "P": -1, "S": 0, "T": -1, "W": -4, "Y": -3, "V": -3},
    "C": {"A": 0, "R": -3, "N": -3, "D": -3, "C": 9, "Q": -3, "E": -4, "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2, "P": -3, "S": -1, "T": -1, "W": -2, "Y": -2, "V": -1},
    "Q": {"A": -1, "R": 1, "N": 0, "D": 0, "C": -3, "Q": 5, "E": 2, "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3, "P": -1, "S": 0, "T": -1, "W": -2, "Y": -1, "V": -2},
    "E": {"A": -1, "R": 0, "N": 0, "D": 2, "C": -4, "Q": 2, "E": 5, "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "G": {"A": 0, "R": -2, "N": 0, "D": -1, "C": -3, "Q": -2, "E": -2, "G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S": 0, "T": -2, "W": -2, "Y": -3, "V": -3},
    "H": {"A": -2, "R": 0, "N": 1, "D": -1, "C": -3, "Q": 0, "E": 0, "G": -2, "H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1, "P": -2, "S": -1, "T": -2, "W": -2, "Y": 2, "V": -3},
    "I": {"A": -1, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -3, "E": -3, "G": -4, "H": -3, "I": 4, "L": 2, "K": -3, "M": 1, "F": 0, "P": -3, "S": -2, "T": -1, "W": -3, "Y": -1, "V": 3},
    "L": {"A": -1, "R": -2, "N": -3, "D": -4, "C": -1, "Q": -2, "E": -3, "G": -4, "H": -3, "I": 2, "L": 4, "K": -2, "M": 2, "F": 0, "P": -3, "S": -2, "T": -1, "W": -2, "Y": -1, "V": 1},
    "K": {"A": -1, "R": 2, "N": 0, "D": -1, "C": -3, "Q": 1, "E": 1, "G": -2, "H": -1, "I": -3, "L": -2, "K": 5, "M": -1, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "M": {"A": -1, "R": -1, "N": -2, "D": -3, "C": -1, "Q": 0, "E": -2, "G": -3, "H": -2, "I": 1, "L": 2, "K": -1, "M": 5, "F": 0, "P": -2, "S": -1, "T": -1, "W": -1, "Y": -1, "V": 1},
    "F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3, "G": -3, "H": -1, "I": 0, "L": 0, "K": -3, "M": 0, "F": 6, "P": -4, "S": -2, "T": -2, "W": 1, "Y": 3, "V": -1},
    "P": {"A": -1, "R": -2, "N": -2, "D": -1, "C": -3, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -3, "L": -3, "K": -1, "M": -2, "F": -4, "P": 7, "S": -1, "T": -1, "W": -4, "Y": -3, "V": -2},
    "S": {"A": 1, "R": -1, "N": 1, "D": 0, "C": -1, "Q": 0, "E": 0, "G": 0, "H": -1, "I": -2, "L": -2, "K": 0, "M": -1, "F": -2, "P": -1, "S": 4, "T": 1, "W": -3, "Y": -2, "V": -2},
    "T": {"A": 0, "R": -1, "N": 0, "D": -1, "C": -1, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 5, "W": -2, "Y": -2, "V": 0},
    "W": {"A": -3, "R": -3, "N": -4, "D": -4, "C": -2, "Q": -2, "E": -3, "G": -2, "H": -2, "I": -3, "L": -2, "K": -3, "M": -1, "F": 1, "P": -4, "S": -3, "T": -2, "W": 11, "Y": 2, "V": -3},
    "Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2, "G": -3, "H": 2, "I": -1, "L": -1, "K": -2, "M": -1, "F": 3, "P": -3, "S": -2, "T": -2, "W": 2, "Y": 7, "V": -1},
    "V": {"A": 0, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -2, "E": -2, "G": -3, "H": -3, "I": 3, "L": 1, "K": -2, "M": 1, "F": -1, "P": -2, "S": -2, "T": 0, "W": -3, "Y": -1, "V": 4},
}


@dataclass
class ObjectiveEnsemble:
    scaler: StandardScaler
    models: List[MLPRegressor]
    noise_sigma: float


@dataclass
class FitReport:
    name: str
    train_n: int
    rmse: float
    mae: float
    r2: float
    max_iter_hits: int
    ensemble_size: int


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def mutation_id(wt: str, pos: int, mut: str) -> str:
    return f"{wt}{pos}{mut}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def build_input_fingerprint(
    focus_round: int,
    train_labels_path: Path,
    site_cards_path: Path,
    context_pack_path: Path,
) -> Dict[str, Any]:
    targets = {
        "train_labels_path": train_labels_path,
        "site_cards_path": site_cards_path,
        "context_pack_path": context_pack_path,
    }
    fp: Dict[str, Any] = {"focus_round": int(focus_round)}
    for key, path in targets.items():
        p = Path(path)
        fp[key] = str(p)
        fp[key.replace("_path", "_sha256")] = file_sha256(p) if p.exists() else None
    return fp


def aa_group(aa: str) -> str:
    aa = str(aa or "").upper()
    if aa in POSITIVE_AA:
        return "pos"
    if aa in NEGATIVE_AA:
        return "neg"
    if aa in POLAR_AA:
        return "polar"
    if aa in AROMATIC_AA:
        return "aromatic"
    if aa in HYDROPHOBIC_AA:
        return "hydrophobic"
    if aa in SPECIAL_AA:
        return "special"
    return "other"


def aa_size_group(aa: str) -> str:
    small = set("AGSTCP")
    medium = set("NDEQHVI")
    bulky = set("RKLFWYM")
    if aa in small:
        return "small"
    if aa in medium:
        return "medium"
    if aa in bulky:
        return "bulky"
    return "other"


def aa_charge(aa: str) -> int:
    aa = str(aa or "").upper()
    if aa in POSITIVE_AA:
        return 1
    if aa in NEGATIVE_AA:
        return -1
    return 0


def supports_functional_exploration(site: Dict[str, Any]) -> bool:
    if bool(site.get("functional_site")) or bool(site.get("ligand_contact")):
        return True
    d_lig = safe_float(site.get("dist_ligand"), float("nan"))
    return bool(math.isfinite(d_lig) and d_lig <= float(GATE_FUNCTIONAL_EXPLORATORY_LIGAND_SHELL))


def is_critical_site(site: Dict[str, Any]) -> bool:
    # Treat only hard policy/API locks as critical. Soft do_not_mutate annotations
    # are modeled as risk penalties, not hard-like critical blocks.
    if bool(site.get("do_not_mutate_hard")):
        return True
    hard = {str(x or "").upper() for x in (site.get("hard_constraints") or [])}
    return any(x in CRITICAL_CONSTRAINTS for x in hard)


def is_hard_block_site(site: Dict[str, Any]) -> bool:
    if bool(site.get("do_not_mutate_hard")):
        return True
    hard = {str(x or "").upper() for x in (site.get("hard_constraints") or [])}
    return any(x in HARD_BLOCK_CONSTRAINTS for x in hard)


def is_conservative_substitution(wt: str, mut: str, blosum_min: int = 0) -> bool:
    if wt == mut:
        return True
    # Do not auto-treat special residues (G/P) as conservative via group membership;
    # require substitution-matrix support there.
    groups = [HYDROPHOBIC_AA, AROMATIC_AA, POLAR_AA, POSITIVE_AA, NEGATIVE_AA]
    for g in groups:
        if wt in g and mut in g:
            return True
    return int(BLOSUM62.get(wt, {}).get(mut, -10)) >= int(blosum_min)


def infer_move_tags(wt: str, mut: str) -> List[str]:
    wt = str(wt or "").upper()
    mut = str(mut or "").upper()
    tags: List[str] = []
    if is_conservative_substitution(wt, mut):
        tags.append("conservative")
    wt_charge = 1 if wt in POSITIVE_AA else (-1 if wt in NEGATIVE_AA else 0)
    mut_charge = 1 if mut in POSITIVE_AA else (-1 if mut in NEGATIVE_AA else 0)
    if wt_charge != mut_charge:
        tags.append("charge_shift")
    if aa_group(wt) != aa_group(mut) and "charge_shift" not in tags:
        tags.append("polarity_shift")
    if (wt in AROMATIC_AA) != (mut in AROMATIC_AA):
        tags.append("aromatic_change")
    if aa_size_group(wt) != aa_size_group(mut):
        tags.append("size_shift")
    if not tags:
        tags.append("size_shift")
    return tags


def primary_move_tag(tags: List[str]) -> str:
    order = ["charge_shift", "aromatic_change", "polarity_shift", "size_shift", "conservative"]
    tag_set = set(tags or [])
    for t in order:
        if t in tag_set:
            return t
    return "conservative"


def infer_role(site: Dict[str, Any], mut: str, move_primary: str) -> str:
    aa = str(mut or "").upper()
    if move_primary in {"charge_shift", "polarity_shift"} or aa in (POSITIVE_AA | NEGATIVE_AA | POLAR_AA):
        return "electrostatics_hbond"
    d_lig = safe_float(site.get("dist_ligand"), 12.0)
    near_lig = bool(site.get("ligand_contact")) or d_lig <= 6.5
    if near_lig and aa in (HYDROPHOBIC_AA | AROMATIC_AA):
        return "binding_shape"
    return "stability"


def site_is_chemistry_coverage_eligible(site: Dict[str, Any]) -> bool:
    if supports_functional_exploration(site):
        return True
    if not bool(GATE_CHEMISTRY_COVERAGE_DISTAL_ENABLE):
        return False
    d_lig = safe_float(site.get("dist_ligand"), float("nan"))
    return bool(math.isfinite(d_lig) and d_lig <= float(GATE_CHEMISTRY_COVERAGE_DISTAL_LIGAND_SHELL))


def chemistry_coverage_preferred_aas(wt: str, site: Dict[str, Any]) -> List[str]:
    wt = str(wt or "").upper()
    near = supports_functional_exploration(site)
    preferred: List[str] = []
    if wt == "P":
        preferred = list("LAVGSTNQ")
    elif wt in AROMATIC_AA:
        preferred = list("RKHSTNQCGAVILM" if near else "STNQCGAVILM")
    elif wt in HYDROPHOBIC_AA:
        preferred = list("RKHSTNQCGAVILM" if near else "STNQCGAVILM")
    elif wt in NEGATIVE_AA:
        preferred = list("NQSTKRH")
    elif wt in POSITIVE_AA:
        preferred = list("EQNSTD")
    elif wt in POLAR_AA:
        preferred = list("RKHEQDAVILMFYWG")
    elif wt in SPECIAL_AA:
        preferred = list("STNQAVIL")
    else:
        preferred = list(AA_LIST)
    # Keep order while removing duplicates and WT identity.
    out: List[str] = []
    seen: Set[str] = set()
    for aa in preferred:
        aa_u = str(aa or "").upper()
        if aa_u == wt or aa_u in seen or aa_u not in AA_LIST:
            continue
        seen.add(aa_u)
        out.append(aa_u)
    return out


def chemistry_coverage_substitutions(
    site: Dict[str, Any],
    wt: str,
    strict_evo: bool,
    evo_allowed: Set[str],
    hard: Set[str],
    already_selected: Set[str],
) -> List[str]:
    if not bool(GATE_CHEMISTRY_COVERAGE_ENABLE):
        return []
    if not site_is_chemistry_coverage_eligible(site):
        return []
    max_extra = int(max(0, GATE_CHEMISTRY_COVERAGE_MAX_EXTRA))
    if max_extra <= 0:
        return []

    out: List[str] = []
    floor = int(GATE_CHEMISTRY_COVERAGE_BLOSUM_MIN)
    for aa in chemistry_coverage_preferred_aas(wt=wt, site=site):
        if aa in already_selected:
            continue
        if strict_evo and evo_allowed and aa not in evo_allowed:
            continue
        if "METAL" in hard and aa not in METAL_COMPAT_AA:
            continue
        if any(x in hard for x in ("DISULFIDE", "DISULFID", "CROSSLINK")) and wt == "C" and aa not in set("CST"):
            continue
        bl = int(BLOSUM62.get(wt, {}).get(aa, -10))
        if (not evo_allowed or aa not in evo_allowed) and bl < floor:
            continue
        out.append(aa)
        if len(out) >= max_extra:
            break
    return out


def focused_site_filter_decision(
    site: Dict[str, Any],
    ligand_shell_max: float,
    functional_distance_min: float,
    min_rsa: float,
    conservation_top_fraction: float,
    functional_site_hard_filter: bool,
    near_functional_hard_filter: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    d_lig = safe_float(site.get("dist_ligand"), float("nan"))
    d_func = safe_float(site.get("dist_functional"), float("nan"))
    rsa = safe_float(site.get("exposure"), float("nan"))
    cons_rank = safe_float(site.get("conservation_rank"), float("nan"))
    hard = {str(h or "").upper() for h in (site.get("hard_constraints") or [])}

    functional_site = bool(site.get("functional_site")) or any(x in FUNCTIONAL_CONSTRAINT_TYPES for x in hard)
    if functional_site and bool(functional_site_hard_filter):
        reasons.append("functional_annotation")

    if bool(site.get("structural_lock")):
        reasons.append("structural_lock")

    if bool(near_functional_hard_filter) and math.isfinite(d_func) and d_func < float(functional_distance_min):
        reasons.append("near_functional_residue")

    if bool(site.get("buried_core")):
        reasons.append("buried_core")
    elif math.isfinite(rsa) and rsa < float(min_rsa):
        reasons.append("buried_core")

    # In mutation-level mode this is a soft check; only reject when simultaneously outside shell and highly conserved.
    if math.isfinite(d_lig) and d_lig > float(ligand_shell_max) and math.isfinite(cons_rank) and cons_rank <= float(conservation_top_fraction):
        reasons.append("outside_ligand_shell")

    return (len(reasons) == 0), sorted(set(reasons))


def load_previous_seen_mutations(
    outdir: Path,
    current_round: int,
    dedupe_scope: str,
    dedupe_lookback_rounds: int,
) -> Tuple[Set[Tuple[str, int, str, str]], Set[str]]:
    seen: Set[Tuple[str, int, str, str]] = set()
    seen_variants: Set[str] = set()
    scope = str(dedupe_scope or "panel").strip().lower()
    if scope == "none":
        return seen, seen_variants

    def ingest_file(p: Path) -> None:
        if not p.exists():
            return
        if p.suffix == ".jsonl":
            for row in load_jsonl(p):
                muts = row_mutations(row)
                if muts:
                    for m in muts:
                        try:
                            seen.add(
                                (
                                    str(m.get("chain") or "A"),
                                    int(m.get("pos")),
                                    str(m.get("wt") or "").upper(),
                                    str(m.get("mut") or "").upper(),
                                )
                            )
                        except Exception:
                            continue
                vid = row_variant_id(row)
                if vid:
                    seen_variants.add(str(vid).upper())
            return

        if p.suffix == ".tsv":
            try:
                import csv

                with p.open() as fh:
                    reader = csv.DictReader(fh, delimiter="\t")
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        clean = {str(k or "").strip(): v for k, v in row.items()}
                        muts = row_mutations(clean)
                        if muts:
                            for m in muts:
                                try:
                                    seen.add(
                                        (
                                            str(m.get("chain") or "A"),
                                            int(m.get("pos")),
                                            str(m.get("wt") or "").upper(),
                                            str(m.get("mut") or "").upper(),
                                        )
                                    )
                                except Exception:
                                    continue
                        vid = row_variant_id(clean)
                        if vid:
                            seen_variants.add(str(vid).upper())
            except Exception:
                return

    # Flat artifact layout.
    lookback = int(max(0, dedupe_lookback_rounds))
    if lookback > 0:
        start_rid = max(0, int(current_round) - lookback)
    else:
        start_rid = 0
    for rid in range(int(start_rid), int(current_round)):
        if scope in ("all",):
            ingest_file(proposals_path(outdir=outdir, round_id=rid))
            ingest_file(proposals_vespag_path(outdir=outdir, round_id=rid))
            ingest_file(panel_path(outdir=outdir, round_id=rid))
        elif scope in ("panel",):
            ingest_file(panel_path(outdir=outdir, round_id=rid))
    return seen, seen_variants


def effective_conservative_blosum_cut(site: Dict[str, Any], mut: str) -> int:
    mut = str(mut or "").upper()
    cut = int(GATE_BLOSUM_MIN)
    if is_critical_site(site):
        cut = int(GATE_CRITICAL_BLOSUM_MIN)
    evo_allowed = set(site.get("evolution_allowed_aas") or [])
    evo_cons = clamp(safe_float(site.get("evolution_conservation"), 0.5), 0.0, 1.0)
    strict_evo = bool(evo_allowed and evo_cons >= float(GATE_STRICT_EVO_CONSERVATION_THRESHOLD))
    if strict_evo and mut and mut not in evo_allowed:
        cut = int(GATE_STRICT_EVO_BLOSUM_MIN)
    return int(cut)


def candidate_mechanistic_risk(site: Dict[str, Any], mut: str) -> float:
    wt = str(site.get("wt") or "").upper()
    mut = str(mut or "").upper()
    risk = 0.0

    if is_critical_site(site):
        risk += 0.85
    elif bool(site.get("do_not_mutate")):
        # Soft functional/catalytic annotations should discourage but not eliminate
        # exploration in single-loop recursion.
        risk += 0.32

    hard = {str(x or "").upper() for x in (site.get("hard_constraints") or [])}
    if "METAL" in hard and mut not in METAL_COMPAT_AA:
        risk += 0.50
    if any(x in hard for x in ("DISULFIDE", "DISULFID", "CROSSLINK")) and wt == "C" and mut not in set("CST"):
        risk += 0.45

    evo_allowed = set(site.get("evolution_allowed_aas") or [])
    evo_cons = clamp(safe_float(site.get("evolution_conservation"), 0.5), 0.0, 1.0)
    policy_cut = effective_conservative_blosum_cut(site, mut)
    policy_conservative = is_conservative_substitution(wt, mut, blosum_min=policy_cut)
    if evo_allowed and mut and mut not in evo_allowed:
        if policy_conservative:
            risk += 0.10 + 0.18 * evo_cons
        else:
            risk += 0.25 + 0.35 * evo_cons

    if wt not in SPECIAL_AA and mut in {"G", "P"}:
        risk += 0.16

    if not policy_conservative:
        risk += 0.10

    if bool(site.get("buried_core")) and not policy_conservative:
        risk += 0.12

    return clamp(risk, 0.0, 1.0)


def candidate_binding_relevance(site: Dict[str, Any], mut: str) -> float:
    _ = mut
    lig_contact = 1.0 if bool(site.get("ligand_contact")) else 0.0
    prolif = site.get("prolif") or {}
    contact_freq = clamp(safe_float(prolif.get("contact_freq"), 0.0), 0.0, 1.0)
    d_lig = safe_float(site.get("dist_ligand"), float("nan"))
    tier = int(safe_float(site.get("tier"), 3.0))
    tier_fallback = 1.0 if tier in (0, 1) else (0.55 if tier == 2 else 0.25)
    if math.isfinite(d_lig):
        near = math.exp(-max(0.0, d_lig) / 5.5)
        return clamp(0.10 + 0.54 * near + 0.24 * contact_freq + 0.12 * lig_contact, 0.0, 1.0)
    return clamp(0.15 + 0.40 * lig_contact + 0.30 * contact_freq + 0.15 * tier_fallback, 0.0, 1.0)


def sequence_plausibility_prior(site: Dict[str, Any], wt: str, mut: str) -> float:
    evo_allowed = set(site.get("evolution_allowed_aas") or [])
    evo_cons = clamp(safe_float(site.get("evolution_conservation"), 0.5), 0.0, 1.0)
    policy_cut = effective_conservative_blosum_cut(site, mut)
    conservative = is_conservative_substitution(wt, mut, blosum_min=policy_cut)
    strict_evo = bool(evo_allowed and evo_cons >= float(GATE_STRICT_EVO_CONSERVATION_THRESHOLD))

    bl = safe_float(BLOSUM62.get(wt, {}).get(mut, -4), -4.0)
    bl_norm = clamp((bl + 4.0) / 15.0, 0.0, 1.0)
    class_same = 1.0 if aa_group(wt) == aa_group(mut) else 0.0
    size_same = 1.0 if aa_size_group(wt) == aa_size_group(mut) else 0.0
    wt_charge = 1 if wt in POSITIVE_AA else (-1 if wt in NEGATIVE_AA else 0)
    mut_charge = 1 if mut in POSITIVE_AA else (-1 if mut in NEGATIVE_AA else 0)
    charge_delta = abs(wt_charge - mut_charge) / 2.0

    if evo_allowed:
        if mut in evo_allowed:
            score = 0.66 + 0.22 * evo_cons
        else:
            score = 0.56 - (0.22 + 0.18 * evo_cons) + (0.04 if conservative else -0.05)
    else:
        score = 0.52 + 0.24 * (1.0 - evo_cons) + (0.05 if conservative else -0.03)

    # Prevent plausibility collapse to a constant when sequence priors are sparse/strict by
    # adding smooth, mutation-specific biochemical terms.
    score += 0.18 * (bl_norm - 0.5)
    score += 0.06 * (class_same - 0.5)
    score += 0.04 * (size_same - 0.5)
    score -= 0.06 * charge_delta

    if wt not in SPECIAL_AA and mut in SPECIAL_AA:
        score -= 0.07
    if strict_evo and mut and mut not in evo_allowed and not conservative:
        score -= 0.08
    if bool(site.get("functional_site")) and not conservative:
        score -= 0.08
    if bool(site.get("buried_core")) and not conservative:
        score -= 0.07
    if is_critical_site(site) and not conservative:
        score -= 0.08
    return float(clamp(score, 0.05, 0.98))


def build_feature_vector(site: Dict[str, Any], mut: str) -> Tuple[List[float], Dict[str, float]]:
    wt = str(site.get("wt") or "").upper()
    tier = safe_float(site.get("tier"), 3.0)
    dist_raw = safe_float(site.get("dist_ligand"), float("nan"))
    dist = float(dist_raw) if math.isfinite(dist_raw) else 12.0
    d_func = safe_float(site.get("dist_functional"), float("nan"))
    d_func = float(d_func) if math.isfinite(d_func) else 12.0
    exposure = clamp(safe_float(site.get("exposure"), 0.5), 0.0, 1.0)
    plddt = clamp(safe_float(site.get("plddt"), 80.0), 0.0, 100.0) / 100.0
    lig_contact = 1.0 if bool(site.get("ligand_contact")) else 0.0
    interface = 1.0 if bool(site.get("interface")) else 0.0
    prolif = site.get("prolif") or {}
    prolif_contact = clamp(safe_float(prolif.get("contact_freq"), 0.0), 0.0, 1.0)

    evo_cons = clamp(safe_float(site.get("evolution_conservation"), 0.5), 0.0, 1.0)
    evo_allowed = set(site.get("evolution_allowed_aas") or [])
    evo_match = 1.0 if (evo_allowed and mut in evo_allowed) else 0.0

    bl = safe_float(BLOSUM62.get(wt, {}).get(mut, -4), -4.0)
    bl_norm = (bl + 4.0) / 15.0
    conservative = 1.0 if is_conservative_substitution(wt, mut, blosum_min=effective_conservative_blosum_cut(site, mut)) else 0.0

    charge_delta = abs((1 if wt in POSITIVE_AA else (-1 if wt in NEGATIVE_AA else 0)) - (1 if mut in POSITIVE_AA else (-1 if mut in NEGATIVE_AA else 0))) / 2.0
    aromatic_flip = 1.0 if ((wt in AROMATIC_AA) != (mut in AROMATIC_AA)) else 0.0
    polarity_flip = 1.0 if (aa_group(wt) != aa_group(mut)) else 0.0
    size_flip = 1.0 if (aa_size_group(wt) != aa_size_group(mut)) else 0.0

    bind = candidate_binding_relevance(site, mut)
    risk = candidate_mechanistic_risk(site, mut)
    plaus = sequence_plausibility_prior(site, wt, mut)

    vec = [
        math.exp(-max(0.0, dist) / 5.5),
        math.exp(-max(0.0, d_func) / 6.0),
        exposure,
        plddt,
        lig_contact,
        interface,
        prolif_contact,
        evo_cons,
        evo_match,
        conservative,
        bl_norm,
        charge_delta,
        aromatic_flip,
        polarity_flip,
        size_flip,
        bind,
        1.0 - risk,
        plaus,
        1.0 - clamp(tier / 3.0, 0.0, 1.0),
    ]

    meta = {
        "bind": bind,
        "risk": risk,
        "plaus": plaus,
    }
    return vec, meta


def extract_training_targets(row: Dict[str, Any], site: Dict[str, Any], mut: str) -> Optional[np.ndarray]:
    y_func = row.get("vespag_shrunk_posterior")
    if y_func is None:
        y_func = row.get("vespag_posterior")
    if y_func is None:
        y_func = row.get("vespag_score_norm")
    if y_func is None:
        return None

    y_func_v = clamp(safe_float(y_func, float("nan")), 0.0, 1.0)
    if not math.isfinite(y_func_v):
        return None

    y_bind_v = clamp(safe_float(row.get("p_bind"), candidate_binding_relevance(site, mut)), 0.0, 1.0)
    y_stability_v = clamp(1.0 - candidate_mechanistic_risk(site, mut), 0.0, 1.0)
    y_seq_raw = row.get("seq_prior_ensemble_plausibility")
    y_seq_v = clamp(safe_float(y_seq_raw, float("nan")), 0.0, 1.0)
    if not math.isfinite(y_seq_v):
        wt = str(row.get("wt") or site.get("wt") or "").upper()
        y_seq_v = sequence_plausibility_prior(site=site, wt=wt, mut=mut)

    return np.asarray([y_func_v, y_bind_v, y_stability_v, y_seq_v], dtype=float)


def load_round_training_data(
    outdir: Path,
    focus_round: int,
    site_index: Dict[Tuple[str, int], Dict[str, Any]],
    min_train_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_path = proposals_vespag_path(outdir=outdir, round_id=focus_round)
    rows = load_jsonl(train_path)
    if not rows:
        raise SystemExit(f"Missing training rows: {train_path}")

    feats: List[List[float]] = []
    targets: List[np.ndarray] = []

    for row in rows:
        muts = row_mutations(row)
        if not muts:
            continue
        for m in muts:
            chain = str(m.get("chain") or "")
            try:
                p = int(m.get("pos"))
            except Exception:
                continue
            mut = str(m.get("mut") or "").upper()
            if not chain or p <= 0 or mut not in AA_LIST:
                continue

            site = site_index.get((chain, p))
            if site is None:
                continue

            vec, _meta = build_feature_vector(site, mut)
            y = extract_training_targets(row, site, mut)
            if y is None:
                continue

            feats.append(vec)
            targets.append(y)

    if len(feats) < int(min_train_samples):
        raise SystemExit(
            f"Insufficient training samples for strict multi-objective training: {len(feats)} < {int(min_train_samples)}"
        )

    X = np.asarray(feats, dtype=float)
    Y = np.asarray(targets, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or Y.shape[1] != len(OBJECTIVE_NAMES):
        raise SystemExit("Training matrix shape mismatch for multi-objective surrogate.")
    return X, Y


def prior_function_target(site: Dict[str, Any], wt: str, mut: str, risk: float, plaus: float) -> float:
    conservative = 1.0 if is_conservative_substitution(wt=wt, mut=mut, blosum_min=0) else 0.0
    d_func = safe_float(site.get("dist_functional"), 12.0)
    near_functional = bool(site.get("functional_site")) or d_func < 5.0
    penalty = 0.18 if (near_functional and conservative < 0.5) else 0.0
    score = (0.48 * plaus) + (0.34 * (1.0 - risk)) + (0.18 * conservative) - penalty
    return float(clamp(score, 0.01, 0.99))


def build_bootstrap_prior_training_data(
    candidate_sites: List[Dict[str, Any]],
    blosum_min: int,
    critical_blosum_min: int,
    strict_evo_conservation_threshold: float,
    strict_evo_blosum_min: int,
    seed: int,
    max_samples: int = 6000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    feats: List[List[float]] = []
    targets: List[np.ndarray] = []

    for site in candidate_sites:
        wt = str(site.get("wt") or "").upper()
        if wt not in AA_LIST:
            continue
        for mut in allowed_mutations_for_site(
            site,
            blosum_min=int(blosum_min),
            critical_blosum_min=int(critical_blosum_min),
            strict_evo_conservation_threshold=float(strict_evo_conservation_threshold),
            strict_evo_blosum_min=int(strict_evo_blosum_min),
        ):
            vec, meta = build_feature_vector(site=site, mut=mut)
            risk = float(meta.get("risk", 0.5))
            plaus = float(meta.get("plaus", 0.5))
            bind = float(meta.get("bind", 0.5))
            y_func = prior_function_target(site=site, wt=wt, mut=mut, risk=risk, plaus=plaus)
            y_bind = clamp(bind, 0.0, 1.0)
            y_stability = clamp(1.0 - risk, 0.0, 1.0)
            y_plaus = clamp(plaus, 0.0, 1.0)
            feats.append(vec)
            targets.append(np.asarray([y_func, y_bind, y_stability, y_plaus], dtype=float))

    total_candidates = int(len(feats))
    if total_candidates <= 0:
        raise SystemExit("Bootstrap prior training set is empty after mutation constraints.")

    if total_candidates > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.asarray(rng.choice(total_candidates, size=int(max_samples), replace=False), dtype=int)
        feats = [feats[i] for i in idx.tolist()]
        targets = [targets[i] for i in idx.tolist()]

    X = np.asarray(feats, dtype=float)
    Y = np.asarray(targets, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or Y.shape[1] != len(OBJECTIVE_NAMES):
        raise SystemExit("Bootstrap prior training matrix shape mismatch.")
    return X, Y, {
        "bootstrap_train_candidates_total": int(total_candidates),
        "bootstrap_train_samples_used": int(X.shape[0]),
    }


class MultiObjectiveDeepEnsemble:
    def __init__(self, objective_names: List[str], n_estimators: int, max_iter: int, random_seed: int):
        self.objective_names = list(objective_names)
        self.n_estimators = int(max(3, n_estimators))
        self.max_iter = int(max(200, max_iter))
        self.random_seed = int(random_seed)
        self.models: Dict[str, ObjectiveEnsemble] = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> List[FitReport]:
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D.")
        if Y.shape[1] != len(self.objective_names):
            raise ValueError("Y objective dimension mismatch.")

        reports: List[FitReport] = []
        self.models = {}

        for j, name in enumerate(self.objective_names):
            y = np.asarray(Y[:, j], dtype=float).reshape(-1)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            n_train = int(y.size)

            if n_train < 400:
                hidden_layers = (64, 32)
            elif n_train < 1500:
                hidden_layers = (96, 48)
            else:
                hidden_layers = (128, 64)

            objective_models: List[MLPRegressor] = []
            preds_each: List[np.ndarray] = []
            max_iter_hits = 0
            for i in range(self.n_estimators):
                rs = int(self.random_seed + (j * 1009) + (i * 31))
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=rs,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(Xs, y)
                if int(getattr(model, "n_iter_", 0)) >= int(self.max_iter):
                    max_iter_hits += 1
                pred_i = np.asarray(model.predict(Xs), dtype=float).reshape(-1)
                objective_models.append(model)
                preds_each.append(pred_i)

            pred_stack = np.vstack(preds_each)
            pred_mean = np.clip(np.mean(pred_stack, axis=0), 0.0, 1.0)
            rmse = float(math.sqrt(mean_squared_error(y, pred_mean)))
            mae = float(mean_absolute_error(y, pred_mean))
            r2 = float(r2_score(y, pred_mean))
            noise_sigma = float(max(0.015, rmse))

            self.models[name] = ObjectiveEnsemble(scaler=scaler, models=objective_models, noise_sigma=noise_sigma)
            reports.append(
                FitReport(
                    name=name,
                    train_n=int(y.size),
                    rmse=rmse,
                    mae=mae,
                    r2=r2,
                    max_iter_hits=int(max_iter_hits),
                    ensemble_size=int(self.n_estimators),
                )
            )

        return reports

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if not self.models:
            raise RuntimeError("Model ensemble not fitted.")

        n = X.shape[0]
        m = len(self.objective_names)
        mu = np.zeros((n, m), dtype=float)
        sigma = np.zeros((n, m), dtype=float)

        for j, name in enumerate(self.objective_names):
            om = self.models.get(name)
            if om is None:
                raise RuntimeError(f"Missing objective model: {name}")
            Xs = om.scaler.transform(X)
            pred_members = []
            for model in om.models:
                pred_members.append(np.asarray(model.predict(Xs), dtype=float).reshape(-1))
            pred_stack = np.vstack(pred_members)
            mean_j = np.clip(np.mean(pred_stack, axis=0), 0.0, 1.0)
            var_j = np.var(pred_stack, axis=0) + (om.noise_sigma ** 2)
            std_j = np.sqrt(np.maximum(var_j, 1e-8))
            mu[:, j] = mean_j
            sigma[:, j] = np.clip(std_j, 1e-3, 0.5)

        return mu, sigma


def pareto_front(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, points.shape[1] if points.ndim == 2 else len(OBJECTIVE_NAMES)), dtype=float)
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_any = np.any(np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1))
        if dominated_by_any:
            keep[i] = False
            continue
        dominates_i = np.all(pts[i] >= pts, axis=1) & np.any(pts[i] > pts, axis=1)
        keep[dominates_i] = False
        keep[i] = True
    front = pts[keep]
    if front.shape[0] == 0:
        return front
    return np.unique(np.round(front, 8), axis=0)


def expected_hvi_for_candidate(
    mu: np.ndarray,
    sigma: np.ndarray,
    front: np.ndarray,
    reference: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    draws = rng.normal(loc=mu.reshape(1, -1), scale=sigma.reshape(1, -1), size=(n_mc, mu.size))
    draws = np.clip(draws, 0.0, 1.0)
    base = np.prod(np.maximum(draws - reference.reshape(1, -1), 0.0), axis=1)

    if front.size == 0:
        return float(np.mean(base)), float(np.std(base))

    ge = np.all(front.reshape(1, front.shape[0], front.shape[1]) >= draws.reshape(n_mc, 1, mu.size), axis=2)
    gt = np.any(front.reshape(1, front.shape[0], front.shape[1]) > draws.reshape(n_mc, 1, mu.size), axis=2)
    dominated = np.any(ge & gt, axis=1)

    mins = np.minimum(front.reshape(1, front.shape[0], front.shape[1]), draws.reshape(n_mc, 1, mu.size))
    overlap = np.prod(np.maximum(mins - reference.reshape(1, 1, -1), 0.0), axis=2)
    overlap_max = np.max(overlap, axis=1)

    improve = np.where(dominated, 0.0, np.maximum(base - overlap_max, 0.0))
    return float(np.mean(improve)), float(np.std(improve))


def constraint_feasibility_probability(mu: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray) -> float:
    probs = []
    for j in range(mu.size):
        sig = max(1e-6, float(sigma[j]))
        z = (float(mu[j]) - float(thresholds[j])) / sig
        probs.append(clamp(norm_cdf(z), 0.0, 1.0))
    return clamp(float(np.prod(np.asarray(probs, dtype=float))), 0.0, 1.0)


def allowed_mutations_for_site(
    site: Dict[str, Any],
    blosum_min: int,
    critical_blosum_min: int,
    strict_evo_conservation_threshold: float,
    strict_evo_blosum_min: int,
) -> List[str]:
    wt = str(site.get("wt") or "").upper()
    if wt not in AA_LIST:
        return []

    # Keep hard blocks only for structural locks; catalytic/functional API flags are
    # handled as modeled risk so the loop can still explore conservative hypotheses.
    if is_hard_block_site(site):
        return []

    hard = {str(x or "").upper() for x in (site.get("hard_constraints") or [])}
    evo_allowed = set(site.get("evolution_allowed_aas") or [])
    evo_cons = clamp(safe_float(site.get("evolution_conservation"), 0.5), 0.0, 1.0)
    strict_evo = bool(evo_allowed and evo_cons >= float(strict_evo_conservation_threshold))
    critical_blosum_cut = int(critical_blosum_min if critical_blosum_min is not None else blosum_min)

    conservative_out: List[str] = []
    exploratory_out: List[Tuple[float, str]] = []
    allow_functional_explore = bool(GATE_FUNCTIONAL_EXPLORATORY_ENABLE) and supports_functional_exploration(site)
    exploratory_cut = int(min(blosum_min, GATE_FUNCTIONAL_EXPLORATORY_BLOSUM_MIN))
    max_exploratory = int(max(0, GATE_FUNCTIONAL_EXPLORATORY_MAX_EXTRA))
    for aa in AA_LIST:
        if aa == wt:
            continue
        policy_blosum_cut = int(blosum_min)
        if is_critical_site(site):
            policy_blosum_cut = int(critical_blosum_cut)
        elif strict_evo and aa not in evo_allowed:
            policy_blosum_cut = int(strict_evo_blosum_min)

        policy_conservative = is_conservative_substitution(wt, aa, blosum_min=policy_blosum_cut)
        if "METAL" in hard and aa not in METAL_COMPAT_AA:
            continue
        if any(x in hard for x in ("DISULFIDE", "DISULFID", "CROSSLINK")) and wt == "C" and aa not in set("CST"):
            continue
        if policy_conservative:
            conservative_out.append(aa)
            continue

        if not allow_functional_explore or strict_evo:
            continue
        if max_exploratory <= 0:
            continue
        if not is_conservative_substitution(wt, aa, blosum_min=exploratory_cut):
            continue
        score = float(BLOSUM62.get(wt, {}).get(aa, -10))
        if aa_charge(wt) != aa_charge(aa):
            score += 0.35
        if aa_group(wt) != aa_group(aa):
            score += 0.20
        if aa_size_group(wt) != aa_size_group(aa):
            score += 0.10
        exploratory_out.append((score, aa))

    if exploratory_out and max_exploratory > 0:
        exploratory_out.sort(key=lambda x: (-x[0], x[1]))
        chosen = [aa for _score, aa in exploratory_out[:max_exploratory]]
        conservative_out.extend(chosen)

    chemistry_out = chemistry_coverage_substitutions(
        site=site,
        wt=wt,
        strict_evo=bool(strict_evo),
        evo_allowed=set(evo_allowed),
        hard=set(hard),
        already_selected=set(conservative_out),
    )
    conservative_out.extend(chemistry_out)

    # Preserve deterministic order while removing duplicates.
    seen: Set[str] = set()
    ordered: List[str] = []
    for aa in conservative_out:
        aa_u = str(aa or "").upper()
        if aa_u in seen:
            continue
        seen.add(aa_u)
        ordered.append(aa_u)
    return ordered


def mutation_labels_for_variant(mutations: List[Dict[str, Any]], site_index: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, str]:
    if not mutations:
        return {"raw": ""}
    raw = mutations_to_id(mutations, include_chain=False)
    labels: Dict[str, str] = {"raw": raw}

    mature_parts: List[str] = []
    ambler_parts: List[str] = []
    for m in mutations:
        chain = str(m.get("chain") or "A")
        pos = int(m.get("pos"))
        wt = str(m.get("wt") or "").upper()
        mut = str(m.get("mut") or "").upper()
        site = site_index.get((chain, pos)) or {}
        numbering = site.get("numbering") if isinstance(site, dict) else None
        if isinstance(numbering, dict):
            mature_pos = numbering.get("mature")
            ambler_pos = numbering.get("ambler")
            if mature_pos is not None:
                try:
                    mature_parts.append(mutation_id(wt, int(mature_pos), mut))
                except Exception:
                    pass
            if ambler_pos is not None:
                try:
                    ambler_parts.append(mutation_id(wt, int(ambler_pos), mut))
                except Exception:
                    pass

    if mature_parts and len(mature_parts) == len(mutations):
        labels["mature"] = ":".join(mature_parts)
    if ambler_parts and len(ambler_parts) == len(mutations):
        labels["ambler"] = ":".join(ambler_parts)
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate single-loop recursive candidates via mutation-level multi-objective deep-ensemble surrogate."
    )
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--focus-round", type=int, default=None)
    ap.add_argument("--target-proposals", type=int, default=120)
    ap.add_argument("--max-per-position", type=int, default=3)
    ap.add_argument(
        "--max-mutations-per-variant",
        type=int,
        default=2,
        help="Maximum mutations per proposed variant (1=single-site only, 2=pairwise multi-point).",
    )
    ap.add_argument(
        "--multi-point-fraction",
        type=float,
        default=0.35,
        help="Target fraction of selected panel allocated to multi-point candidates.",
    )
    ap.add_argument(
        "--multi-seed-size",
        type=int,
        default=120,
        help="Top single-site candidates used to compose multi-point variants.",
    )
    ap.add_argument(
        "--multi-max-candidates",
        type=int,
        default=1200,
        help="Maximum composed multi-point candidate rows retained before panel selection.",
    )
    ap.add_argument(
        "--multi-min-position-separation",
        type=int,
        default=1,
        help="Minimum absolute sequence separation between positions in a pair (same chain).",
    )
    ap.add_argument(
        "--multi-max-position-separation",
        type=int,
        default=0,
        help="Optional maximum absolute sequence separation between positions in a pair (0 disables).",
    )
    ap.add_argument("--blosum-min", type=int, default=0)
    ap.add_argument("--critical-blosum-min", type=int, default=-1)
    ap.add_argument("--strict-evo-conservation-threshold", type=float, default=0.95)
    ap.add_argument("--strict-evo-blosum-min", type=int, default=-1)
    ap.add_argument(
        "--functional-exploratory-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow a bounded nonconservative exploration set at functional/ligand-proximal sites.",
    )
    ap.add_argument(
        "--functional-exploratory-blosum-min",
        type=int,
        default=-2,
        help="BLOSUM floor used for bounded functional exploratory substitutions.",
    )
    ap.add_argument(
        "--functional-exploratory-max-extra",
        type=int,
        default=3,
        help="Maximum number of exploratory nonconservative substitutions admitted per site.",
    )
    ap.add_argument(
        "--functional-exploratory-ligand-shell",
        type=float,
        default=8.0,
        help="Ligand distance threshold (A) to treat a site as eligible for exploratory substitutions.",
    )
    ap.add_argument(
        "--chemistry-coverage-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable protein-agnostic chemistry coverage substitutions to avoid narrow conservative collapse.",
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
        help="Allow chemistry-coverage substitutions at distal but ligand-reachable residues.",
    )
    ap.add_argument(
        "--chemistry-coverage-distal-ligand-shell",
        type=float,
        default=14.0,
        help="Ligand distance threshold (A) for distal chemistry-coverage eligibility.",
    )
    ap.add_argument("--ligand-shell-max", type=float, default=12.0)
    ap.add_argument("--functional-distance-min", type=float, default=4.0)
    ap.add_argument("--min-rsa", type=float, default=0.03)
    ap.add_argument("--conservation-top-fraction", type=float, default=0.05)
    ap.add_argument("--functional-site-hard-filter", action="store_true", default=False)
    ap.add_argument("--near-functional-hard-filter", action="store_true", default=False)
    ap.add_argument(
        "--dedupe-scope",
        choices=["none", "panel", "all"],
        default="panel",
        help="Mutation dedupe scope across prior rounds.",
    )
    ap.add_argument(
        "--dedupe-lookback-rounds",
        type=int,
        default=2,
        help="Dedupe lookback window (0=all prior rounds).",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--site-cards", default=None)
    ap.add_argument("--ensemble-models", type=int, default=5)
    ap.add_argument("--ensemble-max-iter", type=int, default=700)
    ap.add_argument("--min-train-samples", type=int, default=80)
    ap.add_argument("--ehvi-mc", type=int, default=32)
    ap.add_argument("--min-function", type=float, default=0.40)
    ap.add_argument("--min-binding", type=float, default=0.35)
    ap.add_argument("--min-stability", type=float, default=0.40)
    ap.add_argument("--min-plausibility", type=float, default=0.40)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    round_id = int(args.round)
    focus_round = int(args.focus_round) if args.focus_round is not None else (round_id - 1)

    global GATE_BLOSUM_MIN
    global GATE_CRITICAL_BLOSUM_MIN
    global GATE_STRICT_EVO_CONSERVATION_THRESHOLD
    global GATE_STRICT_EVO_BLOSUM_MIN
    global GATE_FUNCTIONAL_EXPLORATORY_ENABLE
    global GATE_FUNCTIONAL_EXPLORATORY_BLOSUM_MIN
    global GATE_FUNCTIONAL_EXPLORATORY_MAX_EXTRA
    global GATE_FUNCTIONAL_EXPLORATORY_LIGAND_SHELL
    global GATE_CHEMISTRY_COVERAGE_ENABLE
    global GATE_CHEMISTRY_COVERAGE_MAX_EXTRA
    global GATE_CHEMISTRY_COVERAGE_BLOSUM_MIN
    global GATE_CHEMISTRY_COVERAGE_DISTAL_ENABLE
    global GATE_CHEMISTRY_COVERAGE_DISTAL_LIGAND_SHELL
    GATE_BLOSUM_MIN = int(args.blosum_min)
    GATE_CRITICAL_BLOSUM_MIN = int(args.critical_blosum_min)
    GATE_STRICT_EVO_CONSERVATION_THRESHOLD = float(args.strict_evo_conservation_threshold)
    GATE_STRICT_EVO_BLOSUM_MIN = int(args.strict_evo_blosum_min)
    GATE_FUNCTIONAL_EXPLORATORY_ENABLE = bool(args.functional_exploratory_enable)
    GATE_FUNCTIONAL_EXPLORATORY_BLOSUM_MIN = int(args.functional_exploratory_blosum_min)
    GATE_FUNCTIONAL_EXPLORATORY_MAX_EXTRA = int(max(0, args.functional_exploratory_max_extra))
    GATE_FUNCTIONAL_EXPLORATORY_LIGAND_SHELL = float(max(0.0, args.functional_exploratory_ligand_shell))
    GATE_CHEMISTRY_COVERAGE_ENABLE = bool(args.chemistry_coverage_enable)
    GATE_CHEMISTRY_COVERAGE_MAX_EXTRA = int(max(0, args.chemistry_coverage_max_extra))
    GATE_CHEMISTRY_COVERAGE_BLOSUM_MIN = int(args.chemistry_coverage_blosum_min)
    GATE_CHEMISTRY_COVERAGE_DISTAL_ENABLE = bool(args.chemistry_coverage_distal_enable)
    GATE_CHEMISTRY_COVERAGE_DISTAL_LIGAND_SHELL = float(max(0.0, args.chemistry_coverage_distal_ligand_shell))

    swarm_root(outdir).mkdir(parents=True, exist_ok=True)

    site_cards_path = Path(args.site_cards) if args.site_cards else outdir / "swarm" / "site_cards.jsonl"
    context_pack_path = outdir / "swarm" / "context_pack.json"

    cards = [compact_site_card(c) for c in load_site_cards(site_cards_path)]
    if not cards:
        raise SystemExit(f"No site cards found at {site_cards_path}")
    site_index = {(str(c.get("chain")), int(c.get("pos"))): c for c in cards if c.get("chain") is not None and c.get("pos") is not None}

    prev_seen_mutations, prev_seen_variants = load_previous_seen_mutations(
        outdir=outdir,
        current_round=round_id,
        dedupe_scope=str(args.dedupe_scope),
        dedupe_lookback_rounds=int(args.dedupe_lookback_rounds),
    )

    candidates_sites = []
    site_filter_rejects: Counter = Counter()
    for site in cards:
        keep_site, reasons = focused_site_filter_decision(
            site=site,
            ligand_shell_max=float(args.ligand_shell_max),
            functional_distance_min=float(args.functional_distance_min),
            min_rsa=float(args.min_rsa),
            conservation_top_fraction=float(args.conservation_top_fraction),
            functional_site_hard_filter=bool(args.functional_site_hard_filter),
            near_functional_hard_filter=bool(args.near_functional_hard_filter),
        )
        if keep_site:
            candidates_sites.append(site)
        else:
            for reason in reasons:
                site_filter_rejects[reason] += 1
    if not candidates_sites:
        raise SystemExit("No candidate sites after mutation-level prefiltering.")

    training_source = "round_labels"
    training_reason = ""
    bootstrap_meta: Dict[str, int] = {}

    try:
        if focus_round < 0:
            raise SystemExit("no_previous_round_labels_for_bootstrap")
        X_train, Y_train = load_round_training_data(
            outdir=outdir,
            focus_round=focus_round,
            site_index=site_index,
            min_train_samples=int(args.min_train_samples),
        )
    except SystemExit as exc:
        training_source = "bootstrap_prior"
        training_reason = str(exc)
        X_train, Y_train, bootstrap_meta = build_bootstrap_prior_training_data(
            candidate_sites=candidates_sites,
            blosum_min=int(args.blosum_min),
            critical_blosum_min=int(args.critical_blosum_min),
            strict_evo_conservation_threshold=float(args.strict_evo_conservation_threshold),
            strict_evo_blosum_min=int(args.strict_evo_blosum_min),
            seed=int(args.seed),
        )

    ensemble = MultiObjectiveDeepEnsemble(
        objective_names=OBJECTIVE_NAMES,
        n_estimators=int(args.ensemble_models),
        max_iter=int(args.ensemble_max_iter),
        random_seed=int(args.seed),
    )
    fit_reports = ensemble.fit(X_train, Y_train)

    front = pareto_front(Y_train)
    if training_source == "bootstrap_prior":
        # In cold-start mode the pseudo-training set is generated from the same candidate manifold.
        # Use absolute expected hypervolume (empty incumbent front) to avoid collapsing acquisition to ~0.
        front = np.zeros((0, len(OBJECTIVE_NAMES)), dtype=float)
    reference = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
    thresholds = np.asarray(
        [
            float(clamp(args.min_function, 0.0, 1.0)),
            float(clamp(args.min_binding, 0.0, 1.0)),
            float(clamp(args.min_stability, 0.0, 1.0)),
            float(clamp(args.min_plausibility, 0.0, 1.0)),
        ],
        dtype=float,
    )
    if training_source == "bootstrap_prior":
        # Cold-start thresholds should be permissive enough to maintain exploration diversity.
        q20 = np.quantile(Y_train, 0.20, axis=0)
        relaxed = np.clip(np.asarray(q20, dtype=float), 0.05, 0.40)
        thresholds = np.minimum(thresholds, relaxed)

    rows: List[Dict[str, Any]] = []
    feat_rows: List[List[float]] = []

    filtered_duplicates = 0
    filtered_constraints = 0

    for site in candidates_sites:
        chain = str(site.get("chain") or "")
        pos = int(site.get("pos") or -1)
        wt = str(site.get("wt") or "").upper()
        if not chain or pos <= 0 or wt not in AA_LIST:
            continue

        allowed = allowed_mutations_for_site(
            site,
            blosum_min=int(args.blosum_min),
            critical_blosum_min=int(args.critical_blosum_min),
            strict_evo_conservation_threshold=float(args.strict_evo_conservation_threshold),
            strict_evo_blosum_min=int(args.strict_evo_blosum_min),
        )
        if not allowed:
            continue

        for mut in allowed:
            key = (chain, pos, wt, mut)
            if key in prev_seen_mutations:
                filtered_duplicates += 1
                continue

            risk = candidate_mechanistic_risk(site, mut)
            if risk >= 0.98:
                filtered_constraints += 1
                continue

            vec, meta = build_feature_vector(site, mut)
            move_tags = infer_move_tags(wt, mut)
            move_primary = primary_move_tag(move_tags)
            role = infer_role(site, mut, move_primary)
            policy_cut = effective_conservative_blosum_cut(site, mut)
            policy_conservative = is_conservative_substitution(wt, mut, blosum_min=policy_cut)
            chemistry_challenger = bool(site_is_chemistry_coverage_eligible(site) and (not policy_conservative))
            mutation = {"chain": chain, "pos": pos, "wt": wt, "mut": mut}
            variant_id = mutations_to_id([mutation], include_chain=False)
            if not variant_id:
                continue
            if variant_id.upper() in prev_seen_variants:
                filtered_duplicates += 1
                continue

            rows.append(
                {
                    "variant_id": variant_id,
                    "mutation_count": 1,
                    "mutations": [mutation],
                    "positions": [(chain, pos)],
                    "chain": chain,
                    "pos": pos,
                    "wt": wt,
                    "mut": mut,
                    "move_tags": move_tags,
                    "move_primary": move_primary,
                    "source_role": role,
                    "chemistry_challenger": bool(chemistry_challenger),
                    "site": site,
                    "meta": meta,
                }
            )
            feat_rows.append(vec)

    if not rows:
        raise SystemExit("No mutation candidates remained after hard constraints and dedupe.")

    X = np.asarray(feat_rows, dtype=float)
    mu_obj, std_obj = ensemble.predict(X)
    if training_source == "bootstrap_prior":
        std_obj = np.maximum(std_obj, 0.08)

    rng = np.random.default_rng(int(args.seed))
    ehvi_vals = np.zeros((len(rows),), dtype=float)
    ehvi_std = np.zeros((len(rows),), dtype=float)
    p_feas = np.zeros((len(rows),), dtype=float)

    for i in range(len(rows)):
        ehvi_mean, ehvi_s = expected_hvi_for_candidate(
            mu=mu_obj[i],
            sigma=std_obj[i],
            front=front,
            reference=reference,
            n_mc=max(8, int(args.ehvi_mc)),
            rng=rng,
        )
        pf = constraint_feasibility_probability(mu=mu_obj[i], sigma=std_obj[i], thresholds=thresholds)
        ehvi_vals[i] = float(ehvi_mean)
        ehvi_std[i] = float(ehvi_s)
        p_feas[i] = float(pf)

    if training_source == "bootstrap_prior":
        # Soften feasibility suppression in cold-start to avoid collapsing exploration
        # when priors are conservative/under-calibrated.
        acquisition = ehvi_vals * np.power(np.clip(p_feas, 1e-8, 1.0), 0.50)
    else:
        acquisition = ehvi_vals * p_feas

    for i, row in enumerate(rows):
        row["objective_mean"] = {k: round(float(mu_obj[i, j]), 6) for j, k in enumerate(OBJECTIVE_NAMES)}
        row["objective_std"] = {k: round(float(std_obj[i, j]), 6) for j, k in enumerate(OBJECTIVE_NAMES)}
        row["expected_hvi"] = round(float(ehvi_vals[i]), 8)
        row["expected_hvi_std"] = round(float(ehvi_std[i]), 8)
        row["feasibility_prob"] = round(float(p_feas[i]), 8)
        row["acquisition"] = round(float(acquisition[i]), 8)
        uncertainty = float(np.mean(std_obj[i, :])) if std_obj.ndim == 2 else 0.0
        row["uncertainty"] = round(float(max(0.0, uncertainty)), 8)
        row["explore_priority"] = round(float(max(0.0, (0.70 * uncertainty) + (0.30 * ehvi_vals[i]))), 8)

    single_rows = list(rows)
    multi_rows: List[Dict[str, Any]] = []
    if int(args.max_mutations_per_variant) > 1 and float(args.multi_point_fraction) > 0.0 and len(single_rows) >= 2:
        seed_n = min(len(single_rows), max(12, int(args.multi_seed_size)))
        seed_rows = sorted(
            single_rows,
            key=lambda rr: (
                safe_float(rr.get("acquisition"), 0.0),
                safe_float(rr.get("expected_hvi"), 0.0),
                safe_float(rr.get("feasibility_prob"), 0.0),
            ),
            reverse=True,
        )[:seed_n]
        seen_multi_vid: Set[str] = set()
        max_multi_candidates = max(0, int(args.multi_max_candidates))
        min_sep = max(0, int(args.multi_min_position_separation))
        max_sep = max(0, int(args.multi_max_position_separation))

        for a, b in combinations(seed_rows, 2):
            ma = (a.get("mutations") or [{}])[0]
            mb = (b.get("mutations") or [{}])[0]
            ca = str(ma.get("chain") or "")
            cb = str(mb.get("chain") or "")
            pa = int(ma.get("pos") or -1)
            pb = int(mb.get("pos") or -1)
            if pa <= 0 or pb <= 0:
                continue
            if ca == cb and pa == pb:
                continue
            if ca == cb:
                sep = abs(pa - pb)
                if sep < min_sep:
                    continue
                if max_sep > 0 and sep > max_sep:
                    continue

            muts = [
                {"chain": ca, "pos": pa, "wt": str(ma.get("wt") or "").upper(), "mut": str(ma.get("mut") or "").upper()},
                {"chain": cb, "pos": pb, "wt": str(mb.get("wt") or "").upper(), "mut": str(mb.get("mut") or "").upper()},
            ]
            vid = mutations_to_id(muts, include_chain=False)
            if not vid:
                continue
            vid_u = str(vid).upper()
            if vid_u in seen_multi_vid or vid_u in prev_seen_variants:
                continue

            mu_a = a.get("objective_mean") if isinstance(a.get("objective_mean"), dict) else {}
            mu_b = b.get("objective_mean") if isinstance(b.get("objective_mean"), dict) else {}
            sd_a = a.get("objective_std") if isinstance(a.get("objective_std"), dict) else {}
            sd_b = b.get("objective_std") if isinstance(b.get("objective_std"), dict) else {}

            mu_func = math.sqrt(
                max(1e-8, safe_float(mu_a.get("function"), 0.0))
                * max(1e-8, safe_float(mu_b.get("function"), 0.0))
            )
            mu_bind = 1.0 - (
                (1.0 - clamp(safe_float(mu_a.get("binding"), 0.0), 0.0, 1.0))
                * (1.0 - clamp(safe_float(mu_b.get("binding"), 0.0), 0.0, 1.0))
            )
            mu_stab = math.sqrt(
                max(1e-8, safe_float(mu_a.get("stability"), 0.0))
                * max(1e-8, safe_float(mu_b.get("stability"), 0.0))
            )
            mu_plaus = math.sqrt(
                max(1e-8, safe_float(mu_a.get("plausibility"), 0.0))
                * max(1e-8, safe_float(mu_b.get("plausibility"), 0.0))
            )
            local_penalty = 0.0
            if ca == cb and abs(pa - pb) <= 2:
                local_penalty = 0.05
            mu_vec = np.asarray(
                [
                    clamp(mu_func - local_penalty, 0.0, 1.0),
                    clamp(mu_bind, 0.0, 1.0),
                    clamp(mu_stab - local_penalty, 0.0, 1.0),
                    clamp(mu_plaus - (0.5 * local_penalty), 0.0, 1.0),
                ],
                dtype=float,
            )

            sd_vec = np.asarray(
                [
                    max(1e-3, math.sqrt(safe_float(sd_a.get("function"), 0.0) ** 2 + safe_float(sd_b.get("function"), 0.0) ** 2) / 2.0 + 0.01),
                    max(1e-3, math.sqrt(safe_float(sd_a.get("binding"), 0.0) ** 2 + safe_float(sd_b.get("binding"), 0.0) ** 2) / 2.0 + 0.01),
                    max(1e-3, math.sqrt(safe_float(sd_a.get("stability"), 0.0) ** 2 + safe_float(sd_b.get("stability"), 0.0) ** 2) / 2.0 + 0.01),
                    max(1e-3, math.sqrt(safe_float(sd_a.get("plausibility"), 0.0) ** 2 + safe_float(sd_b.get("plausibility"), 0.0) ** 2) / 2.0 + 0.01),
                ],
                dtype=float,
            )
            pf = constraint_feasibility_probability(mu=mu_vec, sigma=sd_vec, thresholds=thresholds)
            ehvi_mean, ehvi_s = expected_hvi_for_candidate(
                mu=mu_vec,
                sigma=sd_vec,
                front=front,
                reference=reference,
                n_mc=max(8, int(args.ehvi_mc)),
                rng=rng,
            )
            if training_source == "bootstrap_prior":
                acq = float(ehvi_mean * (max(1e-8, pf) ** 0.50))
            else:
                acq = float(ehvi_mean * pf)

            source_roles = sorted(
                {str(a.get("source_role") or "").strip(), str(b.get("source_role") or "").strip()} - {""}
            )
            move_tags = sorted(
                {str(x) for x in (a.get("move_tags") or [])} | {str(x) for x in (b.get("move_tags") or [])}
            )
            site_a = a.get("site") if isinstance(a.get("site"), dict) else {}
            site_b = b.get("site") if isinstance(b.get("site"), dict) else {}
            hard_union = sorted({str(x) for x in (site_a.get("hard_constraints") or []) + (site_b.get("hard_constraints") or [])})
            soft_union = sorted({str(x) for x in (site_a.get("soft_constraints") or []) + (site_b.get("soft_constraints") or [])})
            evo_allowed_union = sorted(
                {str(x) for x in (site_a.get("evolution_allowed_aas") or []) + (site_b.get("evolution_allowed_aas") or [])}
            )
            bind_meta = 1.0 - ((1.0 - safe_float((a.get("meta") or {}).get("bind"), 0.0)) * (1.0 - safe_float((b.get("meta") or {}).get("bind"), 0.0)))
            risk_meta = clamp(
                0.5 * safe_float((a.get("meta") or {}).get("risk"), 0.5)
                + 0.5 * safe_float((b.get("meta") or {}).get("risk"), 0.5)
                + local_penalty,
                0.0,
                1.0,
            )
            plaus_meta = math.sqrt(
                max(1e-8, safe_float((a.get("meta") or {}).get("plaus"), 0.5))
                * max(1e-8, safe_float((b.get("meta") or {}).get("plaus"), 0.5))
            )

            primary = muts[0]
            secondary = muts[1]
            combined = {
                "variant_id": str(vid),
                "mutation_count": 2,
                "mutations": muts,
                "positions": [(str(primary["chain"]), int(primary["pos"])), (str(secondary["chain"]), int(secondary["pos"]))],
                "chain": str(primary["chain"]),
                "pos": int(primary["pos"]),
                "wt": str(primary["wt"]),
                "mut": str(primary["mut"]),
                "move_tags": move_tags or ["multi_point"],
                "move_primary": "multi_point",
                "source_role": "+".join(source_roles) if source_roles else "multi_point",
                "chemistry_challenger": bool(a.get("chemistry_challenger")) or bool(b.get("chemistry_challenger")),
                "site": dict(site_a) if site_a else {},
                "sites": [dict(site_a), dict(site_b)],
                "meta": {"bind": float(clamp(bind_meta, 0.0, 1.0)), "risk": float(risk_meta), "plaus": float(clamp(plaus_meta, 0.0, 1.0))},
                "tier": int(
                    min(
                        safe_float(site_a.get("tier"), 3.0),
                        safe_float(site_b.get("tier"), 3.0),
                    )
                ),
                "dist_ligand": float(min(safe_float(site_a.get("dist_ligand"), 999.0), safe_float(site_b.get("dist_ligand"), 999.0))),
                "dist_functional": float(min(safe_float(site_a.get("dist_functional"), 999.0), safe_float(site_b.get("dist_functional"), 999.0))),
                "plddt": float(np.mean([safe_float(site_a.get("plddt"), 80.0), safe_float(site_b.get("plddt"), 80.0)])),
                "exposure": float(np.mean([safe_float(site_a.get("exposure"), 0.5), safe_float(site_b.get("exposure"), 0.5)])),
                "ligand_contact": bool(site_a.get("ligand_contact")) or bool(site_b.get("ligand_contact")),
                "interface": bool(site_a.get("interface")) or bool(site_b.get("interface")),
                "functional_site": bool(site_a.get("functional_site")) or bool(site_b.get("functional_site")),
                "hard_constraints": hard_union,
                "soft_constraints": soft_union,
                "evolution_allowed_aas": evo_allowed_union,
                "evolution_conservation": float(np.mean([safe_float(site_a.get("evolution_conservation"), 0.5), safe_float(site_b.get("evolution_conservation"), 0.5)])),
                "conservation_rank": float(min(safe_float(site_a.get("conservation_rank"), 1.0), safe_float(site_b.get("conservation_rank"), 1.0))),
                "numbering": {},
                "objective_mean": {k: round(float(mu_vec[j]), 6) for j, k in enumerate(OBJECTIVE_NAMES)},
                "objective_std": {k: round(float(sd_vec[j]), 6) for j, k in enumerate(OBJECTIVE_NAMES)},
                "expected_hvi": round(float(ehvi_mean), 8),
                "expected_hvi_std": round(float(ehvi_s), 8),
                "feasibility_prob": round(float(pf), 8),
                "acquisition": round(float(acq), 8),
                "uncertainty": round(float(max(0.0, float(np.mean(sd_vec)))), 8),
                "explore_priority": round(float(max(0.0, (0.70 * float(np.mean(sd_vec))) + (0.30 * ehvi_mean))), 8),
                "pair_components": [str(a.get("variant_id") or ""), str(b.get("variant_id") or "")],
            }
            seen_multi_vid.add(vid_u)
            multi_rows.append(combined)
            if max_multi_candidates > 0 and len(multi_rows) >= max_multi_candidates:
                break
        if max_multi_candidates > 0 and len(multi_rows) > max_multi_candidates:
            multi_rows = sorted(
                multi_rows,
                key=lambda rr: (
                    safe_float(rr.get("acquisition"), 0.0),
                    safe_float(rr.get("expected_hvi"), 0.0),
                    safe_float(rr.get("feasibility_prob"), 0.0),
                ),
                reverse=True,
            )[:max_multi_candidates]

    rows = single_rows + multi_rows

    requested_target = max(1, int(args.target_proposals))
    max_per_position = max(1, int(args.max_per_position))
    unique_positions: Set[Tuple[str, int]] = set()
    for r in rows:
        pos_list = r.get("positions") if isinstance(r.get("positions"), list) else []
        if pos_list:
            for cp in pos_list:
                try:
                    unique_positions.add((str(cp[0]), int(cp[1])))
                except Exception:
                    continue
            continue
        unique_positions.add((str(r["chain"]), int(r["pos"])))
    max_possible_by_position = int(len(unique_positions) * max_per_position)
    target = max(1, min(requested_target, len(rows), max_possible_by_position))

    if multi_rows and float(args.multi_point_fraction) > 0.0:
        multi_target = int(round(target * clamp(float(args.multi_point_fraction), 0.0, 1.0)))
        single_target = max(0, target - multi_target)
    else:
        multi_target = 0
        single_target = target

    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    pos_counts: Counter = Counter()
    move_counts: Counter = Counter()
    lane_counts: Counter = Counter()
    selected_multi = 0
    selected_single = 0

    def can_take(r: Dict[str, Any]) -> bool:
        vid = str(r.get("variant_id") or "")
        if not vid or vid in selected_ids:
            return False
        cps = r.get("positions") if isinstance(r.get("positions"), list) else [(r["chain"], r["pos"])]
        for cp in cps:
            key = (str(cp[0]), int(cp[1]))
            if pos_counts[key] >= max_per_position:
                return False
        if safe_float(r.get("feasibility_prob"), 0.0) <= 0.0:
            return False
        mut_count = int(r.get("mutation_count", len(r.get("mutations") or [] or [1])))
        if multi_target > 0 and mut_count > 1 and selected_multi >= multi_target:
            return False
        if single_target > 0 and mut_count <= 1 and selected_single >= single_target:
            return False
        return True

    def take(r: Dict[str, Any], lane: str) -> bool:
        nonlocal selected_multi, selected_single
        if not can_take(r):
            return False
        vid = str(r.get("variant_id") or "")
        cps = r.get("positions") if isinstance(r.get("positions"), list) else [(r["chain"], r["pos"])]
        out = dict(r)
        out["selection_lane"] = str(lane)
        selected.append(out)
        selected_ids.add(vid)
        for cp in cps:
            key = (str(cp[0]), int(cp[1]))
            pos_counts[key] += 1
        move_counts[str(r["move_primary"])] += 1
        lane_counts[str(lane)] += 1
        if int(out.get("mutation_count", len(out.get("mutations") or [] or [1]))) > 1:
            selected_multi += 1
        else:
            selected_single += 1
        return True

    bootstrap_mode = bool(training_source == "bootstrap_prior")
    # In cold-start, avoid spending most budget on one-per-position coverage; keep
    # enough room for within-position mutation alternatives.
    coverage_frac = 0.35 if bootstrap_mode else 0.25
    intra_pos_frac = 0.20 if bootstrap_mode else 0.12
    chemistry_frac = 0.16 if bootstrap_mode else 0.12
    explore_frac = 0.30 if bootstrap_mode else 0.25
    coverage_target = min(target, max(0, int(round(target * coverage_frac))), len(unique_positions))
    intra_pos_target = max(0, int(round(target * intra_pos_frac)))
    intra_pos_target = min(max(0, target - coverage_target), intra_pos_target)
    rem_after_intra = max(0, target - coverage_target - intra_pos_target)
    chemistry_target = min(rem_after_intra, max(0, int(round(target * chemistry_frac))))
    rem_after_chemistry = max(0, rem_after_intra - chemistry_target)
    explore_target = min(rem_after_chemistry, max(0, int(round(target * explore_frac))))
    exploit_target = max(0, target - coverage_target - intra_pos_target - chemistry_target - explore_target)

    rows_by_pos: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        cps = r.get("positions") if isinstance(r.get("positions"), list) else [(r["chain"], r["pos"])]
        for cp in cps:
            pk = (str(cp[0]), int(cp[1]))
            rows_by_pos.setdefault(pk, []).append(r)
    for lst in rows_by_pos.values():
        lst.sort(
            key=lambda rr: (
                safe_float(rr.get("acquisition"), 0.0),
                safe_float(rr.get("expected_hvi"), 0.0),
                safe_float(rr.get("uncertainty"), 0.0),
            ),
            reverse=True,
        )

    coverage_pool = [lst[0] for lst in rows_by_pos.values() if lst]
    coverage_pool.sort(
        key=lambda rr: (
            safe_float(rr.get("acquisition"), 0.0),
            safe_float(rr.get("uncertainty"), 0.0),
            safe_float(rr.get("expected_hvi"), 0.0),
        ),
        reverse=True,
    )

    for r in coverage_pool:
        if lane_counts["coverage"] >= coverage_target or len(selected) >= target:
            break
        take(r, "coverage")

    if intra_pos_target > 0 and len(selected) < target:
        coverage_positions: List[Tuple[str, int]] = []
        for it in selected:
            if str(it.get("selection_lane")) != "coverage":
                continue
            coverage_positions.append((str(it["chain"]), int(it["pos"])))
        coverage_positions = list(dict.fromkeys(coverage_positions))
        coverage_positions.sort(
            key=lambda pk: safe_float((rows_by_pos.get(pk) or [{}])[0].get("acquisition"), 0.0),
            reverse=True,
        )
        while lane_counts["intra_pos"] < intra_pos_target and len(selected) < target:
            progressed = False
            for pk in coverage_positions:
                if lane_counts["intra_pos"] >= intra_pos_target or len(selected) >= target:
                    break
                for cand in rows_by_pos.get(pk, []):
                    if take(cand, "intra_pos"):
                        progressed = True
                        break
            if not progressed:
                break

    chemistry_candidates = sorted(
        [r for r in rows if bool(r.get("chemistry_challenger"))],
        key=lambda rr: (
            safe_float(rr.get("uncertainty"), 0.0),
            safe_float(rr.get("explore_priority"), 0.0),
            safe_float(rr.get("acquisition"), 0.0),
        ),
        reverse=True,
    )
    for r in chemistry_candidates:
        if lane_counts["chemistry"] >= chemistry_target or len(selected) >= target:
            break
        take(r, "chemistry")

    explore_seed: List[Dict[str, Any]] = []
    for lst in rows_by_pos.values():
        if not lst:
            continue
        best_unc = max(
            lst,
            key=lambda rr: (
                safe_float(rr.get("uncertainty"), 0.0),
                safe_float(rr.get("expected_hvi"), 0.0),
                safe_float(rr.get("acquisition"), 0.0),
            ),
        )
        explore_seed.append(best_unc)
    explore_seed.sort(
        key=lambda rr: (
            safe_float(rr.get("uncertainty"), 0.0),
            safe_float(rr.get("explore_priority"), 0.0),
            safe_float(rr.get("acquisition"), 0.0),
        ),
        reverse=True,
    )
    global_explore = sorted(
        rows,
        key=lambda rr: (
            safe_float(rr.get("explore_priority"), 0.0),
            safe_float(rr.get("uncertainty"), 0.0),
            safe_float(rr.get("expected_hvi"), 0.0),
        ),
        reverse=True,
    )
    explore_candidates: List[Dict[str, Any]] = []
    seen_explore: Set[str] = set()
    for r in (explore_seed + global_explore):
        k = str(r.get("variant_id") or "")
        if k in seen_explore:
            continue
        seen_explore.add(k)
        explore_candidates.append(r)

    for r in explore_candidates:
        if lane_counts["explore"] >= explore_target or len(selected) >= target:
            break
        take(r, "explore")

    exploit_candidates = sorted(
        rows,
        key=lambda rr: (
            safe_float(rr.get("acquisition"), 0.0),
            safe_float(rr.get("expected_hvi"), 0.0),
            safe_float(rr.get("feasibility_prob"), 0.0),
        ),
        reverse=True,
    )
    for r in exploit_candidates:
        if lane_counts["exploit"] >= exploit_target or len(selected) >= target:
            break
        take(r, "exploit")

    rebalance_candidates = sorted(
        rows,
        key=lambda rr: (
            safe_float(rr.get("acquisition"), 0.0)
            + (0.25 * safe_float(rr.get("uncertainty"), 0.0))
            + (0.15 * safe_float(rr.get("expected_hvi"), 0.0)),
            safe_float(rr.get("expected_hvi"), 0.0),
        ),
        reverse=True,
    )
    for r in rebalance_candidates:
        if len(selected) >= target:
            break
        take(r, "rebalance")

    if len(selected) < target:
        # Relax single-vs-multi quota if panel cannot be filled under strict split.
        single_target = target
        multi_target = target
        refill = sorted(
            rows,
            key=lambda rr: (
                safe_float(rr.get("acquisition"), 0.0),
                safe_float(rr.get("expected_hvi"), 0.0),
                safe_float(rr.get("feasibility_prob"), 0.0),
            ),
            reverse=True,
        )
        for r in refill:
            if len(selected) >= target:
                break
            take(r, "quota_relax_fill")

    proposals_out_path = proposals_path(outdir=outdir, round_id=round_id)
    manifest_path = round_manifest_path(outdir=outdir, round_id=round_id)
    diagnostics_path = round_diagnostics_path(outdir=outdir, round_id=round_id)

    with proposals_out_path.open("w") as fh:
        for item in selected:
            muts = item.get("mutations") if isinstance(item.get("mutations"), list) else []
            if not muts:
                muts = [
                    {
                        "chain": str(item.get("chain") or "A"),
                        "pos": int(item.get("pos") or -1),
                        "wt": str(item.get("wt") or "").upper(),
                        "mut": str(item.get("mut") or "").upper(),
                    }
                ]
            variant_id = str(item.get("variant_id") or mutations_to_id(muts, include_chain=False))
            if not variant_id:
                continue
            muts = [m for m in muts if int(m.get("pos") or -1) > 0]
            if not muts:
                continue
            anchor = muts[0]
            chain = str(anchor.get("chain") or "A")
            pos = int(anchor.get("pos"))
            wt = str(anchor.get("wt") or "").upper()
            mut = str(anchor.get("mut") or "").upper()

            sites = item.get("sites") if isinstance(item.get("sites"), list) and item.get("sites") else []
            if not sites:
                site0 = item.get("site") if isinstance(item.get("site"), dict) else {}
                if site0:
                    sites = [site0]
            if not sites:
                for m in muts:
                    s = site_index.get((str(m.get("chain") or "A"), int(m.get("pos"))))
                    if s is not None:
                        sites.append(s)

            def min_field(name: str, default: float) -> float:
                vals = [safe_float((s or {}).get(name), float("nan")) for s in sites]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    return float(min(vals))
                return float(safe_float(item.get(name), default))

            def mean_field(name: str, default: float) -> float:
                vals = [safe_float((s or {}).get(name), float("nan")) for s in sites]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    return float(np.mean(np.asarray(vals, dtype=float)))
                return float(safe_float(item.get(name), default))

            hard_constraints = sorted(
                {str(x) for s in sites for x in ((s or {}).get("hard_constraints") or [])}
                | {str(x) for x in (item.get("hard_constraints") or [])}
            )
            soft_constraints = sorted(
                {str(x) for s in sites for x in ((s or {}).get("soft_constraints") or [])}
                | {str(x) for x in (item.get("soft_constraints") or [])}
            )
            evo_allowed = sorted(
                {str(x) for s in sites for x in ((s or {}).get("evolution_allowed_aas") or [])}
                | {str(x) for x in (item.get("evolution_allowed_aas") or [])}
            )
            labels = mutation_labels_for_variant(mutations=muts, site_index=site_index)
            p = {
                "variant_id": variant_id,
                "mutation_count": int(item.get("mutation_count", len(muts))),
                "mutations": muts,
                "chain": chain,
                "pos": pos,
                "wt": wt,
                "mut": mut,
                "tier": int(safe_float(item.get("tier"), min_field("tier", 3.0))),
                "source_role": item["source_role"],
                "move_primary": item["move_primary"],
                "move_tags": item["move_tags"],
                "chemistry_challenger": bool(item.get("chemistry_challenger")),
                "rationale": "Mutation-level multi-objective deep-ensemble acquisition.",
                "site_prefilter_keep": True,
                "site_prefilter_reasons": [],
                "numbering": (site_index.get((chain, pos)) or {}).get("numbering", {}),
                "mutation_labels": labels,
                "priority": round(float(item["acquisition"]), 8),
                "selection_lane": str(item.get("selection_lane") or "unknown"),
                "dist_ligand": min_field("dist_ligand", 999.0),
                "dist_functional": min_field("dist_functional", 999.0),
                "plddt": mean_field("plddt", 80.0),
                "exposure": mean_field("exposure", 0.5),
                "ligand_contact": bool(item.get("ligand_contact")) or any(bool((s or {}).get("ligand_contact")) for s in sites),
                "interface": bool(item.get("interface")) or any(bool((s or {}).get("interface")) for s in sites),
                "functional_site": bool(item.get("functional_site")) or any(bool((s or {}).get("functional_site")) for s in sites),
                "hard_constraints": hard_constraints,
                "soft_constraints": soft_constraints,
                "evolution_allowed_aas": evo_allowed,
                "evolution_conservation": mean_field("evolution_conservation", 0.5),
                "conservation_rank": min_field("conservation_rank", 1.0),
                "seq_prior_ensemble_plausibility": round(float(item["meta"]["plaus"]), 6),
                "stat_model": {
                    "generator": GENERATOR_TAG,
                    "training_source": str(training_source),
                    "acquisition": float(item["acquisition"]),
                    "expected_hvi": float(item["expected_hvi"]),
                    "expected_hvi_std": float(item["expected_hvi_std"]),
                    "feasibility_prob": float(item["feasibility_prob"]),
                    "uncertainty": float(safe_float(item.get("uncertainty"), 0.0)),
                    "explore_priority": float(safe_float(item.get("explore_priority"), 0.0)),
                    "objective_mean": item["objective_mean"],
                    "objective_std": item["objective_std"],
                    "bind_relevance": float(item["meta"]["bind"]),
                    "mechanistic_risk": float(item["meta"]["risk"]),
                },
            }
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    sorted_acq = sorted((safe_float(r.get("acquisition"), 0.0) for r in rows), reverse=True)
    top_k = max(1, min(10, len(sorted_acq)))
    diagnostics = {
        "round": round_id,
        "focus_round": focus_round,
        "generator": GENERATOR_TAG,
        "candidate_sites": len(candidates_sites),
        "site_filter_rejects": dict(site_filter_rejects),
        "candidate_variants_total": len(rows),
        "candidate_single_total": int(len(single_rows)),
        "candidate_multi_total": int(len(multi_rows)),
        "duplicates_filtered": int(filtered_duplicates),
        "constraints_filtered": int(filtered_constraints),
        "selected_total": len(selected),
        "selected_single_total": int(sum(1 for x in selected if int(x.get("mutation_count", 1)) <= 1)),
        "selected_multi_total": int(sum(1 for x in selected if int(x.get("mutation_count", 1)) > 1)),
        "target_requested": requested_target,
        "target_effective": target,
        "max_possible_by_position": max_possible_by_position,
        "max_per_position": max_per_position,
        "max_mutations_per_variant": int(max(1, int(args.max_mutations_per_variant))),
        "multi_point_fraction": float(clamp(float(args.multi_point_fraction), 0.0, 1.0)),
        "multi_seed_size": int(max(1, int(args.multi_seed_size))),
        "multi_max_candidates": int(max(0, int(args.multi_max_candidates))),
        "selection_targets": {
            "coverage": int(coverage_target),
            "intra_pos": int(intra_pos_target),
            "chemistry": int(chemistry_target),
            "explore": int(explore_target),
            "exploit": int(exploit_target),
        },
        "selection_lane_counts": {str(k): int(v) for k, v in sorted(lane_counts.items(), key=lambda kv: kv[0])},
        "move_mix": {str(k): int(v) for k, v in sorted(move_counts.items(), key=lambda kv: kv[0])},
        "model": {
            r.name: {
                "train_n": int(r.train_n),
                "rmse": round(float(r.rmse), 6),
                "mae": round(float(r.mae), 6),
                "r2": round(float(r.r2), 6),
                "max_iter_hits": int(r.max_iter_hits),
                "ensemble_size": int(r.ensemble_size),
            }
            for r in fit_reports
        },
        "objectives": OBJECTIVE_NAMES,
        "constraint_thresholds": {
            "function": float(thresholds[0]),
            "binding": float(thresholds[1]),
            "stability": float(thresholds[2]),
            "plausibility": float(thresholds[3]),
        },
        "mutation_gate_policy": {
            "blosum_min": int(args.blosum_min),
            "critical_blosum_min": int(args.critical_blosum_min),
            "strict_evo_conservation_threshold": float(args.strict_evo_conservation_threshold),
            "strict_evo_blosum_min": int(args.strict_evo_blosum_min),
            "functional_exploratory_enable": bool(args.functional_exploratory_enable),
            "functional_exploratory_blosum_min": int(args.functional_exploratory_blosum_min),
            "functional_exploratory_max_extra": int(max(0, args.functional_exploratory_max_extra)),
            "functional_exploratory_ligand_shell": float(max(0.0, args.functional_exploratory_ligand_shell)),
            "chemistry_coverage_enable": bool(args.chemistry_coverage_enable),
            "chemistry_coverage_max_extra": int(max(0, args.chemistry_coverage_max_extra)),
            "chemistry_coverage_blosum_min": int(args.chemistry_coverage_blosum_min),
            "chemistry_coverage_distal_enable": bool(args.chemistry_coverage_distal_enable),
            "chemistry_coverage_distal_ligand_shell": float(max(0.0, args.chemistry_coverage_distal_ligand_shell)),
            "functional_site_hard_filter": bool(args.functional_site_hard_filter),
            "near_functional_hard_filter": bool(args.near_functional_hard_filter),
            "dedupe_scope": str(args.dedupe_scope),
            "dedupe_lookback_rounds": int(max(0, args.dedupe_lookback_rounds)),
            "hard_block_constraints": sorted(HARD_BLOCK_CONSTRAINTS),
        },
        "pareto_front_size": int(front.shape[0]),
        "training_source": str(training_source),
        "training_reason": str(training_reason),
        "training_samples": int(X_train.shape[0]),
        "bootstrap_meta": bootstrap_meta,
        "expected_hvi_max": round(float(sorted_acq[0]), 10),
        "expected_hvi_mean_top10": round(float(sum(sorted_acq[:top_k]) / float(top_k)), 10),
        "expected_hvi_median": round(float(np.median(sorted_acq)), 10),
        "selected_mean_feasibility": round(float(np.mean([safe_float(x.get("feasibility_prob"), 0.0) for x in selected])) if selected else 0.0, 8),
    }

    if training_source == "round_labels" and focus_round >= 0:
        train_labels_path = proposals_vespag_path(outdir=outdir, round_id=focus_round)
    else:
        train_labels_path = outdir / "swarm" / "bootstrap_prior_labels.jsonl"
    manifest = {
        "round": round_id,
        "focus_round": focus_round,
        "generation_mode": GENERATOR_TAG,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": file_sha256(Path(__file__).resolve()),
        "input_fingerprint": build_input_fingerprint(
            focus_round=focus_round,
            train_labels_path=train_labels_path,
            site_cards_path=site_cards_path,
            context_pack_path=context_pack_path,
        ),
        "target_proposals": int(requested_target),
        "selected_proposals": int(len(selected)),
        "max_per_position": int(max_per_position),
        "generation_config": {
            "target_proposals": int(requested_target),
            "max_per_position": int(max_per_position),
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
            "functional_exploratory_max_extra": int(max(0, args.functional_exploratory_max_extra)),
            "functional_exploratory_ligand_shell": round(float(max(0.0, args.functional_exploratory_ligand_shell)), 6),
            "chemistry_coverage_enable": bool(args.chemistry_coverage_enable),
            "chemistry_coverage_max_extra": int(max(0, args.chemistry_coverage_max_extra)),
            "chemistry_coverage_blosum_min": int(args.chemistry_coverage_blosum_min),
            "chemistry_coverage_distal_enable": bool(args.chemistry_coverage_distal_enable),
            "chemistry_coverage_distal_ligand_shell": round(float(max(0.0, args.chemistry_coverage_distal_ligand_shell)), 6),
            "dedupe_scope": str(args.dedupe_scope),
            "dedupe_lookback_rounds": int(max(0, args.dedupe_lookback_rounds)),
            "ensemble_models": int(args.ensemble_models),
            "ensemble_max_iter": int(args.ensemble_max_iter),
            "min_train_samples": int(args.min_train_samples),
            "ehvi_mc": int(args.ehvi_mc),
            "min_function": round(float(args.min_function), 6),
            "min_binding": round(float(args.min_binding), 6),
            "min_stability": round(float(args.min_stability), 6),
            "min_plausibility": round(float(args.min_plausibility), 6),
            "seed": int(args.seed),
            "functional_site_hard_filter": bool(args.functional_site_hard_filter),
            "near_functional_hard_filter": bool(args.near_functional_hard_filter),
        },
        "model": {
            "ensemble_models": int(args.ensemble_models),
            "ensemble_max_iter": int(args.ensemble_max_iter),
            "min_train_samples": int(args.min_train_samples),
            "ehvi_mc": int(args.ehvi_mc),
            "max_mutations_per_variant": int(max(1, int(args.max_mutations_per_variant))),
            "multi_point_fraction": float(clamp(float(args.multi_point_fraction), 0.0, 1.0)),
            "training_source": str(training_source),
            "training_reason": str(training_reason),
        },
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    print(f"[stat-gen] Wrote proposals: {proposals_out_path}")
    print(f"[stat-gen] Wrote diagnostics: {diagnostics_path}")
    print(f"[stat-gen] Selected {len(selected)} variants from {len(rows)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
