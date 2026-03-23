import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from input_paths import resolve_canonical_fasta, resolve_canonical_protein_pdb, resolve_docking_pose_sdf
except ImportError:
    from scripts.swarm.input_paths import resolve_canonical_fasta, resolve_canonical_protein_pdb, resolve_docking_pose_sdf

AA3_TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def load_fasta(path: Path) -> Tuple[str, Optional[str]]:
    header = None
    seq = []
    if not path.exists():
        return "", None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                continue
            seq.append(line)
    sequence = "".join(seq)
    uniprot = None
    if header and "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            uniprot = parts[1]
    return sequence, uniprot


def parse_pocket_chain(path: Path) -> List[Tuple[str, int, str]]:
    out = []
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if ":" not in s:
            continue
        ch, rest = s.split(":", 1)
        icode = ""
        if rest and rest[-1].isalpha():
            icode = rest[-1]
            rest = rest[:-1]
        try:
            resi = int(rest)
        except Exception:
            continue
        out.append((ch or "A", resi, icode))
    return out


def _is_hydrogen(line: str) -> bool:
    elem = line[76:78].strip()
    if elem:
        return elem.upper() == "H"
    name = line[12:16].strip()
    return name.upper().startswith("H")


def parse_pdb(path: Path):
    per_res = {}
    chains = Counter()
    if not path.exists():
        return per_res, []
    with path.open() as fh:
        for ln in fh:
            if not ln.startswith("ATOM"):
                continue
            if _is_hydrogen(ln):
                continue
            chain = (ln[21].strip() or "A")
            resi = int(ln[22:26])
            icode = (ln[26].strip() or "")
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            b = float(ln[60:66])
            name = ln[12:16].strip()
            key = (chain, resi, icode)
            chains[chain] += 1
            slot = per_res.setdefault(key, {"atoms": [], "b": [], "ca": None})
            if "aa" not in slot:
                resname = (ln[17:20].strip() or "").upper()
                slot["aa"] = AA3_TO1.get(resname)
            slot["atoms"].append((x, y, z))
            slot["b"].append(b)
            if name == "CA":
                slot["ca"] = (x, y, z)
    chain_list = [c for c, _ in chains.most_common()]
    return per_res, chain_list


def parse_sdf_heavy_coords(path: Path) -> List[Tuple[float, float, float]]:
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    # first molecule only
    blocks = text.split("$$$$")
    if not blocks:
        return []
    lines = blocks[0].splitlines()
    if len(lines) < 5:
        return []

    coords = []
    # V3000?
    if any("V3000" in ln for ln in lines[:5]):
        try:
            b = lines.index("M  V30 BEGIN ATOM")
            e = lines.index("M  V30 END ATOM")
            for ln in lines[b+1:e]:
                parts = ln.split()
                if len(parts) < 7:
                    continue
                elem = parts[3]
                if elem.upper() == "H":
                    continue
                x, y, z = map(float, parts[4:7])
                coords.append((x, y, z))
            return coords
        except Exception:
            return []

    # V2000
    try:
        nat = int(lines[3][:3])
        start = 4
        for i in range(nat):
            ln = lines[start + i]
            x = float(ln[0:10]); y = float(ln[10:20]); z = float(ln[20:30])
            elem = ln[31:34].strip()
            if elem.upper() == "H":
                continue
            coords.append((x, y, z))
    except Exception:
        return []
    return coords


def dist2(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return dx*dx + dy*dy + dz*dz


def min_dist_res_to_ligand(res_atoms, lig_atoms) -> Optional[float]:
    if not res_atoms or not lig_atoms:
        return None
    best = float("inf")
    for ra in res_atoms:
        for la in lig_atoms:
            d2 = dist2(ra, la)
            if d2 < best:
                best = d2
    return math.sqrt(best) if best < float("inf") else None


def neighbor_counts(coords: List[Optional[Tuple[float, float, float]]], cutoff=10.0) -> List[Optional[int]]:
    cutoff2 = cutoff * cutoff
    counts = [None] * len(coords)
    for i, ci in enumerate(coords):
        if ci is None:
            continue
        c = 0
        for j, cj in enumerate(coords):
            if i == j or cj is None:
                continue
            if dist2(ci, cj) <= cutoff2:
                c += 1
        counts[i] = c
    return counts


def load_api_constraints(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    mapping = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pos = obj.get("pos")
            if isinstance(pos, int):
                mapping[pos] = obj
    return mapping


def load_numbering_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _iter_uniprot_feature_spans(api_constraints: Dict[int, Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    spans = set()
    for rec in api_constraints.values():
        uni = rec.get("uniprot") if isinstance(rec.get("uniprot"), dict) else {}
        features = []
        features.extend(uni.get("critical") or [])
        features.extend(uni.get("soft") or [])
        for feat in features:
            if not isinstance(feat, dict):
                continue
            t = str(feat.get("type") or "").strip().upper()
            try:
                start = int(feat.get("pos_start"))
            except Exception:
                continue
            try:
                end = int(feat.get("pos_end")) if feat.get("pos_end") is not None else start
            except Exception:
                end = start
            if start > 0 and end >= start:
                spans.add((t, start, end))
    return sorted(spans, key=lambda x: (x[0], x[1], x[2]))


def infer_mature_offset(
    sequence_len: int,
    api_constraints: Dict[int, Dict[str, Any]],
    numbering_cfg: Dict[str, Any],
) -> int:
    # Priority: explicit config > UniProt feature spans > default 0.
    explicit = numbering_cfg.get("mature_offset")
    try:
        if explicit is not None:
            return max(0, min(int(explicit), max(0, sequence_len - 1)))
    except Exception:
        pass

    spans = _iter_uniprot_feature_spans(api_constraints)
    chain_starts: List[int] = []
    signal_ends: List[int] = []
    for t, start, end in spans:
        if t == "CHAIN":
            chain_starts.append(start)
        if t == "SIGNAL":
            signal_ends.append(end)

    if chain_starts:
        start = min(chain_starts)
        if start > 1:
            return max(0, min(start - 1, max(0, sequence_len - 1)))
    if signal_ends:
        end = max(signal_ends)
        if end > 0:
            return max(0, min(end, max(0, sequence_len - 1)))
    return 0


def _normalize_ambler_map(raw: Any) -> Dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in raw.items():
        try:
            kk = int(k)
            vv = int(v)
        except Exception:
            continue
        if kk > 0 and vv > 0:
            out[kk] = vv
    return out


def numbering_for_position(pos: int, mature_offset: int, ambler_map: Dict[int, int]) -> Dict[str, Optional[int]]:
    mature = pos - int(mature_offset)
    if mature <= 0:
        mature = None
    ambler = ambler_map.get(pos)
    return {"raw": int(pos), "mature": mature, "ambler": ambler}


FUNCTIONAL_CONSTRAINT_TYPES = {
    "ACTIVE_SITE",
    "ACT_SITE",
    "BINDING_SITE",
    "BINDING",
    "METAL",
    "METAL_BIND",
    "COFACTOR",
    "NP_BIND",
    "MCSA_CATALYTIC",
    "MOTIF",
    "PROTON_RELAY",
}

HARD_BLOCK_CONSTRAINT_TYPES = {
    "DISULFIDE_BOND",
    "DISULFIDE",
    "DISULFID",
    "CROSSLNK",
    "CROSSLINK",
}


def _safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _prefix_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    m = sum(1 for i in range(n) if a[i] == b[i])
    return m / float(n)


def _validate_fasta_pdb_alignment(sequence: str, per_res: Dict, chain_default: str) -> None:
    compared = 0
    matches = 0
    mismatch_examples = []
    for pos, wt in enumerate(sequence, start=1):
        rec = per_res.get((chain_default, pos, ""))
        if not rec:
            continue
        aa = (rec.get("aa") or "").strip().upper()
        if not aa:
            continue
        compared += 1
        if aa == wt:
            matches += 1
        elif len(mismatch_examples) < 8:
            mismatch_examples.append(f"{pos}:{aa}->{wt}")

    if compared <= 0:
        return
    pid = matches / float(compared)
    if compared >= 30 and pid < 0.70:
        raise SystemExit(
            "ERROR: FASTA/PDB residue identity mismatch is too large for SWARM indexing. "
            f"(chain={chain_default} compared={compared} identity={pid:.3f} "
            f"examples={','.join(mismatch_examples)}) "
            "Likely stale or mixed inputs; regenerate canonical inputs with scripts/swarm/14a_prepare_inputs.py."
        )


def _functional_types_from_api(api: Optional[Dict]) -> List[str]:
    if not isinstance(api, dict):
        return []
    out: List[str] = []
    crit = (api.get("uniprot") or {}).get("critical") or []
    for c in crit:
        if isinstance(c, dict):
            t = str(c.get("type") or "").strip().upper()
            if t:
                out.append(t)
    # InterPro sites are treated as motif/functional anchors.
    sites = (api.get("interpro") or {}).get("sites") or []
    if sites:
        out.append("INTERPRO_SITE")
    mcsa = api.get("mcsa") if isinstance(api.get("mcsa"), dict) else {}
    if (mcsa.get("critical") or []):
        out.append("MCSA_CATALYTIC")
    return sorted(set(out))


def _is_functional_site_from_api(api: Optional[Dict], hard_constraints: List[str], do_not_mutate: bool) -> Tuple[bool, List[str]]:
    reasons = set()
    if do_not_mutate:
        reasons.add("policy_do_not_mutate")
    for h in hard_constraints or []:
        hh = str(h or "").strip().upper()
        if hh in FUNCTIONAL_CONSTRAINT_TYPES:
            reasons.add(f"hard:{hh}")
    for t in _functional_types_from_api(api):
        if t in FUNCTIONAL_CONSTRAINT_TYPES or t == "INTERPRO_SITE":
            reasons.add(f"api:{t}")
    return (len(reasons) > 0), sorted(reasons)


def _is_hard_do_not_mutate(do_not_mutate: bool, hard_constraints: List[str], policy_reason: Optional[str]) -> Tuple[bool, str]:
    """
    Distinguish immutable structural locks from soft functional guidance.

    This keeps `do_not_mutate` provenance from API annotations, but only treats
    covalent/structural constraints as strict hard blocks in downstream generation.
    """
    hard = {str(h or "").strip().upper() for h in (hard_constraints or [])}
    reason = str(policy_reason or "").strip().upper()
    if any(h in HARD_BLOCK_CONSTRAINT_TYPES for h in hard):
        return True, "hard_constraint_structural_lock"
    if bool(do_not_mutate) and reason in HARD_BLOCK_CONSTRAINT_TYPES:
        return True, "policy_reason_structural_lock"
    if bool(do_not_mutate):
        return False, "policy_soft_functional"
    return False, "none"


def _dist_to_functional(
    residue_coords: List[Optional[Tuple[float, float, float]]],
    functional_positions: List[int],
) -> List[Optional[float]]:
    if not residue_coords or not functional_positions:
        return [None] * len(residue_coords)
    out: List[Optional[float]] = [None] * len(residue_coords)
    fcoords: List[Tuple[int, Tuple[float, float, float]]] = []
    for pos in functional_positions:
        if 1 <= pos <= len(residue_coords):
            c = residue_coords[pos - 1]
            if c is not None:
                fcoords.append((pos, c))
    if not fcoords:
        return out
    for i, c in enumerate(residue_coords, start=1):
        if c is None:
            continue
        best = None
        for fpos, fc in fcoords:
            if fpos == i:
                continue
            d = math.sqrt(dist2(c, fc))
            if best is None or d < best:
                best = d
        out[i - 1] = best
    return out


def _conservation_ranks(cons_vals: List[Optional[float]]) -> List[Optional[float]]:
    idx_vals = [(i, v) for i, v in enumerate(cons_vals) if v is not None]
    out: List[Optional[float]] = [None] * len(cons_vals)
    if not idx_vals:
        return out
    idx_vals.sort(key=lambda x: float(x[1]), reverse=True)  # high conservation first
    denom = max(1, len(idx_vals) - 1)
    for rank, (idx, _v) in enumerate(idx_vals):
        out[idx] = float(rank) / float(denom)
    return out


def _infer_proxy_functional_positions(cards: List[Dict]) -> List[int]:
    """
    Fallback functional-site inference when API residue constraints are unavailable.
    Uses ligand geometry and pocket membership to create soft functional anchors.
    """
    if not cards:
        return []

    total = len(cards)
    target = int(round(0.08 * float(total)))
    target = max(8, min(24, target))

    auto_include: List[Tuple[float, int]] = []
    scored: List[Tuple[float, int]] = []
    for card in cards:
        pos = int(card.get("pos") or 0)
        if pos <= 0:
            continue
        d_lig = _safe_float(card.get("dist_ligand"))
        lig_contact = bool(card.get("ligand_contact"))
        fpocket_member = bool(card.get("fpocket_member"))
        tier = int(card.get("tier") or 3)
        exposure = _safe_float(card.get("exposure"))

        if d_lig is None and not lig_contact and not fpocket_member:
            continue

        prox = math.exp(-max(0.0, float(d_lig if d_lig is not None else 12.0)) / 3.5)
        contact_bonus = 0.32 if lig_contact else 0.0
        pocket_bonus = 0.20 if fpocket_member else 0.0
        tier_bonus = 0.12 if tier == 0 else (0.08 if tier == 1 else (0.04 if tier == 2 else 0.0))
        exposure_bonus = 0.04 if (exposure is not None and 0.05 <= exposure <= 0.85) else 0.0

        score = float(prox + contact_bonus + pocket_bonus + tier_bonus + exposure_bonus)
        scored.append((score, pos))

        if d_lig is not None and d_lig <= 4.5:
            auto_include.append((score + 10.0, pos))

    if not scored:
        return []

    chosen: List[int] = []
    seen = set()
    for _s, pos in sorted(auto_include, reverse=True):
        if pos in seen:
            continue
        seen.add(pos)
        chosen.append(pos)

    if len(chosen) < target:
        for _s, pos in sorted(scored, reverse=True):
            if pos in seen:
                continue
            seen.add(pos)
            chosen.append(pos)
            if len(chosen) >= target:
                break

    return sorted(chosen)


def infer_ss(pdb_path: Path, chain_default: str) -> Dict[Tuple[str, int, str], str]:
    # best-effort DSSP if available
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except Exception:
        return {}

    import shutil
    if shutil.which("mkdssp") is None:
        return {}

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("af", str(pdb_path))
        model = next(structure.get_models())
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        return {}

    out = {}
    for (chain_id, res_id), d in dssp.property_dict.items():
        resi = res_id[1]
        icode = res_id[2].strip() if isinstance(res_id[2], str) else ""
        out[(chain_id, resi, icode)] = d.get("ss", "U")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SWARM site_cards.jsonl")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--swarm-dir", default=None)
    ap.add_argument("--api-residue-constraints", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--docked-sdf", default=None)
    ap.add_argument("--pocket-chain", default=None)
    ap.add_argument("--numbering-config", default=None, help="Optional numbering config JSON (e.g., mature_offset, ambler_map).")
    ap.add_argument("--pdbe_predicted_cap", type=float, default=0.5,
                    help="If predicted sites cover more than this fraction of residues, ignore PDBe predicted-site flags.")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(os.environ.get("OUTDIR", "./out"))
    swarm_dir = Path(args.swarm_dir) if args.swarm_dir else outdir / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = resolve_canonical_fasta(outdir, explicit=(Path(args.fasta) if args.fasta else None))
    if fasta_path is None:
        raise SystemExit("ERROR: Could not resolve FASTA. Run scripts/swarm/14a_prepare_inputs.py or pass --fasta.")
    sequence, _ = load_fasta(fasta_path)
    if not sequence:
        raise SystemExit(f"ERROR: FASTA has no sequence: {fasta_path}")

    pdb_path = resolve_canonical_protein_pdb(outdir, explicit=(Path(args.pdb) if args.pdb else None))
    if pdb_path is None:
        raise SystemExit("ERROR: Could not resolve receptor PDB. Run scripts/swarm/14a_prepare_inputs.py or pass --pdb.")

    docked_sdf = resolve_docking_pose_sdf(outdir, explicit=(Path(args.docked_sdf) if args.docked_sdf else None))
    pocket_path = Path(args.pocket_chain) if args.pocket_chain else outdir / "pocket_residues.chain.txt"
    api_path = Path(args.api_residue_constraints) if args.api_residue_constraints else outdir / "swarm_api" / "residue_constraints.jsonl"
    numbering_cfg_path = Path(args.numbering_config) if args.numbering_config else swarm_dir / "numbering_map.json"

    pocket_chain = parse_pocket_chain(pocket_path)
    pocket_set = set(pocket_chain)
    pocket_resi_set = {(ch, resi) for (ch, resi, _ic) in pocket_chain}

    per_res, chains = parse_pdb(pdb_path) if pdb_path else ({}, [])
    chain_default = chains[0] if chains else (pocket_chain[0][0] if pocket_chain else "A")
    _validate_fasta_pdb_alignment(sequence, per_res, chain_default)

    ligand_coords = parse_sdf_heavy_coords(docked_sdf) if docked_sdf and docked_sdf.exists() else []
    api_constraints = load_api_constraints(api_path)
    numbering_cfg = load_numbering_config(numbering_cfg_path)
    mature_offset = infer_mature_offset(len(sequence), api_constraints, numbering_cfg)
    ambler_map = _normalize_ambler_map(numbering_cfg.get("ambler_map"))
    ambler_source = "config_file" if ambler_map else "none"

    # Determine if PDBe predicted-site flags are too broad
    pred_count = 0
    for obj in api_constraints.values():
        if obj.get("pdbe", {}).get("predicted_sites"):
            pred_count += 1
    pred_fraction = (pred_count / len(sequence)) if sequence else 0.0
    ignore_predicted = pred_fraction > args.pdbe_predicted_cap

    ss_map = infer_ss(pdb_path, chain_default) if pdb_path else {}

    # build per-position coords and metrics
    residue_coords = []  # per position (1-based)
    plddt_vals = []
    dist_vals = []

    for pos, wt in enumerate(sequence, start=1):
        key = (chain_default, pos, "")
        res = per_res.get(key)
        if res is None:
            residue_coords.append(None)
            plddt_vals.append(None)
            dist_vals.append(None)
            continue
        ca = res.get("ca")
        if ca is None and res.get("atoms"):
            # centroid fallback
            xs, ys, zs = zip(*res["atoms"])
            ca = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
        residue_coords.append(ca)
        plddt = sum(res["b"]) / len(res["b"]) if res.get("b") else None
        plddt_vals.append(plddt)
        dist = min_dist_res_to_ligand(res.get("atoms", []), ligand_coords) if ligand_coords else None
        dist_vals.append(dist)

    # exposure proxy via neighbor count
    neighbor_cnts = neighbor_counts(residue_coords, cutoff=10.0)
    valid_counts = [c for c in neighbor_cnts if c is not None]
    if valid_counts:
        cmin, cmax = min(valid_counts), max(valid_counts)
    else:
        cmin, cmax = 0, 0

    def exposure_from_count(c: Optional[int]) -> Optional[float]:
        if c is None:
            return None
        if cmax == cmin:
            return 0.5
        return 1.0 - (c - cmin) / (cmax - cmin)

    # Build cards first, then attach global metrics (distance-to-functional, conservation ranks).
    cards: List[Dict] = []
    functional_positions: List[int] = []
    conservation_vals: List[Optional[float]] = []
    for idx, wt in enumerate(sequence, start=1):
        key = (chain_default, idx, "")
        dist = dist_vals[idx - 1]
        fpocket_member = key in pocket_set or (chain_default, idx) in pocket_resi_set
        tier = 3
        if fpocket_member:
            tier = 0
        elif dist is not None:
            if dist <= 6.0:
                tier = 1
            elif dist <= 10.0:
                tier = 2

        fpocket_conf = 0.0
        if fpocket_member:
            fpocket_conf = 1.0
        elif dist is not None:
            if dist <= 6.0:
                fpocket_conf = 0.65
            elif dist <= 10.0:
                fpocket_conf = 0.35
            elif dist <= 12.0:
                fpocket_conf = 0.15

        api = api_constraints.get(idx)
        do_not_mutate = False
        hard_constraints: List[str] = []
        soft_constraints: List[str] = []
        tags: List[str] = []
        risk_flags: List[str] = []
        curated = {}
        if api:
            do_not_mutate = bool(api.get("policy", {}).get("do_not_mutate"))
            do_not_mutate_reason = str(api.get("policy", {}).get("reason") or "")
            crit = api.get("uniprot", {}).get("critical", [])
            hard_constraints = [c.get("type") for c in crit if isinstance(c, dict)]
            soft = api.get("uniprot", {}).get("soft", [])
            soft_constraints = [c.get("type") for c in soft if isinstance(c, dict)]
            pdbe = api.get("pdbe", {})
            interpro = api.get("interpro", {})
            evolution = api.get("evolution", {}) if isinstance(api.get("evolution"), dict) else {}
            mcsa = api.get("mcsa", {}) if isinstance(api.get("mcsa"), dict) else {}
            stability = api.get("stability", {}) if isinstance(api.get("stability"), dict) else {}
            if (mcsa.get("critical") or []) and "MCSA_CATALYTIC" not in hard_constraints:
                hard_constraints.append("MCSA_CATALYTIC")
            curated = {
                "uniprot": api.get("uniprot", {}),
                "pdbe": pdbe,
                "interpro": interpro,
                "mcsa": mcsa,
                "stability": stability,
                "evolution": {
                    "conservation": evolution.get("conservation"),
                    "allowed_aas": evolution.get("allowed_aas") or [],
                },
            }
            if pdbe.get("ligand_sites", {}).get("count", 0) > 0:
                tags.append("pdbe_ligand_contact")
            if pdbe.get("interface", {}).get("is_interface"):
                soft_constraints.append("PDBe_INTERFACE")
                tags.append("pdbe_interface")
                risk_flags.append("pdbe_interface")
            if pdbe.get("predicted_sites") and not ignore_predicted:
                soft_constraints.append("PDBe_PREDICTED_SITE")
                risk_flags.append("pdbe_predicted_site")
            for d in interpro.get("domains", []):
                tags.append(f"domain:{d}")
        else:
            do_not_mutate_reason = ""

        tags.append(f"tier{tier}")
        exposure = exposure_from_count(neighbor_cnts[idx - 1])
        evo_cons = (curated.get("evolution") or {}).get("conservation")
        evo_cons = _safe_float(evo_cons)
        conservation_vals.append(evo_cons)

        functional_site, functional_reasons = _is_functional_site_from_api(
            api=api,
            hard_constraints=[h for h in hard_constraints if h],
            do_not_mutate=do_not_mutate,
        )
        do_not_mutate_hard, do_not_mutate_mode = _is_hard_do_not_mutate(
            do_not_mutate=do_not_mutate,
            hard_constraints=[h for h in hard_constraints if h],
            policy_reason=do_not_mutate_reason,
        )
        if functional_site:
            functional_positions.append(idx)

        card = {
            "pos": idx,
            "chain": chain_default,
            "wt": wt,
            "numbering": numbering_for_position(idx, mature_offset, ambler_map),
            "tier": tier,
            "dist_ligand": dist,
            "plddt": plddt_vals[idx - 1],
            "ss": ss_map.get(key, "U"),
            "exposure": exposure,
            "fpocket_member": bool(fpocket_member),
            "fpocket_occupancy_confidence": round(float(fpocket_conf), 6),
            "neighbor_count_10A": neighbor_cnts[idx - 1],
            "do_not_mutate": do_not_mutate,
            "do_not_mutate_hard": bool(do_not_mutate_hard),
            "do_not_mutate_mode": do_not_mutate_mode,
            "do_not_mutate_reason": do_not_mutate_reason,
            "hard_constraints": [h for h in hard_constraints if h],
            "soft_constraints": [s for s in soft_constraints if s],
            "tags": tags,
            "ligand_contact": bool(dist is not None and dist <= 5.0) or ("pdbe_ligand_contact" in tags),
            "functional_site": bool(functional_site),
            "proxy_functional_site": False,
            "functional_reasons": functional_reasons,
            "evolution_conservation": evo_cons,
            "evolution_allowed_aas": (curated.get("evolution") or {}).get("allowed_aas") or [],
        }
        ddg_fold = _safe_float((curated.get("stability") or {}).get("ddg_fold"))
        if ddg_fold is None and isinstance(api, dict):
            ddg_fold = _safe_float(api.get("ddg_fold"))
        if ddg_fold is not None:
            card["ddg_fold"] = ddg_fold
        if exposure is not None:
            card["buried_core"] = bool(exposure < 0.10)
        if risk_flags:
            card["risk_flags"] = sorted(set(risk_flags))
        if curated:
            card["curated"] = curated
        cards.append(card)

    functional_positions_from_api = bool(functional_positions)

    # API constraints may be missing in offline/minimal runs.
    # Infer soft functional anchors from ligand geometry so recursive SWARM can
    # still focus around pocket-active neighborhoods.
    if not functional_positions:
        proxy_positions = _infer_proxy_functional_positions(cards)
        if proxy_positions:
            proxy_set = set(proxy_positions)
            for card in cards:
                pos = int(card.get("pos") or 0)
                if pos not in proxy_set:
                    continue
                card["proxy_functional_site"] = True
                reasons = set(card.get("functional_reasons") or [])
                reasons.add("proxy:ligand_geometry")
                if bool(card.get("ligand_contact")):
                    reasons.add("proxy:ligand_contact")
                if bool(card.get("fpocket_member")):
                    reasons.add("proxy:fpocket_member")
                card["functional_reasons"] = sorted(reasons)

                soft = set(str(x or "") for x in (card.get("soft_constraints") or []))
                soft.add("POCKET_PROXY")
                card["soft_constraints"] = sorted(x for x in soft if x)

                tags = set(str(x or "") for x in (card.get("tags") or []))
                tags.add("proxy_functional")
                card["tags"] = sorted(x for x in tags if x)
            functional_positions = proxy_positions

    # Global residue-level metrics used by universal first-pass filter.
    dist_to_func = _dist_to_functional(residue_coords, sorted(set(functional_positions)))
    cons_ranks = _conservation_ranks(conservation_vals)
    cons_top_fraction = 0.25
    for i, card in enumerate(cards):
        df = dist_to_func[i]
        card["dist_functional"] = df
        rank = cons_ranks[i]
        card["conservation_rank"] = rank
        card["conservation_top_fraction"] = cons_top_fraction
        if card.get("exposure") is not None:
            card["buried_core"] = bool(float(card["exposure"]) < 0.10)
        structural_lock = False
        wt = str(card.get("wt") or "").upper()
        ss = str(card.get("ss") or "U")
        if wt in {"G", "P"} and ss in {"H", "G", "I", "T", "S", "E"}:
            structural_lock = True
        if "DISULFIDE" in set(card.get("hard_constraints") or []) or "DISULFID" in set(card.get("hard_constraints") or []):
            structural_lock = True
        card["structural_lock"] = structural_lock

        # Residue-level prefilter hints (final hard filtering is in 15e).
        reasons: List[str] = []
        d_lig = _safe_float(card.get("dist_ligand"))
        d_func = _safe_float(card.get("dist_functional"))
        exp = _safe_float(card.get("exposure"))
        if d_lig is None or d_lig > 8.0:
            reasons.append("ligand_shell_outside_8A")
        if card.get("functional_site"):
            reasons.append("functional_annotation")
        elif card.get("proxy_functional_site"):
            reasons.append("proxy_functional_site")
        if d_func is not None and d_func < 1.0:
            reasons.append("near_functional_site")
        if exp is not None and exp < 0.15:
            reasons.append("buried_core_like")
        if rank is not None and rank <= cons_top_fraction:
            reasons.append("high_conservation_rank")
        if structural_lock:
            reasons.append("structural_lock")
        card["site_prefilter_keep"] = len(reasons) == 0
        card["site_prefilter_reasons"] = reasons
        card["site_prefilter_stats"] = {
            "d_lig": d_lig,
            "d_func": d_func,
            "exposure": exp,
            "conservation_rank": rank,
        }

    # write site cards
    out_path = swarm_dir / "site_cards.jsonl"
    cache_dir = swarm_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for card in cards:
            fh.write(json.dumps(card, ensure_ascii=False) + "\n")

    numbering_manifest = {
        "generated_from": str(numbering_cfg_path),
        "mature_offset": int(mature_offset),
        "has_ambler_map": bool(ambler_map),
        "ambler_mapped_positions": int(len(ambler_map)),
        "ambler_source": str(ambler_source),
        "feature_span_count": len(_iter_uniprot_feature_spans(api_constraints)),
        "functional_positions_count": int(len(sorted(set(functional_positions)))),
        "functional_source": "api_constraints" if functional_positions_from_api else ("ligand_geometry_proxy" if functional_positions else "none"),
    }
    (swarm_dir / "numbering_manifest.json").write_text(json.dumps(numbering_manifest, ensure_ascii=False, indent=2))

    # Persist effective numbering config so downstream tools (ground-truth checker, later rounds)
    # see the same explicit numbering configuration used for site-card generation.
    try:
        effective_cfg = {
            "mature_offset": int(mature_offset),
            "ambler_map": {str(k): int(v) for k, v in sorted(ambler_map.items())},
            "_autogenerated": bool(ambler_source != "config_file"),
            "_ambler_source": str(ambler_source),
        }
        numbering_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        numbering_cfg_path.write_text(json.dumps(effective_cfg, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass

    # ---- cache arrays for reuse ----
    def _to_nan_vec(x):
        if x is None:
            return [math.nan, math.nan, math.nan]
        return [float(x[0]), float(x[1]), float(x[2])]

    residue_coords_np = [_to_nan_vec(c) for c in residue_coords]
    dist_np = [float(d) if d is not None else math.nan for d in dist_vals]
    lig_np = [[float(x), float(y), float(z)] for (x, y, z) in ligand_coords] if ligand_coords else []

    try:
        import numpy as np  # optional

        np.save(cache_dir / "residue_coords.npy", np.asarray(residue_coords_np, dtype=float))
        np.save(cache_dir / "ligand_coords.npy", np.asarray(lig_np, dtype=float))
        np.save(cache_dir / "dist_to_ligand.npy", np.asarray(dist_np, dtype=float))
    except Exception:
        # fallback JSON if numpy not available
        (cache_dir / "residue_coords.json").write_text(json.dumps(residue_coords_np))
        (cache_dir / "ligand_coords.json").write_text(json.dumps(lig_np))
        (cache_dir / "dist_to_ligand.json").write_text(json.dumps(dist_np))

    print("Wrote:", out_path)
    print("Wrote:", swarm_dir / "numbering_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
