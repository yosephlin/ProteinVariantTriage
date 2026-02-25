import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

METALS = {
    "ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "NA", "K", "CD", "HG", "PB",
}
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


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def parse_pdb(path: Path) -> Dict[Tuple[str, int, str], Dict]:
    residues: Dict[Tuple[str, int, str], Dict] = {}
    with path.open() as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except Exception:
                continue
            icode = line[26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            try:
                b = float(line[60:66])
            except Exception:
                b = None
            element = line[76:78].strip()
            if not element:
                element = atom_name[0]
            # altloc handling: keep blank or A
            if altloc and altloc not in {"A"}:
                continue
            key = (chain, resseq, icode)
            rec = residues.setdefault(key, {"resname": resname, "atoms": {}, "bfactors": []})
            rec["atoms"][atom_name] = (x, y, z, element)
            if b is not None:
                rec["bfactors"].append(b)
    return residues


def residue_ca(res) -> Optional[Tuple[float, float, float]]:
    if "CA" in res["atoms"]:
        x, y, z, _ = res["atoms"]["CA"]
        return (x, y, z)
    for _, (x, y, z, _) in res["atoms"].items():
        return (x, y, z)
    return None


def residue_plddt(res) -> Optional[float]:
    if res["bfactors"]:
        return sum(res["bfactors"]) / len(res["bfactors"])
    return None


def parse_sdf_coords(path: Path) -> List[Tuple[float, float, float]]:
    coords = []
    if not path.exists():
        return coords
    lines = path.read_text(errors="ignore").splitlines()
    if len(lines) < 4:
        return coords
    try:
        counts = lines[3]
        n_atoms = int(counts[0:3])
    except Exception:
        return coords
    start = 4
    for i in range(start, start + n_atoms):
        if i >= len(lines):
            break
        line = lines[i]
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
        except Exception:
            continue
        elem = line[31:34].strip()
        if elem.upper() == "H":
            continue
        coords.append((x, y, z))
    return coords


def compute_neighbor_counts(residues, radius=10.0) -> Dict[Tuple[str, int, str], int]:
    keys = list(residues.keys())
    ca_coords = {}
    for k in keys:
        ca = residue_ca(residues[k])
        if ca:
            ca_coords[k] = ca
    counts = {k: 0 for k in ca_coords}
    for i, k1 in enumerate(keys):
        if k1 not in ca_coords:
            continue
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            if k2 not in ca_coords:
                continue
            if _dist(ca_coords[k1], ca_coords[k2]) <= radius:
                counts[k1] += 1
                counts[k2] += 1
    return counts


def compute_ligand_contacts(residues, ligand_coords, cutoff=4.5):
    contact = {}
    min_dist = {}
    if not ligand_coords:
        return contact, min_dist
    for key, res in residues.items():
        best = None
        for _, (x, y, z, elem) in res["atoms"].items():
            if elem.upper() == "H":
                continue
            for lc in ligand_coords:
                d = _dist((x, y, z), lc)
                if best is None or d < best:
                    best = d
        if best is None:
            continue
        min_dist[key] = best
        contact[key] = best <= cutoff
    return contact, min_dist


def compute_interface_contacts(residues, cutoff=5.0):
    chains = {}
    for key, res in residues.items():
        chain = key[0]
        for _, (x, y, z, elem) in res["atoms"].items():
            if elem.upper() == "H":
                continue
            chains.setdefault(chain, []).append((x, y, z))
    if len(chains) <= 1:
        return {}
    interfaces = {}
    for key, res in residues.items():
        chain = key[0]
        other_atoms = []
        for ch, atoms in chains.items():
            if ch != chain:
                other_atoms.extend(atoms)
        is_iface = False
        for _, (x, y, z, elem) in res["atoms"].items():
            if elem.upper() == "H":
                continue
            for ox, oy, oz in other_atoms:
                if _dist((x, y, z), (ox, oy, oz)) <= cutoff:
                    is_iface = True
                    break
            if is_iface:
                break
        interfaces[key] = is_iface
    return interfaces


def compute_disulfides(residues, cutoff=2.3):
    cys = []
    for key, res in residues.items():
        if res.get("resname") == "CYS" and "SG" in res["atoms"]:
            x, y, z, _ = res["atoms"]["SG"]
            cys.append((key, (x, y, z)))
    disulfide = {k: False for k, _ in cys}
    for i in range(len(cys)):
        k1, c1 = cys[i]
        for j in range(i + 1, len(cys)):
            k2, c2 = cys[j]
            if _dist(c1, c2) <= cutoff:
                disulfide[k1] = True
                disulfide[k2] = True
    return disulfide


def compute_metal_contacts(residues, cutoff=2.7):
    metals = []
    for key, res in residues.items():
        if res.get("resname") not in AA3_TO_1:
            for _, (x, y, z, elem) in res["atoms"].items():
                if elem.upper() in METALS:
                    metals.append((x, y, z))
    if not metals:
        return {}
    metal_contact = {}
    for key, res in residues.items():
        is_contact = False
        for _, (x, y, z, elem) in res["atoms"].items():
            if elem.upper() == "H":
                continue
            for m in metals:
                if _dist((x, y, z), m) <= cutoff:
                    is_contact = True
                    break
            if is_contact:
                break
        metal_contact[key] = is_contact
    return metal_contact


def compute_dist_to_functional(
    residues: Dict[Tuple[str, int, str], Dict],
    cards: List[Dict],
) -> Dict[Tuple[str, int, str], Optional[float]]:
    def _ca(k):
        res = residues.get(k)
        if not res:
            return None
        return residue_ca(res)

    functional_keys = []
    for card in cards:
        chain = card.get("chain") or "A"
        pos = int(card.get("pos"))
        key = (chain, pos, "")
        if key not in residues:
            for kk in residues.keys():
                if kk[0] == chain and kk[1] == pos:
                    key = kk
                    break
        hard = {str(x or "").upper() for x in (card.get("hard_constraints") or [])}
        if bool(card.get("functional_site")) or bool(card.get("do_not_mutate")) or any(h in FUNCTIONAL_CONSTRAINT_TYPES for h in hard):
            if _ca(key) is not None:
                functional_keys.append(key)

    out: Dict[Tuple[str, int, str], Optional[float]] = {}
    if not functional_keys:
        return out
    for key in residues.keys():
        c = _ca(key)
        if c is None:
            out[key] = None
            continue
        best = None
        for fk in functional_keys:
            if fk == key:
                continue
            fc = _ca(fk)
            if fc is None:
                continue
            d = _dist(c, fc)
            if best is None or d < best:
                best = d
        out[key] = best
    return out


def load_site_cards(path: Path) -> List[Dict]:
    cards = []
    if not path.exists():
        return cards
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except Exception:
                continue
    return cards


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich site_cards with PDB-derived features")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--site-cards", required=True)
    ap.add_argument("--ligand-sdf", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ligand-cutoff", type=float, default=4.5)
    ap.add_argument("--interface-cutoff", type=float, default=5.0)
    ap.add_argument("--metal-cutoff", type=float, default=2.7)
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    site_cards_path = Path(args.site_cards)
    out_path = Path(args.out) if args.out else site_cards_path

    residues = parse_pdb(pdb_path)
    ligand_coords = parse_sdf_coords(Path(args.ligand_sdf)) if args.ligand_sdf else []

    neighbor_counts = compute_neighbor_counts(residues, radius=10.0)
    ligand_contact, ligand_min_dist = compute_ligand_contacts(residues, ligand_coords, cutoff=args.ligand_cutoff)
    interface_contact = compute_interface_contacts(residues, cutoff=args.interface_cutoff)
    disulfide = compute_disulfides(residues)
    metal_contact = compute_metal_contacts(residues, cutoff=args.metal_cutoff)

    cards = load_site_cards(site_cards_path)
    dist_functional = compute_dist_to_functional(residues, cards)
    updated = []
    for card in cards:
        chain = card.get("chain") or "A"
        pos = int(card.get("pos"))
        key = (chain, pos, "")
        if key not in residues:
            for k in residues.keys():
                if k[0] == chain and k[1] == pos:
                    key = k
                    break
        res = residues.get(key)
        if res:
            plddt = residue_plddt(res)
            if plddt is not None:
                card.setdefault("plddt", plddt)
            if key in neighbor_counts:
                card["neighbor_count_10A"] = neighbor_counts[key]
            if key in ligand_min_dist:
                card["dist_ligand"] = float(ligand_min_dist[key])
            card["ligand_contact"] = bool(ligand_contact.get(key)) or bool((card.get("dist_ligand") is not None and float(card.get("dist_ligand")) <= args.ligand_cutoff))
            tags = card.get("tags") or []
            if ligand_contact.get(key):
                if "pdb_ligand_contact" not in tags:
                    tags.append("pdb_ligand_contact")
            if interface_contact.get(key):
                if "pdb_interface" not in tags:
                    tags.append("pdb_interface")
            if disulfide.get(key):
                if "DISULFIDE" not in (card.get("hard_constraints") or []):
                    card.setdefault("hard_constraints", []).append("DISULFIDE")
            if metal_contact.get(key):
                if "METAL" not in (card.get("hard_constraints") or []):
                    card.setdefault("hard_constraints", []).append("METAL")
            card["tags"] = tags
            card["functional_site"] = bool(card.get("functional_site")) or bool(card.get("do_not_mutate")) or any(
                str(h or "").upper() in FUNCTIONAL_CONSTRAINT_TYPES for h in (card.get("hard_constraints") or [])
            )
            card["dist_functional"] = dist_functional.get(key)
            exp = card.get("exposure")
            try:
                exp_f = float(exp) if exp is not None else None
            except Exception:
                exp_f = None
            if exp_f is not None:
                card["buried_core"] = bool(exp_f < 0.10)
            wt = str(card.get("wt") or "").upper()
            ss = str(card.get("ss") or "U")
            card["structural_lock"] = bool(card.get("structural_lock")) or (wt in {"G", "P"} and ss in {"H", "G", "I", "T", "S", "E"})

            # keep residue-level prefilter diagnostics in sync after distance refresh
            reasons = []
            d_lig = card.get("dist_ligand")
            try:
                d_lig = float(d_lig) if d_lig is not None else None
            except Exception:
                d_lig = None
            d_func = card.get("dist_functional")
            try:
                d_func = float(d_func) if d_func is not None else None
            except Exception:
                d_func = None
            rank = card.get("conservation_rank")
            try:
                rank = float(rank) if rank is not None else None
            except Exception:
                rank = None
            top_frac = card.get("conservation_top_fraction")
            try:
                top_frac = float(top_frac) if top_frac is not None else 0.15
            except Exception:
                top_frac = 0.35
            if d_lig is None or d_lig > 8.0:
                reasons.append("ligand_shell_outside_8A")
            if card.get("functional_site"):
                reasons.append("functional_annotation")
            if d_func is not None and d_func < 1.0:
                reasons.append("near_functional_site")
            if exp_f is not None and exp_f < 0.15:
                reasons.append("buried_core_like")
            if rank is not None and rank <= top_frac:
                reasons.append("high_conservation_rank")
            if card.get("structural_lock"):
                reasons.append("structural_lock")
            card["site_prefilter_keep"] = len(reasons) == 0
            card["site_prefilter_reasons"] = reasons
            card["site_prefilter_stats"] = {
                "d_lig": d_lig,
                "d_func": d_func,
                "exposure": exp_f,
                "conservation_rank": rank,
            }
            card["pdb_features"] = {
                "ligand_contact": bool(ligand_contact.get(key)),
                "ligand_min_dist": float(ligand_min_dist.get(key)) if key in ligand_min_dist else None,
                "interface_contact": bool(interface_contact.get(key)),
                "disulfide": bool(disulfide.get(key)),
                "metal_contact": bool(metal_contact.get(key)),
            }
        updated.append(card)

    out_path.write_text("\n".join(json.dumps(c) for c in updated) + "\n")
    print(f"Wrote enriched site cards: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
