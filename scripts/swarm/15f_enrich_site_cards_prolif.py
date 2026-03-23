import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from input_paths import resolve_canonical_protein_pdb, resolve_docking_pose_sdf
except ImportError:
    from scripts.swarm.input_paths import resolve_canonical_protein_pdb, resolve_docking_pose_sdf


def _load_site_cards(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _save_site_cards(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_docking_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _norm_chain(chain: Optional[str]) -> str:
    c = (chain or "").strip()
    return c if c else "A"


def _residue_key_from_string(text: str) -> Optional[Tuple[str, int]]:
    # Handles common formats such as "ASP129.A", "A:129", "129A", "ASP129"
    t = text.strip()
    m = re.search(r"([A-Za-z0-9])[:.](\d+)$", t)
    if m:
        return (_norm_chain(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)\.([A-Za-z0-9])$", t)
    if m:
        return (_norm_chain(m.group(2)), int(m.group(1)))
    m = re.search(r"([A-Za-z]+)?(\d+)(?:\.([A-Za-z0-9]))?$", t)
    if m:
        chain = _norm_chain(m.group(3))
        return (chain, int(m.group(2)))
    return None


def _residue_to_key(residue_obj) -> Optional[Tuple[str, int]]:
    # ProLIF residue objects usually expose chain and number.
    for chain_attr in ("chain", "chain_id"):
        if hasattr(residue_obj, chain_attr):
            for pos_attr in ("number", "resid", "resi"):
                if hasattr(residue_obj, pos_attr):
                    try:
                        chain = _norm_chain(str(getattr(residue_obj, chain_attr)))
                        pos = int(getattr(residue_obj, pos_attr))
                        return (chain, pos)
                    except Exception:
                        pass
    try:
        if hasattr(residue_obj, "name") and hasattr(residue_obj, "number"):
            chain = _norm_chain(str(getattr(residue_obj, "chain", "A")))
            pos = int(getattr(residue_obj, "number"))
            return (chain, pos)
    except Exception:
        pass
    try:
        key = _residue_key_from_string(str(residue_obj))
        if key:
            return key
    except Exception:
        pass
    return None


def _iter_ligands_from_sdf(path: Path, max_poses: int):
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    ligands = []
    for mol in suppl:
        if mol is None:
            continue
        ligands.append(mol)
        if len(ligands) >= max_poses:
            break
    return ligands


def _collect_prolif_features(
    pdb_path: Path,
    sdf_path: Path,
    max_poses: int,
) -> Tuple[Dict[Tuple[str, int], Dict], int]:
    from rdkit import Chem
    with warnings.catch_warnings():
        # Benign upstream dependency noise from current ProLIF/MDAnalysis versions.
        warnings.filterwarnings(
            "ignore",
            message=r".*MDAnalysis\.topology\.tables has been moved to MDAnalysis\.guesser\.tables.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'HBDonor' interaction has been superseded.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'Anionic' interaction has been superseded.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'PiCation' interaction has been superseded.*",
            category=UserWarning,
        )
        import prolif as plf

        prot_rd = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if prot_rd is None:
            raise RuntimeError(f"Could not parse receptor PDB with RDKit: {pdb_path}")

        ligands_rd = _iter_ligands_from_sdf(sdf_path, max_poses=max_poses)
        if not ligands_rd:
            raise RuntimeError(f"No valid ligand poses found in SDF: {sdf_path}")

        prot = plf.Molecule.from_rdkit(prot_rd)
        ligands = [plf.Molecule.from_rdkit(m) for m in ligands_rd]
        fp = plf.Fingerprint()
        fp.run_from_iterable(ligands, prot)

    ifp = getattr(fp, "ifp", None)
    if not ifp:
        return {}, len(ligands)

    # Per residue: contact frequency and interaction-type frequencies across poses.
    pose_contact = Counter()
    interaction_pose_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    for _frame_id, pair_map in ifp.items():
        seen_contact = set()
        seen_types: Dict[Tuple[str, int], set] = defaultdict(set)

        for pair_key, interactions in pair_map.items():
            prot_res = None
            if isinstance(pair_key, tuple) and len(pair_key) >= 2:
                prot_res = pair_key[1]
            else:
                prot_res = pair_key
            key = _residue_to_key(prot_res)
            if key is None:
                continue
            seen_contact.add(key)

            if isinstance(interactions, dict):
                for itype, payload in interactions.items():
                    if payload:
                        seen_types[key].add(str(itype))

        for key in seen_contact:
            pose_contact[key] += 1
        for key, types in seen_types.items():
            for t in types:
                interaction_pose_counts[key][t] += 1

    nposes = len(ligands)
    features: Dict[Tuple[str, int], Dict] = {}
    for key, c in pose_contact.items():
        it_counts = interaction_pose_counts.get(key, Counter())
        it_freq = {k: round(v / nposes, 6) for k, v in it_counts.items()}
        top = [k for k, _ in sorted(it_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
        features[key] = {
            "contact_pose_count": int(c),
            "pose_total": int(nposes),
            "contact_freq": round(c / nposes, 6),
            "interaction_freq": it_freq,
            "top_interactions": top,
        }
    return features, nposes


def _build_feature_dump(features: Dict[Tuple[str, int], Dict]) -> List[Dict]:
    rows = []
    for (chain, pos), feat in sorted(features.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"chain": chain, "pos": pos}
        row.update(feat)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich site_cards with ProLIF interaction fingerprints.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--site-cards", default=None)
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--poses-sdf", default=None)
    ap.add_argument("--out", default=None, help="Output site_cards path (default: overwrite input).")
    ap.add_argument("--features-out", default=None, help="Optional residue-level ProLIF feature JSON output.")
    ap.add_argument("--docking-summary", default=None, help="Optional docking_summary.json for pose-cluster metadata.")
    ap.add_argument("--max-poses", type=int, default=30)
    ap.add_argument("--tag-threshold", type=float, default=0.2, help="Tag residue as prolif_contact if contact_freq >= threshold.")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path("./data")
    site_cards_path = Path(args.site_cards) if args.site_cards else outdir / "swarm" / "site_cards.jsonl"
    out_path = Path(args.out) if args.out else site_cards_path
    pdb_path = resolve_canonical_protein_pdb(outdir, explicit=(Path(args.pdb) if args.pdb else None))
    sdf_path = resolve_docking_pose_sdf(outdir, explicit=(Path(args.poses_sdf) if args.poses_sdf else None))
    docking_summary_path = Path(args.docking_summary) if args.docking_summary else outdir / "docking_summary.json"

    if not site_cards_path.exists():
        raise SystemExit(f"ERROR: site_cards not found: {site_cards_path}")
    if pdb_path is None or not pdb_path.exists():
        raise SystemExit("ERROR: Could not resolve receptor PDB. Pass --pdb.")
    if sdf_path is None or not sdf_path.exists():
        raise SystemExit("ERROR: Could not resolve docking SDF. Pass --poses-sdf.")

    try:
        features, nposes = _collect_prolif_features(pdb_path=pdb_path, sdf_path=sdf_path, max_poses=max(1, args.max_poses))
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ERROR: ProLIF dependencies missing. Install with: pip install prolif rdkit"
        ) from e

    cards = _load_site_cards(site_cards_path)
    docking_summary = _load_docking_summary(docking_summary_path)
    top_cluster_reps = docking_summary.get("top_cluster_representatives") or []
    cluster_count = len(top_cluster_reps) if isinstance(top_cluster_reps, list) else 0
    primary_pose_idx = docking_summary.get("primary_pose_idx")
    score_backend = str(docking_summary.get("score_backend") or "").strip().lower() or None
    card_keys = {(_norm_chain(c.get("chain")), int(c.get("pos"))) for c in cards if c.get("pos") is not None}
    feature_keys = set(features.keys())
    overlap_keys = card_keys.intersection(feature_keys)
    if feature_keys and not overlap_keys:
        feat_pos = sorted(k[1] for k in feature_keys)
        card_pos = sorted(int(c.get("pos")) for c in cards if c.get("pos") is not None)
        raise SystemExit(
            "ERROR: ProLIF/site-card residue indexing mismatch (no overlapping residues). "
            f"prolif_range={feat_pos[0]}-{feat_pos[-1]} site_card_range={card_pos[0]}-{card_pos[-1]} "
            "Likely stale FASTA/PDB pairing; rerun scripts/swarm/14a_prepare_inputs.py then 15b/15d/15f."
        )

    updated = 0
    for card in cards:
        key = (_norm_chain(card.get("chain")), int(card.get("pos")))
        feat = features.get(key)
        contact_freq = 0.0
        contact_pose_count = 0
        top_interactions: List[str] = []
        if not feat:
            feat = {}
        else:
            contact_freq = float(feat.get("contact_freq") or 0.0)
            contact_pose_count = int(feat.get("contact_pose_count") or 0)
            top_interactions = list(feat.get("top_interactions") or [])
            card["prolif"] = feat
        tags = list(card.get("tags") or [])
        if contact_freq >= args.tag_threshold and "prolif_contact" not in tags:
            tags.append("prolif_contact")
        card["tags"] = tags
        # Multi-pose ligand support is a softer signal than binary contact from the
        # single selected docking pose. Carry both forward so generation can treat
        # sparse or unstable pose support cautiously instead of as a hard fact.
        card["ligand_pose_support"] = round(contact_freq, 6)
        card["ligand_pose_contact_count"] = int(contact_pose_count)
        card["ligand_pose_total"] = int(nposes)
        card["ligand_pose_uncertainty"] = round(1.0 - contact_freq, 6)
        card["ligand_pose_top_interactions"] = top_interactions[:4]
        card["docking_pose_backend"] = score_backend or card.get("docking_pose_backend")
        card["docking_cluster_representative_count"] = int(cluster_count)
        card["docking_primary_pose_idx"] = primary_pose_idx if primary_pose_idx is not None else card.get("docking_primary_pose_idx")
        if key in features:
            updated += 1

    _save_site_cards(out_path, cards)

    if feature_keys:
        feature_n = len(feature_keys)
        if feature_n <= 5:
            min_overlap = 1
        else:
            min_overlap = max(3, int(round(0.10 * feature_n)))
        if updated < min_overlap:
            raise SystemExit(
                "ERROR: ProLIF enrichment overlap is unexpectedly low and may indicate stale inputs. "
                f"(updated={updated}, required_min={min_overlap}, feature_residues={feature_n})"
            )

    features_out = Path(args.features_out) if args.features_out else outdir / "swarm" / "prolif_features.json"
    features_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pdb": str(pdb_path),
        "poses_sdf": str(sdf_path),
        "pose_total": int(nposes),
        "max_poses_used": int(max(1, args.max_poses)),
        "residue_features": _build_feature_dump(features),
    }
    features_out.write_text(json.dumps(payload, indent=2))

    print(f"Wrote ProLIF-enriched site cards: {out_path}")
    print(f"Wrote ProLIF residue features: {features_out}")
    print(f"Updated residues with ProLIF features: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
