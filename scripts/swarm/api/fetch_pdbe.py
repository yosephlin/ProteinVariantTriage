import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .http import fetch_json, OfflineError

PDBe_LIGAND_URL = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/ligand_sites/{accession}"
PDBe_INTERFACE_URL = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/interface_residues/{accession}"
PDBe_ANNOT_URL = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/annotations/{accession}"


def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _entry_list(data: Any, accession: str) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if accession in data:
            acc_val = data[accession]
            if isinstance(acc_val, list):
                return acc_val
            if isinstance(acc_val, dict) and isinstance(acc_val.get("data"), list):
                return acc_val["data"]
        if "ligand_sites" in data and isinstance(data["ligand_sites"], list):
            return data["ligand_sites"]
    if isinstance(data, list):
        return data
    return []


def parse_ligand_sites(data: Any, accession: str) -> Dict[int, Dict[str, Any]]:
    entries = _entry_list(data, accession)
    by_pos: Dict[int, Dict[str, Any]] = {}

    for e in entries:
        if not isinstance(e, dict):
            continue
        # New PDBe aggregated format (data list)
        if "residues" in e and isinstance(e.get("residues"), list):
            lig = e.get("accession") or e.get("chem_comp_id") or e.get("name")
            for r in e.get("residues", []):
                try:
                    start = int(r.get("startIndex"))
                    end = int(r.get("endIndex"))
                except Exception:
                    continue
                pdb_entries = r.get("interactingPDBEntries") or []
                pdb_ids = [p.get("pdbId") for p in pdb_entries if isinstance(p, dict) and p.get("pdbId")]
                if not pdb_ids:
                    pdb_ids = r.get("allPDBEntries") or []
                for pos in range(start, end + 1):
                    slot = by_pos.setdefault(pos, {"count": 0, "ligands": set(), "supporting_entries": set()})
                    slot["count"] += 1
                    if lig:
                        slot["ligands"].add(str(lig))
                    for pdb in pdb_ids:
                        slot["supporting_entries"].add(str(pdb))
            continue

        # Legacy flat format
        pos = e.get("residue_number") or e.get("uniprot_residue_number") or e.get("residue_number_uniprot")
        try:
            pos = int(pos)
        except Exception:
            continue

        lig = e.get("chem_comp_id") or e.get("ligand_id") or e.get("chem_comp")
        pdb = e.get("pdb_id") or e.get("pdb")

        slot = by_pos.setdefault(pos, {"count": 0, "ligands": set(), "supporting_entries": set()})
        slot["count"] += 1
        if lig:
            slot["ligands"].add(str(lig))
        if pdb:
            slot["supporting_entries"].add(str(pdb))

    # normalize sets to lists
    out: Dict[int, Dict[str, Any]] = {}
    for pos, d in by_pos.items():
        out[pos] = {
            "count": d["count"],
            "ligands": sorted(d["ligands"]),
            "supporting_entries": len(d["supporting_entries"]),
        }
    return out


def parse_interface_residues(data: Any, accession: str) -> Dict[int, Dict[str, Any]]:
    entries = _entry_list(data, accession)
    by_pos: Dict[int, Dict[str, Any]] = {}

    for e in entries:
        if not isinstance(e, dict):
            continue
        if "residues" in e and isinstance(e.get("residues"), list):
            partner = e.get("accession") or e.get("name")
            for r in e.get("residues", []):
                try:
                    start = int(r.get("startIndex"))
                    end = int(r.get("endIndex"))
                except Exception:
                    continue
                pdb_entries = r.get("interactingPDBEntries") or []
                pdb_ids = [p.get("pdbId") for p in pdb_entries if isinstance(p, dict) and p.get("pdbId")]
                if not pdb_ids:
                    pdb_ids = r.get("allPDBEntries") or []
                for pos in range(start, end + 1):
                    slot = by_pos.setdefault(pos, {"count": 0, "supporting_entries": set(), "partners": set()})
                    slot["count"] += 1
                    if partner:
                        slot["partners"].add(str(partner))
                    for pdb in pdb_ids:
                        slot["supporting_entries"].add(str(pdb))
            continue

        pos = e.get("residue_number") or e.get("uniprot_residue_number") or e.get("residue_number_uniprot")
        try:
            pos = int(pos)
        except Exception:
            continue

        pdb = e.get("pdb_id") or e.get("pdb")
        partner = e.get("partner_id") or e.get("interacting_pdb_id") or e.get("partner")

        slot = by_pos.setdefault(pos, {"count": 0, "supporting_entries": set(), "partners": set()})
        slot["count"] += 1
        if pdb:
            slot["supporting_entries"].add(str(pdb))
        if partner:
            slot["partners"].add(str(partner))

    out: Dict[int, Dict[str, Any]] = {}
    for pos, d in by_pos.items():
        out[pos] = {
            "count": d["count"],
            "supporting_entries": len(d["supporting_entries"]),
            "partners": sorted(d["partners"]),
            "is_interface": d["count"] > 0,
        }
    return out


def parse_annotations(data: Any, accession: str) -> Dict[int, Dict[str, Any]]:
    entries = _entry_list(data, accession)
    by_pos: Dict[int, Dict[str, Any]] = {}

    for e in entries:
        if not isinstance(e, dict):
            continue
        if "residues" in e and isinstance(e.get("residues"), list):
            provider = e.get("name") or e.get("accession") or e.get("dataType")
            site_id = e.get("accession") or e.get("name")
            for r in e.get("residues", []):
                try:
                    start = int(r.get("startIndex"))
                    end = int(r.get("endIndex"))
                except Exception:
                    continue
                pdb_ids = r.get("pdbEntries") or []
                for pos in range(start, end + 1):
                    slot = by_pos.setdefault(pos, {"predicted_sites": []})
                    slot["predicted_sites"].append({
                        "provider": provider,
                        "site_id": site_id,
                        "score": None,
                        "pdb_entries": pdb_ids,
                    })
            continue

        pos = e.get("residue_number") or e.get("uniprot_residue_number") or e.get("residue_number_uniprot")
        try:
            pos = int(pos)
        except Exception:
            continue

        provider = e.get("site_provider") or e.get("provider") or e.get("source")
        site_id = e.get("site_id") or e.get("annotation_id") or e.get("id")
        score = e.get("score") or e.get("confidence")

        slot = by_pos.setdefault(pos, {"predicted_sites": []})
        slot["predicted_sites"].append({
            "provider": provider,
            "site_id": site_id,
            "score": score,
        })

    return by_pos


def fetch_ligand_sites(accession: str, cache_dir: Path, offline: bool = False) -> Any:
    url = PDBe_LIGAND_URL.format(accession=accession)
    data, _ = fetch_json(url, provider="pdbe_ligand_sites", cache_dir=cache_dir, offline=offline)
    return data


def fetch_interface_residues(accession: str, cache_dir: Path, offline: bool = False) -> Any:
    url = PDBe_INTERFACE_URL.format(accession=accession)
    data, _ = fetch_json(url, provider="pdbe_interface_residues", cache_dir=cache_dir, offline=offline)
    return data


def fetch_annotations(accession: str, cache_dir: Path, offline: bool = False) -> Any:
    url = PDBe_ANNOT_URL.format(accession=accession)
    data, _ = fetch_json(url, provider="pdbe_annotations", cache_dir=cache_dir, offline=offline)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and parse PDBe ligand sites")
    ap.add_argument("--accession", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = fetch_ligand_sites(args.accession, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    parsed = parse_ligand_sites(data, args.accession)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pdbe_ligand_sites.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (outdir / "pdbe_ligand_sites_parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("Wrote:", outdir / "pdbe_ligand_sites.json")
    print("Wrote:", outdir / "pdbe_ligand_sites_parsed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
