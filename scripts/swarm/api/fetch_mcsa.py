import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .http import OfflineError, fetch_json

MCSA_RESIDUES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/"


def fetch_mcsa_residues(accession: str, cache_dir: Path, offline: bool = False) -> List[Dict[str, Any]]:
    params = {
        "format": "json",
        "entries.proteins.sequences.uniprot_ids": accession,
    }
    data, _ = fetch_json(
        MCSA_RESIDUES_URL,
        provider="mcsa_residues",
        cache_dir=cache_dir,
        params=params,
        offline=offline,
    )
    if isinstance(data, list):
        return data
    return []


def parse_mcsa_residues(data: List[Dict[str, Any]], accession: str) -> Dict[str, Any]:
    residue_annotations: Dict[int, List[Dict[str, Any]]] = {}
    mcsa_ids = set()
    ref_count = 0
    matched_uniprot_ids = set()
    exact_rows: List[Dict[str, Any]] = []
    fallback_rows: List[Dict[str, Any]] = []

    for item in data or []:
        mcsa_id = item.get("mcsa_id")
        if mcsa_id is not None:
            mcsa_ids.add(mcsa_id)
        main_note = item.get("main_annotation") or ""
        role_summary = item.get("roles_summary") or ""
        role_list = []
        for role in item.get("roles") or []:
            if not isinstance(role, dict):
                continue
            fn = role.get("function")
            if fn:
                role_list.append(str(fn))

        seq_entries = item.get("residue_sequences") or []
        for seq_entry in seq_entries:
            if not isinstance(seq_entry, dict):
                continue
            uniprot_id = (seq_entry.get("uniprot_id") or "").upper()
            pos = seq_entry.get("resid")
            aa = seq_entry.get("code")
            if pos is None:
                continue
            try:
                pos = int(pos)
            except Exception:
                continue
            is_reference = bool(seq_entry.get("is_reference"))
            if is_reference:
                ref_count += 1
            payload = {
                "source": "mcsa",
                "type": "MCSA_CATALYTIC",
                "severity": "critical",
                "mcsa_id": mcsa_id,
                "note": main_note,
                "roles_summary": role_summary,
                "roles": role_list,
                "aa": aa,
                "is_reference": is_reference,
                "uniprot_id": uniprot_id,
            }
            if uniprot_id == accession.upper():
                matched_uniprot_ids.add(uniprot_id)
                payload["match_type"] = "accession"
                exact_rows.append({"pos": pos, "payload": payload})
            elif is_reference:
                payload["match_type"] = "homolog_reference"
                fallback_rows.append({"pos": pos, "payload": payload})

    rows = exact_rows if exact_rows else fallback_rows
    for row in rows:
        residue_annotations.setdefault(int(row["pos"]), []).append(row["payload"])

    residue_count = len(residue_annotations)
    return {
        "summary": {
            "entry_count": len(mcsa_ids),
            "residue_count": residue_count,
            "reference_residue_hits": ref_count,
            "matched_uniprot_ids": sorted(matched_uniprot_ids),
            "used_homolog_reference": bool((not exact_rows) and fallback_rows),
        },
        "residue_annotations": residue_annotations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and parse M-CSA catalytic residues by UniProt accession")
    ap.add_argument("--accession", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        raw = fetch_mcsa_residues(args.accession, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    parsed = parse_mcsa_residues(raw, args.accession)
    (outdir / "mcsa_residues_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    (outdir / "mcsa_residues_parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("Wrote:", outdir / "mcsa_residues_raw.json")
    print("Wrote:", outdir / "mcsa_residues_parsed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
