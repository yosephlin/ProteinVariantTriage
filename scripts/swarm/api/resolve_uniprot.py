import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .http import fetch_json, OfflineError

SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def build_query(
    protein_name: Optional[str],
    organism_id: Optional[str],
    organism_name: Optional[str],
    reviewed_only: bool,
) -> str:
    parts = []
    if protein_name:
        parts.append(f'protein_name:"{protein_name}"')
    if organism_id:
        parts.append(f"organism_id:{organism_id}")
    elif organism_name:
        parts.append(f'organism_name:"{organism_name}"')
    if reviewed_only:
        parts.append("reviewed:true")
    parts.append("fragment:false")
    return " AND ".join(parts)


def _annotation_score(entry: Dict[str, Any]) -> float:
    val = entry.get("annotationScore")
    if val is None:
        val = entry.get("annotation_score")
    try:
        return float(val)
    except Exception:
        return 0.0


def _sequence_length(entry: Dict[str, Any]) -> int:
    seq = entry.get("sequence", {})
    try:
        return int(seq.get("length", 0))
    except Exception:
        return 0


def select_best(results: list) -> Optional[Dict[str, Any]]:
    if not results:
        return None

    def key_fn(e: Dict[str, Any]):
        et = (e.get("entryType") or "").lower()
        reviewed = 1 if (e.get("reviewed") is True or "reviewed" in et or "swiss-prot" in et) else 0
        return (reviewed, _annotation_score(e), _sequence_length(e))

    results_sorted = sorted(results, key=key_fn, reverse=True)
    return results_sorted[0]


def search_uniprot(
    protein_name: str,
    organism_id: Optional[str],
    organism_name: Optional[str],
    reviewed_only: bool,
    cache_dir: Path,
    offline: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    query = build_query(protein_name, organism_id, organism_name, reviewed_only)
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,protein_name,organism_name,reviewed,sequence,length,annotation_score",
    }
    data, meta = fetch_json(SEARCH_URL, provider="uniprot_search", cache_dir=cache_dir, params=params, offline=offline)
    results = data.get("results", []) if isinstance(data, dict) else []
    return select_best(results), meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve UniProt accession by protein name/organism")
    ap.add_argument("--protein-name", required=True)
    ap.add_argument("--organism-id", default=None)
    ap.add_argument("--organism-name", default=None)
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        best, _ = search_uniprot(
            args.protein_name,
            args.organism_id,
            args.organism_name,
            args.reviewed_only,
            cache_dir,
            offline=args.offline,
        )
    except OfflineError as e:
        raise SystemExit(str(e))

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "uniprot_resolved.json"
    out_path.write_text(json.dumps(best or {}, ensure_ascii=False, indent=2))
    if best:
        print("Resolved accession:", best.get("primaryAccession"))
    else:
        print("No results found.")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
