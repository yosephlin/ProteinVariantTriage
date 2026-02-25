import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .http import fetch_json, OfflineError

BASE_URL = "https://www.ebi.ac.uk/interpro/api/protein/uniprot/{accession}"
ALT_URL = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{accession}"


def _extract_intervals(entry: Dict[str, Any]) -> List[Dict[str, int]]:
    intervals: List[Dict[str, int]] = []
    # older format: entry_protein_locations at top level
    for loc in entry.get("entry_protein_locations", []) or []:
        for frag in loc.get("fragments", []) or []:
            try:
                start = int(frag.get("start"))
                end = int(frag.get("end"))
            except Exception:
                continue
            if start and end:
                intervals.append({"start": start, "end": end})

    # new format: proteins list with entry_protein_locations
    if not intervals and isinstance(entry.get("proteins"), list):
        for prot in entry.get("proteins", []) or []:
            for loc in prot.get("entry_protein_locations", []) or []:
                for frag in loc.get("fragments", []) or []:
                    try:
                        start = int(frag.get("start"))
                        end = int(frag.get("end"))
                    except Exception:
                        continue
                    if start and end:
                        intervals.append({"start": start, "end": end})
    return intervals


def _extract_sites(entry: Dict[str, Any]) -> List[Dict[str, int]]:
    sites: List[Dict[str, int]] = []
    for s in entry.get("sites", []) or []:
        loc = s.get("location", {})
        frags = loc.get("fragments", []) if isinstance(loc, dict) else []
        for frag in frags:
            try:
                start = int(frag.get("start"))
                end = int(frag.get("end"))
            except Exception:
                continue
            sites.append({"start": start, "end": end, "type": s.get("type")})
    return sites


def parse_interpro(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = data.get("results", []) if isinstance(data, dict) else []
    domains = []
    for item in results:
        meta = item.get("metadata", {})
        entry_id = meta.get("accession") or meta.get("entry_id")
        name = meta.get("name")
        src_db = meta.get("source_database")
        if isinstance(src_db, dict):
            db = src_db.get("name")
        else:
            db = src_db
        intervals = _extract_intervals(item)
        sites = _extract_sites(item)
        if not entry_id or not intervals:
            continue
        domains.append({
            "source": "interpro",
            "entry_id": entry_id,
            "name": name,
            "db": db,
            "intervals": intervals,
            "sites": sites,
        })
    return domains


def _fetch_pages(url: str, cache_dir: Path, offline: bool) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    next_url: Optional[str] = url
    while next_url:
        data, _ = fetch_json(
            next_url,
            provider="interpro",
            cache_dir=cache_dir,
            offline=offline,
            headers={"Accept": "application/json"},
            params={"page_size": 200} if "?" not in next_url else None,
        )
        if not isinstance(data, dict):
            break
        all_results.append(data)
        next_url = data.get("next")
    return all_results


def fetch_interpro_all(accession: str, cache_dir: Path, offline: bool = False) -> List[Dict[str, Any]]:
    url = BASE_URL.format(accession=accession)
    pages = _fetch_pages(url, cache_dir, offline)
    # If only metadata is returned, try alternate endpoint
    if pages and pages[0].get("results") is None:
        alt = ALT_URL.format(accession=accession)
        pages = _fetch_pages(alt, cache_dir, offline)
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and parse InterPro domains")
    ap.add_argument("--accession", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        pages = fetch_interpro_all(args.accession, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    domains: List[Dict[str, Any]] = []
    for page in pages:
        domains.extend(parse_interpro(page))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "interpro_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2))
    (outdir / "interpro_domains.json").write_text(json.dumps(domains, ensure_ascii=False, indent=2))
    print("Wrote:", outdir / "interpro_pages.json")
    print("Wrote:", outdir / "interpro_domains.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
