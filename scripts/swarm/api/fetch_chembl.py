import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from .http import fetch_json, OfflineError

BASE = "https://www.ebi.ac.uk/chembl/api/data"


def search_target(query: str, cache_dir: Path, offline: bool = False) -> Dict[str, Any]:
    url = f"{BASE}/target/search.json"
    data, _ = fetch_json(
        url,
        provider="chembl_target_search",
        cache_dir=cache_dir,
        params={"q": query},
        offline=offline,
    )
    return data


def select_target(data: Dict[str, Any], organism_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    targets = data.get("targets", []) if isinstance(data, dict) else []
    if not targets:
        return None

    def score(t):
        s = 0
        if t.get("target_type") == "SINGLE PROTEIN":
            s += 3
        if organism_name and organism_name.lower() in (t.get("organism") or "").lower():
            s += 2
        if t.get("target_chembl_id"):
            s += 1
        return s

    targets_sorted = sorted(targets, key=score, reverse=True)
    return targets_sorted[0]


def fetch_activities(target_chembl_id: str, cache_dir: Path, offline: bool = False, max_records: int = 2000) -> List[Dict[str, Any]]:
    url = f"{BASE}/activity.json"
    all_rows = []
    offset = 0
    limit = 500
    while True:
        params = {"target_chembl_id": target_chembl_id, "limit": limit, "offset": offset}
        data, _ = fetch_json(url, provider="chembl_activity", cache_dir=cache_dir, params=params, offline=offline)
        rows = data.get("activities", []) if isinstance(data, dict) else []
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
        if len(all_rows) >= max_records:
            break
    return all_rows


def build_ligand_priors(activities: List[Dict[str, Any]], top_n: int = 50) -> List[Dict[str, Any]]:
    bucket: Dict[str, List[float]] = {}
    for a in activities:
        p = a.get("pchembl_value")
        if p is None:
            continue
        try:
            p = float(p)
        except Exception:
            continue
        mid = a.get("molecule_chembl_id")
        if not mid:
            continue
        bucket.setdefault(mid, []).append(p)

    priors = []
    for mid, vals in bucket.items():
        priors.append({"molecule_chembl_id": mid, "median_pchembl": median(vals), "n": len(vals)})

    priors.sort(key=lambda x: x["median_pchembl"], reverse=True)
    return priors[:top_n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch ChEMBL target + activities")
    ap.add_argument("--query", required=True)
    ap.add_argument("--organism-name", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        search_data = search_target(args.query, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    target = select_target(search_data, organism_name=args.organism_name)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "chembl_target_search.json").write_text(json.dumps(search_data, ensure_ascii=False, indent=2))

    if not target:
        print("No ChEMBL target found.")
        return 0

    target_id = target.get("target_chembl_id")
    activities = fetch_activities(target_id, cache_dir=cache_dir, offline=args.offline)
    priors = build_ligand_priors(activities)

    (outdir / "chembl_target.json").write_text(json.dumps(target, ensure_ascii=False, indent=2))
    (outdir / "chembl_activities.json").write_text(json.dumps(activities, ensure_ascii=False, indent=2))
    (outdir / "chembl_ligand_priors.json").write_text(json.dumps(priors, ensure_ascii=False, indent=2))

    print("Target:", target_id)
    print("Wrote:", outdir / "chembl_target.json")
    print("Wrote:", outdir / "chembl_activities.json")
    print("Wrote:", outdir / "chembl_ligand_priors.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
