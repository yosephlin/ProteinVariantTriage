import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from urllib.parse import quote

from .http import fetch_json, OfflineError

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def fetch_smiles_by_name(name: str, cache_dir: Path, offline: bool = False) -> Dict[str, Any]:
    safe = quote(name, safe="")
    url = f"{BASE}/name/{safe}/property/CanonicalSMILES,IsomericSMILES/JSON"
    data, _ = fetch_json(url, provider="pubchem_name", cache_dir=cache_dir, offline=offline)
    return data


def fetch_smiles_by_cid(cid: str, cache_dir: Path, offline: bool = False) -> Dict[str, Any]:
    url = f"{BASE}/cid/{cid}/property/CanonicalSMILES,IsomericSMILES/JSON"
    data, _ = fetch_json(url, provider="pubchem_cid", cache_dir=cache_dir, offline=offline)
    return data


def extract_smiles(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    props = data.get("PropertyTable", {}).get("Properties", []) if isinstance(data, dict) else []
    if not props:
        return {"canonical_smiles": None, "isomeric_smiles": None}
    p = props[0]
    canonical = p.get("CanonicalSMILES") or p.get("SMILES") or p.get("ConnectivitySMILES")
    isomeric = p.get("IsomericSMILES") or p.get("SMILES")
    return {
        "canonical_smiles": canonical,
        "isomeric_smiles": isomeric,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch PubChem SMILES by name or CID")
    ap.add_argument("--name", default=None)
    ap.add_argument("--cid", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if not args.name and not args.cid:
        raise SystemExit("Provide --name or --cid.")

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.name:
            data = fetch_smiles_by_name(args.name, cache_dir=cache_dir, offline=args.offline)
        else:
            data = fetch_smiles_by_cid(args.cid, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "pubchem_smiles.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    smiles = extract_smiles(data)
    (outdir / "pubchem_smiles_extracted.json").write_text(json.dumps(smiles, ensure_ascii=False, indent=2))
    print("Wrote:", out_path)
    print("Wrote:", outdir / "pubchem_smiles_extracted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
