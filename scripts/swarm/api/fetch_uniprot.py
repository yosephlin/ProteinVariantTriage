import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .http import fetch_json, OfflineError

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

CRITICAL_TYPES = {
    "ACTIVE_SITE",
    "ACT_SITE",
    "BINDING",
    "BINDING_SITE",
    "METAL",
    "DISULFID",
    "DISULFIDE_BOND",
    "CROSSLNK",
    "MOD_RES",
}

SOFT_TYPES = {
    "MUTAGEN",
    "MUTAGENESIS",
    "VARIANT",
    "NATURAL_VARIANT",
    "REGION",
    "DOMAIN",
    "MOTIF",
    "SITE",
    "HELIX",
    "BETA_STRAND",
    "TURN",
    "CHAIN",
}


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _normalize_type(t: str) -> str:
    return t.upper().replace(" ", "_").replace("-", "_")


def _extract_text_list(obj: Any) -> List[str]:
    if not obj:
        return []
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and "value" in item:
                out.append(str(item["value"]))
            elif isinstance(item, str):
                out.append(item)
        return out
    if isinstance(obj, dict) and "value" in obj:
        return [str(obj["value"])]
    if isinstance(obj, str):
        return [obj]
    return []


def _protein_name(entry: Dict[str, Any]) -> Optional[str]:
    desc = entry.get("proteinDescription", {})
    rec = desc.get("recommendedName", {})
    name = _safe_get(rec, ["fullName", "value"], None)
    if name:
        return name
    alt = desc.get("alternativeNames", [])
    if alt:
        return _safe_get(alt[0], ["fullName", "value"], None)
    return None


def _organism_name(entry: Dict[str, Any]) -> Optional[str]:
    org = entry.get("organism", {})
    return org.get("scientificName") or org.get("commonName")


def _function_comment(entry: Dict[str, Any]) -> Optional[str]:
    comments = entry.get("comments", [])
    for c in comments:
        if c.get("commentType") == "FUNCTION":
            texts = _extract_text_list(c.get("texts"))
            if texts:
                return " ".join(texts)
    return None


def _comment_texts_by_type(entry: Dict[str, Any], comment_type: str) -> List[str]:
    out = []
    comments = entry.get("comments", [])
    for c in comments:
        if c.get("commentType") != comment_type:
            continue
        texts = _extract_text_list(c.get("texts"))
        out.extend(texts)
    return out


def _cofactors(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    comments = entry.get("comments", [])
    for c in comments:
        if c.get("commentType") != "COFACTOR":
            continue
        cof = c.get("cofactors") or []
        for item in cof:
            name = None
            if isinstance(item, dict):
                name = _safe_get(item, ["name", "value"], None) or item.get("name")
            if name:
                out.append({"name": name})
    return out


def _catalytic_activity(entry: Dict[str, Any]) -> Optional[str]:
    comments = entry.get("comments", [])
    for c in comments:
        if c.get("commentType") != "CATALYTIC_ACTIVITY":
            continue
        reaction = c.get("reaction") or {}
        # prefer Rhea string if present
        rhea = reaction.get("rheaId") or reaction.get("rheaId")
        if rhea:
            return f"rhea:{rhea}"
        # fall back to reaction text
        txt = reaction.get("name") or reaction.get("reactionString")
        if txt:
            return txt
    return None


def _feature_location(feature: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    loc = feature.get("location", {})
    start = _safe_get(loc, ["start", "value"], None)
    end = _safe_get(loc, ["end", "value"], None)
    try:
        start = int(start) if start is not None else None
    except Exception:
        start = None
    try:
        end = int(end) if end is not None else None
    except Exception:
        end = None
    return start, end


def _feature_positions(ftype: str, start: int, end: int) -> List[int]:
    # Cross-link features denote paired residues; apply constraints to endpoints only.
    if ftype in {"DISULFIDE_BOND", "DISULFID", "CROSSLNK", "CROSSLINK"}:
        if start == end:
            return [start]
        return [start, end]
    return list(range(start, end + 1))


def parse_uniprot_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    seq = _safe_get(entry, ["sequence", "value"], "") or ""
    length = _safe_get(entry, ["sequence", "length"], len(seq))
    accession = entry.get("primaryAccession") or entry.get("uniProtkbId")

    base = {
        "uniprot": {
            "accession": accession,
            "entry_type": entry.get("entryType"),
            "protein_name": _protein_name(entry),
            "organism": _organism_name(entry),
            "sequence_length": length,
            "sequence": seq,
        },
        "function": {
            "summary": _function_comment(entry),
        },
        "cofactors": _cofactors(entry),
        "catalytic_activity": _catalytic_activity(entry),
        "similarity": " ".join(_comment_texts_by_type(entry, "SIMILARITY")) or None,
    }

    residues = []
    for i, aa in enumerate(seq, start=1):
        residues.append({
            "pos": i,
            "wt": aa,
            "uniprot": {"critical": [], "soft": []},
            "pdbe": {
                "ligand_sites": {"count": 0, "ligands": [], "supporting_entries": 0},
                "interface": {"is_interface": False, "count": 0, "supporting_entries": 0, "partners": []},
                "predicted_sites": [],
            },
            "interpro": {"domains": [], "sites": []},
            "policy": {"do_not_mutate": False, "reason": None},
        })

    features = entry.get("features", [])
    for f in features:
        ftype = _normalize_type(str(f.get("type", "")))
        start, end = _feature_location(f)
        if start is None:
            continue
        if end is None:
            end = start
        note = f.get("description") or f.get("note") or ""
        evidences = f.get("evidences") or []

        severity = "low"
        if ftype in CRITICAL_TYPES:
            severity = "critical"
        elif ftype in SOFT_TYPES:
            severity = "medium"

        payload = {
            "source": "uniprot",
            "type": ftype,
            "pos_start": start,
            "pos_end": end,
            "severity": severity,
            "note": note,
            "evidence": {"codes": [e.get("evidenceCode") for e in evidences if isinstance(e, dict)]},
        }

        for pos in _feature_positions(ftype, start, end):
            if pos < 1 or pos > len(residues):
                continue
            if severity == "critical":
                residues[pos - 1]["uniprot"]["critical"].append(payload)
                residues[pos - 1]["policy"]["do_not_mutate"] = True
                residues[pos - 1]["policy"]["reason"] = ftype
            else:
                residues[pos - 1]["uniprot"]["soft"].append(payload)

    return {"context": base, "residues": residues}


def fetch_uniprot(accession: str, cache_dir: Path, offline: bool = False) -> Dict[str, Any]:
    url = UNIPROT_URL.format(accession=accession)
    data, _ = fetch_json(url, provider="uniprot", cache_dir=cache_dir, offline=offline)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and parse UniProt entry")
    ap.add_argument("--accession", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        entry = fetch_uniprot(args.accession, cache_dir=cache_dir, offline=args.offline)
    except OfflineError as e:
        raise SystemExit(str(e))

    parsed = parse_uniprot_entry(entry)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "uniprot_entry.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2))
    (outdir / "uniprot_context.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("Wrote:", outdir / "uniprot_entry.json")
    print("Wrote:", outdir / "uniprot_context.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
