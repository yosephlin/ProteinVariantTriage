from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


AA1 = set("ACDEFGHIKLMNPQRSTVWY")
_MUT_TOKEN_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def parse_mutation_token(token: str, default_chain: str = "A") -> Optional[Dict[str, Any]]:
    s = str(token or "").strip()
    if not s:
        return None
    chain = str(default_chain or "A")
    core = s
    if ":" in s:
        # Accept chain-qualified token forms like "A:A23V".
        parts = [x.strip() for x in s.split(":") if x.strip()]
        if len(parts) == 2 and len(parts[0]) <= 4:
            chain = parts[0]
            core = parts[1]
    m = _MUT_TOKEN_RE.match(core.upper())
    if not m:
        return None
    wt, pos, mut = m.group(1), m.group(2), m.group(3)
    return normalize_mutation_entry(
        {"chain": chain, "pos": safe_int(pos, -1), "wt": wt, "mut": mut},
        default_chain=default_chain,
    )


def normalize_mutation_entry(entry: Dict[str, Any], default_chain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    chain_raw = entry.get("chain", default_chain if default_chain is not None else "A")
    chain = str(chain_raw or "A").strip() or "A"
    pos = safe_int(entry.get("pos"), -1)
    wt = str(entry.get("wt") or "").strip().upper()
    mut = str(entry.get("mut") or "").strip().upper()
    if pos <= 0 or wt not in AA1 or mut not in AA1 or wt == mut:
        return None
    return {"chain": chain, "pos": int(pos), "wt": wt, "mut": mut}


def canonicalize_mutations(mutations: Sequence[Dict[str, Any]], default_chain: Optional[str] = None) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    seen = set()
    for m in mutations:
        mm = normalize_mutation_entry(m, default_chain=default_chain)
        if mm is None:
            continue
        key = (str(mm["chain"]), int(mm["pos"]))
        # Drop conflicting duplicates at same residue; keep the first one encountered.
        if key in seen:
            continue
        seen.add(key)
        norm.append(mm)
    norm.sort(key=lambda x: (str(x["chain"]), int(x["pos"]), str(x["wt"]), str(x["mut"])))
    return norm


def row_mutations(row: Dict[str, Any], default_chain: str = "A") -> List[Dict[str, Any]]:
    row_chain = str(row.get("chain") or default_chain)

    muts_obj = row.get("mutations")
    if isinstance(muts_obj, str):
        try:
            muts_obj = json.loads(muts_obj)
        except Exception:
            muts_obj = None
    if isinstance(muts_obj, list):
        muts = canonicalize_mutations(muts_obj, default_chain=row_chain)
        if muts:
            return muts

    muts_json_obj = row.get("mutations_json")
    if isinstance(muts_json_obj, str):
        try:
            parsed = json.loads(muts_json_obj)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            muts = canonicalize_mutations(parsed, default_chain=row_chain)
            if muts:
                return muts

    variant_id = str(row.get("variant_id") or "").strip()
    if variant_id:
        toks = split_variant_id_tokens(variant_id)
        parsed_tokens = [parse_mutation_token(tok, default_chain=row_chain) for tok in toks]
        muts = canonicalize_mutations([x for x in parsed_tokens if x is not None], default_chain=row_chain)
        if muts:
            return muts

    single = normalize_mutation_entry(
        {
            "chain": row.get("chain", default_chain),
            "pos": row.get("pos"),
            "wt": row.get("wt"),
            "mut": row.get("mut"),
        },
        default_chain=default_chain,
    )
    return [single] if single is not None else []


def mutation_token(m: Dict[str, Any], include_chain: bool = False) -> str:
    core = f"{m['wt']}{int(m['pos'])}{m['mut']}"
    if include_chain:
        return f"{m['chain']}:{core}"
    return core


def mutations_to_id(mutations: Sequence[Dict[str, Any]], include_chain: bool = False) -> str:
    muts = canonicalize_mutations(list(mutations))
    if not muts:
        return ""
    multi_chain = len({str(m["chain"]) for m in muts}) > 1
    include_chain_eff = bool(include_chain and multi_chain)
    return ":".join(mutation_token(m, include_chain=include_chain_eff) for m in muts)


def row_variant_id(row: Dict[str, Any], include_chain: bool = False) -> str:
    vid = str(row.get("variant_id") or "").strip()
    if vid:
        return vid
    muts = row_mutations(row)
    return mutations_to_id(muts, include_chain=include_chain)


def row_anchor_mutation(row: Dict[str, Any], default_chain: str = "A") -> Optional[Dict[str, Any]]:
    muts = row_mutations(row, default_chain=default_chain)
    if not muts:
        return None
    return muts[0]


def row_position_keys(row: Dict[str, Any], default_chain: str = "A") -> List[Tuple[str, int]]:
    muts = row_mutations(row, default_chain=default_chain)
    return [(str(m["chain"]), int(m["pos"])) for m in muts]


def split_variant_id_tokens(variant_id: str) -> List[str]:
    raw = str(variant_id or "").strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(":") if tok.strip()]
