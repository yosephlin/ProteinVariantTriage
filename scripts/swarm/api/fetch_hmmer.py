import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .cache import load_cache, make_meta, save_cache
from .http import OfflineError, fetch_json

HMMER_SEARCH_URL = "https://www.ebi.ac.uk/Tools/hmmer/api/v1/search/phmmer"
HMMER_RESULT_URL = "https://www.ebi.ac.uk/Tools/hmmer/api/v1/result/{job_id}"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def _safe_fasta(seq: str) -> str:
    s = (seq or "").strip()
    if not s:
        return ""
    if s.startswith(">"):
        return s
    return f">query\n{s}\n"


def _cache_key_for_query(sequence: str, database: str) -> str:
    raw = f"{database}:{sequence.strip()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _submit_hmmer_job(sequence: str, database: str = "uniprot", timeout: int = 45) -> str:
    payload = {"database": database, "input": _safe_fasta(sequence)}
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        HMMER_SEARCH_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "swarm-context-pack"},
    )
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    job_id = data.get("id")
    if not job_id:
        raise RuntimeError("HMMER submit response missing job id")
    return str(job_id)


def _poll_hmmer_result(job_id: str, timeout: int = 45, max_wait_s: int = 240) -> Dict[str, Any]:
    url = HMMER_RESULT_URL.format(job_id=job_id)
    started = time.time()
    wait = 1.5
    while True:
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "swarm-context-pack"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        status = str(data.get("status") or "").upper()
        # API commonly returns SUCCESS for completed jobs.
        if status in {"SUCCESS", "DONE", "FINISHED"}:
            return data
        if status in {"FAILURE", "ERROR", "FAILED"}:
            raise RuntimeError(f"HMMER job failed: {status}")
        if (time.time() - started) > max_wait_s:
            raise RuntimeError(f"HMMER job timeout after {max_wait_s}s (status={status or 'unknown'})")
        time.sleep(wait)
        wait = min(5.0, wait * 1.3)


def fetch_phmmer_results(
    query_sequence: str,
    cache_dir: Path,
    offline: bool = False,
    database: str = "uniprot",
) -> Dict[str, Any]:
    provider = "hmmer_phmmer_result"
    key = _cache_key_for_query(query_sequence, database)
    cached = load_cache(cache_dir, provider, key)
    if cached is not None:
        return cached[0]
    if offline:
        raise OfflineError(f"Offline mode enabled and no cache entry for {provider}:{key}")

    last_err: Optional[Exception] = None
    for _ in range(3):
        try:
            job_id = _submit_hmmer_job(query_sequence, database=database)
            data = _poll_hmmer_result(job_id)
            meta = make_meta(HMMER_SEARCH_URL, {"database": database}, 200, {"job_id": job_id})
            save_cache(cache_dir, provider, key, data, meta)
            return data
        except (HTTPError, URLError, RuntimeError) as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"HMMER query failed: {last_err}")


def parse_phmmer_hits(result_data: Dict[str, Any], max_hits: int = 120) -> List[Dict[str, Any]]:
    hits = ((result_data or {}).get("result") or {}).get("hits") or []
    parsed = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        md = h.get("metadata") or {}
        acc = md.get("uniprot_accession") or md.get("accession")
        if not acc:
            continue
        parsed.append(
            {
                "uniprot_accession": acc,
                "identifier": md.get("uniprot_identifier") or md.get("identifier"),
                "description": md.get("description"),
                "species": md.get("species"),
                "taxonomy_id": md.get("taxonomy_id"),
                "evalue": h.get("evalue"),
                "score": h.get("score"),
            }
        )
        if len(parsed) >= max_hits:
            break
    return parsed


def fetch_uniprot_sequences_by_accessions(
    accessions: List[str],
    cache_dir: Path,
    offline: bool = False,
    page_size: int = 100,
) -> Dict[str, str]:
    acc = []
    seen = set()
    for a in accessions:
        a = (a or "").strip()
        if not a or a in seen:
            continue
        seen.add(a)
        acc.append(a)
    if not acc:
        return {}

    seqs: Dict[str, str] = {}
    for i in range(0, len(acc), page_size):
        chunk = acc[i : i + page_size]
        query = "(" + " OR ".join(f"accession:{a}" for a in chunk) + ")"
        data, _ = fetch_json(
            UNIPROT_SEARCH_URL,
            provider="uniprot_sequence_batch",
            cache_dir=cache_dir,
            params={
                "query": query,
                "fields": "accession,sequence",
                "format": "json",
                "size": len(chunk),
            },
            offline=offline,
        )
        for r in data.get("results") or []:
            a = r.get("primaryAccession")
            seq = (r.get("sequence") or {}).get("value")
            if a and seq:
                seqs[a] = seq
    return seqs


def _align_map_pairwise(query_seq: str, homolog_seq: str) -> Tuple[Dict[int, str], float, float]:
    try:
        from Bio import pairwise2
    except Exception:
        if len(query_seq) != len(homolog_seq):
            return {}, 0.0, 0.0
        matches = sum(1 for a, b in zip(query_seq, homolog_seq) if a == b)
        ident = matches / max(1, len(query_seq))
        mapped = {i + 1: aa for i, aa in enumerate(homolog_seq)}
        return mapped, ident, 1.0

    aln = pairwise2.align.globalms(query_seq, homolog_seq, 2, -1, -5, -1, one_alignment_only=True)
    if not aln:
        return {}, 0.0, 0.0
    q_aln, h_aln, _score, _start, _end = aln[0]
    pos = 0
    mapped = {}
    n_match = 0
    n_cov = 0
    for qa, ha in zip(q_aln, h_aln):
        if qa != "-":
            pos += 1
            if ha != "-":
                n_cov += 1
                mapped[pos] = ha
                if qa == ha:
                    n_match += 1
    ident = n_match / max(1, n_cov)
    coverage = n_cov / max(1, len(query_seq))
    return mapped, ident, coverage


def build_position_priors(
    query_seq: str,
    homolog_sequences: Dict[str, str],
    min_identity: float = 0.25,
    min_coverage: float = 0.70,
    max_homologs: int = 64,
) -> Dict[str, Any]:
    per_pos: Dict[int, Counter] = {i: Counter() for i in range(1, len(query_seq) + 1)}
    used = 0
    for _acc, seq in list(homolog_sequences.items())[:max_homologs]:
        mapped, ident, cov = _align_map_pairwise(query_seq, seq)
        if ident < min_identity or cov < min_coverage:
            continue
        used += 1
        for pos, aa in mapped.items():
            if aa and aa != "-":
                per_pos[pos][aa] += 1

    positions = []
    for pos in range(1, len(query_seq) + 1):
        wt = query_seq[pos - 1]
        c = per_pos[pos]
        total = sum(c.values())
        if total <= 0:
            positions.append(
                {
                    "pos": pos,
                    "wt": wt,
                    "homolog_count": 0,
                    "conservation": None,
                    "allowed_aas": [wt],
                    "top_aas": [],
                }
            )
            continue
        top = c.most_common(6)
        conservation = top[0][1] / total
        allowed = []
        for aa, count in top:
            freq = count / total
            # Keep robustly observed variants while avoiding singletons.
            if count >= 2 and freq >= 0.05:
                allowed.append(aa)
        if wt not in allowed:
            allowed.insert(0, wt)
        positions.append(
            {
                "pos": pos,
                "wt": wt,
                "homolog_count": total,
                "conservation": round(conservation, 4),
                "allowed_aas": allowed[:8],
                "top_aas": [{"aa": aa, "count": count, "freq": round(count / total, 4)} for aa, count in top],
            }
        )

    return {"homologs_used": used, "positions": positions}


def build_hmmer_evolution_context(
    query_seq: str,
    cache_dir: Path,
    offline: bool = False,
    max_hits: int = 120,
    min_identity: float = 0.25,
    min_coverage: float = 0.70,
    max_homologs: int = 64,
) -> Dict[str, Any]:
    if not query_seq:
        return {"note": "missing_query_sequence", "positions": []}

    raw = fetch_phmmer_results(query_seq, cache_dir=cache_dir, offline=offline, database="uniprot")
    hits = parse_phmmer_hits(raw, max_hits=max_hits)
    seqs = fetch_uniprot_sequences_by_accessions(
        [h["uniprot_accession"] for h in hits],
        cache_dir=cache_dir,
        offline=offline,
    )
    priors = build_position_priors(
        query_seq=query_seq,
        homolog_sequences=seqs,
        min_identity=min_identity,
        min_coverage=min_coverage,
        max_homologs=max_homologs,
    )
    return {
        "method": "hmmer_phmmer_uniprot_pairwise",
        "query_length": len(query_seq),
        "hits_considered": len(hits),
        "homolog_sequences": len(seqs),
        "homologs_used": priors.get("homologs_used", 0),
        "positions": priors.get("positions", []),
        "top_hits": hits[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build HMMER-derived evolutionary priors for a protein sequence")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-hits", type=int, default=120)
    ap.add_argument("--min-identity", type=float, default=0.25)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    ap.add_argument("--max-homologs", type=int, default=64)
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir) if args.outdir else Path.cwd()
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "swarm_api" / "source_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    seq = ""
    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq += line

    try:
        evo = build_hmmer_evolution_context(
            query_seq=seq,
            cache_dir=cache_dir,
            offline=args.offline,
            max_hits=args.max_hits,
            min_identity=args.min_identity,
            min_coverage=args.min_coverage,
            max_homologs=args.max_homologs,
        )
    except OfflineError as e:
        raise SystemExit(str(e))

    (outdir / "hmmer_evolution.json").write_text(json.dumps(evo, ensure_ascii=False, indent=2))
    print("Wrote:", outdir / "hmmer_evolution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
