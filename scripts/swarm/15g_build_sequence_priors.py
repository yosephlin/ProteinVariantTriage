import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

AA_LIST = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MODEL_WEIGHTS = {
    "esm1v": 0.45,
    "eve": 0.35,
    "evmutation": 0.15,
    "deepsequence": 0.05,
}


def parse_mutation_token(token: str) -> Optional[Tuple[str, int, str]]:
    t = str(token or "").strip().upper()
    if len(t) < 3:
        return None
    wt = t[0]
    mut = t[-1]
    mid = t[1:-1]
    if wt not in AA_LIST or mut not in AA_LIST:
        return None
    if not mid.isdigit():
        return None
    pos = int(mid)
    if pos <= 0:
        return None
    return wt, pos, mut


def _as_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _infer_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open() as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return
    delim = _infer_delimiter(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        for row in reader:
            if isinstance(row, dict):
                yield row


def _rank_percentile(values: List[Tuple[Tuple[str, int, str, str], float]], higher_is_better: bool) -> Dict[Tuple[str, int, str, str], float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda kv: kv[1], reverse=bool(higher_is_better))
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 1.0}
    out: Dict[Tuple[str, int, str, str], float] = {}
    for i, (k, _v) in enumerate(ordered):
        out[k] = float(1.0 - (i / float(n - 1)))
    return out


def load_model_scores(
    path: Optional[Path],
    mutation_col: str,
    score_col: str,
    higher_is_better: bool,
    chain_col: Optional[str] = None,
) -> Dict[Tuple[str, int, str, str], Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    parsed: List[Tuple[Tuple[str, int, str, str], float]] = []
    for row in _read_rows(path):
        mut_token = row.get(mutation_col)
        if not mut_token:
            continue
        parsed_mut = parse_mutation_token(str(mut_token))
        if not parsed_mut:
            continue
        wt, pos, mut = parsed_mut
        score = _as_float(row.get(score_col))
        if score is None:
            continue
        chain = "A"
        if chain_col:
            chain = str(row.get(chain_col) or "A").strip() or "A"
        parsed.append(((chain, pos, wt, mut), score))
    norm = _rank_percentile(parsed, higher_is_better=higher_is_better)
    out: Dict[Tuple[str, int, str, str], Dict[str, float]] = {}
    for key, score in parsed:
        out[key] = {
            "raw": float(score),
            "plausibility": float(norm.get(key, 0.0)),
        }
    return out


def parse_model_weights(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return dict(DEFAULT_MODEL_WEIGHTS)
    out = dict(DEFAULT_MODEL_WEIGHTS)
    for part in str(raw).split(","):
        p = part.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        key = str(k).strip().lower()
        if key not in out:
            continue
        try:
            val = float(v.strip())
        except Exception:
            continue
        if val >= 0.0:
            out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build sequence plausibility priors (ESM1v/EVE/EVmutation/DeepSequence) for SWARM round-0.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out", default=None, help="Output JSONL path (default: OUTDIR/swarm/sequence_priors.jsonl)")

    ap.add_argument("--esm1v", default=None, help="ESM1v scores table (csv/tsv/jsonl)")
    ap.add_argument("--esm1v-mutation-col", default="mutation")
    ap.add_argument("--esm1v-score-col", default="score")
    ap.add_argument("--esm1v-chain-col", default=None)
    ap.add_argument("--esm1v-higher-is-better", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--evmutation", default=None, help="EVmutation scores table (csv/tsv/jsonl)")
    ap.add_argument("--evmutation-mutation-col", default="mutation")
    ap.add_argument("--evmutation-score-col", default="score")
    ap.add_argument("--evmutation-chain-col", default=None)
    ap.add_argument("--evmutation-higher-is-better", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--eve", default=None, help="EVE scores table (csv/tsv/jsonl)")
    ap.add_argument("--eve-mutation-col", default="mutation")
    ap.add_argument("--eve-score-col", default="score")
    ap.add_argument("--eve-chain-col", default=None)
    ap.add_argument("--eve-higher-is-better", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--deepsequence", default=None, help="DeepSequence scores table (csv/tsv/jsonl)")
    ap.add_argument("--deepsequence-mutation-col", default="mutation")
    ap.add_argument("--deepsequence-score-col", default="score")
    ap.add_argument("--deepsequence-chain-col", default=None)
    ap.add_argument("--deepsequence-higher-is-better", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--ensemble-weights",
        default="esm1v=0.45,eve=0.35,evmutation=0.15,deepsequence=0.05",
        help="Comma-separated model weights used for weighted ensemble plausibility.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    out = Path(args.out) if args.out else outdir / "swarm" / "sequence_priors.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    esm1v = load_model_scores(
        Path(args.esm1v) if args.esm1v else None,
        mutation_col=args.esm1v_mutation_col,
        score_col=args.esm1v_score_col,
        higher_is_better=bool(args.esm1v_higher_is_better),
        chain_col=args.esm1v_chain_col,
    )
    ev = load_model_scores(
        Path(args.evmutation) if args.evmutation else None,
        mutation_col=args.evmutation_mutation_col,
        score_col=args.evmutation_score_col,
        higher_is_better=bool(args.evmutation_higher_is_better),
        chain_col=args.evmutation_chain_col,
    )
    eve = load_model_scores(
        Path(args.eve) if args.eve else None,
        mutation_col=args.eve_mutation_col,
        score_col=args.eve_score_col,
        higher_is_better=bool(args.eve_higher_is_better),
        chain_col=args.eve_chain_col,
    )
    ds = load_model_scores(
        Path(args.deepsequence) if args.deepsequence else None,
        mutation_col=args.deepsequence_mutation_col,
        score_col=args.deepsequence_score_col,
        higher_is_better=bool(args.deepsequence_higher_is_better),
        chain_col=args.deepsequence_chain_col,
    )

    model_weights = parse_model_weights(args.ensemble_weights)
    keys = sorted(set(esm1v.keys()) | set(ev.keys()) | set(eve.keys()) | set(ds.keys()), key=lambda x: (x[0], x[1], x[2], x[3]))
    rows: List[Dict[str, Any]] = []
    for chain, pos, wt, mut in keys:
        row: Dict[str, Any] = {
            "chain": chain,
            "pos": int(pos),
            "wt": wt,
            "mut": mut,
            "mutation": f"{wt}{pos}{mut}",
        }
        if (chain, pos, wt, mut) in esm1v:
            row["esm1v_raw"] = esm1v[(chain, pos, wt, mut)]["raw"]
            row["esm1v_plausibility"] = esm1v[(chain, pos, wt, mut)]["plausibility"]
        if (chain, pos, wt, mut) in ev:
            row["evmutation_raw"] = ev[(chain, pos, wt, mut)]["raw"]
            row["evmutation_plausibility"] = ev[(chain, pos, wt, mut)]["plausibility"]
        if (chain, pos, wt, mut) in eve:
            row["eve_raw"] = eve[(chain, pos, wt, mut)]["raw"]
            row["eve_plausibility"] = eve[(chain, pos, wt, mut)]["plausibility"]
        if (chain, pos, wt, mut) in ds:
            row["deepsequence_raw"] = ds[(chain, pos, wt, mut)]["raw"]
            row["deepsequence_plausibility"] = ds[(chain, pos, wt, mut)]["plausibility"]
        model_vals = {
            "esm1v": row.get("esm1v_plausibility"),
            "eve": row.get("eve_plausibility"),
            "evmutation": row.get("evmutation_plausibility"),
            "deepsequence": row.get("deepsequence_plausibility"),
        }
        weighted_sum = 0.0
        weight_total = 0.0
        used = 0
        for mk, mv in model_vals.items():
            if mv is None:
                continue
            w = float(model_weights.get(mk, 0.0))
            if w <= 0.0:
                continue
            weighted_sum += float(mv) * w
            weight_total += w
            used += 1
        row["ensemble_plausibility"] = (weighted_sum / weight_total) if weight_total > 0 else None
        row["num_models"] = used
        rows.append(row)

    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(out),
        "entries": len(rows),
        "model_counts": {
            "esm1v": len(esm1v),
            "evmutation": len(ev),
            "eve": len(eve),
            "deepsequence": len(ds),
        },
        "ensemble_weights": model_weights,
    }
    (out.parent / "sequence_priors_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote sequence priors: {out}")
    print(f"Entries: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
