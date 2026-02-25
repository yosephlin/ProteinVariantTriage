import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

try:
    from artifact_paths import proposals_path, proposals_vespag_path
except ImportError:
    from scripts.swarm.artifact_paths import proposals_path, proposals_vespag_path

try:
    from mutation_utils import row_mutations, row_variant_id
except ImportError:
    from scripts.swarm.mutation_utils import row_mutations, row_variant_id


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def fmt_num(v: Any, ndigits: int = 6) -> str:
    x = safe_float(v)
    if not math.isfinite(x):
        return ""
    return f"{x:.{int(ndigits)}f}"


def fmt_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return "true" if bool(v) else "false"
    s = str(v or "").strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return "true"
    if s in {"0", "false", "no", "n", "f"}:
        return "false"
    return ""


def fmt_list(v: Any, sep: str = ",") -> str:
    if not isinstance(v, list):
        return ""
    out = [str(x).strip() for x in v if str(x).strip()]
    return sep.join(out)


def mutation_id(row: Dict[str, Any]) -> str:
    vid = row_variant_id(row)
    return str(vid or "").upper()


def stat_field(row: Dict[str, Any], key: str) -> Any:
    stat = row.get("stat_model")
    if not isinstance(stat, dict):
        return None
    return stat.get(key)


def stat_obj_field(row: Dict[str, Any], obj_key: str, key: str) -> Any:
    stat = row.get("stat_model")
    if not isinstance(stat, dict):
        return None
    obj = stat.get(obj_key)
    if not isinstance(obj, dict):
        return None
    return obj.get(key)


def binding_field(row: Dict[str, Any], key: str) -> Any:
    bind = row.get("binding_fastdl")
    if not isinstance(bind, dict):
        return None
    return bind.get(key)


def flatten_row(row: Dict[str, Any], rank: int) -> Dict[str, str]:
    muts = row_mutations(row)
    mut_json = json.dumps(muts, ensure_ascii=False) if muts else ""
    labels = row.get("mutation_labels") if isinstance(row.get("mutation_labels"), dict) else {}
    out: Dict[str, str] = {
        "rank": str(int(rank)),
        "variant_id": mutation_id(row),
        "mutation_count": str(int(len(muts) if muts else 0)),
        "mutations_json": mut_json,
        "mutation_raw": str(labels.get("raw") or ""),
        "mutation_mature": str(labels.get("mature") or ""),
        "chain": str(row.get("chain") or ""),
        "pos": str(row.get("pos") or ""),
        "wt": str(row.get("wt") or ""),
        "mut": str(row.get("mut") or ""),
        "source_role": str(row.get("source_role") or ""),
        "move_primary": str(row.get("move_primary") or ""),
        "tier": str(row.get("tier") or ""),
        "selection_lane": str(row.get("selection_lane") or ""),
        "priority": fmt_num(row.get("priority"), 8),
        "acquisition": fmt_num(stat_field(row, "acquisition"), 8),
        "expected_hvi": fmt_num(stat_field(row, "expected_hvi"), 8),
        "expected_hvi_std": fmt_num(stat_field(row, "expected_hvi_std"), 8),
        "feasibility_prob": fmt_num(stat_field(row, "feasibility_prob"), 6),
        "obj_mean_function": fmt_num(stat_obj_field(row, "objective_mean", "function"), 6),
        "obj_mean_binding": fmt_num(stat_obj_field(row, "objective_mean", "binding"), 6),
        "obj_mean_stability": fmt_num(stat_obj_field(row, "objective_mean", "stability"), 6),
        "obj_mean_plausibility": fmt_num(stat_obj_field(row, "objective_mean", "plausibility"), 6),
        "vespag_score_norm": fmt_num(row.get("vespag_score_norm"), 6),
        "vespag_posterior": fmt_num(row.get("vespag_posterior"), 6),
        "vespag_shrunk_posterior": fmt_num(row.get("vespag_shrunk_posterior"), 6),
        "vespag_gate_band": str(row.get("vespag_gate_band") or ""),
        "vespag_gate_pass": fmt_bool(row.get("vespag_gate_pass")),
        "vespag_strict_pass": fmt_bool(row.get("vespag_strict_pass")),
        "vespag_gate_reason": str(row.get("vespag_gate_reason") or ""),
        "p_bind": fmt_num(row.get("p_bind"), 6),
        "p_bind_fastdl": fmt_num(row.get("p_bind_fastdl"), 6),
        "p_stability": fmt_num(row.get("p_stability"), 6),
        "p_plausibility": fmt_num(row.get("p_plausibility"), 6),
        "seq_prior_ensemble_plausibility": fmt_num(row.get("seq_prior_ensemble_plausibility"), 6),
        "dist_ligand": fmt_num(row.get("dist_ligand"), 4),
        "dist_functional": fmt_num(row.get("dist_functional"), 4),
        "ligand_contact": fmt_bool(row.get("ligand_contact")),
        "prolif_contact_freq": fmt_num(row.get("prolif_contact_freq"), 6),
        "functional_site": fmt_bool(row.get("functional_site")),
        "critical": fmt_bool(row.get("critical")),
        "do_not_mutate": fmt_bool(row.get("do_not_mutate")),
        "do_not_mutate_hard": fmt_bool(row.get("do_not_mutate_hard")),
        "hard_constraints": fmt_list(row.get("hard_constraints"), sep="|"),
        "soft_constraints": fmt_list(row.get("soft_constraints"), sep="|"),
        "tags": fmt_list(row.get("tags"), sep="|"),
        "binding_model": str(binding_field(row, "model") or ""),
        "binding_fallback_reason": str(binding_field(row, "fallback_reason") or ""),
        "delta_cnn_affinity": fmt_num(binding_field(row, "delta_cnn_affinity"), 6),
        "delta_cnn_score": fmt_num(binding_field(row, "delta_cnn_score"), 6),
    }
    return out


def sort_key(row: Dict[str, str], key: str):
    s = str(row.get(key) or "")
    try:
        x = float(s)
        if math.isfinite(x):
            return (0, x)
    except Exception:
        pass
    return (1, s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert SWARM proposal JSONL into a readable TSV.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--stage", choices=["raw", "vespag"], default="vespag",
                    help="Default input selection: proposals_rK.jsonl (raw) or proposals_vespag_rK.jsonl (vespag).")
    ap.add_argument("--proposals", default=None, help="Override input JSONL path.")
    ap.add_argument("--out", default=None, help="Output TSV path (default: <input_stem>_readable.tsv).")
    ap.add_argument("--sort-by", default="expected_hvi",
                    help="Column name used for sorting in output TSV.")
    ap.add_argument("--ascending", action="store_true", default=False,
                    help="Sort ascending (default: descending).")
    ap.add_argument("--limit", type=int, default=0, help="Optional row cap after sorting (0 = all).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    round_id = int(args.round)
    if args.proposals:
        in_path = Path(args.proposals)
    elif args.stage == "raw":
        in_path = proposals_path(outdir=outdir, round_id=round_id)
    else:
        in_path = proposals_vespag_path(outdir=outdir, round_id=round_id)

    if not in_path.exists():
        raise SystemExit(f"Proposals JSONL not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_readable.tsv")
    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit(f"No records found in {in_path}")

    flat = [flatten_row(r, i + 1) for i, r in enumerate(rows)]
    sort_by = str(args.sort_by).strip()
    if sort_by and sort_by in flat[0]:
        flat = sorted(flat, key=lambda r: sort_key(r, sort_by), reverse=(not bool(args.ascending)))

    if int(args.limit) > 0:
        flat = flat[: int(args.limit)]

    for i, rec in enumerate(flat, start=1):
        rec["rank"] = str(int(i))

    fieldnames = list(flat[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)

    print(f"Wrote: {out_path} ({len(flat)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
