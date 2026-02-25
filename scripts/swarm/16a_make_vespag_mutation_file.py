import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Set

try:
    from artifact_paths import proposals_path, vespag_mutation_csv_path
except ImportError:
    from scripts.swarm.artifact_paths import proposals_path, vespag_mutation_csv_path

try:
    from mutation_utils import mutations_to_id, row_mutations
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id, row_mutations


def parse_fasta_id(path: Path) -> str:
    if not path.exists():
        return "target"
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:].strip()
                if not header:
                    break
                return header.split()[0]
    return "target"


def iter_proposals(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description="Build VespaG mutation CSV from SWARM proposals.")
    ap.add_argument("--outdir", default="data", help="Pipeline output directory (e.g., data)")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--proposals", default=None, help="Path to proposals.jsonl (overrides --round)")
    ap.add_argument("--fasta", default=None, help="FASTA with header id (default: outdir/enzyme_wt.fasta)")
    ap.add_argument("--protein-id", default=None, help="Override protein_id for VespaG mutation file")
    ap.add_argument("--roles", default=None, help="Comma-separated roles to include (optional)")
    ap.add_argument("--out", default=None, help="Output CSV path")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    proposals = Path(args.proposals) if args.proposals else proposals_path(outdir=outdir, round_id=int(args.round))
    fasta = Path(args.fasta) if args.fasta else outdir / "enzyme_wt.fasta"
    out_csv = Path(args.out) if args.out else vespag_mutation_csv_path(outdir=outdir, round_id=int(args.round))

    protein_id = args.protein_id or parse_fasta_id(fasta)
    roles: Optional[Set[str]] = None
    if args.roles:
        roles = {r.strip() for r in args.roles.split(",") if r.strip()}

    muts: Set[str] = set()
    for p in iter_proposals(proposals):
        if roles and p.get("source_role") not in roles:
            continue
        pmuts = row_mutations(p)
        if not pmuts:
            continue
        # VespaG is a single-mutation predictor; for multi-point proposals we score all
        # constituent singles and compose downstream multi-point function estimates.
        for m in pmuts:
            mid = mutations_to_id([m], include_chain=False)
            if not mid:
                continue
            muts.add(mid)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["protein_id", "mutation_id"])
        for mid in sorted(muts):
            writer.writerow([protein_id, mid])

    print(f"Wrote: {out_csv} ({len(muts)} mutations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
