import argparse
import csv
from pathlib import Path

try:
    from artifact_paths import janus_input_path, panel_path
except ImportError:
    from scripts.swarm.artifact_paths import janus_input_path, panel_path

try:
    from mutation_utils import row_mutations, row_variant_id
except ImportError:
    from scripts.swarm.mutation_utils import row_mutations, row_variant_id


def parse_fasta(path: Path) -> str:
    seq = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq).upper()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build JanusDDG input CSV from selected SWARM panel.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--panel", default=None, help="Default: OUTDIR/swarm/swarm_panel_rK.tsv")
    ap.add_argument("--fasta", default=None, help="Default: OUTDIR/enzyme_wt.fasta")
    ap.add_argument("--out", default=None, help="Default: OUTDIR/swarm/janus_input_rK.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    round_id = int(args.round)
    panel = Path(args.panel) if args.panel else panel_path(outdir=outdir, round_id=round_id)
    fasta = Path(args.fasta) if args.fasta else outdir / "enzyme_wt.fasta"
    out_csv = Path(args.out) if args.out else janus_input_path(outdir=outdir, round_id=round_id)

    if not panel.exists():
        raise SystemExit(f"Panel not found: {panel}")
    if not fasta.exists():
        raise SystemExit(f"FASTA not found: {fasta}")

    wt_seq = parse_fasta(fasta)
    if not wt_seq:
        raise SystemExit("WT sequence is empty.")

    def mutation_token(m: dict) -> str:
        return f"{str(m.get('wt') or '').upper()}{int(m.get('pos'))}{str(m.get('mut') or '').upper()}"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids = set()
    with panel.open() as in_fh, out_csv.open("w", newline="") as out_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        writer = csv.DictWriter(out_fh, fieldnames=["ID", "Sequence", "MTS"])
        writer.writeheader()
        for row in reader:
            muts = row_mutations(row)
            if not muts:
                continue

            fixed_muts = []
            valid = True
            for m in muts:
                try:
                    pos = int(m.get("pos"))
                except Exception:
                    valid = False
                    break
                if pos <= 0 or pos > len(wt_seq):
                    valid = False
                    break
                wt = str(m.get("wt") or "").strip().upper()
                mut = str(m.get("mut") or "").strip().upper()
                if not wt or not mut:
                    valid = False
                    break
                seq_wt = wt_seq[pos - 1]
                if seq_wt != wt:
                    wt = seq_wt
                fixed_muts.append(
                    {
                        "chain": str(m.get("chain") or "A"),
                        "pos": int(pos),
                        "wt": wt,
                        "mut": mut,
                    }
                )
            if not valid or not fixed_muts:
                continue

            # JanusDDG uses underscore-separated mutation tuples for multi-point input.
            mts = "_".join(mutation_token(m) for m in fixed_muts)
            vid = (row.get("variant_id") or "").strip()
            if not vid:
                vid = row_variant_id({"mutations": fixed_muts}) or mts
            if not vid:
                continue
            if vid in seen_ids:
                continue

            writer.writerow({"ID": vid, "Sequence": wt_seq, "MTS": mts})
            seen_ids.add(vid)
            count += 1

    print(f"Wrote: {out_csv} ({count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
