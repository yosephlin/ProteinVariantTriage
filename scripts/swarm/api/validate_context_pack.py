import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate SWARM API context pack")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path.cwd() / "swarm_api"
    ctx_path = outdir / "context_api.json"
    res_path = outdir / "residue_constraints.jsonl"

    if not ctx_path.exists() or not res_path.exists():
        raise SystemExit("Missing context_api.json or residue_constraints.jsonl")

    ctx = json.loads(ctx_path.read_text())
    seq_len = ctx.get("uniprot", {}).get("sequence_length") or ctx.get("target", {}).get("length")

    count = 0
    do_not = 0
    with res_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            count += 1
            try:
                obj = json.loads(line)
                if obj.get("policy", {}).get("do_not_mutate"):
                    do_not += 1
            except Exception:
                pass

    print("Residues in file:", count)
    if seq_len:
        print("Sequence length:", seq_len)
        if int(seq_len) != count:
            print("WARNING: residue count does not match sequence length")
    print("Do-not-mutate residues:", do_not)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
