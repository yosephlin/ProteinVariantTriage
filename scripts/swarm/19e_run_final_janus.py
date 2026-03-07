import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

try:
    from artifact_paths import (
        final_janus_input_path,
        final_janus_scores_path,
        final_swarm_panel_explore_path,
        final_swarm_panel_production_path,
        final_swarm_panel_path,
        final_swarm_panel_summary_path,
        final_with_janus_path,
        final_with_janus_production_path,
        final_with_janus_production_summary_path,
        final_with_janus_summary_path,
        panel_path as round_panel_path,
        panel_with_janus_path as round_panel_with_janus_path,
        swarm_root,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        final_janus_input_path,
        final_janus_scores_path,
        final_swarm_panel_explore_path,
        final_swarm_panel_production_path,
        final_swarm_panel_path,
        final_swarm_panel_summary_path,
        final_with_janus_path,
        final_with_janus_production_path,
        final_with_janus_production_summary_path,
        final_with_janus_summary_path,
        panel_path as round_panel_path,
        panel_with_janus_path as round_panel_with_janus_path,
        swarm_root,
    )


def safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def parse_rounds(spec: str) -> list[int]:
    out: list[int] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("No rounds parsed from --rounds")
    return out


def pick_panel_for_round(outdir: Path, rnd: int) -> Path:
    # Prefer plain selection panel; fall back to Janus-joined if plain missing.
    primary = round_panel_path(outdir=outdir, round_id=int(rnd))
    fallback = round_panel_with_janus_path(outdir=outdir, round_id=int(rnd))
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No panel found for round {rnd}: {primary} or {fallback}")


def panel_rank_score(row: dict) -> float:
    for key in ("triage_score", "utility", "score_total", "p_func", "vespag_shrunk_posterior"):
        x = safe_float(row.get(key))
        if x is not None:
            return x
    return -1e9


def normalize_panel_row(row: dict) -> dict:
    # Some historical artifacts contain padded header names/values; normalize here for stable merges.
    out = {}
    for k, v in (row or {}).items():
        kk = str(k or "").strip()
        if not kk:
            continue
        vv = v.strip() if isinstance(v, str) else v
        out[kk] = vv
    return out


def as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def is_explore_variant(row: dict) -> bool:
    lane = str(row.get("selection_lane") or "").strip().lower()
    gate_band = str(row.get("vespag_gate_band") or row.get("gate_band") or "").strip().lower()
    if lane == "explore":
        return True
    if gate_band == "red":
        return True
    if as_bool(row.get("binding_challenger")) or as_bool(row.get("chemistry_challenger")):
        return True
    if as_bool(row.get("red_rescued")):
        return True
    return False


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(str(x) for x in cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Final Janus scoring over merged SWARM panels across rounds, or rerun from an existing merged panel."
    )
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--rounds", default="0,1", help="Comma-separated round ids to merge (e.g. 0,1)")
    ap.add_argument("--final-dir", default=None, help="Default: OUTDIR/swarm")
    ap.add_argument(
        "--panel",
        default=None,
        help="Optional existing merged panel TSV. If provided, skip round-panel merge and use this panel directly.",
    )
    ap.add_argument("--max-candidates", type=int, default=0, help="Optional cap for merged final panel rows (0=all).")
    ap.add_argument(
        "--janus-panel-mode",
        choices=["all", "production"],
        default="production",
        help="Run Janus on all merged variants or only production split.",
    )
    ap.add_argument("--with-janus", action="store_true", default=True)
    ap.add_argument("--no-janus", dest="with_janus", action="store_false")
    ap.add_argument("--janus-cmd", default=None)
    ap.add_argument("--janus-repo", default=None)
    ap.add_argument("--stability-gate", choices=["bayes"], default="bayes")
    ap.add_argument("--stability-fdr", type=float, default=0.10)
    ap.add_argument("--drop-janus-outliers", action="store_true", default=True)
    ap.add_argument("--keep-janus-outliers", dest="drop_janus_outliers", action="store_false")
    ap.add_argument("--janus-positive-stabilizing", action="store_true", default=True)
    ap.add_argument("--negative-stabilizing", dest="janus_positive_stabilizing", action="store_false")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    rounds = parse_rounds(args.rounds)
    effective_round = int(rounds[-1])
    final_dir = Path(args.final_dir).resolve() if args.final_dir else swarm_root(outdir)
    final_dir.mkdir(parents=True, exist_ok=True)

    final_panel = Path(args.final_dir).resolve() / "swarm_final_panel.tsv" if args.final_dir else final_swarm_panel_path(outdir)
    final_panel_production = (
        Path(args.final_dir).resolve() / "swarm_final_panel_production.tsv"
        if args.final_dir
        else final_swarm_panel_production_path(outdir)
    )
    final_panel_explore = (
        Path(args.final_dir).resolve() / "swarm_final_panel_explore.tsv"
        if args.final_dir
        else final_swarm_panel_explore_path(outdir)
    )
    final_panel_summary = (
        Path(args.final_dir).resolve() / "swarm_final_panel_summary.json"
        if args.final_dir
        else final_swarm_panel_summary_path(outdir)
    )

    panel_override = Path(args.panel).resolve() if args.panel else None
    all_fields: set[str] = set()

    if panel_override is not None:
        if not panel_override.exists():
            raise FileNotFoundError(f"--panel not found: {panel_override}")
        if int(args.max_candidates) > 0:
            print("[final-janus] ignoring --max-candidates because --panel was provided")
        print(f"[final-janus] using explicit merged panel override: {panel_override}")
    else:
        # Load and merge rows.
        rows_by_variant: dict[str, dict] = {}
        source_counts: dict[int, int] = {}
        dedup_replaced = 0

        for rnd in rounds:
            panel_path = pick_panel_for_round(outdir, rnd)
            with panel_path.open() as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                count = 0
                for raw_row in reader:
                    count += 1
                    row = normalize_panel_row(raw_row)
                    vid = (row.get("variant_id") or "").strip()
                    if not vid:
                        continue
                    row["round_source"] = str(rnd)
                    all_fields.update(row.keys())
                    old = rows_by_variant.get(vid)
                    if old is None:
                        rows_by_variant[vid] = row
                        continue
                    old_score = panel_rank_score(old)
                    new_score = panel_rank_score(row)
                    # Prefer higher score; tie-break by newer round.
                    if (new_score > old_score) or (
                        new_score == old_score and int(row["round_source"]) > int(old.get("round_source", "-1"))
                    ):
                        rows_by_variant[vid] = row
                        dedup_replaced += 1
                source_counts[rnd] = count

        merged_rows = list(rows_by_variant.values())
        merged_rows.sort(key=lambda r: panel_rank_score(r), reverse=True)
        merged_before_cap = len(merged_rows)
        if int(args.max_candidates) > 0:
            merged_rows = merged_rows[: int(args.max_candidates)]

        preferred_front = ["variant_id", "chain", "pos", "wt", "mut", "source_role", "round_source"]
        ordered_fields = [f for f in preferred_front if f in all_fields]
        ordered_fields += sorted(f for f in all_fields if f not in ordered_fields)

        production_rows = [r for r in merged_rows if not is_explore_variant(r)]
        explore_rows = [r for r in merged_rows if is_explore_variant(r)]

        for panel_path, rows_out in (
            (final_panel, merged_rows),
            (final_panel_production, production_rows),
            (final_panel_explore, explore_rows),
        ):
            with panel_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ordered_fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                for row in rows_out:
                    writer.writerow(row)

        summary = {
            "rounds": rounds,
            "source_counts": source_counts,
            "merged_unique_variants_before_cap": int(merged_before_cap),
            "merged_unique_variants": len(merged_rows),
            "production_variants": len(production_rows),
            "explore_variants": len(explore_rows),
            "max_candidates": int(args.max_candidates),
            "dedup_replaced": dedup_replaced,
            "final_panel": str(final_panel),
            "final_panel_production": str(final_panel_production),
            "final_panel_explore": str(final_panel_explore),
            "janus_panel_mode": str(args.janus_panel_mode),
        }
        final_panel_summary.write_text(json.dumps(summary, indent=2))
        print(f"Wrote: {final_panel}")
        print(f"Wrote: {final_panel_production}")
        print(f"Wrote: {final_panel_explore}")
        print(f"Wrote: {final_panel_summary}")

    if not args.with_janus:
        return 0

    py = sys.executable
    if args.final_dir:
        fd = Path(args.final_dir).resolve()
        janus_input = fd / "janus_input.csv"
        janus_scores = fd / "janus_scores.csv"
        final_with_janus = fd / "swarm_final_with_janus.tsv"
        final_with_janus_summary = fd / "swarm_final_with_janus_summary.json"
        final_with_janus_production = fd / "swarm_final_with_janus_production.tsv"
        final_with_janus_production_summary = fd / "swarm_final_with_janus_production_summary.json"
    else:
        janus_input = final_janus_input_path(outdir)
        janus_scores = final_janus_scores_path(outdir)
        final_with_janus = final_with_janus_path(outdir)
        final_with_janus_summary = final_with_janus_summary_path(outdir)
        final_with_janus_production = final_with_janus_production_path(outdir)
        final_with_janus_production_summary = final_with_janus_production_summary_path(outdir)

    if panel_override is not None:
        janus_panel = panel_override
        if str(args.janus_panel_mode).strip().lower() == "production":
            janus_out = final_with_janus_production
            janus_summary = final_with_janus_production_summary
        else:
            janus_out = final_with_janus
            janus_summary = final_with_janus_summary
    elif str(args.janus_panel_mode).strip().lower() == "production":
        janus_panel = final_panel_production
        janus_out = final_with_janus_production
        janus_summary = final_with_janus_production_summary
    else:
        janus_panel = final_panel
        janus_out = final_with_janus
        janus_summary = final_with_janus_summary

    cmd_19b = [
        py,
        "scripts/swarm/19b_make_janus_input.py",
        "--outdir", str(outdir),
        "--round", str(effective_round),
        "--panel", str(janus_panel),
        "--out", str(janus_input),
    ]
    run(cmd_19b)

    cmd_19c = [
        py,
        "scripts/swarm/19c_run_janusddg.py",
        "--outdir", str(outdir),
        "--round", str(effective_round),
        "--input", str(janus_input),
        "--output", str(janus_scores),
    ]
    if args.janus_cmd:
        cmd_19c.extend(["--janus-cmd", args.janus_cmd])
    if args.janus_repo:
        cmd_19c.extend(["--janus-repo", args.janus_repo])
    run(cmd_19c)

    cmd_19d = [
        py,
        "scripts/swarm/19d_join_janus_panel.py",
        "--outdir", str(outdir),
        "--round", str(effective_round),
        "--panel", str(janus_panel),
        "--janus", str(janus_scores),
        "--out", str(janus_out),
        "--summary", str(janus_summary),
        "--stability-gate", str(args.stability_gate),
        "--stability-fdr", str(args.stability_fdr),
    ]
    if args.drop_janus_outliers:
        cmd_19d.append("--drop-janus-outliers")
    else:
        cmd_19d.append("--keep-janus-outliers")
    if args.janus_positive_stabilizing:
        cmd_19d.append("--janus-positive-stabilizing")
    else:
        cmd_19d.append("--negative-stabilizing")
    run(cmd_19d)

    # Preserve legacy final paths for downstream readers while exposing explicit production artifact names.
    if str(args.janus_panel_mode).strip().lower() == "production":
        if janus_out != final_with_janus:
            final_with_janus.write_text(janus_out.read_text())
        if janus_summary != final_with_janus_summary:
            final_with_janus_summary.write_text(janus_summary.read_text())

    print("[final-janus] complete")
    print(f"  - panel_mode={args.janus_panel_mode}")
    if panel_override is not None:
        print(f"  - merged panel override: {panel_override}")
    print(f"  - {janus_out}")
    print(f"  - {janus_summary}")
    if str(args.janus_panel_mode).strip().lower() == "production":
        print(f"  - legacy mirror: {final_with_janus}")
        print(f"  - legacy mirror: {final_with_janus_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
