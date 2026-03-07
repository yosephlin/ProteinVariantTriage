import argparse
import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path

try:
    from artifact_paths import janus_input_path, janus_scores_path
except ImportError:
    from scripts.swarm.artifact_paths import janus_input_path, janus_scores_path


def build_command(template: str, input_csv: Path, output_csv: Path) -> tuple[list[str], bool, bool]:
    uses_input = "{input}" in template
    uses_output = "{output}" in template
    # Quote substituted paths so templates don't need explicit shell quoting.
    rendered = template.format(
        input=shlex.quote(str(input_csv)),
        output=shlex.quote(str(output_csv)),
    )
    cmd = shlex.split(rendered)
    if not uses_input:
        cmd.append(str(input_csv))
    return cmd, uses_input, uses_output


def infer_default_janus_output(input_csv: Path, cwd: Path | None) -> Path | None:
    # JanusDDG commonly writes either:
    #   results/result_<input_name>.csv
    # or
    #   results/Result_<input_name>.csv
    root = cwd if cwd is not None else Path.cwd()
    candidates = [
        root / "results" / f"result_{input_csv.name}",
        root / "results" / f"Result_{input_csv.name}",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for _ in csv.DictReader(fh))


def load_csv_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return ids
        # Robust to whitespace-padded headers.
        fields = {f.strip(): f for f in reader.fieldnames}
        id_col = None
        for cand in ("ID", "id", "variant_id", "mutation_id", "MTS"):
            if cand in fields:
                id_col = fields[cand]
                break
        if id_col is None:
            for f in reader.fieldnames:
                if "id" in f.lower():
                    id_col = f
                    break
        if id_col is None:
            return ids
        for row in reader:
            v = (row.get(id_col) or "").strip()
            if v:
                ids.add(v)
    return ids


def _ensure_conda_streaming(cmd: list[str]) -> list[str]:
    """Make `conda run` stream child output in notebooks for easier debugging."""
    if not cmd:
        return cmd
    exe = Path(cmd[0]).name
    if exe not in {"conda", "mamba"}:
        return cmd
    if len(cmd) >= 2 and cmd[1] == "run" and "--no-capture-output" not in cmd:
        return [cmd[0], "run", "--no-capture-output", *cmd[2:]]
    return cmd


def _is_conda_run(cmd: list[str]) -> bool:
    return bool(cmd) and Path(cmd[0]).name in {"conda", "mamba"} and len(cmd) >= 2 and cmd[1] == "run"


def _sanitize_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # Prevent the notebook/kernel Python env from leaking into Janus env resolution.
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "_CE_M", "_CE_CONDA"):
        env.pop(k, None)
    return env


def _find_python_token_index(cmd: list[str]) -> int | None:
    for i, token in enumerate(cmd):
        exe = Path(token).name
        if exe.startswith("python"):
            return i
    return None


def _build_python_preflight(cmd: list[str]) -> list[str] | None:
    """
    Build a preflight command that validates Janus runtime imports in the exact
    interpreter context used by Janus invocation.
    """
    py_idx = _find_python_token_index(cmd)
    if py_idx is None:
        return None
    return cmd[: py_idx + 1] + [
        "-c",
        (
            "import sys, torch, esm, pandas, scipy, sklearn; "
            "print(sys.executable); "
            "print(torch.__version__)"
        ),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run JanusDDG on selected SWARM panel.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--input", default=None, help="Default: OUTDIR/swarm/janus_input_rK.csv")
    ap.add_argument("--output", default=None, help="Canonical output path (default: OUTDIR/swarm/janus_scores_rK.csv)")
    ap.add_argument(
        "--janus-cmd",
        default=None,
        help="Command template. Supports optional {input} and {output}; if {input} is omitted it is appended.",
    )
    ap.add_argument("--janus-repo", default=None,
                    help="Repo/workdir for Janus command; default from JANUS_REPO env.")
    ap.add_argument("--python", default=None,
                    help="Python executable for default Janus command when --janus-cmd is omitted.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    round_id = int(args.round)
    input_csv = Path(args.input) if args.input else janus_input_path(outdir=outdir, round_id=round_id)
    output_csv = Path(args.output) if args.output else janus_scores_path(outdir=outdir, round_id=round_id)

    if not input_csv.exists():
        raise SystemExit(f"Janus input not found: {input_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    janus_repo = Path(args.janus_repo or os.environ.get("JANUS_REPO", "")).expanduser() if (args.janus_repo or os.environ.get("JANUS_REPO")) else None

    template = args.janus_cmd or os.environ.get("JANUS_CMD")
    if not template:
        if not janus_repo:
            raise SystemExit(
                "Janus command is not configured. Set --janus-cmd (with {input}/{output}) "
                "or set JANUS_CMD / JANUS_REPO."
            )
        pyexe = args.python or os.environ.get("PYTHON", "python")
        # JanusDDG README invocation style.
        template = f"{pyexe} src/main.py {{input}}"

    cmd, _uses_input, uses_output = build_command(template, input_csv, output_csv)
    cmd = _ensure_conda_streaming(cmd)
    env = _sanitize_subprocess_env()
    # Prevent stale-score reuse when command writes to Janus default results path.
    if not uses_output and output_csv.exists():
        output_csv.unlink()

    preflight = _build_python_preflight(cmd)
    if preflight:
        print(">", " ".join(preflight))
        p0 = subprocess.run(
            preflight,
            cwd=str(janus_repo) if janus_repo else None,
            text=True,
            capture_output=True,
            env=env,
        )
        if p0.returncode != 0:
            if p0.stdout:
                print(p0.stdout)
            if p0.stderr:
                print(p0.stderr)
            raise SystemExit(
                "Janus preflight failed: target env cannot import one or more Janus "
                "runtime dependencies (torch, esm, pandas, scipy, sklearn). "
                "Verify JANUS_CMD / JANUS_REPO / env setup."
            )
        if p0.stdout:
            print(p0.stdout)

    print(">", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(janus_repo) if janus_repo else None,
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise SystemExit(f"Janus command failed with exit code {proc.returncode}")
    if proc.stdout:
        print(proc.stdout)

    if not output_csv.exists():
        inferred = infer_default_janus_output(input_csv, janus_repo)
        if not uses_output and inferred:
            shutil.copy2(inferred, output_csv)
        else:
            raise SystemExit(
                f"Janus run finished but output missing: {output_csv}. "
                f"Tried inferred defaults in {(janus_repo if janus_repo else Path.cwd()) / 'results'}"
            )

    # Basic integrity check: output should not have fewer rows than input.
    in_rows = count_csv_rows(input_csv)
    out_rows = count_csv_rows(output_csv)
    if in_rows > 0 and out_rows < in_rows:
        raise SystemExit(
            f"Janus output appears incomplete: input rows={in_rows}, output rows={out_rows}, output={output_csv}"
        )
    in_ids = load_csv_ids(input_csv)
    out_ids = load_csv_ids(output_csv)
    missing_ids = sorted(in_ids - out_ids)
    if missing_ids:
        sample = ", ".join(missing_ids[:8])
        raise SystemExit(
            f"Janus output ID mismatch: missing {len(missing_ids)}/{len(in_ids)} input IDs in output. "
            f"Examples: {sample}"
        )

    print(f"Wrote: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
