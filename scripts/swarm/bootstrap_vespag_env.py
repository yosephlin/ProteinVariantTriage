import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path


def resolve_conda_bin(preferred: str) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    if os.environ.get("CONDA_EXE"):
        candidates.append(os.environ["CONDA_EXE"])
    which = shutil.which("conda")
    if which:
        candidates.append(which)
    home = Path.home()
    candidates.extend(
        [
            str(home / "miniconda3" / "Scripts" / "conda.exe"),
            str(home / "anaconda3" / "Scripts" / "conda.exe"),
            str(home / "miniconda" / "Scripts" / "conda.exe"),
            str(home / "mambaforge" / "Scripts" / "conda.exe"),
            r"C:\ProgramData\anaconda3\Scripts\conda.exe",
        ]
    )
    seen = set()
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.exists():
            return s
    raise SystemExit("Could not resolve conda executable. Set CONDA_EXE or VESPAG_CONDA_BIN.")


def conda_env_exists(conda_bin: str, env_name: str) -> bool:
    cmd = [conda_bin, "env", "list", "--json"]
    data = json.loads(subprocess.check_output(cmd, text=True))
    for env_path in data.get("envs") or []:
        if Path(env_path).name.lower() == env_name.lower():
            return True
    return False


def probe_env(conda_bin: str, env_name: str) -> dict:
    code = (
        "import importlib.util, json, sys; "
        "mods=['vespag','typer','torch','transformers','sentencepiece','h5py','polars','rich']; "
        "print(json.dumps({'python': sys.version.split()[0], "
        "'missing': [m for m in mods if importlib.util.find_spec(m) is None]}))"
    )
    out = subprocess.check_output(
        [conda_bin, "run", "--no-capture-output", "-n", env_name, "python", "-c", code],
        text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Create or repair a dedicated VespaG conda env for offline notebook use.")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--env-name", default=os.environ.get("VESPAG_CONDA_ENV", "vespag-py312"))
    ap.add_argument("--python-version", default="3.12")
    ap.add_argument("--conda-bin", default=os.environ.get("CONDA_EXE", "conda"))
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    vespag_repo = repo_root / "VespaG"
    if not (vespag_repo / "pyproject.toml").exists():
        raise SystemExit(f"Local VespaG repo not found at {vespag_repo}")

    conda_bin = resolve_conda_bin(args.conda_bin)
    if not conda_env_exists(conda_bin, args.env_name):
        subprocess.check_call([conda_bin, "create", "-n", args.env_name, "-y", f"python={args.python_version}", "pip"])

    probe = probe_env(conda_bin, args.env_name)
    if probe.get("missing"):
        subprocess.check_call(
            [
                conda_bin,
                "run",
                "--no-capture-output",
                "-n",
                args.env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(vespag_repo),
            ]
        )
        probe = probe_env(conda_bin, args.env_name)

    print(
        json.dumps(
            {
                "env_name": args.env_name,
                "python": probe.get("python"),
                "missing": probe.get("missing", []),
                "ready": not bool(probe.get("missing")),
            }
        )
    )
    return 0 if not probe.get("missing") else 1


if __name__ == "__main__":
    raise SystemExit(main())
