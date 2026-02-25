import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Set

try:
    from artifact_paths import (
        swarm_root,
        vespag_mutation_csv_path,
        vespag_scores_csv_path,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        swarm_root,
        vespag_mutation_csv_path,
        vespag_scores_csv_path,
    )


def parse_fasta_ids(fasta_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                token = line[1:].split()[0]
                if token:
                    ids.add(token)
    return ids


def embeddings_has_all_ids(path: Path, required_ids: Set[str]) -> bool:
    if not required_ids:
        return True
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as h5:
            keys = set(str(k) for k in h5.keys())
        return required_ids.issubset(keys)
    except Exception:
        return False


def find_embeddings(cache_dir: Path, required_ids: Optional[Set[str]] = None) -> Optional[Path]:
    if not cache_dir.exists():
        return None
    h5s = sorted(cache_dir.glob("*.h5"))
    if not h5s:
        return None
    required_ids = required_ids or set()
    preferred = sorted(h5s, key=lambda p: ("embed" not in p.name.lower(), p.name.lower()))
    for p in preferred:
        if is_valid_h5(p) and embeddings_has_all_ids(p, required_ids):
            return p
    return None


def is_valid_h5(path: Path) -> bool:
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as h5:
            keys = list(h5.keys())
            return len(keys) > 0
    except Exception:
        return False


def run(cmd, cwd: Optional[Path] = None, env: Optional[dict] = None):
    print(">", " ".join(str(c) for c in cmd))
    if cwd is not None:
        print("  cwd:", str(cwd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def with_cpu_env(env: dict) -> dict:
    cpu_env = env.copy()
    # Force torch device selection inside VespaG to CPU.
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    return cpu_env


def resolve_installed_model_weights() -> Optional[Path]:
    try:
        import vespag  # type: ignore
        pkg_dir = Path(vespag.__file__).resolve().parent
        candidates = [
            pkg_dir / "model_weights",
            pkg_dir.parent / "model_weights",
        ]
        for c in candidates:
            if c.exists():
                return c
    except Exception:
        return None
    return None


def resolve_conda_env_model_weights(conda_bin: str, conda_env: str) -> Optional[Path]:
    if not conda_env:
        return None
    if not shutil.which(conda_bin):
        return None
    cmd = [
        conda_bin,
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        "-c",
        (
            "from pathlib import Path; "
            "import vespag; "
            "pkg=Path(vespag.__file__).resolve().parent; "
            "cands=[pkg/'model_weights', pkg.parent/'model_weights']; "
            "hits=[str(c) for c in cands if c.exists()]; "
            "print(hits[0] if hits else '')"
        ),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None
    if not out:
        return None
    p = Path(out.splitlines()[-1].strip())
    if p.exists():
        return p
    return None


def resolve_conda_bin(preferred: str) -> Optional[str]:
    candidates = []
    if preferred:
        candidates.append(preferred)
    env_bin = os.environ.get("CONDA_EXE")
    if env_bin:
        candidates.append(env_bin)
    path_bin = shutil.which(preferred) if preferred else None
    if path_bin:
        candidates.append(path_bin)
    which_conda = shutil.which("conda")
    if which_conda:
        candidates.append(which_conda)

    home = Path.home()
    candidates.extend(
        [
            str(home / "miniconda" / "bin" / "conda"),
            str(home / "miniconda3" / "bin" / "conda"),
            str(home / "anaconda3" / "bin" / "conda"),
            str(home / "miniforge3" / "bin" / "conda"),
        ]
    )

    seen = set()
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        if p.exists() and os.access(str(p), os.X_OK):
            return s
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def prepare_runtime_dir(outdir: Path, model_weights_dir: Optional[Path]) -> Path:
    runtime_dir = outdir / "swarm" / "vespag_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if model_weights_dir and model_weights_dir.exists():
        link_or_copy(model_weights_dir, runtime_dir / "model_weights")
    return runtime_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Run VespaG scoring for a SWARM round.")
    ap.add_argument("--outdir", default="data", help="Pipeline output directory")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--fasta", default=None, help="FASTA path (default: outdir/enzyme_wt.fasta)")
    ap.add_argument("--mutation-file", default=None, help="Mutation CSV path (default: outdir/swarm/vespag_roundX_mutations.csv)")
    ap.add_argument("--cache-dir", default=None, help="Cache dir for embeddings (default: outdir/swarm/vespag_cache)")
    ap.add_argument("--embeddings", default=None, help="Override embeddings .h5 path")
    ap.add_argument("--out", default=None, help="VespaG raw output dir (default: outdir/swarm)")
    ap.add_argument("--vespag-bin", default="vespag", help="vespag CLI entrypoint")
    ap.add_argument("--conda-bin", default=os.environ.get("CONDA_EXE", "conda"), help="Conda executable used for env-scoped VespaG calls.")
    ap.add_argument("--conda-env", default=os.environ.get("VESPAG_CONDA_ENV", "swarm"), help="Conda env containing VespaG when not in current runtime.")
    ap.add_argument("--vespag-cwd", default=None, help="Working directory for vespag runtime")
    ap.add_argument("--model-weights-dir", default=None, help="Directory containing VespaG model_weights")
    ap.add_argument("--hf-home", default=None, help="HuggingFace cache dir (default: outdir/swarm/hf_cache)")
    ap.add_argument("--cpu-embeddings", action="store_true", default=True, help="Generate transformer embeddings on CPU to avoid GPU OOM")
    ap.add_argument("--gpu-embeddings", dest="cpu_embeddings", action="store_false", help="Allow GPU for embedding generation")
    ap.add_argument("--retry-cpu-on-fail", action="store_true", default=True, help="Retry scoring command on CPU if GPU launch fails")
    ap.add_argument("--no-retry-cpu-on-fail", dest="retry_cpu_on_fail", action="store_false")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--single-csv", action="store_true", default=True)
    ap.add_argument("--no-single-csv", dest="single_csv", action="store_false")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    fasta = (Path(args.fasta).resolve() if args.fasta else outdir / "enzyme_wt.fasta")
    mutation_file = (
        Path(args.mutation_file).resolve()
        if args.mutation_file
        else vespag_mutation_csv_path(outdir=outdir, round_id=int(args.round))
    )
    cache_dir = (Path(args.cache_dir).resolve() if args.cache_dir else outdir / "swarm" / "vespag_cache")
    out = (Path(args.out).resolve() if args.out else swarm_root(outdir))
    canonical_scores = vespag_scores_csv_path(outdir=outdir, round_id=int(args.round))
    project_root = Path(__file__).resolve().parents[2]
    local_repo_weights = project_root / "VespaG" / "model_weights"
    conda_bin = resolve_conda_bin(args.conda_bin)
    installed_weights = resolve_installed_model_weights()
    conda_env_weights = resolve_conda_env_model_weights(conda_bin, args.conda_env) if conda_bin else None
    explicit_weights = Path(args.model_weights_dir) if args.model_weights_dir else None

    model_weights_dir = None
    if explicit_weights and explicit_weights.exists():
        model_weights_dir = explicit_weights
    elif local_repo_weights.exists():
        model_weights_dir = local_repo_weights
    elif installed_weights and installed_weights.exists():
        model_weights_dir = installed_weights
    elif conda_env_weights and conda_env_weights.exists():
        model_weights_dir = conda_env_weights

    default_runtime = prepare_runtime_dir(outdir, model_weights_dir)
    vespag_cwd = Path(args.vespag_cwd) if args.vespag_cwd else default_runtime

    has_cli_in_path = shutil.which(args.vespag_bin) is not None
    use_module_runner = False
    use_conda_runner = False
    if not has_cli_in_path:
        try:
            import importlib.util
            use_module_runner = importlib.util.find_spec("vespag") is not None
        except Exception:
            use_module_runner = False
        if (not use_module_runner) and args.conda_env and conda_bin:
            use_conda_runner = True
        if not use_module_runner and not use_conda_runner:
            raise SystemExit(
                f"vespag CLI not found in current env and no usable conda env runner. "
                f"Checked current PATH for '{args.vespag_bin}' and conda env '{args.conda_env}'."
            )

    def vespag_cmd(*parts: str) -> list[str]:
        if use_module_runner:
            return [sys.executable, "-m", "vespag", *parts]
        if use_conda_runner:
            return [str(conda_bin), "run", "--no-capture-output", "-n", args.conda_env, args.vespag_bin, *parts]
        return [args.vespag_bin, *parts]

    if not (vespag_cwd / "model_weights").exists():
        raise SystemExit(
            "VespaG model_weights not found for runtime. "
            "Provide --model-weights-dir or place weights at VespaG/model_weights."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    fasta_ids = parse_fasta_ids(fasta)
    env = os.environ.copy()
    default_hf = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).resolve()
    hf_home = Path(args.hf_home).resolve() if args.hf_home else default_hf
    hf_home.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)

    embeddings = Path(args.embeddings) if args.embeddings else find_embeddings(cache_dir, required_ids=fasta_ids)
    if embeddings is not None and embeddings.exists() and (
        (not is_valid_h5(embeddings)) or (not embeddings_has_all_ids(embeddings, fasta_ids))
    ):
        print(f"[vespag] removing invalid embeddings cache: {embeddings}")
        try:
            embeddings.unlink()
        except Exception:
            pass
        embeddings = None

    if embeddings is None:
        embed_cmd = vespag_cmd(
            "predict",
            "-i", str(fasta),
            "-o", str(cache_dir),
            "--no-csv",
            "--h5-output",
        )
        embed_env = with_cpu_env(env) if args.cpu_embeddings else env
        run(embed_cmd, cwd=vespag_cwd, env=embed_env)
        embeddings = find_embeddings(cache_dir, required_ids=fasta_ids)

    if (
        embeddings is None
        or not embeddings.exists()
        or not is_valid_h5(embeddings)
        or not embeddings_has_all_ids(embeddings, fasta_ids)
    ):
        raise SystemExit("Failed to locate VespaG embeddings (.h5) in cache dir.")

    cmd = vespag_cmd(
        "predict",
        "-i", str(fasta),
        "-e", str(embeddings),
        "--mutation-file", str(mutation_file),
        "-o", str(out),
    )
    if args.single_csv:
        cmd.append("--single-csv")
    if args.normalize:
        cmd.append("--normalize")
    try:
        run(cmd, cwd=vespag_cwd, env=env)
    except subprocess.CalledProcessError:
        if not args.retry_cpu_on_fail:
            raise
        print("[vespag] scoring failed on current device; retrying on CPU")
        run(cmd, cwd=vespag_cwd, env=with_cpu_env(env))

    produced_candidates = [
        out / "vespag_scores_all.csv",
        out / "vespag_scores.csv",
    ]
    produced_csv = next((p for p in produced_candidates if p.exists()), None)
    if produced_csv is None:
        any_csv = sorted(out.glob("*.csv"))
        produced_csv = any_csv[0] if any_csv else None
    if produced_csv is None:
        raise SystemExit(f"VespaG completed but no score CSV found in {out}")

    canonical_scores.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced_csv, canonical_scores)
    if not args.out:
        try:
            produced_csv.unlink()
        except Exception:
            pass
    print(f"[vespag] canonical scores: {canonical_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
