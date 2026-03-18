import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from artifact_paths import (
        minimal_af2_panel_path,
        minimal_af2_panel_vespag_path,
        minimal_af2_vespag_mutation_csv_path,
        minimal_af2_vespag_scores_csv_path,
        minimal_af2_vespag_summary_path,
        swarm_root,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        minimal_af2_panel_path,
        minimal_af2_panel_vespag_path,
        minimal_af2_vespag_mutation_csv_path,
        minimal_af2_vespag_scores_csv_path,
        minimal_af2_vespag_summary_path,
        swarm_root,
    )

try:
    from mutation_utils import mutations_to_id
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def normalize_minmax(vals: np.ndarray) -> np.ndarray:
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        return np.full_like(vals, 0.5)
    return (vals - lo) / (hi - lo)


def beta_params_from_weighted_moments(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    ww = np.maximum(w, eps)
    ww = ww / np.sum(ww)
    mean = float(np.sum(ww * x))
    var = float(np.sum(ww * ((x - mean) ** 2)))

    mean = min(1.0 - eps, max(eps, mean))
    var = min(max(var, 1e-5), mean * (1.0 - mean) - 1e-5)

    kappa = (mean * (1.0 - mean) / var) - 1.0
    kappa = min(max(kappa, 1.0), 1000.0)
    a = mean * kappa
    b = (1.0 - mean) * kappa
    return float(max(0.2, min(500.0, a))), float(max(0.2, min(500.0, b)))


def log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) + log_norm


def fit_beta_mixture_posteriors(x: np.ndarray, iters: int = 80) -> Tuple[np.ndarray, Dict[str, float]]:
    x = np.clip(x.astype(float), 1e-6, 1.0 - 1e-6)
    n = len(x)
    if n < 8:
        return np.full(n, 0.5), {
            "pi_low": 0.5,
            "pi_high": 0.5,
            "a_low": 2.0,
            "b_low": 5.0,
            "a_high": 5.0,
            "b_high": 2.0,
        }

    med = float(np.median(x))
    r_high = (x >= med).astype(float)
    r_low = 1.0 - r_high
    pi_high = max(0.1, min(0.9, float(np.mean(r_high))))
    pi_low = 1.0 - pi_high
    a_low, b_low = beta_params_from_weighted_moments(x, np.maximum(r_low, 1e-3))
    a_high, b_high = beta_params_from_weighted_moments(x, np.maximum(r_high, 1e-3))

    for _ in range(max(5, iters)):
        l_low = np.log(max(1e-8, pi_low)) + log_beta_pdf(x, a_low, b_low)
        l_high = np.log(max(1e-8, pi_high)) + log_beta_pdf(x, a_high, b_high)
        mx = np.maximum(l_low, l_high)
        den = np.exp(l_low - mx) + np.exp(l_high - mx)
        r_high = np.exp(l_high - mx) / np.maximum(1e-12, den)
        r_low = 1.0 - r_high

        pi_high = float(np.mean(r_high))
        pi_high = max(0.05, min(0.95, pi_high))
        pi_low = 1.0 - pi_high
        a_low, b_low = beta_params_from_weighted_moments(x, r_low)
        a_high, b_high = beta_params_from_weighted_moments(x, r_high)

    mean_low = a_low / (a_low + b_low)
    mean_high = a_high / (a_high + b_high)
    if mean_high < mean_low:
        pi_low, pi_high = pi_high, pi_low
        a_low, a_high = a_high, a_low
        b_low, b_high = b_high, b_low

    l_low = np.log(max(1e-8, pi_low)) + log_beta_pdf(x, a_low, b_low)
    l_high = np.log(max(1e-8, pi_high)) + log_beta_pdf(x, a_high, b_high)
    mx = np.maximum(l_low, l_high)
    den = np.exp(l_low - mx) + np.exp(l_high - mx)
    post_high = np.exp(l_high - mx) / np.maximum(1e-12, den)

    return post_high, {
        "pi_low": round(float(pi_low), 6),
        "pi_high": round(float(pi_high), 6),
        "a_low": round(float(a_low), 6),
        "b_low": round(float(b_low), 6),
        "a_high": round(float(a_high), 6),
        "b_high": round(float(b_high), 6),
    }


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


def parse_fasta_id(path: Path) -> str:
    if not path.exists():
        return "target"
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:].strip()
                if header:
                    return header.split()[0]
                break
    return "target"


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


def is_valid_h5(path: Path) -> bool:
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as h5:
            return len(list(h5.keys())) > 0
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


def with_cpu_env(env: dict) -> dict:
    cpu_env = env.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    return cpu_env


def resolve_installed_model_weights() -> Optional[Path]:
    try:
        import vespag  # type: ignore
        pkg_dir = Path(vespag.__file__).resolve().parent
        for c in [pkg_dir / "model_weights", pkg_dir.parent / "model_weights"]:
            if c.exists():
                return c
    except Exception:
        return None
    return None


def resolve_conda_bin(preferred: str) -> Optional[str]:
    candidates: List[str] = []
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
    candidates.extend([
        str(home / "miniconda" / "bin" / "conda"),
        str(home / "miniconda3" / "bin" / "conda"),
        str(home / "anaconda3" / "bin" / "conda"),
        str(home / "miniforge3" / "bin" / "conda"),
    ])
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


def resolve_conda_env_model_weights(conda_bin: Optional[str], conda_env: str) -> Optional[Path]:
    if not conda_bin or not conda_env:
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
            "from pathlib import Path; import vespag; "
            "pkg=Path(vespag.__file__).resolve().parent; "
            "cands=[pkg/'model_weights', pkg.parent/'model_weights']; "
            "hits=[str(c) for c in cands if c.exists()]; print(hits[0] if hits else '')"
        ),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None
    if not out:
        return None
    p = Path(out.splitlines()[-1].strip())
    return p if p.exists() else None


def conda_env_exists(conda_bin: Optional[str], conda_env: str) -> bool:
    if not conda_bin or not conda_env:
        return False
    cmd = [conda_bin, "env", "list", "--json"]
    try:
        payload = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = json.loads(payload)
    except Exception:
        return False
    envs = data.get("envs") or []
    target = conda_env.strip().lower()
    for env_path in envs:
        name = Path(str(env_path)).name.strip().lower()
        if name == target:
            return True
    return False


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


def has_local_repo_vespag(project_root: Path) -> bool:
    pkg_root = project_root / "VespaG"
    return (pkg_root / "vespag" / "__init__.py").exists()


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def ensure_local_repo_runner_ready(local_repo_vespag: Path) -> None:
    if sys.version_info >= (3, 13):
        raise SystemExit(
            f"Local VespaG bootstrap into Python {sys.version_info.major}.{sys.version_info.minor} is not supported here. "
            "Use a dedicated conda env with Python 3.12 and pass --conda-env."
        )
    required = ["vespag", "typer"]
    missing = [name for name in required if not module_available(name)]
    if not missing:
        return
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(local_repo_vespag)]
    print(
        "[vespag-panel] bootstrapping local VespaG package into the active environment "
        f"(missing: {', '.join(missing)})"
    )
    subprocess.check_call(cmd)


def pick_column(fieldnames: List[str], preferred: List[str], contains: List[str]) -> Optional[str]:
    lowers = {f.lower(): f for f in fieldnames}
    for p in preferred:
        if p in lowers:
            return lowers[p]
    for f in fieldnames:
        lf = f.lower()
        if any(c in lf for c in contains):
            return f
    return None


def detect_score_column(rows: List[Dict[str, str]], fieldnames: List[str]) -> Optional[str]:
    blocked = ("pos", "position", "index", "protein", "mutation", "variant", "id")
    candidates: List[str] = []
    for f in fieldnames:
        lf = f.lower()
        if any(b in lf for b in blocked):
            continue
        if "score" in lf or "pred" in lf or "effect" in lf or "vespag" in lf:
            candidates.append(f)
    if not candidates:
        return None
    for f in candidates:
        for r in rows[:20]:
            v = r.get(f)
            if v is None:
                continue
            try:
                float(v)
                return f
            except Exception:
                continue
    return candidates[0]


def parse_panel_rows(panel_path: Path) -> List[Dict[str, str]]:
    with panel_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader)


def component_mutation_ids(raw: str) -> List[str]:
    muts = json.loads(raw)
    mids: List[str] = []
    for m in muts:
        mids.append(mutations_to_id([m], include_chain=False).upper())
    return mids


def geometric_mean(values: Sequence[float]) -> float:
    vals = [float(max(1e-8, min(1.0, v))) for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in vals) / float(len(vals))))


def build_mutation_csv(panel_rows: Sequence[Dict[str, str]], protein_id: str, out_csv: Path) -> int:
    muts: Set[str] = set()
    for row in panel_rows:
        muts.update(component_mutation_ids(str(row["mutations_json"])))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["protein_id", "mutation_id"])
        for mid in sorted(muts):
            writer.writerow([protein_id, mid])
    return len(muts)


def score_panel_rows(
    panel_rows: Sequence[Dict[str, str]],
    score_map: Dict[str, float],
    post_map: Dict[str, float],
    min_score: float,
    min_posterior: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    scored_rows: List[Dict[str, str]] = []
    stats = {"direct": 0, "composed_from_singles": 0, "missing": 0, "pass": 0, "fail": 0}
    legacy_drop = {
        "vespag_gate_pass",
        "janus_stability_ok",
        "janus_stability_score",
        "p_bind",
        "p_bind_binary",
        "p_bind_ternary",
        "p_bind_coupling_abs",
    }
    for row in panel_rows:
        comp_ids = component_mutation_ids(str(row["mutations_json"]))
        comp_scores = [score_map.get(mid) for mid in comp_ids]
        comp_posts = [post_map.get(mid) for mid in comp_ids]

        out = {k: v for k, v in row.items() if k not in legacy_drop}
        if len(comp_ids) == 1 and comp_scores[0] is not None and comp_posts[0] is not None:
            score = float(comp_scores[0])
            post = float(comp_posts[0])
            source = "direct"
        elif comp_ids and all(v is not None for v in comp_scores) and all(v is not None for v in comp_posts):
            score = float(np.mean(np.asarray(comp_scores, dtype=float)))
            post = float(geometric_mean([float(v) for v in comp_posts if v is not None]))
            source = "composed_from_singles"
        else:
            score = float("nan")
            post = float("nan")
            source = "missing"

        passed = bool(
            math.isfinite(score)
            and math.isfinite(post)
            and score >= float(min_score)
            and post >= float(min_posterior)
        )

        out["vespag_score_norm"] = "" if not math.isfinite(score) else round(score, 6)
        out["vespag_function_prob"] = "" if not math.isfinite(post) else round(post, 6)
        out["vespag_score_source"] = source
        out["vespag_pass"] = passed
        scored_rows.append(out)

        stats[source] = stats.get(source, 0) + 1
        stats["pass" if passed else "fail"] += 1

    scored_rows.sort(
        key=lambda r: (
            1 if str(r.get("vespag_pass")).lower() == "true" else 0,
            safe_float(r.get("vespag_function_prob"), -1.0),
            safe_float(r.get("triage_score"), -1.0),
            safe_float(r.get("protein_agnostic_score"), -1.0),
        ),
        reverse=True,
    )
    return scored_rows, stats


def write_tsv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score minimal AF2 panel with VespaG before AF2.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--panel-out", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--mutation-file", default=None)
    ap.add_argument("--vespag-scores", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--vespag-bin", default="vespag")
    ap.add_argument("--conda-bin", default=os.environ.get("CONDA_EXE", "conda"))
    ap.add_argument("--conda-env", default=os.environ.get("VESPAG_CONDA_ENV", os.environ.get("CONDA_DEFAULT_ENV", "")))
    ap.add_argument("--vespag-cwd", default=None)
    ap.add_argument("--model-weights-dir", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--cpu-embeddings", action="store_true", default=True)
    ap.add_argument("--gpu-embeddings", dest="cpu_embeddings", action="store_false")
    ap.add_argument("--retry-cpu-on-fail", action="store_true", default=True)
    ap.add_argument("--no-retry-cpu-on-fail", dest="retry_cpu_on_fail", action="store_false")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--single-csv", action="store_true", default=True)
    ap.add_argument("--no-single-csv", dest="single_csv", action="store_false")
    ap.add_argument("--min-score", type=float, default=0.40)
    ap.add_argument("--min-posterior", type=float, default=0.35)
    ap.add_argument("--mixture-iters", type=int, default=80)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    panel_path = Path(args.panel).resolve() if args.panel else minimal_af2_panel_path(outdir)
    panel_out = Path(args.panel_out).resolve() if args.panel_out else minimal_af2_panel_vespag_path(outdir)
    fasta = Path(args.fasta).resolve() if args.fasta else outdir / "enzyme_wt.fasta"
    mutation_file = Path(args.mutation_file).resolve() if args.mutation_file else minimal_af2_vespag_mutation_csv_path(outdir)
    canonical_scores = Path(args.vespag_scores).resolve() if args.vespag_scores else minimal_af2_vespag_scores_csv_path(outdir)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else outdir / "swarm" / "vespag_cache"
    summary_path = minimal_af2_vespag_summary_path(outdir)

    if not panel_path.exists():
        raise SystemExit(f"Minimal AF2 panel not found: {panel_path}")
    if not fasta.exists():
        raise SystemExit(f"WT FASTA not found: {fasta}")

    panel_rows = parse_panel_rows(panel_path)
    if not panel_rows:
        raise SystemExit(f"Panel is empty: {panel_path}")

    protein_id = parse_fasta_id(fasta)
    mutation_count = build_mutation_csv(panel_rows, protein_id=protein_id, out_csv=mutation_file)

    project_root = Path(__file__).resolve().parents[2]
    local_repo_weights = project_root / "VespaG" / "model_weights"
    local_repo_vespag = project_root / "VespaG"
    conda_bin = resolve_conda_bin(args.conda_bin)
    requested_conda_env = str(args.conda_env or "").strip()
    conda_env_ok = conda_env_exists(conda_bin, requested_conda_env)
    installed_weights = resolve_installed_model_weights()
    conda_env_weights = (
        resolve_conda_env_model_weights(conda_bin, requested_conda_env)
        if conda_bin and conda_env_ok
        else None
    )
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
    use_local_repo_runner = False
    if not has_cli_in_path:
        use_module_runner = module_available("vespag")
        if (not use_module_runner) and has_local_repo_vespag(project_root):
            use_local_repo_runner = True
        if (not use_module_runner) and (not use_local_repo_runner) and requested_conda_env and conda_bin and conda_env_ok:
            use_conda_runner = True
        if not use_module_runner and not use_local_repo_runner and not use_conda_runner:
            missing_env_msg = ""
            if requested_conda_env and not conda_env_ok:
                missing_env_msg = (
                    f" Requested conda env '{requested_conda_env}' was not found; "
                    "clear VESPAG_CONDA_ENV/--conda-env or point it at a real env."
                )
            raise SystemExit(
                f"vespag CLI not found in current env and no usable conda env runner. "
                f"Checked current PATH for '{args.vespag_bin}' and conda env '{requested_conda_env or '<unset>'}'."
                f"{missing_env_msg}"
            )

    def vespag_cmd(*parts: str) -> List[str]:
        if use_module_runner:
            return [sys.executable, "-m", "vespag", *parts]
        if use_local_repo_runner:
            return [sys.executable, "-m", "vespag", *parts]
        if use_conda_runner:
            return [str(conda_bin), "run", "--no-capture-output", "-n", requested_conda_env, args.vespag_bin, *parts]
        return [args.vespag_bin, *parts]

    if use_local_repo_runner:
        ensure_local_repo_runner_ready(local_repo_vespag)
        if module_available("vespag"):
            use_module_runner = True
            use_local_repo_runner = False

    if not (vespag_cwd / "model_weights").exists():
        raise SystemExit(
            "VespaG model_weights not found for runtime. "
            "Provide --model-weights-dir or place weights at VespaG/model_weights."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    swarm_root(outdir).mkdir(parents=True, exist_ok=True)
    fasta_ids = parse_fasta_ids(fasta)
    env = os.environ.copy()
    default_hf = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).resolve()
    hf_home = Path(args.hf_home).resolve() if args.hf_home else default_hf
    hf_home.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    if use_local_repo_runner:
        existing_py = env.get("PYTHONPATH", "")
        py_entries = [str(local_repo_vespag)]
        if existing_py:
            py_entries.append(existing_py)
        env["PYTHONPATH"] = os.pathsep.join(py_entries)

    def run(cmd: Sequence[str], env_run: Optional[dict] = None) -> None:
        print(">", " ".join(str(c) for c in cmd))
        print("  cwd:", str(vespag_cwd))
        subprocess.check_call(cmd, cwd=str(vespag_cwd), env=env_run)

    if not canonical_scores.exists():
        embeddings = find_embeddings(cache_dir, required_ids=fasta_ids)
        if embeddings is not None and embeddings.exists() and ((not is_valid_h5(embeddings)) or (not embeddings_has_all_ids(embeddings, fasta_ids))):
            try:
                embeddings.unlink()
            except Exception:
                pass
            embeddings = None

        if embeddings is None:
            embed_cmd = vespag_cmd("predict", "-i", str(fasta), "-o", str(cache_dir), "--no-csv", "--h5-output")
            run(embed_cmd, env_run=with_cpu_env(env) if args.cpu_embeddings else env)
            embeddings = find_embeddings(cache_dir, required_ids=fasta_ids)

        if embeddings is None or not embeddings.exists() or not is_valid_h5(embeddings) or not embeddings_has_all_ids(embeddings, fasta_ids):
            raise SystemExit("Failed to locate VespaG embeddings (.h5) in cache dir.")

        score_cmd = vespag_cmd(
            "predict",
            "-i", str(fasta),
            "-e", str(embeddings),
            "--mutation-file", str(mutation_file),
            "-o", str(swarm_root(outdir)),
        )
        if args.single_csv:
            score_cmd.append("--single-csv")
        if args.normalize:
            score_cmd.append("--normalize")
        try:
            run(score_cmd, env_run=env)
        except subprocess.CalledProcessError:
            if not args.retry_cpu_on_fail:
                raise
            print("[vespag-panel] scoring failed on current device; retrying on CPU")
            run(score_cmd, env_run=with_cpu_env(env))

        produced_candidates = [
            swarm_root(outdir) / "vespag_scores_all.csv",
            swarm_root(outdir) / "vespag_scores.csv",
        ]
        produced_csv = next((p for p in produced_candidates if p.exists()), None)
        if produced_csv is None:
            any_csv = sorted(swarm_root(outdir).glob("*.csv"))
            produced_csv = any_csv[0] if any_csv else None
        if produced_csv is None:
            raise SystemExit(f"VespaG completed but no score CSV found in {swarm_root(outdir)}")
        shutil.copy2(produced_csv, canonical_scores)
    else:
        print(f"[vespag-panel] reusing existing score CSV: {canonical_scores}")

    with canonical_scores.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    mut_col = pick_column(fieldnames, ["mutation_id"], ["mutation", "variant", "mut"])
    if not mut_col:
        raise SystemExit("Could not find mutation id column in VespaG CSV.")
    score_col = detect_score_column(rows, fieldnames)
    if not score_col:
        raise SystemExit("Could not infer score column in VespaG CSV.")
    prot_col = pick_column(fieldnames, ["protein_id"], ["protein", "id"])

    score_map: Dict[str, float] = {}
    for r in rows:
        if prot_col and protein_id:
            val = (r.get(prot_col) or "").strip()
            if val and val != protein_id:
                continue
        mid = (r.get(mut_col) or "").strip().upper()
        if not mid:
            continue
        sc = safe_float(r.get(score_col), float("nan"))
        if not math.isfinite(sc):
            continue
        score_map[mid] = sc
    if not score_map:
        raise SystemExit("No VespaG scores were mapped to mutation ids.")

    score_vals = np.array(list(score_map.values()), dtype=float)
    if float(np.min(score_vals)) < 0.0 or float(np.max(score_vals)) > 1.0:
        score_vals = normalize_minmax(score_vals)
    else:
        score_vals = np.clip(score_vals, 0.0, 1.0)
    for i, k in enumerate(list(score_map.keys())):
        score_map[k] = float(score_vals[i])

    p_post, mix_params = fit_beta_mixture_posteriors(score_vals, iters=args.mixture_iters)
    post_map = {k: float(p_post[i]) for i, k in enumerate(list(score_map.keys()))}

    scored_rows, stats = score_panel_rows(
        panel_rows=panel_rows,
        score_map=score_map,
        post_map=post_map,
        min_score=float(args.min_score),
        min_posterior=float(args.min_posterior),
    )
    write_tsv(panel_out, scored_rows)

    summary = {
        "panel_path": str(panel_path),
        "panel_out": str(panel_out),
        "mutation_file": str(mutation_file),
        "vespag_scores_csv": str(canonical_scores),
        "protein_id": protein_id,
        "unique_single_mutations": int(mutation_count),
        "panel_rows": int(len(panel_rows)),
        "min_score": float(args.min_score),
        "min_posterior": float(args.min_posterior),
        "score_column": score_col,
        "mapped_single_scores": int(len(score_map)),
        "mixture": mix_params,
    }
    summary.update(stats)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote: {panel_out}")
    print(f"Wrote: {summary_path}")
    print(f"[vespag-panel] unique_single_mutations={mutation_count} panel_rows={len(panel_rows)} pass={stats['pass']} fail={stats['fail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
