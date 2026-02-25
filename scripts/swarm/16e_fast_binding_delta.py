#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from artifact_paths import (
        binding_fastdl_cache_path,
        binding_fastdl_mutants_dir,
        binding_fastdl_summary_path,
        proposals_vespag_path,
    )
except ImportError:
    from scripts.swarm.artifact_paths import (
        binding_fastdl_cache_path,
        binding_fastdl_mutants_dir,
        binding_fastdl_summary_path,
        proposals_vespag_path,
    )

try:
    from mutation_utils import mutations_to_id, row_mutations
except ImportError:
    from scripts.swarm.mutation_utils import mutations_to_id, row_mutations


AA1_TO_AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def logistic(x: float) -> float:
    xx = float(x)
    if xx >= 0.0:
        z = math.exp(-xx)
        return 1.0 / (1.0 + z)
    z = math.exp(xx)
    return z / (1.0 + z)


def rank_quantile(values: List[float], v: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return 1.0
    lo, hi = 0, len(vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if vals[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    rank = max(0, lo - 1)
    return rank / (len(vals) - 1)


def mutation_id(wt: str, pos: int, mut: str) -> str:
    return f"{wt}{pos}{mut}"


def canonical_row_mutations(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    muts = row_mutations(row)
    out: List[Dict[str, Any]] = []
    for m in muts:
        chain = str(m.get("chain") or "A")
        pos = int(m.get("pos"))
        wt = str(m.get("wt") or "").upper()
        mut = str(m.get("mut") or "").upper()
        out.append({"chain": chain, "pos": pos, "wt": wt, "mut": mut})
    return out


def row_variant_id(row: Dict[str, Any]) -> str:
    vid = str(row.get("variant_id") or "").strip()
    if vid:
        return vid
    return mutations_to_id(canonical_row_mutations(row), include_chain=False)


def fallback_bind_probability(row: Dict[str, Any]) -> float:
    existing = safe_float(row.get("p_bind_fastdl", row.get("p_bind")), float("nan"))
    if math.isfinite(existing):
        return clamp(existing, 0.0, 1.0)

    stat = row.get("stat_model") if isinstance(row.get("stat_model"), dict) else {}
    obj_mean = stat.get("objective_mean") if isinstance(stat.get("objective_mean"), dict) else {}
    model_bind = safe_float(obj_mean.get("binding"), float("nan"))
    if not math.isfinite(model_bind):
        model_bind = safe_float(stat.get("bind_relevance"), float("nan"))

    lig_contact = 1.0 if bool(row.get("ligand_contact")) else 0.0
    dist_lig = safe_float(row.get("dist_ligand"), float("nan"))
    prolif_freq = safe_float(row.get("prolif_contact_freq"), 0.0)
    if math.isfinite(dist_lig):
        near = math.exp(-max(0.0, dist_lig) / 5.5)
    else:
        near = 0.35
    proxy_bind = clamp(0.10 + (0.55 * near) + (0.22 * prolif_freq) + (0.13 * lig_contact), 0.0, 1.0)

    if math.isfinite(model_bind):
        return clamp((0.65 * model_bind) + (0.35 * proxy_bind), 0.0, 1.0)
    return proxy_bind


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open() as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_gnina_bin() -> str:
    local = Path.cwd() / "gnina.1.3.2"
    if local.exists():
        return str(local)
    return shutil.which("gnina") or "gnina"


def default_ld_library_path() -> Optional[str]:
    p = Path.home() / "miniconda" / "lib"
    if p.exists():
        return str(p)
    return None


def pick_ligand_pose(outdir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Ligand pose path not found: {p}")
        return p
    candidates = [
        outdir / "docked_top.sdf",
        outdir / "docked_gnina_rescored.sdf",
        outdir / "ligand.sdf",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "Could not resolve ligand pose. Expected one of: "
        f"{', '.join(str(x) for x in candidates)}"
    )


def pick_existing_path(candidates: List[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_resname_csv(raw: str) -> Set[str]:
    out: Set[str] = set()
    for tok in str(raw or "").split(","):
        t = str(tok or "").strip().upper()
        if t:
            out.add(t)
    return out


def append_hetero_from_template(
    protein_pdb: Path,
    template_pdb: Path,
    out_pdb: Path,
    keep_resnames: Set[str],
) -> int:
    prot_lines = protein_pdb.read_text().splitlines()
    templ_lines = template_pdb.read_text().splitlines()

    filtered_prot: List[str] = []
    for ln in prot_lines:
        if ln.startswith("END"):
            continue
        if ln.startswith("CONECT"):
            continue
        filtered_prot.append(ln)

    hetero_lines: List[str] = []
    for ln in templ_lines:
        if not ln.startswith("HETATM"):
            continue
        resname = str(ln[17:20]).strip().upper()
        if keep_resnames and resname not in keep_resnames:
            continue
        hetero_lines.append(ln.rstrip("\n"))

    out_lines = list(filtered_prot)
    out_lines.extend(hetero_lines)
    out_lines.append("END")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text("\n".join(out_lines) + "\n")
    return int(len(hetero_lines))


def gnina_env(ld_library_path: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if ld_library_path:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{ld_library_path}:{old}" if old else ld_library_path
    return env


def parse_gnina_stdout(stdout: str) -> Dict[str, float]:
    out = {}
    patterns = {
        "cnn_affinity": r"CNNaffinity:\s*([-+]?\d*\.?\d+)",
        "cnn_score": r"CNNscore:\s*([-+]?\d*\.?\d+)",
        "cnn_variance": r"CNNvariance:\s*([-+]?\d*\.?\d+)",
        "vina_affinity": r"Affinity:\s*([-+]?\d*\.?\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            out[key] = float(m.group(1))
    if "cnn_affinity" not in out:
        raise RuntimeError("GNINA output did not contain CNNaffinity.")
    return out


def run_gnina_score(
    receptor_pdb: Path,
    ligand_pose: Path,
    gnina_bin: str,
    gnina_cpu: int,
    cnn_model: str,
    autobox_add: float,
    ld_library_path: Optional[str],
) -> Dict[str, float]:
    cmd = [
        gnina_bin,
        "-r",
        str(receptor_pdb),
        "-l",
        str(ligand_pose),
        "--score_only",
        "--cnn_scoring",
        "rescore",
        "--autobox_ligand",
        str(ligand_pose),
        "--autobox_add",
        str(float(autobox_add)),
        "--no_gpu",
        "--cpu",
        str(max(1, int(gnina_cpu))),
    ]
    if cnn_model:
        cmd.extend(["--cnn", str(cnn_model)])

    env = gnina_env(ld_library_path=ld_library_path)
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr[-400:] if stderr else stdout[-400:]
        raise RuntimeError(f"GNINA score_only failed ({proc.returncode}): {msg}")
    return parse_gnina_stdout(proc.stdout or "")


def build_mutant_pdb(
    wt_pdb: Path,
    mutant_pdb: Path,
    mutations: List[Dict[str, Any]],
) -> None:
    from pdbfixer import PDBFixer  # type: ignore
    try:
        from openmm.app import PDBFile  # type: ignore
    except ModuleNotFoundError:
        from simtk.openmm.app import PDBFile  # type: ignore

    fixer = PDBFixer(filename=str(wt_pdb))
    specs_by_chain: Dict[str, List[str]] = {}
    for mm in mutations:
        chain = str(mm.get("chain") or "A")
        pos = int(mm.get("pos"))
        wt = str(mm.get("wt") or "").upper()
        mut = str(mm.get("mut") or "").upper()
        mut3 = AA1_TO_AA3.get(mut)
        if mut3 is None:
            raise ValueError(f"Unknown mutant amino acid: {mut}")

        old3 = AA1_TO_AA3.get(wt)
        for chain_obj in fixer.topology.chains():
            if str(chain_obj.id) != str(chain):
                continue
            for residue in chain_obj.residues():
                try:
                    rid = int(str(residue.id))
                except Exception:
                    continue
                if rid == int(pos):
                    old3 = str(residue.name).upper()
                    break
            break
        if not old3:
            raise ValueError(f"Unknown wild-type amino acid for mutation {wt}{pos}{mut}")
        specs_by_chain.setdefault(str(chain), []).append(f"{old3}-{int(pos)}-{mut3}")

    for chain_id, specs in specs_by_chain.items():
        if specs:
            fixer.applyMutations(specs, str(chain_id))

    # Do not back-fill missing loops; only reconstruct local atoms after mutation.
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    mutant_pdb.parent.mkdir(parents=True, exist_ok=True)
    with mutant_pdb.open("w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def relax_mutant_pdb(
    mutant_pdb: Path,
    mutations: List[Dict[str, Any]],
    max_iterations: int,
    heavy_atom_restraint_k: float,
) -> None:
    try:
        from openmm import CustomExternalForce, VerletIntegrator, unit  # type: ignore
        from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation  # type: ignore
    except ModuleNotFoundError:
        from simtk import unit  # type: ignore
        from simtk.openmm import CustomExternalForce, VerletIntegrator  # type: ignore
        from simtk.openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation  # type: ignore

    max_iterations = max(0, int(max_iterations))
    if max_iterations <= 0:
        return

    pdb = PDBFile(str(mutant_pdb))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield=forcefield, pH=7.0)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    k = float(heavy_atom_restraint_k)
    if k > 0.0:
        # Keep the global fold fixed while allowing local mutation-pocket relaxation.
        restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter(
            "k",
            float(
                (k * unit.kilocalorie_per_mole / (unit.angstrom**2)).value_in_unit(
                    unit.kilojoule_per_mole / (unit.nanometer**2)
                )
            ),
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        mutated_sites = {
            (str(mm.get("chain") or "A").strip(), str(int(mm.get("pos"))).strip())
            for mm in mutations
        }
        for idx, atom in enumerate(modeller.topology.atoms()):
            residue = atom.residue
            is_target_residue = (
                str(getattr(residue.chain, "id", "")).strip(),
                str(getattr(residue, "id", "")).strip(),
            ) in mutated_sites
            if is_target_residue:
                continue
            if atom.element is None or str(atom.element.symbol).upper() == "H":
                continue
            xyz = modeller.positions[idx].value_in_unit(unit.nanometer)
            restraint.addParticle(idx, [float(xyz.x), float(xyz.y), float(xyz.z)])

        if restraint.getNumParticles() > 0:
            system.addForce(restraint)

    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    state = simulation.context.getState(getPositions=True)
    with mutant_pdb.open("w") as fh:
        PDBFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)


def score_variant_task(task: Dict[str, Any]) -> Dict[str, Any]:
    variant_id = str(task["variant_id"])
    mutations = canonical_row_mutations({"mutations": task.get("mutations") or []})
    if not mutations:
        raise ValueError(f"No mutations provided for variant: {variant_id}")
    anchor = mutations[0]
    chain = str(anchor["chain"])
    pos = int(anchor["pos"])
    wt = str(anchor["wt"]).upper()
    mut = str(anchor["mut"]).upper()
    mutant_pdb = Path(task["mutant_pdb"])
    wt_pdb = Path(task["wt_pdb"])
    ligand_pose = Path(task["ligand_pose"])

    out: Dict[str, Any] = {
        "variant_id": variant_id,
        "chain": chain,
        "pos": pos,
        "wt": wt,
        "mut": mut,
        "mutations": mutations,
    }
    try:
        if not mutant_pdb.exists():
            build_mutant_pdb(
                wt_pdb=wt_pdb,
                mutant_pdb=mutant_pdb,
                mutations=mutations,
            )
        if bool(task.get("relax_mutants", True)):
            try:
                relax_mutant_pdb(
                    mutant_pdb=mutant_pdb,
                    mutations=mutations,
                    max_iterations=int(task.get("relax_max_iterations", 200)),
                    heavy_atom_restraint_k=float(task.get("relax_heavy_restraint_k", 25.0)),
                )
                out["mutant_relaxed"] = True
            except Exception as exc:
                out["mutant_relaxed"] = False
                out["mutant_relax_error"] = str(exc)
        else:
            out["mutant_relaxed"] = False
        binary_metrics = run_gnina_score(
            receptor_pdb=mutant_pdb,
            ligand_pose=ligand_pose,
            gnina_bin=str(task["gnina_bin"]),
            gnina_cpu=int(task["gnina_cpu"]),
            cnn_model=str(task["cnn_model"]),
            autobox_add=float(task["autobox_add"]),
            ld_library_path=task.get("ld_library_path"),
        )
        out["binary_ok"] = True
        out["binary_cnn_affinity"] = float(binary_metrics.get("cnn_affinity"))
        out["binary_cnn_score"] = float(safe_float(binary_metrics.get("cnn_score"), 0.0))
        out["binary_cnn_variance"] = float(safe_float(binary_metrics.get("cnn_variance"), 0.0))

        run_ternary = bool(task.get("run_ternary"))
        if run_ternary:
            ternary_template = Path(str(task.get("wt_pdb_ternary") or ""))
            ternary_mutant_pdb = Path(str(task.get("mutant_pdb_ternary") or ""))
            ternary_keep_resnames = parse_resname_csv(str(task.get("ternary_keep_resnames") or ""))
            if ternary_template.exists():
                appended = append_hetero_from_template(
                    protein_pdb=mutant_pdb,
                    template_pdb=ternary_template,
                    out_pdb=ternary_mutant_pdb,
                    keep_resnames=ternary_keep_resnames,
                )
                out["ternary_hetero_appended"] = int(appended)
                if appended > 0:
                    ternary_receptor = ternary_mutant_pdb
                else:
                    ternary_receptor = mutant_pdb
            else:
                out["ternary_hetero_appended"] = 0
                ternary_receptor = mutant_pdb

            try:
                ternary_metrics = run_gnina_score(
                    receptor_pdb=ternary_receptor,
                    ligand_pose=ligand_pose,
                    gnina_bin=str(task["gnina_bin"]),
                    gnina_cpu=int(task["gnina_cpu"]),
                    cnn_model=str(task["cnn_model"]),
                    autobox_add=float(task["autobox_add"]),
                    ld_library_path=task.get("ld_library_path"),
                )
                out["ternary_ok"] = True
                out["ternary_cnn_affinity"] = float(ternary_metrics.get("cnn_affinity"))
                out["ternary_cnn_score"] = float(safe_float(ternary_metrics.get("cnn_score"), 0.0))
                out["ternary_cnn_variance"] = float(safe_float(ternary_metrics.get("cnn_variance"), 0.0))
            except Exception as exc:
                out["ternary_ok"] = False
                out["ternary_error"] = str(exc)
        else:
            out["ternary_ok"] = False
            out["ternary_skipped"] = True

        # Backward-compatible keys for any stale readers.
        out["cnn_affinity"] = float(out.get("binary_cnn_affinity", 0.0))
        out["cnn_score"] = float(out.get("binary_cnn_score", 0.0))
        out["cnn_variance"] = float(out.get("binary_cnn_variance", 0.0))
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
    return out


def run_gnina_version(gnina_bin: str, ld_library_path: Optional[str]) -> str:
    proc = subprocess.run(
        [gnina_bin, "--version"],
        text=True,
        capture_output=True,
        env=gnina_env(ld_library_path=ld_library_path),
    )
    if proc.returncode != 0:
        msg = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
        raise SystemExit(
            f"Unable to execute GNINA binary '{gnina_bin}'. "
            f"Set --gnina-bin and/or --ld-library-path. Details: {msg[-400:]}"
        )
    out = (proc.stdout or "").strip()
    if not out:
        out = (proc.stderr or "").strip()
    return out.splitlines()[0] if out else "gnina"


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    payload = obj.get("scores")
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in payload.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = dict(v)
    return out


def cache_signature(
    *,
    gnina_bin: str,
    cnn_model: str,
    autobox_add: float,
    gnina_cpu: int,
    relax_mutants: bool,
    relax_max_iterations: int,
    relax_heavy_restraint_k: float,
    binding_context: str,
    wt_pdb_binary: str,
    wt_pdb_ternary: str,
    ternary_keep_resnames: str,
) -> str:
    obj = {
        "gnina_bin": str(gnina_bin),
        "cnn_model": str(cnn_model),
        "autobox_add": float(autobox_add),
        "gnina_cpu": int(gnina_cpu),
        "relax_mutants": bool(relax_mutants),
        "relax_max_iterations": int(relax_max_iterations),
        "relax_heavy_restraint_k": float(relax_heavy_restraint_k),
        "binding_context": str(binding_context),
        "wt_pdb_binary": str(wt_pdb_binary),
        "wt_pdb_ternary": str(wt_pdb_ternary),
        "ternary_keep_resnames": str(ternary_keep_resnames),
        "stage_version": "fast_binding_delta_v3_context_aware",
    }
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def load_cache_with_signature(path: Path, expected_signature: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    if str(metadata.get("cache_signature") or "") != str(expected_signature):
        return {}
    payload = obj.get("scores")
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in payload.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = dict(v)
    return out


def save_cache(path: Path, scores: Dict[str, Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "metadata": metadata,
        "scores": scores,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fast binding alteration stage: local mutation + GNINA CNN score-only delta (WT->mutant)."
    )
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--proposals", default=None, help="Default: OUTDIR/swarm/proposals_vespag_rK.jsonl")
    ap.add_argument("--out", default=None, help="Default: overwrite proposals file in-place")
    ap.add_argument("--summary", default=None, help="Default: OUTDIR/swarm/binding_fastdl_summary_rK.json")

    ap.add_argument("--wt-pdb", default=None, help="Legacy alias for --wt-pdb-binary.")
    ap.add_argument("--wt-pdb-binary", default=None, help="Binary context receptor (no cofactor); default OUTDIR/reference_protein.pdb")
    ap.add_argument("--wt-pdb-ternary", default=None, help="Ternary context receptor (cofactor-present). Optional.")
    ap.add_argument(
        "--binding-context",
        choices=["auto", "single", "dual"],
        default="auto",
        help="Binding scoring contexts: single=binary only, dual=binary+ternary, auto=dual when ternary receptor exists.",
    )
    ap.add_argument(
        "--ternary-keep-resnames",
        default="",
        help=(
            "Comma-separated HETATM residue names copied from ternary template into mutant ternary receptor. "
            "Leave empty to disable explicit hetero copying."
        ),
    )
    ap.add_argument("--ligand-pose", default=None, help="Default: OUTDIR/docked_top.sdf (or docked_gnina_rescored.sdf, ligand.sdf)")
    ap.add_argument("--gnina-bin", default=None, help="Default: ./gnina.1.3.2 if present, else 'gnina' in PATH")
    ap.add_argument("--ld-library-path", default=None, help="Optional LD_LIBRARY_PATH prefix for GNINA runtime libs.")
    ap.add_argument("--cnn-model", default="fast", help="GNINA built-in CNN model. 'fast' is lower-latency.")
    ap.add_argument("--gnina-cpu", type=int, default=1, help="CPU threads used per GNINA invocation.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel mutation+score workers.")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N scored variants.")
    ap.add_argument("--autobox-add", type=float, default=6.0)
    ap.add_argument("--relax-mutants", action="store_true", default=True,
                    help="Run restrained local minimization on each mutant receptor before GNINA scoring.")
    ap.add_argument("--no-relax-mutants", dest="relax_mutants", action="store_false")
    ap.add_argument("--relax-max-iterations", type=int, default=200,
                    help="OpenMM energy-minimization iterations per mutant.")
    ap.add_argument("--relax-heavy-restraint-k", type=float, default=25.0,
                    help="Harmonic restraint strength (kcal/mol/A^2) on non-mutated heavy atoms during minimization.")

    ap.add_argument(
        "--score-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score all proposals, not only green/amber gate.",
    )
    ap.add_argument("--max-variants", type=int, default=0, help="Optional cap of variants to score (0=all eligible).")
    ap.add_argument("--binding-affinity-scale", type=float, default=0.035,
                    help="Scale (delta CNN affinity) for absolute binding calibration logistic.")
    ap.add_argument("--binding-score-scale", type=float, default=0.020,
                    help="Scale (delta CNN score) for absolute binding calibration logistic.")
    ap.add_argument(
        "--context-blend-weight-binary",
        type=float,
        default=0.60,
        help="Blend weight for binary context when both binary and ternary are available.",
    )
    ap.add_argument(
        "--context-blend-weight-ternary",
        type=float,
        default=0.40,
        help="Blend weight for ternary context when both binary and ternary are available.",
    )
    ap.add_argument(
        "--context-coupling-penalty",
        type=float,
        default=0.10,
        help="Penalty on absolute binary/ternary disagreement in blended p_bind.",
    )
    ap.add_argument("--use-cache", action="store_true", default=True)
    ap.add_argument("--no-cache", dest="use_cache", action="store_false")
    ap.add_argument("--strict", action="store_true", default=False, help="Fail if any scored variant errors.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    round_id = int(args.round)
    proposals_path = Path(args.proposals).resolve() if args.proposals else proposals_vespag_path(outdir=outdir, round_id=round_id)
    out_path = Path(args.out).resolve() if args.out else proposals_path
    summary_path = Path(args.summary).resolve() if args.summary else binding_fastdl_summary_path(outdir=outdir, round_id=round_id)
    cache_path = binding_fastdl_cache_path(outdir=outdir, round_id=round_id)

    wt_pdb_binary = (
        Path(args.wt_pdb_binary).resolve()
        if args.wt_pdb_binary
        else (Path(args.wt_pdb).resolve() if args.wt_pdb else (outdir / "reference_protein.pdb").resolve())
    )
    if not wt_pdb_binary.exists():
        raise SystemExit(f"Binary WT PDB not found: {wt_pdb_binary}")

    if args.wt_pdb_ternary:
        wt_pdb_ternary = Path(args.wt_pdb_ternary).resolve()
        if not wt_pdb_ternary.exists():
            raise SystemExit(f"Ternary WT PDB not found: {wt_pdb_ternary}")
    else:
        wt_pdb_ternary = pick_existing_path(
            [
                (outdir / "reference_protein_ternary.pdb").resolve(),
                (outdir / "reference_complex.pdb").resolve(),
                (outdir / "reference_holo.pdb").resolve(),
            ]
        )
        if wt_pdb_ternary is None:
            wt_pdb_ternary = wt_pdb_binary

    context_mode = str(args.binding_context).strip().lower()
    if context_mode == "auto":
        run_dual_context = bool(wt_pdb_ternary.exists() and wt_pdb_ternary != wt_pdb_binary)
    elif context_mode == "dual":
        run_dual_context = True
    else:
        run_dual_context = False

    ternary_keep_resnames = parse_resname_csv(args.ternary_keep_resnames)

    ligand_pose = pick_ligand_pose(outdir=outdir, explicit=args.ligand_pose).resolve()

    gnina_bin = str(args.gnina_bin) if args.gnina_bin else default_gnina_bin()
    ld_library_path = args.ld_library_path if args.ld_library_path else default_ld_library_path()
    gnina_version = run_gnina_version(gnina_bin=gnina_bin, ld_library_path=ld_library_path)
    cache_sig = cache_signature(
        gnina_bin=str(gnina_bin),
        cnn_model=str(args.cnn_model),
        autobox_add=float(args.autobox_add),
        gnina_cpu=int(max(1, int(args.gnina_cpu))),
        relax_mutants=bool(args.relax_mutants),
        relax_max_iterations=int(args.relax_max_iterations),
        relax_heavy_restraint_k=float(args.relax_heavy_restraint_k),
        binding_context=("dual" if run_dual_context else "single"),
        wt_pdb_binary=str(wt_pdb_binary),
        wt_pdb_ternary=str(wt_pdb_ternary),
        ternary_keep_resnames=",".join(sorted(ternary_keep_resnames)),
    )

    rows = load_jsonl(proposals_path)
    if not rows:
        raise SystemExit(f"No proposals found at {proposals_path}")

    eligible: List[Dict[str, Any]] = []
    for row in rows:
        band = str(row.get("vespag_gate_band") or "").strip().lower()
        if not bool(args.score_all) and band not in ("green", "amber"):
            continue
        muts = canonical_row_mutations(row)
        if not muts:
            continue
        aa_ok = all(str(m["wt"]).upper() in AA1_TO_AA3 and str(m["mut"]).upper() in AA1_TO_AA3 for m in muts)
        if not aa_ok:
            continue
        anchor = muts[0]
        rec = dict(row)
        rec["mutations"] = muts
        rec["variant_id"] = row_variant_id(rec)
        rec["pos"] = int(anchor["pos"])
        rec["wt"] = str(anchor["wt"]).upper()
        rec["mut"] = str(anchor["mut"]).upper()
        rec["chain"] = str(anchor["chain"])
        eligible.append(rec)

    if not eligible:
        raise SystemExit("No eligible proposals for fast binding scoring.")

    def score_priority(rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
        stat = rec.get("stat_model") if isinstance(rec.get("stat_model"), dict) else {}
        acq = safe_float(stat.get("acquisition"), 0.0)
        ehvi = safe_float(stat.get("expected_hvi"), 0.0)
        posterior = safe_float(
            rec.get("vespag_shrunk_posterior", rec.get("vespag_posterior", rec.get("vespag_score_norm", 0.0))),
            0.0,
        )
        band = str(rec.get("vespag_gate_band") or "").strip().lower()
        rescue_bonus = 0.03 if bool(args.score_all) and band == "red" else 0.0
        return (
            float(acq + (0.25 * ehvi) + rescue_bonus),
            float(ehvi),
            float(posterior),
            float(safe_float(stat.get("feasibility_prob"), 0.0)),
        )

    def uncertainty_priority(rec: Dict[str, Any]) -> float:
        stat = rec.get("stat_model") if isinstance(rec.get("stat_model"), dict) else {}
        obj_std = stat.get("objective_std") if isinstance(stat.get("objective_std"), dict) else {}
        std_vals: List[float] = []
        for k in ("function", "binding", "stability", "plausibility"):
            std_vals.append(max(0.0, safe_float(obj_std.get(k), 0.0)))
        mean_std = float(sum(std_vals) / float(max(1, len(std_vals))))
        ehvi_std = max(0.0, safe_float(stat.get("expected_hvi_std"), 0.0))
        return float(mean_std + (0.20 * ehvi_std))

    eligible_pre_cap_total = int(len(eligible))
    eligible_pre_cap_band_counts = dict(
        Counter(str(rec.get("vespag_gate_band") or "unknown").strip().lower() or "unknown" for rec in eligible)
    )
    eligible.sort(key=score_priority, reverse=True)
    forced_functional_kept = 0
    if int(args.max_variants) > 0:
        max_variants = int(args.max_variants)
        if bool(args.score_all) and len(eligible) > max_variants:
            keep: List[Dict[str, Any]] = []
            must_keep = [
                rec
                for rec in eligible
                if bool(rec.get("functional_site")) or bool(rec.get("ligand_contact"))
            ]
            must_keep.sort(key=score_priority, reverse=True)
            if must_keep:
                keep.extend(must_keep[:max_variants])
            forced_functional_kept = len(keep)
            chosen = {str(x.get("variant_id") or "") for x in keep}
            remaining_all = [rec for rec in eligible if str(rec.get("variant_id") or "") not in chosen]

            buckets: Dict[str, List[Dict[str, Any]]] = {"green": [], "amber": [], "red": []}
            for rec in remaining_all:
                band = str(rec.get("vespag_gate_band") or "").strip().lower()
                if band not in buckets:
                    band = "red"
                buckets[band].append(rec)
            for vals in buckets.values():
                vals.sort(key=score_priority, reverse=True)

            n_left = max(0, max_variants - len(keep))
            red_quota = min(len(buckets["red"]), max(1 if n_left > 0 else 0, int(round(0.20 * n_left))))
            amber_quota = min(len(buckets["amber"]), max(1 if n_left > 0 else 0, int(round(0.20 * n_left))))
            green_quota = min(len(buckets["green"]), max(1 if n_left > 0 else 0, int(round(0.20 * n_left))))
            if n_left > 0:
                keep.extend(buckets["red"][:red_quota])
                keep.extend(buckets["amber"][:amber_quota])
                keep.extend(buckets["green"][:green_quota])
            chosen = {str(x.get("variant_id") or "") for x in keep}
            remaining = [rec for rec in remaining_all if str(rec.get("variant_id") or "") not in chosen]

            n_left = max(0, max_variants - len(keep))
            if n_left > 0:
                remaining_priority = sorted(remaining, key=score_priority, reverse=True)
                n_priority = max(0, int(round(0.70 * n_left)))
                keep.extend(remaining_priority[:n_priority])
                chosen = {str(x.get("variant_id") or "") for x in keep}

                remaining_uncertainty = sorted(
                    [rec for rec in remaining if str(rec.get("variant_id") or "") not in chosen],
                    key=lambda rec: (
                        uncertainty_priority(rec),
                        safe_float((rec.get("stat_model") or {}).get("expected_hvi"), 0.0),
                        safe_float((rec.get("stat_model") or {}).get("acquisition"), 0.0),
                    ),
                    reverse=True,
                )
                n_left = max(0, max_variants - len(keep))
                keep.extend(remaining_uncertainty[:n_left])

                chosen = {str(x.get("variant_id") or "") for x in keep}
                if len(keep) < max_variants:
                    backfill = [rec for rec in remaining_priority if str(rec.get("variant_id") or "") not in chosen]
                    keep.extend(backfill[: max(0, max_variants - len(keep))])
            eligible = keep[:max_variants]
        else:
            eligible = eligible[:max_variants]
    eligible_post_cap_total = int(len(eligible))
    eligible_post_cap_band_counts = dict(
        Counter(str(rec.get("vespag_gate_band") or "unknown").strip().lower() or "unknown" for rec in eligible)
    )

    wt_metrics_binary = run_gnina_score(
        receptor_pdb=wt_pdb_binary,
        ligand_pose=ligand_pose,
        gnina_bin=gnina_bin,
        gnina_cpu=max(1, int(args.gnina_cpu)),
        cnn_model=str(args.cnn_model),
        autobox_add=float(args.autobox_add),
        ld_library_path=ld_library_path,
    )
    wt_aff_binary = float(wt_metrics_binary["cnn_affinity"])
    wt_score_binary = float(safe_float(wt_metrics_binary.get("cnn_score"), 0.0))
    if run_dual_context:
        wt_metrics_ternary = run_gnina_score(
            receptor_pdb=wt_pdb_ternary,
            ligand_pose=ligand_pose,
            gnina_bin=gnina_bin,
            gnina_cpu=max(1, int(args.gnina_cpu)),
            cnn_model=str(args.cnn_model),
            autobox_add=float(args.autobox_add),
            ld_library_path=ld_library_path,
        )
        wt_aff_ternary = float(wt_metrics_ternary["cnn_affinity"])
        wt_score_ternary = float(safe_float(wt_metrics_ternary.get("cnn_score"), 0.0))
    else:
        wt_aff_ternary = wt_aff_binary
        wt_score_ternary = wt_score_binary

    cache_scores = load_cache_with_signature(cache_path, expected_signature=cache_sig) if bool(args.use_cache) else {}
    mutant_dir = binding_fastdl_mutants_dir(outdir=outdir, round_id=round_id)
    mutant_dir.mkdir(parents=True, exist_ok=True)

    to_score: List[Dict[str, Any]] = []
    scored: Dict[str, Dict[str, Any]] = {}
    cache_hits = 0

    for row in eligible:
        vid = str(row["variant_id"])
        cached = cache_scores.get(vid)
        if cached is not None and bool(cached.get("ok")):
            scored[vid] = dict(cached)
            cache_hits += 1
            continue
        task = {
            "variant_id": vid,
            "chain": str(row["chain"]),
            "pos": int(row["pos"]),
            "wt": str(row["wt"]),
            "mut": str(row["mut"]),
            "mutations": row.get("mutations") or [{"chain": row["chain"], "pos": row["pos"], "wt": row["wt"], "mut": row["mut"]}],
            "wt_pdb": str(wt_pdb_binary),
            "wt_pdb_ternary": str(wt_pdb_ternary),
            "run_ternary": bool(run_dual_context),
            "ternary_keep_resnames": ",".join(sorted(ternary_keep_resnames)),
            "ligand_pose": str(ligand_pose),
            "gnina_bin": str(gnina_bin),
            "gnina_cpu": int(max(1, int(args.gnina_cpu))),
            "cnn_model": str(args.cnn_model),
            "autobox_add": float(args.autobox_add),
            "ld_library_path": ld_library_path,
            "mutant_pdb": str(mutant_dir / f"{vid}.pdb"),
            "mutant_pdb_ternary": str(mutant_dir / f"{vid}__ternary.pdb"),
            "relax_mutants": bool(args.relax_mutants),
            "relax_max_iterations": int(args.relax_max_iterations),
            "relax_heavy_restraint_k": float(args.relax_heavy_restraint_k),
        }
        to_score.append(task)

    workers = max(1, int(args.workers))
    progress_every = max(1, int(args.progress_every))
    progress_total = int(len(to_score))
    progress_start = float(time.time())
    if progress_total > 0:
        print(
            "[fast-binding] "
            f"starting scored stage: total={progress_total} cache_hits={cache_hits} "
            f"workers={workers} gnina_cpu={max(1, int(args.gnina_cpu))} "
            f"relax_mutants={bool(args.relax_mutants)}"
        )
    if to_score:
        if workers == 1:
            for i, task in enumerate(to_score, start=1):
                out = score_variant_task(task)
                scored[str(out["variant_id"])] = out
                if i == progress_total or (i % progress_every == 0):
                    elapsed = max(0.0, float(time.time() - progress_start))
                    rate = float(i / elapsed) if elapsed > 1e-9 else 0.0
                    eta = float((progress_total - i) / rate) if rate > 1e-9 else float("inf")
                    eta_str = f"{int(round(eta))}s" if math.isfinite(eta) else "n/a"
                    print(f"[fast-binding] progress {i}/{progress_total} | elapsed={int(round(elapsed))}s | eta={eta_str}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(score_variant_task, task): str(task["variant_id"]) for task in to_score}
                done_n = 0
                for fut in as_completed(futs):
                    vid = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as exc:
                        out = {"variant_id": vid, "ok": False, "error": str(exc)}
                    scored[str(vid)] = dict(out)
                    done_n += 1
                    if done_n == progress_total or (done_n % progress_every == 0):
                        elapsed = max(0.0, float(time.time() - progress_start))
                        rate = float(done_n / elapsed) if elapsed > 1e-9 else 0.0
                        eta = float((progress_total - done_n) / rate) if rate > 1e-9 else float("inf")
                        eta_str = f"{int(round(eta))}s" if math.isfinite(eta) else "n/a"
                        print(f"[fast-binding] progress {done_n}/{progress_total} | elapsed={int(round(elapsed))}s | eta={eta_str}")

    for vid, payload in scored.items():
        cache_scores[vid] = dict(payload)
    if bool(args.use_cache):
        save_cache(
            path=cache_path,
            scores=cache_scores,
            metadata={
                "round": int(args.round),
                "gnina_version": gnina_version,
                "cnn_model": str(args.cnn_model),
                "binding_context": ("dual" if run_dual_context else "single"),
                "wt_pdb_binary": str(wt_pdb_binary),
                "wt_pdb_ternary": str(wt_pdb_ternary),
                "ternary_keep_resnames": ",".join(sorted(ternary_keep_resnames)),
                "ligand_pose": str(ligand_pose),
                "cache_signature": cache_sig,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    deltas_aff_binary: List[float] = []
    deltas_score_binary: List[float] = []
    deltas_aff_ternary: List[float] = []
    deltas_score_ternary: List[float] = []
    errors: List[Dict[str, Any]] = []
    ternary_errors: List[Dict[str, Any]] = []
    for row in eligible:
        vid = str(row["variant_id"])
        out = scored.get(vid, {})
        if not bool(out.get("ok")):
            errors.append(
                {
                    "variant_id": vid,
                    "error": str(out.get("error") or "unknown_error"),
                }
            )
            continue
        delta_aff_binary = float(out.get("binary_cnn_affinity", out.get("cnn_affinity", wt_aff_binary)) - wt_aff_binary)
        delta_score_binary = float(safe_float(out.get("binary_cnn_score", out.get("cnn_score", 0.0)), 0.0) - wt_score_binary)
        out["delta_cnn_affinity_binary"] = delta_aff_binary
        out["delta_cnn_score_binary"] = delta_score_binary
        out["delta_cnn_affinity"] = delta_aff_binary
        out["delta_cnn_score"] = delta_score_binary
        deltas_aff_binary.append(delta_aff_binary)
        deltas_score_binary.append(delta_score_binary)

        if run_dual_context:
            if bool(out.get("ternary_ok")):
                delta_aff_ternary = float(out.get("ternary_cnn_affinity", wt_aff_ternary) - wt_aff_ternary)
                delta_score_ternary = float(safe_float(out.get("ternary_cnn_score", 0.0), 0.0) - wt_score_ternary)
                out["delta_cnn_affinity_ternary"] = delta_aff_ternary
                out["delta_cnn_score_ternary"] = delta_score_ternary
                deltas_aff_ternary.append(delta_aff_ternary)
                deltas_score_ternary.append(delta_score_ternary)
            else:
                ternary_errors.append(
                    {
                        "variant_id": vid,
                        "error": str(out.get("ternary_error") or "ternary_context_failed"),
                    }
                )

    if bool(args.strict) and errors:
        raise SystemExit(f"Fast binding stage failed for {len(errors)} variants in --strict mode.")

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    aff_mean_binary, aff_std_binary = mean_std(deltas_aff_binary)
    score_mean_binary, _ = mean_std(deltas_score_binary)
    aff_mean_ternary, aff_std_ternary = mean_std(deltas_aff_ternary)
    score_mean_ternary, _ = mean_std(deltas_score_ternary)

    values_aff_binary = [float(x) for x in deltas_aff_binary]
    values_score_binary = [float(x) for x in deltas_score_binary]
    values_aff_ternary = [float(x) for x in deltas_aff_ternary]
    values_score_ternary = [float(x) for x in deltas_score_ternary]

    updated = 0
    fallback_updated = 0
    relaxed_success = 0
    relaxed_failed = 0
    affinity_scale = max(1e-6, float(args.binding_affinity_scale))
    score_scale = max(1e-6, float(args.binding_score_scale))
    w_binary = max(0.0, float(args.context_blend_weight_binary))
    w_ternary = max(0.0, float(args.context_blend_weight_ternary))
    w_sum = max(1e-9, w_binary + w_ternary)
    w_binary /= w_sum
    w_ternary /= w_sum
    coupling_penalty = max(0.0, float(args.context_coupling_penalty))
    for row in rows:
        muts = canonical_row_mutations(row)
        if not muts:
            continue
        vid = row_variant_id(row)
        if not vid:
            continue
        row["variant_id"] = vid
        row["mutations"] = muts
        anchor = muts[0]
        row["chain"] = str(anchor["chain"])
        row["pos"] = int(anchor["pos"])
        row["wt"] = str(anchor["wt"]).upper()
        row["mut"] = str(anchor["mut"]).upper()
        out = scored.get(vid)
        if out is not None and bool(out.get("ok")):
            delta_aff_binary = float(out.get("delta_cnn_affinity_binary", out.get("delta_cnn_affinity", 0.0)))
            delta_score_binary = float(out.get("delta_cnn_score_binary", out.get("delta_cnn_score", 0.0)))
            q_aff_binary = rank_quantile(values_aff_binary, delta_aff_binary)
            q_score_binary = rank_quantile(values_score_binary, delta_score_binary)
            abs_aff_binary = logistic(delta_aff_binary / affinity_scale)
            abs_score_binary = logistic(delta_score_binary / score_scale)
            centered_aff_binary = logistic((delta_aff_binary - aff_mean_binary) / max(aff_std_binary, affinity_scale))
            p_bind_binary = float(
                clamp(
                    (0.45 * abs_aff_binary)
                    + (0.15 * abs_score_binary)
                    + (0.25 * centered_aff_binary)
                    + (0.10 * q_aff_binary)
                    + (0.05 * q_score_binary),
                    0.0,
                    1.0,
                )
            )

            ternary_available = bool(run_dual_context and bool(out.get("ternary_ok")) and values_aff_ternary)
            if ternary_available:
                delta_aff_ternary = float(out.get("delta_cnn_affinity_ternary", 0.0))
                delta_score_ternary = float(out.get("delta_cnn_score_ternary", 0.0))
                q_aff_ternary = rank_quantile(values_aff_ternary, delta_aff_ternary)
                q_score_ternary = rank_quantile(values_score_ternary, delta_score_ternary)
                abs_aff_ternary = logistic(delta_aff_ternary / affinity_scale)
                abs_score_ternary = logistic(delta_score_ternary / score_scale)
                centered_aff_ternary = logistic((delta_aff_ternary - aff_mean_ternary) / max(aff_std_ternary, affinity_scale))
                p_bind_ternary = float(
                    clamp(
                        (0.45 * abs_aff_ternary)
                        + (0.15 * abs_score_ternary)
                        + (0.25 * centered_aff_ternary)
                        + (0.10 * q_aff_ternary)
                        + (0.05 * q_score_ternary),
                        0.0,
                        1.0,
                    )
                )
                coupling_abs = abs(float(p_bind_ternary) - float(p_bind_binary))
                p_bind_fastdl = float(
                    clamp(
                        (w_binary * p_bind_binary) + (w_ternary * p_bind_ternary) - (coupling_penalty * coupling_abs),
                        0.0,
                        1.0,
                    )
                )
            else:
                delta_aff_ternary = float("nan")
                delta_score_ternary = float("nan")
                q_aff_ternary = float("nan")
                q_score_ternary = float("nan")
                abs_aff_ternary = float("nan")
                abs_score_ternary = float("nan")
                centered_aff_ternary = float("nan")
                p_bind_ternary = float("nan")
                coupling_abs = 0.0
                p_bind_fastdl = float(p_bind_binary)

            row["p_bind"] = round(p_bind_fastdl, 6)
            row["p_bind_fastdl"] = round(p_bind_fastdl, 6)
            row["p_bind_binary"] = round(float(p_bind_binary), 6)
            row["p_bind_binary_fastdl"] = round(float(p_bind_binary), 6)
            if math.isfinite(float(p_bind_ternary)):
                row["p_bind_ternary"] = round(float(p_bind_ternary), 6)
                row["p_bind_ternary_fastdl"] = round(float(p_bind_ternary), 6)
                row["p_bind_coupling_abs"] = round(float(coupling_abs), 6)
            else:
                row["p_bind_ternary"] = None
                row["p_bind_ternary_fastdl"] = None
                row["p_bind_coupling_abs"] = round(float(coupling_abs), 6)
            relaxed_ok = bool(out.get("mutant_relaxed"))
            if bool(args.relax_mutants):
                if relaxed_ok:
                    relaxed_success += 1
                else:
                    relaxed_failed += 1
            row["binding_fastdl"] = {
                "model": "gnina_score_only_delta",
                "gnina_version": gnina_version,
                "cnn_model": str(args.cnn_model),
                "binding_context": ("dual" if run_dual_context else "single"),
                "mutant_relaxed": relaxed_ok,
                "mutant_relax_error": str(out.get("mutant_relax_error") or ""),
                "wt_cnn_affinity_binary": round(float(wt_aff_binary), 6),
                "wt_cnn_score_binary": round(float(wt_score_binary), 6),
                "mut_cnn_affinity_binary": round(float(out.get("binary_cnn_affinity", wt_aff_binary)), 6),
                "mut_cnn_score_binary": round(float(safe_float(out.get("binary_cnn_score"), 0.0)), 6),
                "mut_cnn_variance_binary": round(float(safe_float(out.get("binary_cnn_variance"), 0.0)), 6),
                "delta_cnn_affinity_binary": round(delta_aff_binary, 6),
                "delta_cnn_score_binary": round(delta_score_binary, 6),
                "abs_affinity_prob_binary": round(float(abs_aff_binary), 6),
                "abs_score_prob_binary": round(float(abs_score_binary), 6),
                "centered_affinity_prob_binary": round(float(centered_aff_binary), 6),
                "delta_affinity_quantile_binary": round(float(q_aff_binary), 6),
                "delta_score_quantile_binary": round(float(q_score_binary), 6),
                "delta_affinity_round_mean_binary": round(float(aff_mean_binary), 6),
                "delta_affinity_round_std_binary": round(float(aff_std_binary), 6),
                "delta_score_round_mean_binary": round(float(score_mean_binary), 6),
                "p_bind_binary": round(float(p_bind_binary), 6),
                "p_bind_ternary": (round(float(p_bind_ternary), 6) if math.isfinite(float(p_bind_ternary)) else None),
                "p_bind_coupling_abs": round(float(coupling_abs), 6),
                "p_bind_blended": round(float(p_bind_fastdl), 6),
                "blend_weight_binary": round(float(w_binary), 6),
                "blend_weight_ternary": round(float(w_ternary), 6),
                "coupling_penalty": round(float(coupling_penalty), 6),
                "ternary_ok": bool(out.get("ternary_ok")),
                "ternary_error": str(out.get("ternary_error") or ""),
                "ternary_hetero_appended": int(safe_float(out.get("ternary_hetero_appended"), 0.0)),
            }
            if run_dual_context:
                row["binding_fastdl"].update(
                    {
                        "wt_cnn_affinity_ternary": round(float(wt_aff_ternary), 6),
                        "wt_cnn_score_ternary": round(float(wt_score_ternary), 6),
                        "mut_cnn_affinity_ternary": (
                            round(float(out.get("ternary_cnn_affinity", wt_aff_ternary)), 6)
                            if bool(out.get("ternary_ok"))
                            else None
                        ),
                        "mut_cnn_score_ternary": (
                            round(float(safe_float(out.get("ternary_cnn_score"), 0.0)), 6)
                            if bool(out.get("ternary_ok"))
                            else None
                        ),
                        "mut_cnn_variance_ternary": (
                            round(float(safe_float(out.get("ternary_cnn_variance"), 0.0)), 6)
                            if bool(out.get("ternary_ok"))
                            else None
                        ),
                        "delta_cnn_affinity_ternary": (round(float(delta_aff_ternary), 6) if math.isfinite(float(delta_aff_ternary)) else None),
                        "delta_cnn_score_ternary": (round(float(delta_score_ternary), 6) if math.isfinite(float(delta_score_ternary)) else None),
                        "abs_affinity_prob_ternary": (round(float(abs_aff_ternary), 6) if math.isfinite(float(abs_aff_ternary)) else None),
                        "abs_score_prob_ternary": (round(float(abs_score_ternary), 6) if math.isfinite(float(abs_score_ternary)) else None),
                        "centered_affinity_prob_ternary": (
                            round(float(centered_aff_ternary), 6) if math.isfinite(float(centered_aff_ternary)) else None
                        ),
                        "delta_affinity_quantile_ternary": (round(float(q_aff_ternary), 6) if math.isfinite(float(q_aff_ternary)) else None),
                        "delta_score_quantile_ternary": (round(float(q_score_ternary), 6) if math.isfinite(float(q_score_ternary)) else None),
                        "delta_affinity_round_mean_ternary": round(float(aff_mean_ternary), 6),
                        "delta_affinity_round_std_ternary": round(float(aff_std_ternary), 6),
                        "delta_score_round_mean_ternary": round(float(score_mean_ternary), 6),
                    }
                )
            updated += 1
            continue

        fallback_bind = fallback_bind_probability(row)
        row["p_bind"] = round(fallback_bind, 6)
        row["p_bind_fastdl"] = round(fallback_bind, 6)
        row["p_bind_binary"] = round(fallback_bind, 6)
        row["p_bind_binary_fastdl"] = round(fallback_bind, 6)
        if run_dual_context:
            row["p_bind_ternary"] = round(fallback_bind, 6)
            row["p_bind_ternary_fastdl"] = round(fallback_bind, 6)
            row["p_bind_coupling_abs"] = 0.0
        else:
            row["p_bind_ternary"] = None
            row["p_bind_ternary_fastdl"] = None
            row["p_bind_coupling_abs"] = 0.0
        fallback_reason = "not_scored"
        if out is not None and not bool(out.get("ok")):
            fallback_reason = str(out.get("error") or "score_error")
        row["binding_fastdl"] = {
            "model": "fallback_surrogate_bind",
            "gnina_version": gnina_version,
            "cnn_model": str(args.cnn_model),
            "binding_context": ("dual" if run_dual_context else "single"),
            "fallback_reason": fallback_reason,
            "proxy_bind": round(float(fallback_bind), 6),
            "source": "stat_model_objective_or_dist_contact_proxy",
        }
        fallback_updated += 1

    write_jsonl(out_path, rows)

    summary = {
        "round": int(args.round),
        "stage": "fast_binding_delta_v3_context_aware",
        "proposals_path": str(proposals_path),
        "out_path": str(out_path),
        "binding_context": ("dual" if run_dual_context else "single"),
        "wt_pdb_binary": str(wt_pdb_binary),
        "wt_pdb_ternary": str(wt_pdb_ternary),
        "ternary_keep_resnames": ",".join(sorted(ternary_keep_resnames)),
        "ligand_pose": str(ligand_pose),
        "gnina_bin": str(gnina_bin),
        "gnina_version": gnina_version,
        "cnn_model": str(args.cnn_model),
        "relax_mutants": bool(args.relax_mutants),
        "relax_max_iterations": int(args.relax_max_iterations),
        "relax_heavy_restraint_k": float(args.relax_heavy_restraint_k),
        "binding_affinity_scale": float(args.binding_affinity_scale),
        "binding_score_scale": float(args.binding_score_scale),
        "relax_success_total": int(relaxed_success),
        "relax_failed_total": int(relaxed_failed),
        "workers": int(workers),
        "gnina_cpu_per_job": int(max(1, int(args.gnina_cpu))),
        "score_all": bool(args.score_all),
        "eligible_pre_cap_total": int(eligible_pre_cap_total),
        "eligible_pre_cap_band_counts": eligible_pre_cap_band_counts,
        "eligible_total": int(eligible_post_cap_total),
        "eligible_band_counts": eligible_post_cap_band_counts,
        "forced_functional_kept": int(forced_functional_kept),
        "cache_hits": int(cache_hits),
        "scored_total": int(len(scored)),
        "updated_total": int(updated),
        "fallback_updated_total": int(fallback_updated),
        "error_total": int(len(errors)),
        "ternary_error_total": int(len(ternary_errors)),
        "errors": errors[:100],
        "ternary_errors": ternary_errors[:100],
        "wt_cnn_affinity_binary": round(float(wt_aff_binary), 6),
        "wt_cnn_score_binary": round(float(wt_score_binary), 6),
        "wt_cnn_affinity_ternary": round(float(wt_aff_ternary), 6),
        "wt_cnn_score_ternary": round(float(wt_score_ternary), 6),
        "delta_affinity_mean_binary": round(float(aff_mean_binary), 6),
        "delta_affinity_std_binary": round(float(aff_std_binary), 6),
        "delta_score_mean_binary": round(float(score_mean_binary), 6),
        "delta_affinity_mean_ternary": round(float(aff_mean_ternary), 6),
        "delta_affinity_std_ternary": round(float(aff_std_ternary), 6),
        "delta_score_mean_ternary": round(float(score_mean_ternary), 6),
        "context_blend_weight_binary": round(float(w_binary), 6),
        "context_blend_weight_ternary": round(float(w_ternary), 6),
        "context_coupling_penalty": round(float(coupling_penalty), 6),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"[fast-binding] Wrote: {out_path}")
    print(f"[fast-binding] Wrote: {summary_path}")
    print(
        "[fast-binding] "
        f"eligible={len(eligible)} scored={len(scored)} updated={updated} "
        f"errors={len(errors)} cache_hits={cache_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
