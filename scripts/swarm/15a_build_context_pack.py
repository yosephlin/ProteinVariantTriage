import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from input_paths import (
        infer_uniprot_accession,
        load_input_manifest,
        resolve_canonical_fasta,
        resolve_canonical_protein_pdb,
        resolve_docking_pose_sdf,
    )
except ImportError:
    from scripts.swarm.input_paths import (
        infer_uniprot_accession,
        load_input_manifest,
        resolve_canonical_fasta,
        resolve_canonical_protein_pdb,
        resolve_docking_pose_sdf,
    )


def load_fasta(path: Path) -> Tuple[str, Optional[str]]:
    header = None
    seq = []
    if not path.exists():
        return "", None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                continue
            seq.append(line)
    sequence = "".join(seq)
    uniprot = None
    if header and "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            uniprot = parts[1]
    return sequence, uniprot


def parse_pdb_plddt_stats(pdb_path: Path) -> Dict[str, float]:
    bvals = []
    if not pdb_path or not pdb_path.exists():
        return {"min": 0.0, "mean": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    with pdb_path.open() as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                try:
                    b = float(ln[60:66])
                    bvals.append(b)
                except Exception:
                    pass
    if not bvals:
        return {"min": 0.0, "mean": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    bvals.sort()
    n = len(bvals)

    def q(p):
        if n == 1:
            return bvals[0]
        idx = int(round(p * (n - 1)))
        return bvals[max(0, min(n - 1, idx))]

    return {
        "min": bvals[0],
        "mean": sum(bvals) / n,
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
    }


def estimate_primary_chain_residue_count(pdb_path: Optional[Path]) -> int:
    if not pdb_path or not pdb_path.exists():
        return 0
    chain_counts: Dict[str, set] = {}
    with pdb_path.open() as fh:
        for ln in fh:
            if not ln.startswith("ATOM"):
                continue
            chain = (ln[21].strip() or "A")
            try:
                resi = int(ln[22:26])
            except Exception:
                continue
            icode = (ln[26].strip() if len(ln) > 26 else "")
            chain_counts.setdefault(chain, set()).add((resi, icode))
    if not chain_counts:
        return 0
    primary_chain = max(chain_counts.keys(), key=lambda c: len(chain_counts[c]))
    return len(chain_counts[primary_chain])


def parse_box_params(path: Path) -> Optional[Dict[str, List[float]]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        center = data.get("center")
        size = data.get("size")
        return {"center": center, "size": size}
    except Exception:
        return None


def parse_pocket_chain(path: Path) -> List[str]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        out.append(s)
    return out


def fpocket_present(outdir: Path) -> bool:
    for p in outdir.glob("*_out"):
        if (p / "pockets").exists():
            return True
    return False


def _has_smiles(smiles: Optional[object]) -> bool:
    if not smiles:
        return False
    if isinstance(smiles, str):
        return True
    if isinstance(smiles, dict):
        for k in ("canonical_smiles", "isomeric_smiles", "SMILES", "ConnectivitySMILES"):
            if smiles.get(k):
                return True
    return False


def _smiles_from_rdkit(path: Path) -> Optional[Dict[str, str]]:
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = None
    suffix = path.suffix.lower()
    try:
        if suffix in (".sdf", ".mol", ".mol2"):
            suppl = Chem.SDMolSupplier(str(path), removeHs=False)
            for m in suppl:
                if m is not None:
                    mol = m
                    break
        elif suffix in (".pdb", ".pdbqt"):
            mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    except Exception:
        mol = None
    if mol is None:
        return None
    try:
        canon = Chem.MolToSmiles(mol, canonical=True)
        iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not canon and not iso:
            return None
        return {"canonical_smiles": canon or iso, "isomeric_smiles": iso or canon}
    except Exception:
        return None


def _smiles_from_obabel(path: Path) -> Optional[Dict[str, str]]:
    if not shutil.which("obabel"):
        return None
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        return None
    try:
        out = subprocess.check_output(
            ["obabel", f"-i{suffix}", str(path), "-osmi"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    first = out.splitlines()[0].strip()
    if not first:
        return None
    smi = first.split()[0]
    if not smi:
        return None
    return {"canonical_smiles": smi, "isomeric_smiles": smi}


def extract_smiles_from_structures(paths: List[Path]) -> Optional[Dict[str, str]]:
    seen = set()
    for path in paths:
        if not path or not path.exists():
            continue
        if path in seen:
            continue
        seen.add(path)
        smiles = _smiles_from_rdkit(path)
        if smiles:
            return smiles
    for path in paths:
        if not path or not path.exists():
            continue
        smiles = _smiles_from_obabel(path)
        if smiles:
            return smiles
    return None


def prune_api_context_for_swarm(api_context: Dict) -> Dict:
    # Keep only API sections that directly influence round-0 mutation generation priors.
    # Residue-level traffic-light constraints are consumed from residue_constraints.jsonl,
    # not from this context pack.
    if not isinstance(api_context, dict):
        return {}
    out: Dict[str, Dict] = {}
    uni = api_context.get("uniprot")
    if isinstance(uni, dict):
        out["uniprot"] = {"accession": uni.get("accession")}
    chembl = api_context.get("chembl")
    if isinstance(chembl, dict):
        out["chembl"] = {
            "target": chembl.get("target"),
            "ligand_priors": (chembl.get("ligand_priors") or [])[:50] if isinstance(chembl.get("ligand_priors"), list) else chembl.get("ligand_priors"),
            "note": chembl.get("note"),
        }
    lig = api_context.get("ligand")
    if isinstance(lig, dict):
        out["ligand"] = {"smiles": lig.get("smiles")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build local SWARM context pack (pre-ESM)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--swarm-dir", default=None)
    ap.add_argument("--api-context", default=None)
    ap.add_argument("--uniprot-id", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--docked-sdf", default=None)
    ap.add_argument("--docking-summary", default=None)
    ap.add_argument("--box-params", default=None)
    ap.add_argument("--pocket-chain", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(os.environ.get("OUTDIR", "./out"))
    swarm_dir = Path(args.swarm_dir) if args.swarm_dir else outdir / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)

    explicit_fasta = Path(args.fasta) if args.fasta else None
    fasta_path = resolve_canonical_fasta(outdir, explicit=explicit_fasta) or (outdir / "enzyme_wt.fasta")
    sequence, uniprot_from_fasta = load_fasta(fasta_path)

    uniprot_id = args.uniprot_id or uniprot_from_fasta or infer_uniprot_accession(outdir, fasta_path=fasta_path)

    explicit_pdb = Path(args.pdb) if args.pdb else None
    pdb_path = resolve_canonical_protein_pdb(outdir, explicit=explicit_pdb)
    pdb_res_count = estimate_primary_chain_residue_count(pdb_path)
    if sequence and pdb_res_count:
        short = min(len(sequence), pdb_res_count)
        long = max(len(sequence), pdb_res_count)
        if short / float(long) < 0.70:
            raise SystemExit(
                "ERROR: FASTA/PDB size mismatch is too large for stable SWARM context. "
                f"(fasta_len={len(sequence)} pdb_primary_chain_len={pdb_res_count}). "
                "Regenerate canonical inputs with scripts/swarm/14a_prepare_inputs.py."
            )
    explicit_sdf = Path(args.docked_sdf) if args.docked_sdf else None
    docked_sdf = resolve_docking_pose_sdf(outdir, explicit=explicit_sdf) or (outdir / "ligand.sdf")
    docking_summary = Path(args.docking_summary) if args.docking_summary else outdir / "docking_summary.json"
    box_params = Path(args.box_params) if args.box_params else outdir / "box_params.json"
    pocket_chain = Path(args.pocket_chain) if args.pocket_chain else outdir / "pocket_residues.chain.txt"
    input_manifest = load_input_manifest(outdir)

    api_context = None
    api_context_path = None
    if args.api_context:
        api_context_path = Path(args.api_context)
    else:
        cand = outdir / "swarm_api" / "context_api.json"
        if cand.exists():
            api_context_path = cand
    if api_context_path and api_context_path.exists():
        try:
            api_context = json.loads(api_context_path.read_text())
            if not uniprot_id:
                uniprot_id = api_context.get("uniprot", {}).get("accession")
        except Exception:
            api_context = None

    context = {
        "target": {
            "uniprot_id": uniprot_id,
            "sequence": sequence,
            "length": len(sequence),
            "chains_present": ["A"],
        },
        "structures": {
            "receptor_pdb_path": str(pdb_path) if pdb_path else None,
            "af_plddt_stats": parse_pdb_plddt_stats(pdb_path) if pdb_path else {},
        },
        "ligand": {
            "docked_pose_path": str(docked_sdf),
            "vina_gnina_summary": json.loads(docking_summary.read_text()) if docking_summary.exists() else {},
        },
        "pocket": {
            "pocket_residues_chain": parse_pocket_chain(pocket_chain),
            "box_center_size": parse_box_params(box_params) or {},
            "fpocket_present": fpocket_present(outdir),
        },
        "rules": {
            "disallow_mutating_active_site": True,
            "disallow_break_disulfide": True,
            "disallow_gly_pro_in_helix": True,
        },
        "input_manifest_present": bool(input_manifest),
        "inputs": {
            "fasta_path": str(fasta_path),
            "receptor_pdb_path": str(pdb_path) if pdb_path else None,
            "ligand_pose_sdf_path": str(docked_sdf),
            "pocket_chain_path": str(pocket_chain),
        },
    }

    if api_context is not None:
        pruned_api_context = prune_api_context_for_swarm(api_context)
        context["api_context"] = pruned_api_context
        context["api_context_sources"] = {
            "kept_for_round0_generation": ["uniprot", "chembl", "ligand"],
            "dropped_or_compacted": [k for k in sorted(api_context.keys()) if k not in set(pruned_api_context.keys())],
        }
        # surface optional ligand/chembl info at top-level for convenience
        if "chembl" in pruned_api_context:
            context["chembl"] = pruned_api_context["chembl"]
        api_lig = pruned_api_context.get("ligand", {})
        if isinstance(api_lig, dict) and api_lig.get("smiles"):
            context.setdefault("ligand", {})
            context["ligand"]["smiles"] = api_lig.get("smiles")

    if not _has_smiles(context.get("ligand", {}).get("smiles")):
        candidate_paths = [
            outdir / "ligand.sdf",
            docked_sdf,
            outdir / "docked_gnina_rescored.sdf",
            outdir / "docked_vina_poses.sdf",
            outdir / "ligand.pdb",
            outdir / "ligand.pdbqt",
        ]
        smiles = extract_smiles_from_structures(candidate_paths)
        if smiles:
            context.setdefault("ligand", {})
            context["ligand"]["smiles"] = smiles
        else:
            context.setdefault("ligand", {})
            context["ligand"].setdefault("smiles", {"note": "missing_smiles"})

    (swarm_dir / "context_pack.json").write_text(json.dumps(context, ensure_ascii=False, indent=2))
    print("Wrote:", swarm_dir / "context_pack.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
