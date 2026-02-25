import json
import re
from pathlib import Path
from typing import Dict, Optional


UNIPROT_ACCESSION_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_input_manifest(outdir: Path) -> Dict:
    return _read_json(outdir / "swarm" / "input_manifest.json")


def resolve_outdir_path(outdir: Path, raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = outdir / p
    return p


def _existing_file(path: Optional[Path]) -> Optional[Path]:
    if path and path.exists() and path.is_file() and path.stat().st_size > 0:
        return path
    return None


def resolve_canonical_protein_pdb(outdir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    p = _existing_file(explicit)
    if p:
        return p

    manifest = load_input_manifest(outdir)
    mp = resolve_outdir_path(outdir, ((manifest.get("protein") or {}).get("canonical_pdb") or ""))
    p = _existing_file(mp)
    if p:
        return p

    p = _existing_file(outdir / "reference_protein.pdb")
    if p:
        return p

    for name in ("receptor.pdb", "protein.pdb", "enzyme.pdb", "structure.pdb"):
        p = _existing_file(outdir / name)
        if p:
            return p

    for glob_pat in ("AF-*-F1-model_*.pdb", "AF-*-model_*.pdb"):
        for cand in sorted(outdir.glob(glob_pat)):
            p = _existing_file(cand)
            if p:
                return p

    for cand in sorted(outdir.glob("*.pdb")):
        low = cand.name.lower()
        if any(t in low for t in ("ligand", "docked", "pose", "pocket")):
            continue
        p = _existing_file(cand)
        if p:
            return p
    return None


def resolve_canonical_fasta(outdir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    p = _existing_file(explicit)
    if p:
        return p

    manifest = load_input_manifest(outdir)
    mf = resolve_outdir_path(outdir, ((manifest.get("fasta") or {}).get("canonical_fasta") or ""))
    p = _existing_file(mf)
    if p:
        return p

    p = _existing_file(outdir / "enzyme_wt.fasta")
    if p:
        return p
    return None


def resolve_docking_pose_sdf(outdir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    p = _existing_file(explicit)
    if p:
        return p

    for name in ("docked_gnina_rescored.sdf", "docked_top.sdf", "docked_vina_poses.sdf"):
        p = _existing_file(outdir / name)
        if p:
            return p

    manifest = load_input_manifest(outdir)
    ms = resolve_outdir_path(outdir, ((manifest.get("ligand") or {}).get("canonical_sdf") or ""))
    p = _existing_file(ms)
    if p:
        return p

    p = _existing_file(outdir / "ligand.sdf")
    if p:
        return p
    return None


def resolve_ligand_sdf(outdir: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    p = _existing_file(explicit)
    if p:
        return p

    manifest = load_input_manifest(outdir)
    ms = resolve_outdir_path(outdir, ((manifest.get("ligand") or {}).get("canonical_sdf") or ""))
    p = _existing_file(ms)
    if p:
        return p

    p = _existing_file(outdir / "ligand.sdf")
    if p:
        return p
    return None


def infer_uniprot_accession(outdir: Path, fasta_path: Optional[Path] = None) -> Optional[str]:
    manifest = load_input_manifest(outdir)

    mf = (manifest.get("fasta") or {}).get("uniprot_accession")
    if isinstance(mf, str) and mf.strip():
        return mf.strip()

    mp = (manifest.get("protein") or {}).get("id")
    if isinstance(mp, str):
        mid = mp.strip().upper()
        if UNIPROT_ACCESSION_RE.match(mid):
            return mid

    p = _existing_file(fasta_path) if fasta_path is not None else resolve_canonical_fasta(outdir)
    if p:
        with p.open() as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                header = line.strip()[1:]
                if "|" in header:
                    parts = header.split("|")
                    if len(parts) >= 2 and parts[1]:
                        return parts[1].strip()
                m = re.search(r"\b([A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b", header)
                if m:
                    return m.group(1)
                break

    pdb = resolve_canonical_protein_pdb(outdir)
    if pdb:
        m = re.search(r"AF-([A-NR-Z][0-9][A-Z0-9]{3}[0-9])-F1-model_", pdb.name)
        if m:
            return m.group(1)
    return None
