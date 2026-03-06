import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}

PROTEIN_HET_KEEP = {"MSE", "SEC", "PYL"}
WATER_NAMES = {"HOH", "WAT", "DOD"}
METALS = {
    "ZN",
    "MG",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CA",
    "NA",
    "K",
    "CD",
    "HG",
    "PB",
}


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "ProteinVariantTriage/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, timeout: int = 60) -> Dict:
    payload = _http_get_bytes(url, timeout=timeout)
    obj = json.loads(payload.decode("utf-8"))
    return obj if isinstance(obj, dict) else {}


def _try_download(url: str, out_path: Path) -> bool:
    try:
        data = _http_get_bytes(url)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False
    if not data:
        return False
    out_path.write_bytes(data)
    return out_path.exists() and out_path.stat().st_size > 0


def _run(cmd: List[str]) -> None:
    subprocess.check_call(cmd)


def _infer_uniprot_from_fasta(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if "|" in header:
                parts = header.split("|")
                if len(parts) >= 2 and parts[1]:
                    return parts[1]
            m = re.search(r"\b([A-NR-Z0-9]{6,10})\b", header)
            if m:
                return m.group(1)
            break
    return None


def _download_uniprot_fasta(accession: str, out_path: Path) -> None:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    data = _http_get_bytes(url)
    text = data.decode("utf-8", errors="ignore")
    if not text.startswith(">"):
        raise RuntimeError(f"UniProt FASTA response did not look like FASTA for accession {accession}")
    out_path.write_text(text)


def _download_alphafold_model(accession: str, out_path: Path, versions: List[str]) -> Tuple[str, str]:
    attempts: List[Tuple[str, str]] = []
    for ver in versions:
        ver = ver.strip()
        if not ver:
            continue
        pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_{ver}.pdb"
        cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_{ver}.cif"
        attempts.append((pdb_url, "pdb"))
        attempts.append((cif_url, "cif"))

    tmp = out_path.parent / f".tmp_af_{accession}"
    for url, fmt in attempts:
        target = tmp.with_suffix(f".{fmt}")
        if _try_download(url, target):
            if fmt == "pdb":
                out_path.write_bytes(target.read_bytes())
            else:
                _convert_mmcif_to_pdb(target, out_path)
            target.unlink(missing_ok=True)
            return (url, fmt)
    raise RuntimeError(f"Could not download AlphaFold model for {accession} using versions={versions}")


def _download_rcsb_model(pdb_id: str, tmp_dir: Path) -> Tuple[Path, str]:
    pid = str(pdb_id).strip().lower()
    if not re.fullmatch(r"[0-9a-z]{4}", pid):
        raise RuntimeError(f"Invalid RCSB PDB id: {pdb_id}")

    candidates = [
        (f"https://files.rcsb.org/download/{pid}.pdb", "pdb"),
        (f"https://files.rcsb.org/download/{pid}.cif", "cif"),
        (f"https://files.rcsb.org/download/pdb_{pid:0>8}.cif", "cif"),
    ]
    for url, kind in candidates:
        target = tmp_dir / f"{pid}.{kind}"
        if _try_download(url, target):
            return target, url
    raise RuntimeError(f"Could not download RCSB model for {pdb_id}")


def _download_rcsb_fasta(pdb_id: str, out_path: Path) -> bool:
    pid = str(pdb_id).strip().upper()
    url = f"https://www.rcsb.org/fasta/entry/{pid}/download"
    try:
        data = _http_get_bytes(url)
    except Exception:
        return False
    text = data.decode("utf-8", errors="ignore")
    if not text.startswith(">"):
        return False
    out_path.write_text(text)
    return True


def _resolve_obabel_format(path: Path) -> Optional[str]:
    ext = path.suffix.lower().lstrip(".")
    if not ext:
        return None
    if ext in {"cif", "mmcif", "mcif"}:
        return "mcif"
    if ext in {"smi", "smiles"}:
        return "smi"
    return ext


def _convert_mmcif_to_pdb(src: Path, dst: Path) -> None:
    # Preferred: Open Babel conversion. Fallback: Biopython MMCIFParser.
    if shutil.which("obabel"):
        in_fmt = _resolve_obabel_format(src) or "mcif"
        try:
            _run(["obabel", f"-i{in_fmt}", str(src), "-opdb", "-O", str(dst)])
            if dst.exists() and dst.stat().st_size > 0:
                return
        except Exception:
            pass
    try:
        from Bio.PDB import MMCIFParser, PDBIO  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Cannot convert mmCIF to PDB for {src}: missing converter ({e})")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", str(src))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst))
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"mmCIF to PDB conversion produced empty output: {dst}")


def _line_elem(line: str, atom_name: str) -> str:
    elem = line[76:78].strip() if len(line) >= 78 else ""
    if elem:
        return elem.upper()
    return atom_name[:1].upper()


def _normalize_protein_pdb(src: Path, dst: Path) -> Dict[str, int]:
    kept = 0
    skipped_altloc = 0
    skipped_water = 0
    skipped_non_protein_het = 0
    skipped_h = 0
    models_seen = 0
    in_first_model = False
    has_model_records = False
    lines_out: List[str] = []

    with src.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            rec = line[:6].strip().upper()

            if rec == "MODEL":
                has_model_records = True
                models_seen += 1
                if models_seen == 1:
                    in_first_model = True
                else:
                    break
                continue
            if rec == "ENDMDL":
                if in_first_model:
                    break
                continue
            if has_model_records and not in_first_model:
                continue

            if rec not in {"ATOM", "HETATM", "TER", "END"}:
                continue
            if rec in {"TER", "END"}:
                lines_out.append(line)
                continue

            if len(line) < 54:
                continue

            atom_name = line[12:16].strip()
            altloc = line[16].strip() if len(line) > 16 else ""
            resname = (line[17:20].strip() if len(line) > 20 else "").upper()
            chain = line[21].strip() if len(line) > 21 else ""
            elem = _line_elem(line, atom_name)

            if altloc and altloc not in {"A", "1"}:
                skipped_altloc += 1
                continue
            if elem == "H" or atom_name.upper().startswith("H"):
                skipped_h += 1
                continue
            if resname in WATER_NAMES:
                skipped_water += 1
                continue

            is_protein = rec == "ATOM" or resname in AA3_TO_1 or resname in PROTEIN_HET_KEEP
            is_metal = elem in METALS
            if rec == "HETATM" and not (is_protein or is_metal):
                skipped_non_protein_het += 1
                continue

            # Normalize chain + altloc.
            line_chars = list(line.ljust(80))
            line_chars[16] = " "
            line_chars[21] = chain if chain else "A"
            lines_out.append("".join(line_chars).rstrip())
            kept += 1

    if kept == 0:
        raise RuntimeError(f"No atoms were kept after PDB normalization: {src}")
    if not any(l.startswith("END") for l in lines_out):
        lines_out.append("END")
    dst.write_text("\n".join(lines_out) + "\n")

    return {
        "atoms_kept": kept,
        "skipped_altloc": skipped_altloc,
        "skipped_hydrogen": skipped_h,
        "skipped_water": skipped_water,
        "skipped_non_protein_het": skipped_non_protein_het,
        "has_model_records": 1 if has_model_records else 0,
        "models_seen": models_seen,
    }


def _sequence_from_pdb(path: Path, preferred_chain: Optional[str] = None) -> Tuple[str, Optional[str]]:
    seqres: Dict[str, List[str]] = {}
    with path.open() as fh:
        for raw in fh:
            if not raw.startswith("SEQRES"):
                continue
            chain = (raw[11].strip() or "A")
            aas = raw[19:].split()
            slot = seqres.setdefault(chain, [])
            for aa in aas:
                aa1 = AA3_TO_1.get(aa.upper())
                if aa1:
                    slot.append(aa1)
    if seqres:
        chain = preferred_chain if preferred_chain in seqres else sorted(seqres.keys())[0]
        return ("".join(seqres[chain]), chain)

    # Fallback to ATOM-based residue walk.
    chain_res: Dict[str, List[Tuple[int, str, str]]] = {}
    seen = set()
    with path.open() as fh:
        for raw in fh:
            if not raw.startswith("ATOM"):
                continue
            chain = (raw[21].strip() or "A")
            resname = raw[17:20].strip().upper()
            aa1 = AA3_TO_1.get(resname)
            if not aa1:
                continue
            try:
                resseq = int(raw[22:26])
            except Exception:
                continue
            icode = raw[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            chain_res.setdefault(chain, []).append((resseq, icode, aa1))

    if not chain_res:
        return ("", None)
    chain = preferred_chain if preferred_chain in chain_res else sorted(chain_res.keys())[0]
    ordered = sorted(chain_res[chain], key=lambda x: (x[0], x[1]))
    return ("".join(aa for _, _, aa in ordered), chain)


def _write_fasta(sequence: str, out_path: Path, header: str) -> None:
    if not sequence:
        raise RuntimeError("Cannot write FASTA with empty sequence")
    lines = [f">{header}"]
    for i in range(0, len(sequence), 80):
        lines.append(sequence[i : i + 80])
    out_path.write_text("\n".join(lines) + "\n")


def _read_fasta_sequence(path: Path) -> Tuple[str, Optional[str]]:
    if not path.exists():
        return "", None
    header = None
    seq = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                header = s[1:].strip() or None
                continue
            seq.append(s)
    return "".join(seq), header


def _prefix_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / float(n)


def _sequences_compatible(fasta_seq: str, pdb_seq: str) -> bool:
    fa = (fasta_seq or "").strip().upper()
    pb = (pdb_seq or "").strip().upper()
    if not fa or not pb:
        return True
    if fa == pb:
        return True
    shorter = min(len(fa), len(pb))
    longer = max(len(fa), len(pb))
    if shorter >= 0.85 * longer and (fa in pb or pb in fa):
        return True
    pid = _prefix_identity(fa, pb)
    if abs(len(fa) - len(pb)) <= 10 and pid >= 0.90:
        return True
    if (shorter / float(longer)) >= 0.95 and pid >= 0.80:
        return True
    return False


def _first_nonempty_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if s:
                return s
    return None


def _pubchem_cid_from_name(name: str) -> str:
    q = quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    obj = _http_get_json(url)
    cids = ((obj.get("IdentifierList") or {}).get("CID") or [])
    if not cids:
        raise RuntimeError(f"No PubChem CID found for name='{name}'")
    return str(cids[0])


def _download_pubchem_sdf(cid: str, out_sdf: Path) -> str:
    cid_s = str(cid).strip()
    urls = [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid_s}/SDF?record_type=3d",
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{cid_s}/SDF",
    ]
    for url in urls:
        if _try_download(url, out_sdf):
            return url
    raise RuntimeError(f"Failed to download PubChem SDF for CID={cid_s}")


def _chembl_smiles(chembl_id: str) -> str:
    cid = str(chembl_id).strip().upper()
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{cid}.json"
    obj = _http_get_json(url)
    smiles = ((obj.get("molecule_structures") or {}).get("canonical_smiles") or "").strip()
    if not smiles:
        raise RuntimeError(f"No canonical SMILES found for ChEMBL id={cid}")
    return smiles


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def _write_rdkit_sdf_no_kekulize(mol, out_sdf: Path) -> None:
    from rdkit import Chem

    block = Chem.MolToMolBlock(mol, kekulize=False)
    with open(out_sdf, "w") as fh:
        fh.write(block)
        for prop in mol.GetPropNames():
            fh.write(f">  <{prop}>\n{mol.GetProp(prop)}\n\n")
        fh.write("$$$$\n")


def _smiles_to_sdf(smiles: str, out_sdf: Path) -> None:
    smi = str(smiles or "").strip()
    if not smi:
        raise RuntimeError("Empty SMILES string")
    if _rdkit_available():
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError("RDKit could not parse SMILES")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 13
        if AllChem.EmbedMolecule(mol, params) != 0:
            # Fallback to 2D coordinates still keeps a valid structure for downstream conversion.
            AllChem.Compute2DCoords(mol)
        else:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=300)
            except Exception:
                pass
        mol = Chem.RemoveHs(mol)
        _write_rdkit_sdf_no_kekulize(mol, out_sdf)
        return

    if not shutil.which("obabel"):
        raise RuntimeError("Cannot build ligand from SMILES: neither RDKit nor Open Babel is available")
    _run(["obabel", f"-:{smi}", "-osdf", "-O", str(out_sdf), "--gen3d"])


def _convert_ligand_to_sdf(src: Path, out_sdf: Path) -> None:
    ext = src.suffix.lower()
    if ext == ".sdf":
        out_sdf.write_bytes(src.read_bytes())
        return
    if ext in {".smi", ".smiles", ".txt"}:
        line = _first_nonempty_line(src)
        if not line:
            raise RuntimeError(f"No SMILES found in {src}")
        smi = line.split()[0]
        _smiles_to_sdf(smi, out_sdf)
        return

    if _rdkit_available():
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = None
        try:
            if ext == ".mol2":
                mol = Chem.MolFromMol2File(str(src), sanitize=True, removeHs=False)
            elif ext == ".mol":
                mol = Chem.MolFromMolFile(str(src), sanitize=True, removeHs=False)
            elif ext == ".pdb":
                mol = Chem.MolFromPDBFile(str(src), sanitize=True, removeHs=False)
        except Exception:
            mol = None
        if mol is not None:
            if mol.GetNumConformers() == 0:
                try:
                    mol = Chem.AddHs(mol)
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 13
                    AllChem.EmbedMolecule(mol, params)
                    AllChem.UFFOptimizeMolecule(mol, maxIters=300)
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    pass
            _write_rdkit_sdf_no_kekulize(mol, out_sdf)
            return

    if not shutil.which("obabel"):
        raise RuntimeError(f"Cannot convert ligand to SDF from {src}: missing converter")
    in_fmt = _resolve_obabel_format(src)
    if not in_fmt:
        raise RuntimeError(f"Cannot infer ligand input format for {src}")
    cmd = ["obabel", f"-i{in_fmt}", str(src), "-osdf", "-O", str(out_sdf)]
    if ext in {".smi", ".smiles", ".txt"}:
        cmd.append("--gen3d")
    _run(cmd)


def _sdf_to_pdb(src_sdf: Path, out_pdb: Path) -> None:
    if _rdkit_available():
        from rdkit import Chem

        suppl = Chem.SDMolSupplier(str(src_sdf), removeHs=False)
        mol = None
        for m in suppl:
            if m is not None:
                mol = m
                break
        if mol is not None:
            block = Chem.MolToPDBBlock(mol)
            out_pdb.write_text(block)
            return

    if not shutil.which("obabel"):
        _sdf_to_pdb_fallback(src_sdf, out_pdb)
        if out_pdb.exists() and out_pdb.stat().st_size > 0:
            return
        raise RuntimeError("Cannot convert ligand SDF to PDB: missing converter")
    _run(["obabel", str(src_sdf), "-opdb", "-O", str(out_pdb)])


def _sdf_to_pdb_fallback(src_sdf: Path, out_pdb: Path) -> None:
    text = src_sdf.read_text(errors="ignore")
    block = text.split("$$$$", 1)[0]
    lines = [ln.rstrip("\n") for ln in block.splitlines()]
    if len(lines) < 4:
        raise RuntimeError(f"SDF parse failed (too short): {src_sdf}")

    atoms: List[Tuple[str, float, float, float]] = []
    if any("V3000" in ln for ln in lines[:6]):
        try:
            b = lines.index("M  V30 BEGIN ATOM")
            e = lines.index("M  V30 END ATOM")
            for ln in lines[b + 1 : e]:
                parts = ln.split()
                if len(parts) < 7:
                    continue
                elem = parts[3].strip() or "C"
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                atoms.append((elem, x, y, z))
        except Exception as exc:
            raise RuntimeError(f"V3000 SDF parse failed: {src_sdf}") from exc
    else:
        try:
            nat = int(lines[3][:3])
        except Exception as exc:
            raise RuntimeError(f"V2000 SDF header parse failed: {src_sdf}") from exc
        for i in range(4, 4 + nat):
            if i >= len(lines):
                break
            ln = lines[i]
            try:
                x = float(ln[0:10])
                y = float(ln[10:20])
                z = float(ln[20:30])
                elem = (ln[31:34].strip() or "C")
            except Exception:
                continue
            atoms.append((elem, x, y, z))

    if not atoms:
        raise RuntimeError(f"SDF had no atoms for fallback conversion: {src_sdf}")

    out_lines: List[str] = []
    out_lines.append("HETATM generated from SDF")
    out_lines.append("INPUT_PREP")
    out_lines.append("")
    for idx, (elem, x, y, z) in enumerate(atoms, start=1):
        atom_name = f"{elem[:2].upper():>2}"
        resname = "LIG"
        out_lines.append(
            f"HETATM{idx:5d} {atom_name:>4} {resname:>3} A{1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {elem[:2].upper():>2}"
        )
    out_lines.append("END")
    out_pdb.write_text("\n".join(out_lines) + "\n")


def _ligand_smiles_from_sdf(src_sdf: Path) -> Optional[str]:
    if _rdkit_available():
        try:
            from rdkit import Chem

            suppl = Chem.SDMolSupplier(str(src_sdf), removeHs=False)
            for m in suppl:
                if m is None:
                    continue
                smi = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
                if smi:
                    return smi
        except Exception:
            pass
    if shutil.which("obabel"):
        try:
            out = subprocess.check_output(
                ["obabel", "-isdf", str(src_sdf), "-osmi"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            out = ""
        if out:
            first = out.splitlines()[0].strip()
            if first:
                return first.split()[0]
    return None


def _count_heavy_atoms_in_sdf(path: Path) -> int:
    if _rdkit_available():
        try:
            from rdkit import Chem

            suppl = Chem.SDMolSupplier(str(path), removeHs=False)
            for m in suppl:
                if m is None:
                    continue
                return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
        except Exception:
            pass
    text = path.read_text(errors="ignore")
    block = text.split("$$$$", 1)[0]
    lines = block.splitlines()
    if len(lines) < 4:
        return 0
    if any("V3000" in ln for ln in lines[:5]):
        try:
            b = lines.index("M  V30 BEGIN ATOM")
            e = lines.index("M  V30 END ATOM")
            count = 0
            for ln in lines[b + 1 : e]:
                parts = ln.split()
                if len(parts) >= 7:
                    elem = parts[3].upper()
                    if elem != "H":
                        count += 1
            return count
        except Exception:
            return 0
    try:
        nat = int(lines[3][:3])
        count = 0
        for i in range(4, 4 + nat):
            if i >= len(lines):
                break
            elem = lines[i][31:34].strip().upper()
            if elem and elem != "H":
                count += 1
        return count
    except Exception:
        return 0


def _resolve_spec_arg(
    cli_value: Optional[str],
    spec: Dict,
    section: str,
    key: str,
) -> Optional[str]:
    if cli_value is not None:
        return cli_value
    sec = spec.get(section)
    if not isinstance(sec, dict):
        return None
    v = sec.get(key)
    return str(v) if isinstance(v, str) else None


def _resolve_input_manifest(outdir: Path) -> Dict:
    return _read_json(outdir / "swarm" / "input_manifest.json")


def _resolve_existing_manifest_path(outdir: Path, raw: object) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw.strip())
    if not p.is_absolute():
        p = outdir / p
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare canonical protein/ligand inputs for SWARM + fpocket from local files or external APIs."
    )
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--input-spec", default=None, help="Optional JSON spec for protein/ligand sources.")

    ap.add_argument(
        "--protein-source",
        choices=["auto", "local_pdb", "local_mmcif", "alphafold_uniprot", "rcsb_pdb"],
        default=None,
    )
    ap.add_argument("--protein-path", default=None)
    ap.add_argument("--protein-id", default=None, help="UniProt accession for AlphaFold or 4-char PDB id for RCSB.")
    ap.add_argument("--protein-chain", default=None, help="Preferred chain for sequence extraction when needed.")
    ap.add_argument("--af-versions", default="v6,v5,v4,v3")

    ap.add_argument(
        "--fasta-source",
        choices=["auto", "local_fasta", "uniprot_accession", "rcsb_pdb", "none"],
        default=None,
    )
    ap.add_argument("--fasta-path", default=None)
    ap.add_argument("--fasta-accession", default=None, help="UniProt accession for FASTA fetch.")

    ap.add_argument(
        "--ligand-source",
        choices=[
            "auto",
            "local_sdf",
            "local_mol2",
            "local_mol",
            "local_pdb",
            "local_smiles",
            "pubchem_cid",
            "pubchem_name",
            "chembl_id",
            "smiles",
        ],
        default=None,
    )
    ap.add_argument("--ligand-path", default=None)
    ap.add_argument("--ligand-id", default=None, help="CID, PubChem name, or ChEMBL id depending on --ligand-source.")
    ap.add_argument("--ligand-smiles", default=None)

    ap.add_argument("--protein-out", default="reference_protein.pdb")
    ap.add_argument("--fasta-out", default="enzyme_wt.fasta")
    ap.add_argument("--ligand-sdf-out", default="ligand.sdf")
    ap.add_argument("--ligand-pdb-out", default="ligand.pdb")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(os.environ.get("OUTDIR", "./data"))
    outdir.mkdir(parents=True, exist_ok=True)
    swarm_dir = outdir / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = swarm_dir / "_tmp_input_prep"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    spec = _read_json(Path(args.input_spec)) if args.input_spec else {}
    existing_manifest = _resolve_input_manifest(outdir)

    protein_source = _resolve_spec_arg(args.protein_source, spec, "protein", "source") or "auto"
    protein_path_raw = _resolve_spec_arg(args.protein_path, spec, "protein", "path")
    protein_id = _resolve_spec_arg(args.protein_id, spec, "protein", "id")
    preferred_chain = _resolve_spec_arg(args.protein_chain, spec, "protein", "chain")
    af_versions = [v.strip() for v in str(args.af_versions).split(",") if v.strip()]

    fasta_source = _resolve_spec_arg(args.fasta_source, spec, "fasta", "source") or "auto"
    fasta_path_raw = _resolve_spec_arg(args.fasta_path, spec, "fasta", "path")
    fasta_accession = _resolve_spec_arg(args.fasta_accession, spec, "fasta", "accession")

    ligand_source = _resolve_spec_arg(args.ligand_source, spec, "ligand", "source") or "auto"
    ligand_path_raw = _resolve_spec_arg(args.ligand_path, spec, "ligand", "path")
    ligand_id = _resolve_spec_arg(args.ligand_id, spec, "ligand", "id")
    ligand_smiles = _resolve_spec_arg(args.ligand_smiles, spec, "ligand", "smiles")

    protein_out = outdir / args.protein_out
    fasta_out = outdir / args.fasta_out
    ligand_sdf_out = outdir / args.ligand_sdf_out
    ligand_pdb_out = outdir / args.ligand_pdb_out

    warnings: List[str] = []
    protein_meta: Dict[str, object] = {}
    fasta_meta: Dict[str, object] = {}
    ligand_meta: Dict[str, object] = {}

    # ---------- Protein ----------
    src_pdb = tmp_dir / "protein_source.pdb"
    if protein_source == "auto":
        manifest_protein = _resolve_existing_manifest_path(
            outdir,
            (existing_manifest.get("protein", {}) or {}).get("canonical_pdb"),
        )
        auto_candidates = [
            protein_out,
            outdir / "reference_protein.pdb",
            outdir / "AF-{}.pdb".format(protein_id) if protein_id else None,
            manifest_protein,
        ]
        af_glob = sorted(outdir.glob("AF-*-model_*.pdb"))
        for p in af_glob:
            auto_candidates.append(p)
        explicit_pdb = outdir / "protein.pdb"
        if explicit_pdb.exists():
            auto_candidates.append(explicit_pdb)
        for c in auto_candidates:
            if not c:
                continue
            cp = Path(c)
            if cp.exists() and cp.is_file() and cp.stat().st_size > 0:
                src_pdb.write_bytes(cp.read_bytes())
                protein_source = "local_pdb"
                protein_path_raw = str(cp)
                break
        if protein_source == "auto":
            if protein_id:
                if re.fullmatch(r"[0-9A-Za-z]{4}", protein_id):
                    protein_source = "rcsb_pdb"
                else:
                    protein_source = "alphafold_uniprot"
            else:
                raise RuntimeError(
                    "protein-source=auto could not resolve a local protein structure; provide --protein-source and --protein-id/path"
                )

    if protein_source in {"local_pdb", "local_mmcif"}:
        if not protein_path_raw:
            raise RuntimeError(f"protein-source={protein_source} requires --protein-path")
        source_path = Path(protein_path_raw)
        if not source_path.exists():
            raise RuntimeError(f"Protein path not found: {source_path}")
        if protein_source == "local_pdb":
            src_pdb.write_bytes(source_path.read_bytes())
            protein_meta["source_path"] = str(source_path)
        else:
            _convert_mmcif_to_pdb(source_path, src_pdb)
            protein_meta["source_path"] = str(source_path)
    elif protein_source == "alphafold_uniprot":
        if not protein_id:
            # try infer from existing fasta first
            protein_id = _infer_uniprot_from_fasta(fasta_out) or _infer_uniprot_from_fasta(outdir / "enzyme_wt.fasta")
        if not protein_id:
            raise RuntimeError("protein-source=alphafold_uniprot requires UniProt accession via --protein-id or FASTA header")
        af_url, af_kind = _download_alphafold_model(protein_id, src_pdb, af_versions)
        protein_meta["download_url"] = af_url
        protein_meta["download_kind"] = af_kind
        protein_meta["uniprot_accession"] = protein_id
    elif protein_source == "rcsb_pdb":
        if not protein_id:
            raise RuntimeError("protein-source=rcsb_pdb requires --protein-id (4-character PDB id)")
        raw_model, model_url = _download_rcsb_model(protein_id, tmp_dir)
        if raw_model.suffix.lower() == ".cif":
            _convert_mmcif_to_pdb(raw_model, src_pdb)
        else:
            src_pdb.write_bytes(raw_model.read_bytes())
        protein_meta["download_url"] = model_url
        protein_meta["pdb_id"] = protein_id.upper()
    else:
        raise RuntimeError(f"Unsupported protein source: {protein_source}")

    norm_stats = _normalize_protein_pdb(src_pdb, protein_out)
    protein_meta.update(norm_stats)
    if int(norm_stats.get("has_model_records", 0)) > 0:
        warnings.append("Protein structure contained MODEL records; only the first model was retained.")
    pdb_seq_for_check, pdb_seq_chain = _sequence_from_pdb(protein_out, preferred_chain)

    # ---------- FASTA ----------
    fasta_seq = ""
    fasta_header = "enzyme_wt"
    if fasta_source == "auto":
        if fasta_path_raw:
            fasta_source = "local_fasta"
        elif fasta_out.exists() and fasta_out.stat().st_size > 0:
            existing_seq, _existing_header = _read_fasta_sequence(fasta_out)
            if _sequences_compatible(existing_seq, pdb_seq_for_check):
                fasta_source = "local_fasta"
                fasta_path_raw = str(fasta_out)
            else:
                warnings.append(
                    "Ignoring existing FASTA because it is inconsistent with canonical protein PDB "
                    f"(len_fasta={len(existing_seq)} len_pdb_chain={len(pdb_seq_for_check)} chain={pdb_seq_chain or 'A'})."
                )
                if protein_source == "alphafold_uniprot" and protein_id:
                    fasta_source = "uniprot_accession"
                    fasta_accession = protein_id
                elif protein_source == "rcsb_pdb" and protein_id:
                    fasta_source = "rcsb_pdb"
                    fasta_accession = protein_id
                else:
                    fasta_source = "none"
        elif protein_source == "alphafold_uniprot" and protein_id:
            fasta_source = "uniprot_accession"
            fasta_accession = protein_id
        elif protein_source == "rcsb_pdb" and protein_id:
            fasta_source = "rcsb_pdb"
            fasta_accession = protein_id
        else:
            fasta_source = "none"

    if fasta_source == "local_fasta":
        if not fasta_path_raw:
            raise RuntimeError("fasta-source=local_fasta requires --fasta-path")
        src_fasta = Path(fasta_path_raw)
        if not src_fasta.exists():
            raise RuntimeError(f"FASTA path not found: {src_fasta}")
        fasta_out.write_bytes(src_fasta.read_bytes())
        lines = [ln.strip() for ln in fasta_out.read_text().splitlines() if ln.strip()]
        if lines and lines[0].startswith(">"):
            fasta_header = lines[0][1:].strip() or fasta_header
        fasta_seq = "".join(ln for ln in lines if not ln.startswith(">"))
        if not _sequences_compatible(fasta_seq, pdb_seq_for_check):
            pid = _prefix_identity(fasta_seq.upper(), pdb_seq_for_check.upper())
            raise RuntimeError(
                "FASTA/PDB mismatch detected. Refusing to proceed to avoid stale/misaligned SWARM outputs. "
                f"(len_fasta={len(fasta_seq)} len_pdb_chain={len(pdb_seq_for_check)} "
                f"prefix_identity={pid:.3f} chain={pdb_seq_chain or 'A'})"
            )
    elif fasta_source == "uniprot_accession":
        acc = fasta_accession or protein_id
        if not acc:
            raise RuntimeError("fasta-source=uniprot_accession requires --fasta-accession or --protein-id")
        _download_uniprot_fasta(acc, fasta_out)
        lines = [ln.strip() for ln in fasta_out.read_text().splitlines() if ln.strip()]
        fasta_header = lines[0][1:].strip() if lines and lines[0].startswith(">") else f"sp|{acc}|"
        fasta_seq = "".join(ln for ln in lines if not ln.startswith(">"))
    elif fasta_source == "rcsb_pdb":
        pid = fasta_accession or protein_id
        if not pid:
            raise RuntimeError("fasta-source=rcsb_pdb requires --fasta-accession or --protein-id")
        if not _download_rcsb_fasta(pid, fasta_out):
            warnings.append("RCSB FASTA download failed; using sequence inferred from normalized PDB.")
            seq, seq_chain = _sequence_from_pdb(protein_out, preferred_chain)
            if not seq:
                raise RuntimeError("Could not infer sequence from normalized protein PDB")
            fasta_header = f"{pid.upper()}|chain_{seq_chain or 'A'}"
            _write_fasta(seq, fasta_out, fasta_header)
        lines = [ln.strip() for ln in fasta_out.read_text().splitlines() if ln.strip()]
        fasta_header = lines[0][1:].strip() if lines and lines[0].startswith(">") else f"{pid.upper()}"
        fasta_seq = "".join(ln for ln in lines if not ln.startswith(">"))
    elif fasta_source == "none":
        seq, seq_chain = _sequence_from_pdb(protein_out, preferred_chain)
        if not seq:
            raise RuntimeError("No FASTA source available and sequence inference from PDB failed")
        fasta_header = f"derived|chain_{seq_chain or 'A'}"
        _write_fasta(seq, fasta_out, fasta_header)
        fasta_seq = seq
        warnings.append("FASTA was derived from PDB coordinates; missing residues not present in structure may be absent.")
    else:
        raise RuntimeError(f"Unsupported FASTA source: {fasta_source}")

    inferred_uniprot = _infer_uniprot_from_fasta(fasta_out)
    if inferred_uniprot and not protein_id:
        protein_id = inferred_uniprot

    fasta_meta = {
        "source": fasta_source,
        "header": fasta_header,
        "length": len(fasta_seq),
        "uniprot_accession": inferred_uniprot,
        "canonical_fasta": str(fasta_out),
        "pdb_chain_for_alignment": pdb_seq_chain,
        "pdb_chain_sequence_length": len(pdb_seq_for_check),
        "pdb_fasta_compatible": bool(_sequences_compatible(fasta_seq, pdb_seq_for_check)),
    }

    # ---------- Ligand ----------
    ligand_tmp_sdf = tmp_dir / "ligand_source.sdf"
    ligand_tmp_meta: Dict[str, object] = {}

    if ligand_source == "auto":
        if ligand_path_raw:
            lp = Path(ligand_path_raw)
            ext = lp.suffix.lower()
            if ext == ".sdf":
                ligand_source = "local_sdf"
            elif ext == ".mol2":
                ligand_source = "local_mol2"
            elif ext == ".mol":
                ligand_source = "local_mol"
            elif ext == ".pdb":
                ligand_source = "local_pdb"
            elif ext in {".smi", ".smiles", ".txt"}:
                ligand_source = "local_smiles"
            else:
                raise RuntimeError(f"Unsupported ligand file extension for auto source: {lp.suffix}")
        elif ligand_smiles:
            ligand_source = "smiles"
        elif ligand_id:
            if re.fullmatch(r"\d+", str(ligand_id).strip()):
                ligand_source = "pubchem_cid"
            elif str(ligand_id).upper().startswith("CHEMBL"):
                ligand_source = "chembl_id"
            else:
                ligand_source = "pubchem_name"
        else:
            local_candidates = [
                outdir / "ligand.sdf",
                outdir / "ligand.mol2",
                outdir / "ligand.mol",
                outdir / "ligand.pdb",
            ]
            found = next((p for p in local_candidates if p.exists() and p.stat().st_size > 0), None)
            if found:
                ligand_path_raw = str(found)
                ext = found.suffix.lower()
                ligand_source = {
                    ".sdf": "local_sdf",
                    ".mol2": "local_mol2",
                    ".mol": "local_mol",
                    ".pdb": "local_pdb",
                }[ext]
            else:
                raise RuntimeError(
                    "ligand-source=auto could not resolve ligand input; provide --ligand-source with --ligand-id/path/smiles"
                )

    if ligand_source in {"local_sdf", "local_mol2", "local_mol", "local_pdb", "local_smiles"}:
        if not ligand_path_raw:
            raise RuntimeError(f"ligand-source={ligand_source} requires --ligand-path")
        lp = Path(ligand_path_raw)
        if not lp.exists():
            raise RuntimeError(f"Ligand path not found: {lp}")
        _convert_ligand_to_sdf(lp, ligand_tmp_sdf)
        ligand_tmp_meta["source_path"] = str(lp)
    elif ligand_source == "pubchem_cid":
        if not ligand_id:
            raise RuntimeError("ligand-source=pubchem_cid requires --ligand-id")
        url = _download_pubchem_sdf(str(ligand_id), ligand_tmp_sdf)
        ligand_tmp_meta["download_url"] = url
        ligand_tmp_meta["pubchem_cid"] = str(ligand_id)
    elif ligand_source == "pubchem_name":
        if not ligand_id:
            raise RuntimeError("ligand-source=pubchem_name requires --ligand-id (name)")
        cid = _pubchem_cid_from_name(str(ligand_id))
        url = _download_pubchem_sdf(cid, ligand_tmp_sdf)
        ligand_tmp_meta["download_url"] = url
        ligand_tmp_meta["pubchem_name"] = str(ligand_id)
        ligand_tmp_meta["pubchem_cid"] = cid
    elif ligand_source == "chembl_id":
        if not ligand_id:
            raise RuntimeError("ligand-source=chembl_id requires --ligand-id")
        smi = _chembl_smiles(str(ligand_id))
        _smiles_to_sdf(smi, ligand_tmp_sdf)
        ligand_tmp_meta["chembl_id"] = str(ligand_id).upper()
        ligand_tmp_meta["smiles"] = smi
    elif ligand_source == "smiles":
        smi = ligand_smiles or ""
        if not smi:
            raise RuntimeError("ligand-source=smiles requires --ligand-smiles")
        _smiles_to_sdf(smi, ligand_tmp_sdf)
        ligand_tmp_meta["smiles"] = smi
    else:
        raise RuntimeError(f"Unsupported ligand source: {ligand_source}")

    # Canonicalize ligand outputs.
    ligand_sdf_out.write_bytes(ligand_tmp_sdf.read_bytes())
    heavy_atoms = _count_heavy_atoms_in_sdf(ligand_sdf_out)
    if heavy_atoms <= 0:
        raise RuntimeError(f"Canonical ligand SDF has no heavy atoms: {ligand_sdf_out}")
    _sdf_to_pdb(ligand_sdf_out, ligand_pdb_out)
    ligand_smiles_out = _ligand_smiles_from_sdf(ligand_sdf_out)
    if not ligand_smiles_out:
        warnings.append("Could not derive ligand SMILES from canonical SDF.")

    ligand_meta = {
        "source": ligand_source,
        "canonical_sdf": str(ligand_sdf_out),
        "canonical_pdb": str(ligand_pdb_out),
        "heavy_atom_count": int(heavy_atoms),
        "smiles": ligand_smiles_out,
    }
    ligand_meta.update(ligand_tmp_meta)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protein": {
            "source": protein_source,
            "id": protein_id,
            "canonical_pdb": str(protein_out),
            "preferred_chain": preferred_chain,
            **protein_meta,
        },
        "fasta": fasta_meta,
        "ligand": ligand_meta,
        "warnings": warnings,
        "notes": [
            "Canonical receptor is normalized for fpocket/SWARM (first model, filtered altloc/hydrogen/water/non-protein HETATM).",
            "Canonical ligand is stored as ligand.sdf + ligand.pdb for downstream conversion/docking.",
        ],
    }
    manifest_path = swarm_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # Cleanup temp directory best-effort.
    for p in tmp_dir.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"Wrote canonical protein PDB: {protein_out}")
    print(f"Wrote canonical FASTA: {fasta_out}")
    print(f"Wrote canonical ligand SDF: {ligand_sdf_out}")
    print(f"Wrote canonical ligand PDB: {ligand_pdb_out}")
    print(f"Wrote input manifest: {manifest_path}")
    if warnings:
        print("[input-prep] Warnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
