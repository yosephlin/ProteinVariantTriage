import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .fetch_uniprot import fetch_uniprot, parse_uniprot_entry
from .fetch_pdbe import (
    fetch_ligand_sites,
    parse_ligand_sites,
    fetch_interface_residues,
    parse_interface_residues,
    fetch_annotations,
    parse_annotations,
)
from .fetch_interpro import fetch_interpro_all, parse_interpro
from .fetch_chembl import search_target, select_target, fetch_activities, build_ligand_priors
from .fetch_pubchem import fetch_smiles_by_name, fetch_smiles_by_cid, extract_smiles
from .resolve_uniprot import search_uniprot
from .fetch_mcsa import fetch_mcsa_residues, parse_mcsa_residues
from .fetch_hmmer import build_hmmer_evolution_context
from .http import OfflineError


def merge_pdbe(residues, pdbe_map: Dict[int, Dict[str, Any]]):
    for pos, info in pdbe_map.items():
        if 1 <= pos <= len(residues):
            residues[pos - 1]["pdbe"]["ligand_sites"] = info


def merge_pdbe_interface(residues, pdbe_map: Dict[int, Dict[str, Any]]):
    for pos, info in pdbe_map.items():
        if 1 <= pos <= len(residues):
            residues[pos - 1]["pdbe"]["interface"] = info


def merge_pdbe_annotations(residues, pdbe_map: Dict[int, Dict[str, Any]]):
    for pos, info in pdbe_map.items():
        if 1 <= pos <= len(residues):
            residues[pos - 1]["pdbe"]["predicted_sites"] = info.get("predicted_sites", [])


def build_context(
    accession: str,
    outdir: Path,
    cache_dir: Path,
    offline: bool = False,
    include_chembl: bool = False,
    chembl_query: str = "",
    chembl_organism: str = "",
    ligand_name: str = "",
    ligand_cid: str = "",
    include_mcsa: bool = False,
    include_hmmer: bool = False,
    hmmer_max_hits: int = 120,
    hmmer_min_identity: float = 0.25,
    hmmer_min_coverage: float = 0.70,
    hmmer_max_homologs: int = 64,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entry = fetch_uniprot(accession, cache_dir=cache_dir, offline=offline)
    parsed = parse_uniprot_entry(entry)
    context = parsed["context"]
    residues = parsed["residues"]

    # PDBe ligand sites (best-effort)
    try:
        pdbe_raw = fetch_ligand_sites(accession, cache_dir=cache_dir, offline=offline)
        pdbe_map = parse_ligand_sites(pdbe_raw, accession)
        merge_pdbe(residues, pdbe_map)
        context["pdbe"] = {
            "ligand_sites": {
                "residue_count": len(pdbe_map),
            }
        }
        # summarize known ligands
        lig_counts = {}
        for info in pdbe_map.values():
            for lig in info.get("ligands", []) or []:
                lig_counts[lig] = lig_counts.get(lig, 0) + 1
        top_ligs = sorted(lig_counts.items(), key=lambda kv: kv[1], reverse=True)
        context["pdbe"]["known_ligands"] = [{"id": k, "residue_hits": v} for k, v in top_ligs[:25]]
    except OfflineError:
        context["pdbe"] = {"ligand_sites": {"residue_count": 0, "note": "offline_no_cache"}}
    except Exception:
        context["pdbe"] = {"ligand_sites": {"residue_count": 0, "note": "parse_error"}}

    # PDBe interface residues (best-effort)
    try:
        pdbe_int_raw = fetch_interface_residues(accession, cache_dir=cache_dir, offline=offline)
        pdbe_int_map = parse_interface_residues(pdbe_int_raw, accession)
        merge_pdbe_interface(residues, pdbe_int_map)
        context["pdbe"]["interface_residues"] = {
            "residue_count": len(pdbe_int_map),
        }
    except OfflineError:
        context.setdefault("pdbe", {})
        context["pdbe"]["interface_residues"] = {"residue_count": 0, "note": "offline_no_cache"}
    except Exception:
        context.setdefault("pdbe", {})
        context["pdbe"]["interface_residues"] = {"residue_count": 0, "note": "parse_error"}

    # PDBe annotations (best-effort)
    try:
        pdbe_ann_raw = fetch_annotations(accession, cache_dir=cache_dir, offline=offline)
        pdbe_ann_map = parse_annotations(pdbe_ann_raw, accession)
        merge_pdbe_annotations(residues, pdbe_ann_map)
        context["pdbe"]["annotations"] = {
            "residue_count": len(pdbe_ann_map),
        }
    except OfflineError:
        context.setdefault("pdbe", {})
        context["pdbe"]["annotations"] = {"residue_count": 0, "note": "offline_no_cache"}
    except Exception:
        context.setdefault("pdbe", {})
        context["pdbe"]["annotations"] = {"residue_count": 0, "note": "parse_error"}

    # InterPro domains (best-effort)
    try:
        pages = fetch_interpro_all(accession, cache_dir=cache_dir, offline=offline)
        domains = []
        for page in pages:
            domains.extend(parse_interpro(page))
        context["interpro"] = {
            "domains": domains,
            "domain_count": len(domains),
        }
        # attach per-residue domain labels
        for d in domains:
            entry_id = d.get("entry_id")
            for interval in d.get("intervals", []) or []:
                start = interval.get("start")
                end = interval.get("end")
                if not start or not end:
                    continue
                for pos in range(int(start), int(end) + 1):
                    if 1 <= pos <= len(residues):
                        residues[pos - 1]["interpro"]["domains"].append(entry_id)
            # attach interpro sites (motifs)
            for site in d.get("sites", []) or []:
                sstart = site.get("start")
                send = site.get("end")
                if not sstart or not send:
                    continue
                for pos in range(int(sstart), int(send) + 1):
                    if 1 <= pos <= len(residues):
                        residues[pos - 1]["interpro"].setdefault("sites", []).append({
                            "entry_id": entry_id,
                            "type": site.get("type"),
                            "start": sstart,
                            "end": send,
                        })
    except OfflineError:
        context["interpro"] = {"domains": [], "domain_count": 0, "note": "offline_no_cache"}
    except Exception:
        context["interpro"] = {"domains": [], "domain_count": 0, "note": "parse_error"}

    # M-CSA catalytic residues (best-effort)
    if include_mcsa:
        try:
            mcsa_raw = fetch_mcsa_residues(accession, cache_dir=cache_dir, offline=offline)
            mcsa_parsed = parse_mcsa_residues(mcsa_raw, accession)
            context["mcsa"] = mcsa_parsed.get("summary") or {}
            for pos, items in (mcsa_parsed.get("residue_annotations") or {}).items():
                try:
                    ipos = int(pos)
                except Exception:
                    continue
                if not (1 <= ipos <= len(residues)):
                    continue
                r = residues[ipos - 1]
                r.setdefault("mcsa", {})
                r["mcsa"].setdefault("critical", [])
                r["mcsa"].setdefault("transferred", [])
                for it in items:
                    mtype = it.get("match_type")
                    if mtype == "accession":
                        r["mcsa"]["critical"].append(it)
                        # Keep compatibility with existing hard-constraint handling.
                        r["uniprot"]["critical"].append(
                            {
                                "source": "mcsa",
                                "type": "MCSA_CATALYTIC",
                                "pos_start": ipos,
                                "pos_end": ipos,
                                "severity": "critical",
                                "note": it.get("note", ""),
                                "evidence": {"codes": []},
                            }
                        )
                        r["policy"]["do_not_mutate"] = True
                        if not r["policy"].get("reason"):
                            r["policy"]["reason"] = "MCSA_CATALYTIC"
                    else:
                        r["mcsa"]["transferred"].append(it)
                        r["uniprot"]["soft"].append(
                            {
                                "source": "mcsa",
                                "type": "MCSA_HOMOLOG_CATALYTIC",
                                "pos_start": ipos,
                                "pos_end": ipos,
                                "severity": "medium",
                                "note": it.get("note", ""),
                                "evidence": {"codes": []},
                            }
                        )
        except OfflineError:
            context["mcsa"] = {"note": "offline_no_cache"}
        except Exception:
            context["mcsa"] = {"note": "parse_error"}
    else:
        context["mcsa"] = {"note": "not_requested"}

    # HMMER-derived evolutionary priors (best-effort)
    if include_hmmer:
        query_seq = context.get("uniprot", {}).get("sequence") or ""
        try:
            evo = build_hmmer_evolution_context(
                query_seq=query_seq,
                cache_dir=cache_dir,
                offline=offline,
                max_hits=hmmer_max_hits,
                min_identity=hmmer_min_identity,
                min_coverage=hmmer_min_coverage,
                max_homologs=hmmer_max_homologs,
            )
            context["evolution"] = {
                "method": evo.get("method"),
                "hits_considered": evo.get("hits_considered"),
                "homolog_sequences": evo.get("homolog_sequences"),
                "homologs_used": evo.get("homologs_used"),
                "position_count": len(evo.get("positions") or []),
                "top_hits": evo.get("top_hits") or [],
            }
            for p in evo.get("positions") or []:
                ipos = p.get("pos")
                if not isinstance(ipos, int) or not (1 <= ipos <= len(residues)):
                    continue
                r = residues[ipos - 1]
                r["evolution"] = {
                    "homolog_count": p.get("homolog_count"),
                    "conservation": p.get("conservation"),
                    "allowed_aas": p.get("allowed_aas") or [r.get("wt")],
                    "top_aas": p.get("top_aas") or [],
                }
        except OfflineError:
            context["evolution"] = {"note": "offline_no_cache"}
        except Exception:
            context["evolution"] = {"note": "parse_error"}
    else:
        context["evolution"] = {"note": "not_requested"}

    # summary stats
    critical_positions = sum(1 for r in residues if r["policy"]["do_not_mutate"])
    context["summary"] = {
        "total_residues": len(residues),
        "critical_positions": critical_positions,
    }

    # policy memo (deterministic summary)
    critical_types = {}
    pdbe_iface = 0
    pdbe_lig = 0
    for r in residues:
        for c in r.get("uniprot", {}).get("critical", []):
            t = c.get("type")
            if t:
                critical_types[t] = critical_types.get(t, 0) + 1
        if r.get("pdbe", {}).get("interface", {}).get("is_interface"):
            pdbe_iface += 1
        if r.get("pdbe", {}).get("ligand_sites", {}).get("count", 0) > 0:
            pdbe_lig += 1
    memo_parts = [
        f"critical_positions={critical_positions}",
        f"critical_types={{ {', '.join(f'{k}:{v}' for k, v in sorted(critical_types.items()))} }}",
        f"pdbe_interface_residues={pdbe_iface}",
        f"pdbe_ligand_contact_residues={pdbe_lig}",
    ]
    context["policy_memo"] = "; ".join(memo_parts)

    # Optional: ChEMBL ligand priors
    if include_chembl:
        try:
            query = chembl_query or accession
            chembl_search = search_target(query, cache_dir=cache_dir, offline=offline)
            target = select_target(chembl_search, organism_name=chembl_organism or context.get("uniprot", {}).get("organism"))
            if target:
                target_id = target.get("target_chembl_id")
                activities = fetch_activities(target_id, cache_dir=cache_dir, offline=offline)
                priors = build_ligand_priors(activities)
                context["chembl"] = {
                    "target": target,
                    "ligand_priors": priors,
                }
            else:
                context["chembl"] = {"note": "no_target_found"}
        except OfflineError:
            context["chembl"] = {"note": "offline_no_cache"}
        except Exception:
            context["chembl"] = {"note": "parse_error"}
    else:
        context["chembl"] = {"note": "not_requested"}

    # Optional: PubChem ligand smiles
    if ligand_name or ligand_cid:
        try:
            if ligand_name:
                pc = fetch_smiles_by_name(ligand_name, cache_dir=cache_dir, offline=offline)
            else:
                pc = fetch_smiles_by_cid(ligand_cid, cache_dir=cache_dir, offline=offline)
            smiles = extract_smiles(pc)
            context.setdefault("ligand", {})
            context["ligand"]["smiles"] = smiles
        except OfflineError:
            context.setdefault("ligand", {})
            context["ligand"]["smiles"] = {"note": "offline_no_cache"}
        except Exception:
            context.setdefault("ligand", {})
            context["ligand"]["smiles"] = {"note": "parse_error"}
    else:
        context.setdefault("ligand", {})
        context["ligand"].setdefault("smiles", {"note": "not_requested"})

    # write outputs
    (outdir / "context_api.json").write_text(json.dumps(context, ensure_ascii=False, indent=2))
    with (outdir / "residue_constraints.jsonl").open("w", encoding="utf-8") as fh:
        for r in residues:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SWARM API context pack")
    ap.add_argument("--accession", default=None)
    ap.add_argument("--protein-name", default=None)
    ap.add_argument("--organism-id", default=None)
    ap.add_argument("--organism-name", default=None)
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--include-chembl", action="store_true")
    ap.add_argument("--include-mcsa", action="store_true")
    ap.add_argument("--include-hmmer", action="store_true")
    ap.add_argument("--hmmer-max-hits", type=int, default=120)
    ap.add_argument("--hmmer-min-identity", type=float, default=0.25)
    ap.add_argument("--hmmer-min-coverage", type=float, default=0.70)
    ap.add_argument("--hmmer-max-homologs", type=int, default=64)
    ap.add_argument("--chembl-query", default=None)
    ap.add_argument("--chembl-organism", default=None)
    ap.add_argument("--ligand-name", default=None)
    ap.add_argument("--ligand-cid", default=None)
    args = ap.parse_args()

    if args.outdir:
        base_out = Path(args.outdir)
    else:
        base_out = Path(os.environ.get("OUTDIR", "./out"))

    outdir = base_out / "swarm_api"
    cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "source_cache"

    accession = args.accession
    if not accession:
        if not args.protein_name:
            raise SystemExit("Provide --accession or --protein-name for UniProt resolution.")
        best, _ = search_uniprot(
            args.protein_name,
            args.organism_id,
            args.organism_name,
            args.reviewed_only,
            cache_dir,
            offline=args.offline,
        )
        if not best:
            raise SystemExit("No UniProt match found for the provided query.")
        accession = best.get("primaryAccession")

    build_context(
        accession,
        outdir,
        cache_dir,
        offline=args.offline,
        include_chembl=args.include_chembl,
        chembl_query=args.chembl_query or "",
        chembl_organism=args.chembl_organism or "",
        ligand_name=args.ligand_name or "",
        ligand_cid=args.ligand_cid or "",
        include_mcsa=args.include_mcsa,
        include_hmmer=args.include_hmmer,
        hmmer_max_hits=args.hmmer_max_hits,
        hmmer_min_identity=args.hmmer_min_identity,
        hmmer_min_coverage=args.hmmer_min_coverage,
        hmmer_max_homologs=args.hmmer_max_homologs,
    )
    print("Wrote:", outdir / "context_api.json")
    print("Wrote:", outdir / "residue_constraints.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
